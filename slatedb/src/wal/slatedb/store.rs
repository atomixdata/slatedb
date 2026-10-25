use std::collections::VecDeque;
use std::ops::{Bound, Range, RangeBounds};
use std::sync::Arc;

use bytes::Bytes;
use fail_parallel::{fail_point, FailPointRegistry};
use futures::future::join_all;
use futures::StreamExt;
use log::{debug, warn};
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt, PutMode, PutOptions};
use serde::Serialize;
use slatedb_common::object_metadata::IdentifiedObjectMetadata;
use slatedb_common::ObjectMetadata;

use crate::blob::ReadOnlyBlob;
use crate::db_state::{SsTableInfo, SstType};
use crate::error::SlateDBError;
use crate::flatbuffer_types::SsTableIndexOwned;
use crate::format::block::Block;
use crate::format::sst::{EncodedSsTable, SsTableFormat};
use crate::object_store_tag::{ObjectStoreCallTag, TableStoreKind};
use crate::paths::PathResolver;
use crate::sst_io::{read_with_validation_retry, ReadOnlyObject};
use crate::wal::slatedb::iterator::{excluded_range_end, WalExcludedRanges};
use crate::wal::slatedb::sst_builder::EncodedWalSsTableBuilder;

const WAL_SST_CACHED_FOOTER_SIZE: u64 = 128 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub(crate) struct WalFileId(u64);

impl WalFileId {
    pub(crate) fn value(self) -> u64 {
        self.0
    }
}

impl From<u64> for WalFileId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<WalFileId> for u64 {
    fn from(value: WalFileId) -> Self {
        value.value()
    }
}

#[derive(Clone, Debug)]
struct CachedFooter {
    offset: u64,
    data: Bytes,
}

impl CachedFooter {
    fn from_object_size(object_size: u64, data: Bytes) -> Result<Self, SlateDBError> {
        let data_len = u64::try_from(data.len()).map_err(|_| SlateDBError::InvalidDBState)?;
        let offset = object_size
            .checked_sub(data_len)
            .ok_or(SlateDBError::InvalidDBState)?;
        Ok(Self { offset, data })
    }

    fn object_size(&self) -> u64 {
        self.offset
            .checked_add(u64::try_from(self.data.len()).expect("footer size must fit in u64"))
            .expect("cached footer cannot extend beyond u64::MAX")
    }

    fn read_range(&self, range: &Range<u64>) -> Option<Bytes> {
        if range.start < self.offset || range.end > self.object_size() {
            return None;
        }

        let start = usize::try_from(range.start - self.offset).ok()?;
        let end = usize::try_from(range.end - self.offset).ok()?;
        Some(self.data.slice(start..end))
    }
}

struct WalReadOnlyObject {
    inner: ReadOnlyObject,
    cached_footer: Option<CachedFooter>,
}

impl ReadOnlyBlob for WalReadOnlyObject {
    async fn len(&self) -> Result<u64, SlateDBError> {
        match &self.cached_footer {
            Some(cached_footer) => Ok(cached_footer.object_size()),
            None => self.inner.len().await,
        }
    }

    async fn read_range(&self, range: Range<u64>) -> Result<Bytes, SlateDBError> {
        if let Some(data) = self
            .cached_footer
            .as_ref()
            .and_then(|cached_footer| cached_footer.read_range(&range))
        {
            return Ok(data);
        }
        self.inner.read_range(range).await
    }

    async fn read(&self) -> Result<Bytes, SlateDBError> {
        if let Some(cached_footer) = &self.cached_footer {
            if cached_footer.offset == 0 {
                return Ok(cached_footer.data.clone());
            }
        }
        self.inner.read().await
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WalFileHandle {
    pub(crate) id: WalFileId,
    pub(crate) format_version: u16,
    pub(crate) info: SsTableInfo,
    cached_footer: Option<CachedFooter>,
}

impl WalFileHandle {
    fn new(id: WalFileId, format_version: u16, info: SsTableInfo) -> Self {
        Self {
            id,
            format_version,
            info,
            cached_footer: None,
        }
    }

    fn with_cached_footer(mut self, offset: u64, data: Bytes) -> Self {
        self.cached_footer = Some(CachedFooter { offset, data });
        self
    }
}

/// Storage adapter for SlateDB's object-store-backed WAL files.
///
/// WALs continue to use the shared SST format and builders. This type owns only
/// WAL object-store operations and deliberately has no `DbCache` integration.
/// Its object store may be a [`CachedObjectStore`](crate::cached_object_store::CachedObjectStore);
/// WAL calls carry [`SstType::Wal`] so that wrapper can apply its WAL policy.
pub(crate) struct WalTableStore {
    object_store: Arc<dyn ObjectStore>,
    sst_format: SsTableFormat,
    path_resolver: PathResolver,
    #[allow(dead_code)]
    fp_registry: Arc<FailPointRegistry>,
    kind: TableStoreKind,
}

impl WalTableStore {
    pub(crate) fn new<P: Into<Path>>(
        object_store: Arc<dyn ObjectStore>,
        sst_format: SsTableFormat,
        root_path: P,
        kind: TableStoreKind,
    ) -> Self {
        Self::new_with_fp_registry(
            object_store,
            sst_format,
            PathResolver::from_root(root_path),
            Arc::new(FailPointRegistry::new()),
            kind,
        )
    }

    pub(crate) fn new_with_fp_registry(
        object_store: Arc<dyn ObjectStore>,
        sst_format: SsTableFormat,
        path_resolver: PathResolver,
        fp_registry: Arc<FailPointRegistry>,
        kind: TableStoreKind,
    ) -> Self {
        Self {
            object_store,
            sst_format,
            path_resolver,
            fp_registry,
            kind,
        }
    }

    pub(crate) fn table_builder(&self) -> EncodedWalSsTableBuilder {
        self.sst_format.wal_table_builder()
    }

    pub(crate) fn estimate_encoded_size(&self, num_entries: usize, size_entries: usize) -> usize {
        self.sst_format
            .estimate_encoded_size_wal(num_entries, size_entries)
    }

    /// Writes a WAL SST with create-if-absent semantics required for fencing.
    pub(crate) async fn write_sst(
        &self,
        wal_id: WalFileId,
        encoded_sst: &EncodedSsTable,
    ) -> Result<WalFileHandle, SlateDBError> {
        fail_point!(self.fp_registry.clone(), "write-wal-sst-io-error", |_| {
            Err(slatedb_io_error())
        });

        self.write_create(wal_id, encoded_sst.remaining_as_bytes())
            .await?;
        Ok(WalFileHandle::new(
            wal_id,
            encoded_sst.format_version,
            encoded_sst.info.clone(),
        ))
    }

    /// Writes a zero-byte WAL object as a fencing marker.
    pub(crate) async fn write_wal_fence(&self, wal_id: WalFileId) -> Result<(), SlateDBError> {
        fail_point!(self.fp_registry.clone(), "write-wal-sst-io-error", |_| {
            Err(slatedb_io_error())
        });
        self.write_create(wal_id, Bytes::new()).await
    }

    async fn write_create(&self, wal_id: WalFileId, data: Bytes) -> Result<(), SlateDBError> {
        let path = self.path(wal_id);
        let opts = PutOptions {
            mode: PutMode::Create,
            extensions: ObjectStoreCallTag::new(self.kind, SstType::Wal).into(),
            ..PutOptions::default()
        };
        self.object_store
            .put_opts(&path, data.into(), opts)
            .await
            .map_err(|error| match error {
                object_store::Error::AlreadyExists { .. } => {
                    debug!("path already exists [path={}]", path);
                    SlateDBError::Fenced
                }
                error => SlateDBError::from(error),
            })?;
        Ok(())
    }

    pub(crate) async fn open_sst(&self, wal_id: WalFileId) -> Result<WalFileHandle, SlateDBError> {
        let path = self.path(wal_id);
        let (info, version, cached_footer) =
            read_with_validation_retry(ObjectStoreCallTag::new(self.kind, SstType::Wal), |tag| {
                let path = path.clone();
                async move {
                    let cached_footer = self.read_cached_footer(&path, tag.clone()).await?;
                    let obj = WalReadOnlyObject {
                        inner: ReadOnlyObject {
                            object_store: Arc::clone(&self.object_store),
                            path: path.clone(),
                            tag,
                        },
                        cached_footer: Some(cached_footer.clone()),
                    };
                    let (info, version) = self
                        .sst_format
                        .read_info_and_version(&obj)
                        .await
                        .map_err(|error| error.with_path(&path))?;
                    Ok((info, version, cached_footer))
                }
            })
            .await?;
        Ok(WalFileHandle::new(wal_id, version, info)
            .with_cached_footer(cached_footer.offset, cached_footer.data))
    }

    async fn read_cached_footer(
        &self,
        path: &Path,
        tag: ObjectStoreCallTag,
    ) -> Result<CachedFooter, SlateDBError> {
        // Determine the object size before requesting the footer instead of using a suffix
        // range directly. S3 responds to a suffix-range request for a zero-byte object with
        // HTTP 200 and no Content-Range header. object_store rejects that response as the
        // private GetResultError::NotPartial, wraps it in Error::Generic, and SlateDB's object
        // store retry layer retries Generic errors indefinitely by default. A HEAD lets us
        // recognize zero-byte WAL fence files without issuing the problematic range request.
        let head_options = GetOptions {
            head: true,
            extensions: tag.clone().into(),
            ..GetOptions::default()
        };
        let object_size = self
            .object_store
            .get_opts(path, head_options)
            .await?
            .meta
            .size;
        if object_size == 0 {
            return CachedFooter::from_object_size(0, Bytes::new());
        }

        let offset = object_size.saturating_sub(WAL_SST_CACHED_FOOTER_SIZE);
        let (returned_object_size, data) = self
            .read_footer_range(path, GetRange::Bounded(offset..object_size), tag)
            .await?;
        CachedFooter::from_object_size(returned_object_size, data)
    }

    async fn read_footer_range(
        &self,
        path: &Path,
        range: GetRange,
        tag: ObjectStoreCallTag,
    ) -> object_store::Result<(u64, Bytes)> {
        let options = GetOptions {
            range: Some(range),
            extensions: tag.into(),
            ..GetOptions::default()
        };
        let result = self.object_store.get_opts(path, options).await?;
        let object_size = result.meta.size;
        let data = result.bytes().await?;
        Ok((object_size, data))
    }

    /// Lists the WAL SSTs whose IDs fall in `id_range`, sorted by ID.
    ///
    /// WAL file names are zero-padded IDs, so a range with a start lists only
    /// the objects after the ID before it. The WAL directory holds every WAL
    /// that GC hasn't deleted yet, tens of thousands at high write rates, and
    /// some object stores list only about a thousand keys a second.
    pub(crate) async fn list_wal_ssts<R: RangeBounds<WalFileId>>(
        &self,
        id_range: R,
    ) -> Result<Vec<IdentifiedObjectMetadata<WalFileId>>, SlateDBError> {
        let mut wal_list = Vec::new();
        let wal_path = self.path_resolver.wal_path();
        let offset = match id_range.start_bound() {
            Bound::Included(id) => id.value().checked_sub(1).map(|prev| self.path(prev.into())),
            Bound::Excluded(id) => Some(self.path(*id)),
            Bound::Unbounded => None,
        };
        let mut files_stream = match &offset {
            Some(offset) => self.object_store.list_with_offset(Some(&wal_path), offset),
            None => self.object_store.list(Some(&wal_path)),
        };

        while let Some(file) = files_stream.next().await.transpose()? {
            match self.path_resolver.parse_wal_file_id(&file.location) {
                Ok(Some(id)) if id_range.contains(&id) => {
                    wal_list.push(IdentifiedObjectMetadata::from_object_meta(id, file));
                }
                Ok(Some(_)) => {}
                Err(error) => {
                    warn!(
                        "error while parsing WAL file id [location={}, error={}]",
                        file.location, error
                    );
                }
                Ok(None) => {
                    warn!(
                        "unexpected file found in WAL directory [location={}]",
                        file.location
                    );
                }
            }
        }

        wal_list.sort_by_key(|metadata| metadata.id);
        Ok(wal_list)
    }

    pub(crate) async fn delete_sst(&self, wal_id: WalFileId) -> Result<(), SlateDBError> {
        let path = self.path(wal_id);
        debug!("deleting WAL SST [path={}]", path);
        self.object_store
            .delete(&path)
            .await
            .map_err(SlateDBError::from)
    }

    /// Deletes the WAL SSTs in one `delete_stream` call, so an object store with
    /// a bulk delete API deletes them with few requests. Returns the IDs that it
    /// did not delete. A missing object counts as deleted.
    pub(crate) async fn delete_ssts(&self, wal_ids: &[WalFileId]) -> Vec<WalFileId> {
        let paths: Vec<Path> = wal_ids.iter().map(|id| self.path(*id)).collect();
        let mut deleted = std::collections::HashSet::with_capacity(paths.len());
        let mut results = self
            .object_store
            .delete_stream(futures::stream::iter(paths.clone().into_iter().map(Ok)).boxed());
        while let Some(result) = results.next().await {
            match result {
                Ok(path) => {
                    deleted.insert(path);
                }
                Err(object_store::Error::NotFound { path, .. }) => {
                    deleted.insert(Path::from(path));
                }
                Err(e) => warn!("error deleting WAL SSTs [error={}]", e),
            }
        }
        wal_ids
            .iter()
            .zip(paths)
            .filter(|(_, path)| !deleted.contains(path))
            .map(|(id, _)| *id)
            .collect()
    }

    pub(crate) async fn metadata(&self, wal_id: WalFileId) -> Result<ObjectMetadata, SlateDBError> {
        let path = self.path(wal_id);
        let options = GetOptions {
            head: true,
            extensions: ObjectStoreCallTag::new(self.kind, SstType::Wal).into(),
            ..GetOptions::default()
        };
        Ok(ObjectMetadata::new(
            self.object_store.get_opts(&path, options).await?.meta,
        ))
    }

    pub(crate) async fn read_index(
        &self,
        handle: &WalFileHandle,
    ) -> Result<Arc<SsTableIndexOwned>, SlateDBError> {
        let path = self.path(handle.id);
        let index =
            read_with_validation_retry(ObjectStoreCallTag::new(self.kind, SstType::Wal), |tag| {
                let obj = WalReadOnlyObject {
                    inner: ReadOnlyObject {
                        object_store: Arc::clone(&self.object_store),
                        path: path.clone(),
                        tag: tag.clone(),
                    },
                    cached_footer: if tag.retry.is_none() {
                        handle.cached_footer.clone()
                    } else {
                        None
                    },
                };
                async move {
                    self.sst_format
                        .read_index(&handle.info, &obj)
                        .await
                        .map_err(|error| error.with_path(&obj.inner.path))
                }
            })
            .await?;
        Ok(Arc::new(index))
    }

    pub(crate) fn block_range_for_target_bytes(
        &self,
        handle: &WalFileHandle,
        index: &SsTableIndexOwned,
        first_block: usize,
        target_bytes: usize,
    ) -> Range<usize> {
        assert!(target_bytes > 0);

        let index = index.borrow();
        let block_meta = index.block_meta();
        let num_blocks = block_meta.len();
        assert!(first_block < num_blocks);
        let target_bytes = u64::try_from(target_bytes).unwrap_or(u64::MAX);

        let mut blocks = first_block..first_block + 1;
        loop {
            let byte_range = self
                .sst_format
                .block_range(blocks.clone(), &handle.info, &index);
            if byte_range.end.saturating_sub(byte_range.start) >= target_bytes
                || blocks.end == num_blocks
            {
                return blocks;
            }
            blocks.end += 1;
        }
    }

    pub(crate) fn block_range_size(
        &self,
        handle: &WalFileHandle,
        index: &SsTableIndexOwned,
        blocks: Range<usize>,
    ) -> usize {
        if blocks.is_empty() {
            return 0;
        }
        let byte_range = self
            .sst_format
            .block_range(blocks, &handle.info, &index.borrow());
        usize::try_from(byte_range.end.saturating_sub(byte_range.start)).unwrap_or(usize::MAX)
    }

    pub(crate) async fn read_blocks_using_index(
        &self,
        handle: &WalFileHandle,
        index: Arc<SsTableIndexOwned>,
        blocks: Range<usize>,
    ) -> Result<VecDeque<Arc<Block>>, SlateDBError> {
        let object_store = Arc::clone(&self.object_store);
        let path = self.path(handle.id);
        let index = &index;
        let blocks =
            read_with_validation_retry(ObjectStoreCallTag::new(self.kind, SstType::Wal), |tag| {
                let obj = WalReadOnlyObject {
                    inner: ReadOnlyObject {
                        object_store: Arc::clone(&object_store),
                        path: path.clone(),
                        tag: tag.clone(),
                    },
                    cached_footer: if tag.retry.is_none() {
                        handle.cached_footer.clone()
                    } else {
                        None
                    },
                };
                let blocks = blocks.clone();
                async move {
                    self.sst_format
                        .read_blocks(&handle.info, index, blocks, &obj)
                        .await
                        .map_err(|error| error.with_path(&obj.inner.path))
                }
            })
            .await?;
        Ok(blocks.into_iter().map(Arc::new).collect())
    }

    /// Walks accepted WAL history from `start` and returns the frontier, i.e.
    /// the first missing WAL ID.
    ///
    /// `start` is the first WAL ID to check, usually `replay_after_wal_id + 1`.
    /// It must be at least 1 and above the replay boundary, because GC can
    /// delete files at or below the boundary during the walk.
    ///
    /// A zero-byte fence that starts an excluded range accounts for the whole
    /// range, so the walk skips the IDs inside it.
    pub(crate) async fn walk_to_frontier(
        &self,
        start: WalFileId,
        excluded_ranges: &WalExcludedRanges,
    ) -> Result<WalFileId, SlateDBError> {
        fail_point!(Arc::clone(&self.fp_registry), "probe-wal-ssts", |_| {
            Err(SlateDBError::from(std::io::Error::other("oops")))
        });
        let offset = self.path(WalFileId::from(start.value() - 1));
        let mut objects = Vec::new();
        let mut listing = self
            .object_store
            .list_with_offset(Some(&self.path_resolver.wal_path()), &offset);
        while let Some(object) = listing.next().await.transpose()? {
            if let Ok(Some(id)) = self.path_resolver.parse_wal_file_id(&object.location) {
                objects.push((id.value(), object.size));
            }
        }
        // The walk below requires IDs to be in ascending order.
        // `LocalFileSystem` lists in directory order, so sort here.
        objects.sort_unstable();

        // Load the ranges after the listing. A takeover records its range
        // before its fence, so every listed fence has its range in this load.
        let ranges = excluded_ranges.load().await?;
        let mut next = start.value();
        for (wal_id, size) in objects {
            if wal_id < next {
                // The ID is below `start` or inside an excluded range.
                continue;
            }
            if wal_id > next {
                break;
            }
            next = match excluded_range_end(&ranges, wal_id) {
                Some(end) if size == 0 => end + 1,
                _ => wal_id + 1,
            };
        }
        Ok(WalFileId::from(next))
    }

    /// Walks like [`Self::walk_to_frontier`], but lists only the WAL objects
    /// near the last one. A takeover after thousands of WAL uploads then lists
    /// one page, not one page for each thousand objects.
    ///
    /// The walk finds the last WAL object with [`Self::last_seen_wal_id`]. Then
    /// it lists from `2 * window` IDs below that object. Use it only in a writer
    /// that holds the epoch, with `start` at the replay boundary of its
    /// manifest. GC deletes WAL objects only below that boundary. An older
    /// writer leaves gaps only among its last `window` uploads. A takeover
    /// covers those gaps with an excluded range that starts at a fence. So every
    /// ID from `start` to the last `window` IDs exists or is in such a range. If
    /// a range below the listing does not start at a fence, the walk lists from
    /// `start`.
    pub(crate) async fn walk_to_frontier_from_tail(
        &self,
        start: WalFileId,
        excluded_ranges: &WalExcludedRanges,
        window: u64,
    ) -> Result<WalFileId, SlateDBError> {
        let start = start.value();
        let last = self
            .last_seen_wal_id(WalFileId::from(start - 1))
            .await?
            .value();
        let mut from = last.saturating_sub(2 * window).max(start);
        let ranges = excluded_ranges.load().await?;
        // Walk from the start of a range that `from` falls inside, so the walk
        // sees its fence.
        if let Some(range) = ranges
            .iter()
            .find(|range| range.start < from && from <= range.end)
        {
            from = range.start.max(start);
        }
        for range in ranges
            .iter()
            .filter(|range| range.start < from && range.end >= start)
        {
            if range.start < start || !self.is_fence(WalFileId::from(range.start)).await? {
                from = start;
                break;
            }
        }
        self.walk_to_frontier(WalFileId::from(from), excluded_ranges)
            .await
    }

    /// Find the highest WAL SST id present in the object store at or above
    /// `start_after + 1`, returning `start_after` if none exist.
    ///
    /// Two phases:
    ///   1. Parallel exponential probe at offsets `2^0, 2^1, ..., 2^k` from
    ///      `start_after`. One RTT per round of 8 exponents. Brackets the
    ///      frontier between two adjacent powers of two.
    ///   2. Sequential binary search inside the bracketed range to find the
    ///      exact frontier.
    ///
    /// The search is correct where "id exists" is monotone-decreasing in id.
    /// With pipelined uploads, that is true below the last `window` uploads of
    /// an older writer. So the result is within `window` of the last WAL object
    /// (see [`Self::walk_to_frontier_from_tail`]). Total HEAD count is
    /// `O(log N)` for a gap of size N, vs `O(N)` for a windowed scan.
    async fn last_seen_wal_id(&self, start_after: WalFileId) -> Result<WalFileId, SlateDBError> {
        const ROUND_SIZE: u32 = 8;
        const MAX_EXP: u32 = 48;

        let mut lo_offset = None;
        let mut hi_offset = None;
        let mut next_exp = 0;

        while hi_offset.is_none() {
            if next_exp >= MAX_EXP {
                return Err(SlateDBError::InvalidDBState);
            }
            let end_exp = (next_exp + ROUND_SIZE).min(MAX_EXP);
            let exps: Vec<u32> = (next_exp..end_exp).collect();
            let probes = exps.iter().map(|&exp| {
                let offset = 1u64 << exp;
                let path = self.path(WalFileId::from(start_after.value() + offset));
                let object_store = Arc::clone(&self.object_store);
                async move { wal_object_exists(&object_store, &path).await }
            });
            let results = join_all(probes).await;

            for (exp, result) in exps.iter().zip(results) {
                let offset = 1u64 << exp;
                if result? {
                    lo_offset = Some(offset);
                } else {
                    hi_offset = Some(offset);
                    break;
                }
            }
            next_exp = end_exp;
        }

        let hi = hi_offset.expect("loop exits only after finding an upper bound");
        let Some(lo) = lo_offset else {
            return Ok(start_after);
        };

        let mut left = lo + 1;
        let mut right = hi;
        while left < right {
            let mid = left + (right - left) / 2;
            if wal_object_exists(
                &self.object_store,
                &self.path(WalFileId::from(start_after.value() + mid)),
            )
            .await?
            {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        Ok(WalFileId::from(start_after.value() + left - 1))
    }

    /// Returns whether the WAL object at `wal_id` is a zero-byte fence.
    async fn is_fence(&self, wal_id: WalFileId) -> Result<bool, SlateDBError> {
        match self.object_store.head(&self.path(wal_id)).await {
            Ok(meta) => Ok(meta.size == 0),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(error) => Err(SlateDBError::from(error)),
        }
    }

    fn path(&self, wal_id: WalFileId) -> Path {
        self.path_resolver.wal_sst_path(&wal_id)
    }
}

async fn wal_object_exists(
    object_store: &Arc<dyn ObjectStore>,
    path: &Path,
) -> Result<bool, SlateDBError> {
    match object_store.head(path).await {
        Ok(_) => Ok(true),
        Err(object_store::Error::NotFound { .. }) => Ok(false),
        Err(error) => Err(SlateDBError::from(error)),
    }
}

#[allow(dead_code)]
fn slatedb_io_error() -> SlateDBError {
    SlateDBError::from(std::io::Error::other("oops"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_iterator::DataBlockIterator;
    use crate::error::RetryReason;
    use crate::iter::IterationOrder;
    use crate::manifest::WalExcludedRange;
    use crate::test_utils::{FlakyObjectStore, RecordingObjectStore};
    use crate::types::RowEntry;
    use object_store::memory::InMemory;

    fn test_store() -> WalTableStore {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("test-db"),
            TableStoreKind::Main,
        )
    }

    #[tokio::test]
    async fn test_list_wal_ssts() {
        let store = test_store();
        // Ten and up cross a digit, which the zero-padded names keep in order.
        for id in 1..=12u64 {
            store.write_wal_fence(id.into()).await.unwrap();
        }
        let ids = |wals: Vec<IdentifiedObjectMetadata<WalFileId>>| {
            wals.into_iter()
                .map(|wal| wal.id.value())
                .collect::<Vec<_>>()
        };

        assert_eq!(
            ids(store.list_wal_ssts(..).await.unwrap()),
            (1..=12).collect::<Vec<_>>()
        );
        assert_eq!(
            ids(store.list_wal_ssts(WalFileId::from(9)..).await.unwrap()),
            vec![9, 10, 11, 12]
        );
        assert_eq!(
            ids(store
                .list_wal_ssts(WalFileId::from(10)..=WalFileId::from(11))
                .await
                .unwrap()),
            vec![10, 11]
        );
        assert_eq!(
            ids(store
                .list_wal_ssts((Bound::Excluded(WalFileId::from(10)), Bound::Unbounded))
                .await
                .unwrap()),
            vec![11, 12]
        );
        assert_eq!(
            ids(store.list_wal_ssts(..WalFileId::from(3)).await.unwrap()),
            vec![1, 2]
        );
        assert_eq!(
            ids(store.list_wal_ssts(WalFileId::from(0)..).await.unwrap()).len(),
            12
        );
        assert!(store
            .list_wal_ssts(WalFileId::from(13)..)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn open_sst_caches_last_500kb() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let observed = Arc::new(FlakyObjectStore::new(inner, 0));
        let recording = Arc::new(RecordingObjectStore::new(observed.clone()));
        let object_store: Arc<dyn ObjectStore> = recording.clone();
        let store = WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("cached-footer"),
            TableStoreKind::Main,
        );

        let value = vec![b'x'; 4096];
        let mut builder =
            EncodedWalSsTableBuilder::new(32 * 1024, store.sst_format.sst_codec.clone());
        for seq in 0..300 {
            builder
                .add(RowEntry::new_value(b"key", &value, seq))
                .await
                .unwrap();
        }
        let encoded = builder.build().await.unwrap();
        let encoded_bytes = encoded.remaining_as_bytes();
        assert!(encoded_bytes.len() > WAL_SST_CACHED_FOOTER_SIZE as usize);
        store.write_sst(1.into(), &encoded).await.unwrap();

        let handle = store.open_sst(1.into()).await.unwrap();
        let cached_footer = handle.cached_footer.as_ref().unwrap();
        assert_eq!(
            cached_footer.data.len(),
            WAL_SST_CACHED_FOOTER_SIZE as usize
        );
        assert_eq!(
            cached_footer.offset,
            encoded_bytes.len() as u64 - WAL_SST_CACHED_FOOTER_SIZE
        );
        assert_eq!(
            cached_footer.data,
            encoded_bytes.slice(cached_footer.offset as usize..)
        );
        assert_eq!(observed.get_range_attempts(), 1);
        assert_eq!(observed.head_attempts(), 1);
        assert_eq!(
            recording.recorded_get_ranges(false),
            vec![Some(GetRange::Bounded(
                encoded_bytes.len() as u64 - WAL_SST_CACHED_FOOTER_SIZE..encoded_bytes.len() as u64
            ))]
        );

        let index = store.read_index(&handle).await.unwrap();
        let last_block = index.borrow().block_meta().len() - 1;
        store
            .read_blocks_using_index(&handle, index, last_block..last_block + 1)
            .await
            .unwrap();

        // The index and final data block are both covered by the cached tail.
        assert_eq!(observed.get_range_attempts(), 1);
        assert_eq!(observed.head_attempts(), 1);
    }

    #[tokio::test]
    async fn open_sst_caches_entire_small_sst_with_bounded_range() {
        let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let observed = Arc::new(FlakyObjectStore::new(inner, 0));
        let recording = Arc::new(RecordingObjectStore::new(observed.clone()));
        let object_store: Arc<dyn ObjectStore> = recording.clone();
        let store = WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("cached-footer-fallback"),
            TableStoreKind::Main,
        );

        let mut builder = store.table_builder();
        builder
            .add(RowEntry::new_value(b"key", b"value", 1))
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        let encoded_bytes = encoded.remaining_as_bytes();
        store.write_sst(1.into(), &encoded).await.unwrap();

        let handle = store.open_sst(1.into()).await.unwrap();
        let cached_footer = handle.cached_footer.as_ref().unwrap();
        assert_eq!(cached_footer.offset, 0);
        assert_eq!(cached_footer.data, encoded_bytes);
        assert_eq!(observed.get_range_attempts(), 1);
        assert_eq!(observed.head_attempts(), 1);
        assert_eq!(
            recording.recorded_get_ranges(false),
            vec![Some(GetRange::Bounded(0..encoded_bytes.len() as u64))]
        );

        let index = store.read_index(&handle).await.unwrap();
        store
            .read_blocks_using_index(&handle, index, 0..1)
            .await
            .unwrap();

        // One HEAD and one bounded GET were enough to open and read the small WAL SST.
        assert_eq!(observed.get_range_attempts(), 1);
        assert_eq!(observed.head_attempts(), 1);
    }

    #[tokio::test]
    async fn validation_retry_bypasses_the_cached_footer() {
        let recording = Arc::new(RecordingObjectStore::new(Arc::new(InMemory::new())));
        let object_store: Arc<dyn ObjectStore> = recording.clone();
        let store = WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("cached-footer-validation-retry"),
            TableStoreKind::Main,
        );

        let mut builder = store.table_builder();
        builder
            .add(RowEntry::new_value(b"key", b"value", 1))
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        store.write_sst(1.into(), &encoded).await.unwrap();

        let mut handle = store.open_sst(1.into()).await.unwrap();
        recording.clear();

        let cached_footer = handle.cached_footer.as_mut().unwrap();
        let index_offset =
            usize::try_from(handle.info.index_offset - cached_footer.offset).unwrap();
        let mut corrupted = cached_footer.data.to_vec();
        corrupted[index_offset] ^= 0xff;
        cached_footer.data = Bytes::from(corrupted);

        store.read_index(&handle).await.unwrap();
        assert_eq!(
            recording.get_retries(false),
            vec![Some(RetryReason::CrcMismatch)]
        );
    }

    #[tokio::test]
    async fn walk_to_frontier_skips_excluded_ranges_and_stops_at_the_first_gap() {
        // given: data at 1, a fence at 2 that starts the excluded range 2 to 4,
        // an orphan at 3, and a missing ID at 4. Data at 5, a gap at 6, and an
        // orphan at 7 follow. A stale range 5 to 9 starts at a data file.
        let store = test_store();
        let write_data = |wal_id: u64| {
            let store = &store;
            async move {
                let mut builder = store.table_builder();
                builder
                    .add(RowEntry::new_value(b"key", b"value", wal_id))
                    .await
                    .unwrap();
                let encoded = builder.build().await.unwrap();
                store.write_sst(wal_id.into(), &encoded).await.unwrap();
            }
        };
        write_data(1).await;
        store.write_wal_fence(2.into()).await.unwrap();
        for wal_id in [3, 5, 7] {
            write_data(wal_id).await;
        }
        let ranges = WalExcludedRanges::known(&[
            WalExcludedRange { start: 2, end: 4 },
            WalExcludedRange { start: 5, end: 9 },
        ]);

        // when/then: the walk skips the range at the fence, ignores the range
        // at the data file, and stops at the gap.
        assert_eq!(
            store
                .walk_to_frontier(1.into(), &ranges)
                .await
                .unwrap()
                .value(),
            6
        );
        // Without the ranges, the walk stops at the missing ID inside the
        // range.
        assert_eq!(
            store
                .walk_to_frontier(1.into(), &WalExcludedRanges::none())
                .await
                .unwrap()
                .value(),
            4
        );
        // A walk that starts at an existing ID continues past it.
        assert_eq!(
            store
                .walk_to_frontier(7.into(), &ranges)
                .await
                .unwrap()
                .value(),
            8
        );
    }

    /// A store over `observed`, with a fence at every ID in `ids`.
    async fn store_with_fences(
        observed: &Arc<FlakyObjectStore>,
        ids: impl IntoIterator<Item = u64>,
    ) -> WalTableStore {
        let store = WalTableStore::new(
            Arc::clone(observed) as Arc<dyn ObjectStore>,
            SsTableFormat::default(),
            Path::from("test-db"),
            TableStoreKind::Main,
        );
        for id in ids {
            store.write_wal_fence(id.into()).await.unwrap();
        }
        store
    }

    #[tokio::test]
    async fn walk_to_frontier_from_tail_probes_then_lists_once() {
        // given: 300 contiguous WALs, then an older writer's tail with a gap
        // at 301 and an upload that landed at 302.
        let observed = Arc::new(FlakyObjectStore::new(Arc::new(InMemory::new()), 0));
        let store = store_with_fences(&observed, (1..=300).chain([302])).await;
        let heads = observed.head_attempts();

        // when
        let frontier = store
            .walk_to_frontier_from_tail(1.into(), &WalExcludedRanges::none(), 4)
            .await
            .unwrap();

        // then: the frontier is the gap, found with O(log n) HEADs and one
        // listing.
        assert_eq!(frontier.value(), 301);
        assert!(observed.head_attempts() - heads <= 2 * 8 + 9);
        assert_eq!(observed.list_with_offset_attempts(), 1);
    }

    #[tokio::test]
    async fn walk_to_frontier_from_tail_lists_only_near_the_last_wal() {
        // given: a gap at 50 outside any excluded range, which a writer whose
        // replay boundary is below it never sees, and 300 WALs.
        let observed = Arc::new(FlakyObjectStore::new(Arc::new(InMemory::new()), 0));
        let store = store_with_fences(&observed, (1..=300).filter(|id| *id != 50)).await;

        // when/then: the full walk stops at the gap, the walk from the tail
        // starts its listing past it.
        let ranges = WalExcludedRanges::none();
        assert_eq!(
            store
                .walk_to_frontier(1.into(), &ranges)
                .await
                .unwrap()
                .value(),
            50
        );
        assert_eq!(
            store
                .walk_to_frontier_from_tail(1.into(), &ranges, 4)
                .await
                .unwrap()
                .value(),
            301
        );
    }

    #[tokio::test]
    async fn walk_to_frontier_from_tail_lists_from_the_start_when_a_range_has_no_fence() {
        // given: a takeover that recorded the range 50 to 53 but stopped
        // before its fence at 50, and WALs on both sides.
        let observed = Arc::new(FlakyObjectStore::new(Arc::new(InMemory::new()), 0));
        let store = store_with_fences(&observed, (1..=300).filter(|id| *id != 50)).await;
        let ranges = WalExcludedRanges::known(&[WalExcludedRange { start: 50, end: 53 }]);

        // when/then: the walk can't skip the range, so it walks from the start
        // like the full walk.
        assert_eq!(
            store
                .walk_to_frontier_from_tail(1.into(), &ranges, 4)
                .await
                .unwrap()
                .value(),
            store
                .walk_to_frontier(1.into(), &ranges)
                .await
                .unwrap()
                .value()
        );
    }

    #[tokio::test]
    async fn walk_to_frontier_from_tail_skips_a_fenced_range_near_the_tail() {
        // given: a takeover's range 96 to 99 with its fence at 96 and missing
        // IDs inside, the new writer's WALs from 100 to 105, and the frontier
        // at 106. The last WAL is 105, so the listing would start at 97, inside
        // the range.
        let observed = Arc::new(FlakyObjectStore::new(Arc::new(InMemory::new()), 0));
        let store = store_with_fences(&observed, (1..=96).chain(100..=105)).await;
        let ranges = WalExcludedRanges::known(&[WalExcludedRange { start: 96, end: 99 }]);

        // when/then: both walks skip the range at its fence.
        assert_eq!(
            store
                .walk_to_frontier_from_tail(1.into(), &ranges, 4)
                .await
                .unwrap()
                .value(),
            106
        );
        assert_eq!(
            store
                .walk_to_frontier(1.into(), &ranges)
                .await
                .unwrap()
                .value(),
            106
        );
    }

    #[tokio::test]
    async fn walk_to_frontier_from_tail_without_wals_returns_the_start() {
        let observed = Arc::new(FlakyObjectStore::new(Arc::new(InMemory::new()), 0));
        let store = store_with_fences(&observed, []).await;
        assert_eq!(
            store
                .walk_to_frontier_from_tail(7.into(), &WalExcludedRanges::none(), 4)
                .await
                .unwrap()
                .value(),
            7
        );
    }

    #[tokio::test]
    async fn writes_and_reads_wal_sst_without_a_cache() {
        let store = test_store();
        let rows = [
            RowEntry::new_value(b"first", b"value-1", 10),
            RowEntry::new_value(b"second", b"value-2", 11),
        ];
        assert!(store.estimate_encoded_size(2, 26) > 0);

        let mut builder = store.table_builder();
        for row in rows.iter().cloned() {
            builder.add(row).await.unwrap();
        }
        let encoded = builder.build().await.unwrap();
        let written = store.write_sst(1.into(), &encoded).await.unwrap();
        let opened = store.open_sst(1.into()).await.unwrap();

        assert_eq!(written.id, opened.id);
        assert_eq!(written.info, opened.info);
        assert_eq!(written.format_version, opened.format_version);
        assert_eq!(opened.id.value(), 1);

        let index = store.read_index(&opened).await.unwrap();
        let block_count = index.borrow().block_meta().len();
        assert_eq!(block_count, 1);
        assert_eq!(
            index.borrow().block_meta().get(0).first_key().bytes(),
            &10u64.to_be_bytes()
        );

        let block_range = store.block_range_for_target_bytes(&opened, &index, 0, usize::MAX);
        assert_eq!(block_range, 0..block_count);
        assert!(store.block_range_size(&opened, &index, block_range.clone()) > 0);

        let blocks = store
            .read_blocks_using_index(&opened, Arc::clone(&index), block_range)
            .await
            .unwrap();
        let mut actual = Vec::new();
        for block in blocks {
            let mut iter =
                DataBlockIterator::new(block, opened.format_version, IterationOrder::Ascending)
                    .unwrap();
            while let Some(row) = iter.next().await.unwrap() {
                actual.push(row);
            }
        }
        assert_eq!(actual, rows);
    }

    #[tokio::test]
    async fn uses_create_semantics_for_wals_and_fences() {
        let store = test_store();

        store.write_wal_fence(1.into()).await.unwrap();
        assert_eq!(
            store
                .walk_to_frontier(1.into(), &WalExcludedRanges::none())
                .await
                .unwrap()
                .value(),
            2
        );
        assert!(matches!(
            store.write_wal_fence(1.into()).await,
            Err(SlateDBError::Fenced)
        ));
        assert!(matches!(
            store.open_sst(1.into()).await,
            Err(SlateDBError::EmptySSTable)
        ));

        let mut builder = store.table_builder();
        builder
            .add(RowEntry::new_value(b"key", b"value", 12))
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        assert!(matches!(
            store.write_sst(1.into(), &encoded).await,
            Err(SlateDBError::Fenced)
        ));
    }
    #[tokio::test]
    async fn object_store_calls_carry_wal_tag() {
        let recording = Arc::new(RecordingObjectStore::new(Arc::new(InMemory::new())));
        let object_store: Arc<dyn ObjectStore> = recording.clone();
        let store = WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("tagged-wal-store"),
            TableStoreKind::Reader,
        );

        let mut builder = store.table_builder();
        builder
            .add(RowEntry::new_value(b"key", b"value", 1))
            .await
            .unwrap();
        let encoded = builder.build().await.unwrap();
        store.write_sst(1.into(), &encoded).await.unwrap();

        assert_eq!(recording.write_kinds(), vec![Some(TableStoreKind::Reader)]);
        assert_eq!(recording.write_sst_types(), vec![Some(SstType::Wal)]);

        recording.clear();
        store.metadata(1.into()).await.unwrap();
        let handle = store.open_sst(1.into()).await.unwrap();
        let index = store.read_index(&handle).await.unwrap();
        store
            .read_blocks_using_index(&handle, index, 0..1)
            .await
            .unwrap();

        let read_kinds = recording.get_kinds(false);
        let read_sst_types = recording.get_sst_types(false);
        let head_kinds = recording.get_kinds(true);
        let head_sst_types = recording.get_sst_types(true);
        assert!(!head_kinds.is_empty());
        assert!(head_kinds
            .iter()
            .all(|kind| *kind == Some(TableStoreKind::Reader)));
        assert!(head_sst_types
            .iter()
            .all(|sst_type| *sst_type == Some(SstType::Wal)));
        assert!(!read_kinds.is_empty());
        assert!(read_kinds
            .iter()
            .all(|kind| *kind == Some(TableStoreKind::Reader)));
        assert!(read_sst_types
            .iter()
            .all(|sst_type| *sst_type == Some(SstType::Wal)));
    }
}
