use std::collections::VecDeque;
use std::ops::{Range, RangeBounds};
use std::sync::Arc;

use bytes::Bytes;
use fail_parallel::{fail_point, FailPointRegistry};
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

    pub(crate) async fn list_wal_ssts<R: RangeBounds<WalFileId>>(
        &self,
        id_range: R,
    ) -> Result<Vec<IdentifiedObjectMetadata<WalFileId>>, SlateDBError> {
        let mut wal_list = Vec::new();
        let wal_path = self.path_resolver.wal_path();
        let mut files_stream = self.object_store.list(Some(&wal_path));

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

    fn path(&self, wal_id: WalFileId) -> Path {
        self.path_resolver.wal_sst_path(&wal_id)
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
