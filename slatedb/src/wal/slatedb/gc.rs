use crate::garbage_collector::stats::GcStats;
use crate::garbage_collector::{retain_allowed_by_gc_filter, GcFilter, GC_DELETE_CONCURRENCY};
use crate::wal::slatedb::store::{WalFileId, WalTableStore};
use crate::wal::{WalError, WalFileRange, WalGc};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use log::error;
use parking_lot::Mutex;
use slatedb_common::clock::SystemClock;
use slatedb_common::object_metadata::IdentifiedObjectMetadata;
use std::collections::{BTreeSet, HashSet, VecDeque};
use std::ops::{Bound, Range};
use std::sync::Arc;
use std::time::Duration;

/// Selects which class of SlateDB WAL object is collected.
///
/// Regular WAL SSTs and zero-byte WAL fence objects share the same WAL
/// directory and `WalFileId` identifier space, but they have separate
/// retention policies. This mode keeps a single implementation while
/// allowing regular WAL GC and fence WAL GC to run on independent schedules.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WalGcMode {
    /// Collect non-empty WAL SSTs that are old enough for retention and unreferenced by active
    /// manifests.
    Regular,

    /// Collect zero-byte WAL fence objects under the same safety checks as regular WAL GC.
    Fence,
}

impl WalGcMode {
    pub(crate) fn resource(self) -> &'static str {
        match self {
            WalGcMode::Regular => "WAL",
            WalGcMode::Fence => "WAL fence",
        }
    }
}

/// The maximum number of concurrent deletes when GC collects WAL SSTs by ID.
///
/// The WAL directory grows at the WAL flush rate. With a remote WAL, that rate
/// is hundreds of files per second, so GC must delete at that rate too.
const WAL_ID_DELETE_CONCURRENCY: usize = 32;

/// The maximum number of WAL IDs that one collection by ID deletes. The
/// remaining IDs wait for the next collection.
const WAL_ID_MAX_PER_COLLECTION: u64 = 200_000;

/// The state that lets regular WAL GC collect by ID instead of listing.
///
/// WAL IDs are sequential, and the latest manifest references every WAL from
/// its replay boundary on. After one listing, GC knows which IDs to delete
/// without another listing of the WAL directory.
#[derive(Default)]
struct IdProgress {
    /// GC deleted, or tried to delete, every WAL ID below this ID, except the
    /// IDs in `retry`. `None` until the first collection lists the directory.
    next_id: Option<u64>,
    /// The IDs below `next_id` that a checkpoint still referenced or that GC
    /// could not delete, in ascending order.
    retry: Vec<Range<u64>>,
    /// The fence objects that the first listing found at or above the replay
    /// boundary. GC by ID never deletes them.
    fences: BTreeSet<u64>,
    /// The replay boundary that each collection saw, oldest first.
    boundaries: VecDeque<(DateTime<Utc>, u64)>,
}

/// The result of a collection that lists the WAL directory.
struct ListedCollection {
    /// The listed WAL SSTs below the requested ID that are still in the store.
    remaining: Vec<u64>,
    /// The listed fence objects at or above the requested ID.
    fences: Vec<u64>,
}

#[derive(Clone)]
pub(crate) struct SlateDbWalGc {
    wal_store: Arc<WalTableStore>,
    stats: Arc<GcStats>,
    mode: WalGcMode,
    gc_filter: Option<Arc<dyn GcFilter>>,
    system_clock: Arc<dyn SystemClock>,
    /// Whether regular WAL GC collects by ID after its first listing.
    collect_by_id: bool,
    id_progress: Arc<Mutex<IdProgress>>,
}

impl std::fmt::Debug for SlateDbWalGc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlateDbWalGc")
            .field("mode", &self.mode)
            .finish()
    }
}

impl SlateDbWalGc {
    pub(crate) fn new(
        wal_store: Arc<WalTableStore>,
        stats: Arc<GcStats>,
        mode: WalGcMode,
        gc_filter: Option<Arc<dyn GcFilter>>,
        system_clock: Arc<dyn SystemClock>,
    ) -> Self {
        Self {
            wal_store,
            stats,
            mode,
            gc_filter,
            system_clock,
            collect_by_id: false,
            id_progress: Arc::new(Mutex::new(IdProgress::default())),
        }
    }

    /// Makes regular WAL GC collect by ID after its first listing. See
    /// [`crate::config::GarbageCollectorOptions::wal_collect_by_id`].
    pub(crate) fn with_collect_by_id(mut self, collect_by_id: bool) -> Self {
        self.collect_by_id = collect_by_id;
        self
    }

    fn is_wal_sst_eligible_for_deletion(
        utc_now: &DateTime<Utc>,
        wal_sst: &IdentifiedObjectMetadata<WalFileId>,
        min_age: &chrono::Duration,
        referenced_ranges: &[WalFileRange],
    ) -> bool {
        if utc_now.signed_duration_since(wal_sst.metadata.last_modified) <= *min_age {
            return false;
        }

        let wal_sst_id = wal_sst.id.value();
        !referenced_ranges
            .iter()
            .any(|range| Self::range_contains(range, wal_sst_id))
    }

    fn range_contains(range: &WalFileRange, wal_sst_id: u64) -> bool {
        let after_start = match &range.0 {
            Bound::Included(start) => wal_sst_id >= *start,
            Bound::Excluded(start) => wal_sst_id > *start,
            Bound::Unbounded => true,
        };
        let before_end = match &range.1 {
            Bound::Included(end) => wal_sst_id <= *end,
            Bound::Excluded(end) => wal_sst_id < *end,
            Bound::Unbounded => true,
        };
        after_start && before_end
    }

    /// Returns the replay boundary of the latest manifest. That manifest
    /// references every WAL from this ID on.
    fn replay_boundary(referenced_ranges: &[WalFileRange]) -> Option<u64> {
        referenced_ranges.iter().find_map(|range| match range {
            WalFileRange(Bound::Included(start), Bound::Unbounded) => Some(*start),
            WalFileRange(Bound::Excluded(start), Bound::Unbounded) => Some(start + 1),
            _ => None,
        })
    }

    fn wal_sst_min_age(&self, min_age: Duration) -> chrono::Duration {
        chrono::Duration::from_std(min_age).expect("invalid duration")
    }

    /// Deletes the given WAL SSTs from the table store and returns the IDs
    /// that it did not delete.
    ///
    /// In case of dryrun, the actual deletion doesn't happen.
    async fn maybe_delete_wal_ssts(
        &self,
        sst_ids: Vec<WalFileId>,
        dry_run: bool,
    ) -> Vec<WalFileId> {
        if dry_run {
            if !sst_ids.is_empty() {
                log::info!(
                    "dry run: skipping {} deletion [count={}]",
                    self.mode.resource(),
                    sst_ids.len()
                );
                if matches!(self.mode, WalGcMode::Fence) {
                    log::info!(
                        "WAL fence GC is dry-run by default. This is a conservative setting. \
                        Set wal_fence_options.dry_run=false and use a conservative min_age to enable. \
                        Silence this log with wal_fence_options=None. See #352 for details."
                    );
                }
            }
            for id in &sst_ids {
                log::debug!(
                    "dry run: would delete {} but skipped [id={:?}]",
                    self.mode.resource(),
                    id
                );
            }
            return sst_ids;
        }

        self.delete_wal_ssts(sst_ids, GC_DELETE_CONCURRENCY).await
    }

    /// Deletes WAL SSTs with at most `concurrency` deletes in flight. Returns
    /// the IDs that GC could not delete. A missing object counts as deleted.
    async fn delete_wal_ssts(&self, sst_ids: Vec<WalFileId>, concurrency: usize) -> Vec<WalFileId> {
        futures::stream::iter(sst_ids)
            .map(|id| async move { (id, self.wal_store.delete_sst(id).await) })
            .buffer_unordered(concurrency)
            .filter_map(|(id, result)| async move {
                match result {
                    Ok(()) => {
                        match self.mode {
                            WalGcMode::Regular => self.stats.gc_wal_count.increment(1),
                            WalGcMode::Fence => self.stats.gc_wal_fence_count.increment(1),
                        }
                        None
                    }
                    Err(e) if e.has_object_store_not_found() => None,
                    Err(e) => {
                        error!("error deleting WAL SST [id={:?}, error={}]", id, e);
                        Some(id)
                    }
                }
            })
            .collect()
            .await
    }

    /// Lists the WAL directory, then deletes the WAL objects of this mode that
    /// are old enough and that no manifest references.
    ///
    /// `below` selects which listed IDs the result reports.
    async fn collect_listed(
        &self,
        referenced_ranges: &[WalFileRange],
        min_age: Duration,
        dry_run: bool,
        below: u64,
    ) -> Result<ListedCollection, WalError> {
        let utc_now = self.system_clock.now();
        let min_age = self.wal_sst_min_age(min_age);
        let mut fences = Vec::new();
        let listed = self
            .wal_store
            .list_wal_ssts(..)
            .await?
            .into_iter()
            .filter(|wal_sst| {
                let is_fence = wal_sst.metadata.size == 0;
                if is_fence && wal_sst.id.value() >= below {
                    fences.push(wal_sst.id.value());
                }
                match self.mode {
                    // In regular mode, only consider WAL SSTs with size > 0 for deletion.
                    WalGcMode::Regular => !is_fence,
                    // In fence mode, only consider zero-byte WAL SSTs for deletion.
                    WalGcMode::Fence => is_fence,
                }
            })
            .collect::<Vec<_>>();
        let mut remaining = listed
            .iter()
            .map(|wal_sst| wal_sst.id.value())
            .filter(|id| *id < below)
            .collect::<Vec<_>>();
        let ssts_to_delete = listed
            .into_iter()
            .filter(|wal_sst| {
                Self::is_wal_sst_eligible_for_deletion(
                    &utc_now,
                    wal_sst,
                    &min_age,
                    referenced_ranges,
                )
            })
            .collect::<Vec<_>>();
        let ssts_to_delete = retain_allowed_by_gc_filter(&self.gc_filter, ssts_to_delete).await;
        let sst_ids_to_delete = ssts_to_delete
            .into_iter()
            .map(|wal_sst| wal_sst.id)
            .collect::<Vec<_>>();
        let attempted = sst_ids_to_delete
            .iter()
            .map(|id| id.value())
            .collect::<HashSet<_>>();
        let not_deleted = self
            .maybe_delete_wal_ssts(sst_ids_to_delete, dry_run)
            .await
            .into_iter()
            .map(|id| id.value())
            .collect::<HashSet<_>>();
        remaining.retain(|id| !attempted.contains(id) || not_deleted.contains(id));

        Ok(ListedCollection { remaining, fences })
    }

    /// Deletes the IDs in `candidates` that are below `limit`, that no manifest
    /// references, and that are not known fences. Returns the IDs to try again
    /// in a later collection.
    async fn collect_ids(
        &self,
        candidates: impl Iterator<Item = u64>,
        limit: u64,
        fences: &BTreeSet<u64>,
        referenced_ranges: &[WalFileRange],
        dry_run: bool,
    ) -> Vec<Range<u64>> {
        let mut keep = Vec::new();
        let mut delete = Vec::new();
        for id in candidates {
            if fences.contains(&id) {
                continue;
            }
            if id >= limit
                || referenced_ranges
                    .iter()
                    .any(|range| Self::range_contains(range, id))
            {
                keep.push(id);
            } else {
                delete.push(WalFileId::from(id));
            }
        }
        if dry_run {
            if !delete.is_empty() {
                log::info!(
                    "dry run: skipping WAL deletion by ID [count={}]",
                    delete.len()
                );
            }
        } else {
            let failed = self
                .delete_wal_ssts(delete, WAL_ID_DELETE_CONCURRENCY)
                .await;
            keep.extend(failed.into_iter().map(|id| id.value()));
        }
        keep.sort_unstable();
        ids_to_ranges(keep)
    }

    /// Collects regular WAL SSTs by ID. The first collection lists the
    /// directory to find the existing WALs. Later collections delete every ID
    /// from where the previous collection stopped up to the replay boundary
    /// that GC saw at least `min_age` ago.
    async fn collect_by_id(
        &self,
        referenced_ranges: &[WalFileRange],
        boundary: u64,
        min_age: Duration,
        dry_run: bool,
    ) -> Result<(), WalError> {
        let utc_now = self.system_clock.now();
        let plan = {
            let mut progress = self.id_progress.lock();
            progress.boundaries.push_back((utc_now, boundary));
            let cutoff = utc_now - self.wal_sst_min_age(min_age);
            while progress
                .boundaries
                .get(1)
                .is_some_and(|(seen, _)| *seen <= cutoff)
            {
                progress.boundaries.pop_front();
            }
            let aged_boundary = progress
                .boundaries
                .front()
                .filter(|(seen, _)| *seen <= cutoff)
                .map(|(_, boundary)| *boundary);
            progress.next_id.map(|next_id| {
                (
                    next_id,
                    aged_boundary,
                    std::mem::take(&mut progress.retry),
                    progress.fences.clone(),
                )
            })
        };

        let Some((next_id, aged_boundary, retry, fences)) = plan else {
            let listed = self
                .collect_listed(referenced_ranges, min_age, dry_run, boundary)
                .await?;
            let mut progress = self.id_progress.lock();
            progress.next_id = Some(boundary);
            progress.retry = ids_to_ranges(listed.remaining);
            progress.fences = listed.fences.into_iter().collect();
            return Ok(());
        };

        let limit = aged_boundary.unwrap_or(next_id);
        let end = limit.clamp(next_id, next_id.saturating_add(WAL_ID_MAX_PER_COLLECTION));
        let candidates = retry.into_iter().flatten().chain(next_id..end);
        let retry = self
            .collect_ids(candidates, end, &fences, referenced_ranges, dry_run)
            .await;
        let mut progress = self.id_progress.lock();
        progress.next_id = Some(end);
        progress.retry = retry;
        Ok(())
    }
}

/// Groups ascending IDs into ranges of consecutive IDs.
fn ids_to_ranges(ids: impl IntoIterator<Item = u64>) -> Vec<Range<u64>> {
    let mut ranges: Vec<Range<u64>> = Vec::new();
    for id in ids {
        match ranges.last_mut() {
            Some(last) if last.end == id => last.end += 1,
            _ => ranges.push(id..id + 1),
        }
    }
    ranges
}

#[async_trait]
impl WalGc for SlateDbWalGc {
    async fn collect(
        &self,
        referenced_ranges: Vec<WalFileRange>,
        min_age: Duration,
        dry_run: bool,
    ) -> Result<(), WalError> {
        // A GC filter needs the metadata of each WAL, which only a listing has.
        if let (WalGcMode::Regular, true, None, Some(boundary)) = (
            self.mode,
            self.collect_by_id,
            &self.gc_filter,
            Self::replay_boundary(&referenced_ranges),
        ) {
            return self
                .collect_by_id(&referenced_ranges, boundary, min_age, dry_run)
                .await;
        }
        self.collect_listed(&referenced_ranges, min_age, dry_run, 0)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::sst::SsTableFormat;
    use crate::tablestore::TableStoreKind;
    use crate::RowEntry;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::ObjectStore;
    use slatedb_common::clock::MockSystemClock;
    use slatedb_common::metrics::MetricsRecorderHelper;
    use std::time::Duration;

    fn build_wal_store() -> Arc<WalTableStore> {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        Arc::new(WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("/"),
            TableStoreKind::GC,
        ))
    }

    fn build_collector(
        wal_store: Arc<WalTableStore>,
        clock: Arc<MockSystemClock>,
        mode: WalGcMode,
    ) -> SlateDbWalGc {
        SlateDbWalGc::new(
            wal_store,
            Arc::new(GcStats::new(&MetricsRecorderHelper::noop())),
            mode,
            None,
            clock,
        )
    }

    async fn write_regular_wal(wal_store: &WalTableStore, wal_id: u64) {
        let mut sst = wal_store.table_builder();
        sst.add(RowEntry::new_value(b"key", b"value", wal_id))
            .await
            .unwrap();
        let sst = sst.build().await.unwrap();
        wal_store.write_sst(wal_id.into(), &sst).await.unwrap();
    }

    async fn write_fence_wal(wal_store: &WalTableStore, wal_id: u64) {
        wal_store.write_wal_fence(wal_id.into()).await.unwrap();
    }

    async fn wal_ids(wal_store: &WalTableStore) -> Vec<u64> {
        wal_store
            .list_wal_ssts(..)
            .await
            .unwrap()
            .into_iter()
            .map(|wal| wal.id.value())
            .collect()
    }

    async fn make_all_wals_older_than(
        wal_store: &WalTableStore,
        clock: &MockSystemClock,
        min_age: Duration,
    ) {
        let newest_wal = wal_store
            .list_wal_ssts(..)
            .await
            .unwrap()
            .into_iter()
            .map(|wal| wal.metadata.last_modified)
            .max()
            .expect("expected at least one WAL");
        let min_age_millis =
            i64::try_from(min_age.as_millis()).expect("min_age should fit in i64 milliseconds");
        clock.set(newest_wal.timestamp_millis() + min_age_millis + 1_000);
    }

    fn protect_outer_wals() -> Vec<WalFileRange> {
        vec![
            WalFileRange(Bound::Included(1), Bound::Excluded(2)),
            WalFileRange(Bound::Included(4), Bound::Unbounded),
        ]
    }

    #[tokio::test]
    async fn regular_mode_deletes_unreferenced_range_and_keeps_referenced_wals() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=4 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_collector(wal_store.clone(), clock, WalGcMode::Regular);

        collector
            .collect(protect_outer_wals(), Duration::ZERO, false)
            .await
            .unwrap();

        assert_eq!(wal_ids(&wal_store).await, vec![1, 4]);
    }

    #[tokio::test]
    async fn regular_mode_does_not_touch_fence_wals() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        write_regular_wal(&wal_store, 1).await;
        write_fence_wal(&wal_store, 2).await;
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_collector(wal_store.clone(), clock, WalGcMode::Regular);

        collector
            .collect(vec![], Duration::ZERO, false)
            .await
            .unwrap();

        assert_eq!(wal_ids(&wal_store).await, vec![2]);
    }

    #[tokio::test]
    async fn regular_mode_respects_min_age() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        write_regular_wal(&wal_store, 1).await;
        let last_modified = wal_store.metadata(1.into()).await.unwrap().last_modified;
        let min_age = Duration::from_secs(60 * 60);
        let collector = build_collector(wal_store.clone(), clock.clone(), WalGcMode::Regular);

        clock.set((last_modified + chrono::Duration::minutes(30)).timestamp_millis());
        collector.collect(vec![], min_age, false).await.unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![1]);

        clock.set((last_modified + chrono::Duration::minutes(61)).timestamp_millis());
        collector.collect(vec![], min_age, false).await.unwrap();
        assert!(wal_ids(&wal_store).await.is_empty());
    }

    #[tokio::test]
    async fn fence_mode_deletes_unreferenced_range_and_keeps_referenced_wals() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=4 {
            write_fence_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_collector(wal_store.clone(), clock, WalGcMode::Fence);

        collector
            .collect(protect_outer_wals(), Duration::ZERO, false)
            .await
            .unwrap();

        assert_eq!(wal_ids(&wal_store).await, vec![1, 4]);
    }

    #[tokio::test]
    async fn fence_mode_does_not_touch_regular_wals() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        write_fence_wal(&wal_store, 1).await;
        write_regular_wal(&wal_store, 2).await;
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_collector(wal_store.clone(), clock, WalGcMode::Fence);

        collector
            .collect(vec![], Duration::ZERO, false)
            .await
            .unwrap();

        assert_eq!(wal_ids(&wal_store).await, vec![2]);
    }

    #[tokio::test]
    async fn fence_mode_respects_min_age() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        write_fence_wal(&wal_store, 1).await;
        let last_modified = wal_store.metadata(1.into()).await.unwrap().last_modified;
        let min_age = Duration::from_secs(60 * 60);
        let collector = build_collector(wal_store.clone(), clock.clone(), WalGcMode::Fence);

        clock.set((last_modified + chrono::Duration::minutes(30)).timestamp_millis());
        collector.collect(vec![], min_age, false).await.unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![1]);

        clock.set((last_modified + chrono::Duration::minutes(61)).timestamp_millis());
        collector.collect(vec![], min_age, false).await.unwrap();
        assert!(wal_ids(&wal_store).await.is_empty());
    }

    fn build_id_collector(
        wal_store: Arc<WalTableStore>,
        clock: Arc<MockSystemClock>,
    ) -> SlateDbWalGc {
        build_collector(wal_store, clock, WalGcMode::Regular).with_collect_by_id(true)
    }

    fn from_boundary(boundary: u64) -> Vec<WalFileRange> {
        vec![WalFileRange(Bound::Included(boundary), Bound::Unbounded)]
    }

    fn advance(clock: &MockSystemClock, secs: i64) {
        clock.set(clock.now().timestamp_millis() + secs * 1_000);
    }

    #[tokio::test]
    async fn test_collect_by_id_deletes_below_the_boundary_seen_min_age_ago() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=4 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        let min_age = Duration::from_secs(60);
        make_all_wals_older_than(&wal_store, &clock, min_age).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());

        // The first collection lists the directory, like the listing GC.
        collector
            .collect(from_boundary(3), min_age, false)
            .await
            .unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4]);

        for wal_id in 5..=8 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        // No boundary is min_age old yet.
        advance(&clock, 30);
        collector
            .collect(from_boundary(7), min_age, false)
            .await
            .unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5, 6, 7, 8]);

        // Boundary 3 is old enough, and GC already collected below it.
        advance(&clock, 40);
        collector
            .collect(from_boundary(7), min_age, false)
            .await
            .unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5, 6, 7, 8]);

        // Boundary 7 from the second collection is old enough now.
        advance(&clock, 30);
        collector
            .collect(from_boundary(8), min_age, false)
            .await
            .unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![7, 8]);
    }

    #[tokio::test]
    async fn test_collect_by_id_keeps_checkpoint_wals_until_released() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=6 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock);

        collector
            .collect(from_boundary(2), Duration::ZERO, false)
            .await
            .unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![2, 3, 4, 5, 6]);

        let checkpoint = WalFileRange(Bound::Excluded(2), Bound::Excluded(5));
        let mut ranges = from_boundary(6);
        ranges.push(checkpoint);
        collector
            .collect(ranges, Duration::ZERO, false)
            .await
            .unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 6]);

        collector
            .collect(from_boundary(6), Duration::ZERO, false)
            .await
            .unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![6]);
    }

    #[tokio::test]
    async fn test_collect_by_id_keeps_fences_from_the_first_listing() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        write_regular_wal(&wal_store, 1).await;
        write_fence_wal(&wal_store, 2).await;
        write_fence_wal(&wal_store, 3).await;
        write_regular_wal(&wal_store, 4).await;
        write_regular_wal(&wal_store, 5).await;
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock);

        collector
            .collect(from_boundary(2), Duration::ZERO, false)
            .await
            .unwrap();
        collector
            .collect(from_boundary(5), Duration::ZERO, false)
            .await
            .unwrap();

        assert_eq!(wal_ids(&wal_store).await, vec![2, 3, 5]);
    }

    #[tokio::test]
    async fn test_collect_by_id_skips_missing_ids() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in [1, 2, 5] {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock);

        collector
            .collect(from_boundary(1), Duration::ZERO, false)
            .await
            .unwrap();
        collector
            .collect(from_boundary(6), Duration::ZERO, false)
            .await
            .unwrap();

        assert!(wal_ids(&wal_store).await.is_empty());
        let progress = collector.id_progress.lock();
        assert_eq!(progress.next_id, Some(6));
        assert!(progress.retry.is_empty());
    }
}
