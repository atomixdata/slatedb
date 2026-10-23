use crate::garbage_collector::stats::GcStats;
use crate::garbage_collector::{retain_allowed_by_gc_filter, GcFilter, GC_DELETE_CONCURRENCY};
use crate::wal::slatedb::store::{WalFileId, WalTableStore};
use crate::wal::{WalError, WalFileRange, WalGc, WalGcContext};
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
/// GC by ID must keep up with the WAL flush rate. With a WAL on an object
/// store, this rate can be hundreds of files per second.
const WAL_ID_DELETE_CONCURRENCY: usize = 32;

/// The maximum number of WAL IDs one collection by ID deletes. The remaining
/// IDs wait for the next collection.
const WAL_ID_MAX_PER_COLLECTION: u64 = 200_000;

/// The state that regular WAL GC by ID keeps between collections. See
/// [`crate::config::GarbageCollectorOptions::wal_collect_by_id`].
///
/// GC by ID must not delete a WAL fence object, and only a listing shows where
/// the fences are. A writer bumps `writer_epoch`, then writes its fences, then
/// serves writes. With one upload slot, it writes one fence above every WAL ID
/// in use. With more slots, it also fences the missing IDs of the range that
/// its takeover excludes. Thus it writes its fences before it first advances
/// `replay_after_wal_id`. While its epoch is the latest, no other
/// fence that a writer depends on appears below a replay boundary. All IDs
/// below the boundary were in use when the boundary passed them. An older
/// writer that is still opening can write its fence into a free ID, but then
/// that writer fails to open. Thus GC lists after it sees the writer of the
/// latest epoch advance the replay boundary. That listing finds every fence
/// that matters, until the epoch changes again.
#[derive(Default)]
struct IdState {
    /// The `writer_epoch` and `replay_after_wal_id` of the first manifest that
    /// GC read with that epoch.
    epoch_start: Option<(u64, u64)>,
    /// A listing sets this after the replay boundary moved past `epoch_start` in
    /// the same epoch. GC clears it when the epoch changes.
    progress: Option<IdProgress>,
}

/// Where GC by ID stands within one writer epoch.
struct IdProgress {
    /// GC deleted, or tried to delete, every WAL ID below this ID except the
    /// IDs in `retry`.
    next_id: u64,
    /// The WAL SSTs below `next_id` that GC did not delete yet, in ascending
    /// order. They were too young, a checkpoint referenced them, or the delete
    /// failed.
    retry: Vec<Range<u64>>,
    /// The fence objects that the listing found at or above the replay
    /// boundary. GC by ID never deletes them.
    fences: BTreeSet<u64>,
    /// The replay boundaries that GC read in this epoch, with the time that it
    /// read each one, oldest first. GC deletes only below the newest boundary
    /// that is at least `min_age` old.
    boundaries: VecDeque<(DateTime<Utc>, u64)>,
}

/// The result of a collection that lists the WAL directory.
struct ListedCollection {
    /// The listed WAL SSTs below the requested ID that are still in the store.
    remaining: Vec<u64>,
    /// The listed fence objects at or above the requested ID.
    fences: Vec<u64>,
}

/// What a collection by ID does. GC decides it while it holds the lock.
enum IdPlan {
    /// List the directory. If `arm` is set, start to delete by ID after the
    /// listing.
    List { arm: bool },
    /// Delete by ID.
    Delete {
        next_id: u64,
        end: u64,
        limit: u64,
        retry: Vec<Range<u64>>,
        fences: BTreeSet<u64>,
    },
}

#[derive(Clone)]
pub(crate) struct SlateDbWalGc {
    wal_store: Arc<WalTableStore>,
    stats: Arc<GcStats>,
    mode: WalGcMode,
    gc_filter: Option<Arc<dyn GcFilter>>,
    system_clock: Arc<dyn SystemClock>,
    /// Whether regular WAL GC deletes by ID after a listing found the fences.
    collect_by_id: bool,
    id_state: Arc<Mutex<IdState>>,
}

impl std::fmt::Debug for SlateDbWalGc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlateDbWalGc")
            .field("mode", &self.mode)
            .field("collect_by_id", &self.collect_by_id)
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
            id_state: Arc::new(Mutex::new(IdState::default())),
        }
    }

    /// Makes regular WAL GC delete by ID after a listing found the fences. See
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
        !Self::is_referenced(referenced_ranges, wal_sst_id)
    }

    fn is_referenced(referenced_ranges: &[WalFileRange], wal_sst_id: u64) -> bool {
        referenced_ranges
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

    /// Returns the `replay_after_wal_id` of the latest manifest. It is the start
    /// of the only unbounded range, which references every WAL from that ID on.
    fn replay_boundary(referenced_ranges: &[WalFileRange]) -> Option<u64> {
        referenced_ranges.iter().find_map(|range| match range {
            WalFileRange(Bound::Included(start), Bound::Unbounded) => Some(*start),
            WalFileRange(Bound::Excluded(start), Bound::Unbounded) => start.checked_add(1),
            _ => None,
        })
    }

    fn wal_sst_min_age(&self, min_age: Duration) -> chrono::Duration {
        chrono::Duration::from_std(min_age).expect("invalid duration")
    }

    /// Deletes the given WAL SSTs from the table store and returns the IDs it
    /// did not delete.
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

    /// Deletes WAL SSTs with at most `concurrency` deletes in flight and
    /// returns the IDs it could not delete. A missing object counts as deleted.
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
        let mut remaining = Vec::new();
        let mut candidates = Vec::new();
        for wal_sst in self.wal_store.list_wal_ssts(..).await? {
            let id = wal_sst.id.value();
            let is_fence = wal_sst.metadata.size == 0;
            if is_fence && id >= below {
                fences.push(id);
            }
            if !is_fence && id < below {
                remaining.push(id);
            }
            let of_this_mode = match self.mode {
                // In regular mode, only consider WAL SSTs with size > 0 for deletion.
                WalGcMode::Regular => !is_fence,
                // In fence mode, only consider zero-byte WAL SSTs for deletion.
                WalGcMode::Fence => is_fence,
            };
            if of_this_mode
                && Self::is_wal_sst_eligible_for_deletion(
                    &utc_now,
                    &wal_sst,
                    &min_age,
                    referenced_ranges,
                )
            {
                candidates.push(wal_sst);
            }
        }
        let ssts_to_delete = retain_allowed_by_gc_filter(&self.gc_filter, candidates).await;
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
            if id >= limit || Self::is_referenced(referenced_ranges, id) {
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

    /// Collects regular WAL SSTs, by ID where [`IdState`] allows it and by
    /// listing otherwise.
    async fn collect_by_id(
        &self,
        referenced_ranges: &[WalFileRange],
        writer_epoch: u64,
        boundary: u64,
        min_age: Duration,
        dry_run: bool,
    ) -> Result<(), WalError> {
        let utc_now = self.system_clock.now();
        let plan = {
            let mut state = self.id_state.lock();
            if state.epoch_start.map(|(epoch, _)| epoch) != Some(writer_epoch) {
                // A new writer fenced the database. The fences that GC knows
                // can miss its fence.
                if state.progress.is_some() {
                    log::info!(
                        "WAL GC lists again until writer epoch {} advances the replay boundary",
                        writer_epoch
                    );
                }
                state.epoch_start = Some((writer_epoch, boundary));
                state.progress = None;
            }
            let epoch_start_boundary = state.epoch_start.map(|(_, start)| start);
            match &mut state.progress {
                None => IdPlan::List {
                    arm: epoch_start_boundary != Some(boundary),
                },
                Some(progress) => {
                    progress.boundaries.push_back((utc_now, boundary));
                    let cutoff = utc_now - self.wal_sst_min_age(min_age);
                    while progress
                        .boundaries
                        .get(1)
                        .is_some_and(|(seen, _)| *seen <= cutoff)
                    {
                        progress.boundaries.pop_front();
                    }
                    // Every WAL below a boundary that GC read min_age ago is at
                    // least min_age old. If there is no such boundary, no WAL is
                    // old enough.
                    let limit = progress
                        .boundaries
                        .front()
                        .filter(|(seen, _)| *seen <= cutoff)
                        .map_or(0, |(_, boundary)| *boundary);
                    let next_id = progress.next_id;
                    IdPlan::Delete {
                        next_id,
                        end: limit
                            .clamp(next_id, next_id.saturating_add(WAL_ID_MAX_PER_COLLECTION)),
                        limit,
                        retry: std::mem::take(&mut progress.retry),
                        fences: progress.fences.clone(),
                    }
                }
            }
        };

        match plan {
            IdPlan::List { arm } => {
                let listed = self
                    .collect_listed(referenced_ranges, min_age, dry_run, boundary)
                    .await?;
                if arm {
                    let mut state = self.id_state.lock();
                    if state.epoch_start.map(|(epoch, _)| epoch) == Some(writer_epoch) {
                        log::info!(
                            "WAL GC deletes by ID for writer epoch {} [replay_after_wal_id={}, fences={}]",
                            writer_epoch,
                            boundary,
                            listed.fences.len()
                        );
                        state.progress = Some(IdProgress {
                            next_id: boundary,
                            retry: ids_to_ranges(listed.remaining),
                            fences: listed.fences.into_iter().collect(),
                            boundaries: VecDeque::from([(utc_now, boundary)]),
                        });
                    }
                }
            }
            IdPlan::Delete {
                next_id,
                end,
                limit,
                retry,
                fences,
            } => {
                let candidates = retry.into_iter().flatten().chain(next_id..end);
                let retry = self
                    .collect_ids(candidates, limit, &fences, referenced_ranges, dry_run)
                    .await;
                let mut state = self.id_state.lock();
                // A run for a newer epoch can reset the progress meanwhile.
                let same_epoch = state.epoch_start.map(|(epoch, _)| epoch) == Some(writer_epoch);
                if let (true, Some(progress)) = (same_epoch, &mut state.progress) {
                    progress.next_id = end;
                    progress.retry = retry;
                }
            }
        }
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
        self.collect_listed(&referenced_ranges, min_age, dry_run, 0)
            .await?;
        Ok(())
    }

    async fn collect_with_context(
        &self,
        context: WalGcContext,
        min_age: Duration,
        dry_run: bool,
    ) -> Result<(), WalError> {
        // A GC filter needs the metadata of each WAL, which only a listing has.
        if let (WalGcMode::Regular, true, None, Some(boundary)) = (
            self.mode,
            self.collect_by_id,
            &self.gc_filter,
            Self::replay_boundary(&context.referenced_ranges),
        ) {
            return self
                .collect_by_id(
                    &context.referenced_ranges,
                    context.writer_epoch,
                    boundary,
                    min_age,
                    dry_run,
                )
                .await;
        }
        self.collect(context.referenced_ranges, min_age, dry_run)
            .await
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

    /// The context of a run that read a latest manifest with `writer_epoch` and
    /// `replay_after_wal_id = boundary`, plus any checkpoint ranges.
    fn latest(writer_epoch: u64, boundary: u64, checkpoints: &[WalFileRange]) -> WalGcContext {
        let mut ranges = checkpoints.to_vec();
        ranges.push(WalFileRange(Bound::Included(boundary), Bound::Unbounded));
        WalGcContext::new(ranges, writer_epoch)
    }

    fn advance(clock: &MockSystemClock, secs: i64) {
        clock.set(clock.now().timestamp_millis() + secs * 1_000);
    }

    async fn collect(collector: &SlateDbWalGc, context: WalGcContext, min_age: Duration) {
        collector
            .collect_with_context(context, min_age, false)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_collect_by_id_keeps_a_fence_written_after_gc_first_saw_the_epoch() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=4 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());

        // Writer 2 bumped the epoch but did not write its fence yet when GC
        // first reads epoch 2. So this listing cannot see the fence.
        collect(&collector, latest(2, 4, &[]), Duration::ZERO).await;
        assert_eq!(wal_ids(&wal_store).await, vec![4]);

        // Writer 2 fences 5, writes 6 and 7 and flushes through 7.
        write_fence_wal(&wal_store, 5).await;
        for wal_id in [6, 7, 8] {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        collect(&collector, latest(2, 7, &[]), Duration::ZERO).await;
        // Writer 2 flushes through 8. GC now deletes by ID up to 8.
        advance(&clock, 1);
        collect(&collector, latest(2, 8, &[]), Duration::ZERO).await;

        assert_eq!(wal_ids(&wal_store).await, vec![5, 8]);
    }

    #[tokio::test]
    async fn test_collect_by_id_keeps_a_newer_writers_fence() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=5 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());

        // Writer 1 advances its boundary from 2 to 3, so GC deletes by ID.
        collect(&collector, latest(1, 2, &[]), Duration::ZERO).await;
        collect(&collector, latest(1, 3, &[]), Duration::ZERO).await;
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5]);

        // Writer 2 fences 6, writes 7 and 8 and flushes through 8. Writer 1 can
        // still upload WAL 6.
        write_fence_wal(&wal_store, 6).await;
        for wal_id in [7, 8] {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        collect(&collector, latest(2, 8, &[]), Duration::ZERO).await;
        advance(&clock, 1);
        collect(&collector, latest(2, 8, &[]), Duration::ZERO).await;

        assert_eq!(wal_ids(&wal_store).await, vec![6, 8]);
        assert!(matches!(
            wal_store.write_wal_fence(6.into()).await,
            Err(crate::error::SlateDBError::Fenced)
        ));
    }

    #[tokio::test]
    async fn test_collect_by_id_keeps_the_fences_of_a_parallel_upload_takeover() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=5 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());

        // Writer 1 advances its boundary from 2 to 3, so GC deletes by ID.
        collect(&collector, latest(1, 2, &[]), Duration::ZERO).await;
        collect(&collector, latest(1, 3, &[]), Duration::ZERO).await;
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5]);

        // Writer 1 uploads up to four WALs at a time. WAL 7 lands, but WAL 6
        // does not. Writer 2 takes over with the excluded range 6 to 9. It
        // fences 6, 8 and 9, and it keeps the orphan 7. Then it writes 10 and
        // 11 and flushes through 11.
        write_regular_wal(&wal_store, 7).await;
        for wal_id in [6, 8, 9] {
            write_fence_wal(&wal_store, wal_id).await;
        }
        for wal_id in [10, 11] {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        collect(&collector, latest(2, 11, &[]), Duration::ZERO).await;
        advance(&clock, 1);
        collect(&collector, latest(2, 11, &[]), Duration::ZERO).await;

        // GC deleted the WALs below the boundary, the orphan too, and kept
        // every fence. A late upload of writer 1 to a fenced ID fails.
        assert_eq!(wal_ids(&wal_store).await, vec![6, 8, 9, 11]);
        for wal_id in [6, 8, 9] {
            assert!(matches!(
                wal_store.write_wal_fence(wal_id.into()).await,
                Err(crate::error::SlateDBError::Fenced)
            ));
        }
    }

    #[tokio::test]
    async fn test_collect_by_id_deletes_below_the_boundary_read_min_age_ago() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=4 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        let min_age = Duration::from_secs(60);
        make_all_wals_older_than(&wal_store, &clock, min_age).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());

        // The first two runs list. The second run saw the boundary move.
        collect(&collector, latest(1, 2, &[]), min_age).await;
        collect(&collector, latest(1, 3, &[]), min_age).await;
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4]);

        for wal_id in 5..=8 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        // Boundary 3 is not min_age old yet.
        advance(&clock, 30);
        collect(&collector, latest(1, 7, &[]), min_age).await;
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5, 6, 7, 8]);

        // Boundary 3 is old enough, and nothing below it is left.
        advance(&clock, 40);
        collect(&collector, latest(1, 7, &[]), min_age).await;
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5, 6, 7, 8]);

        // Boundary 7 from the third run is old enough now.
        advance(&clock, 30);
        collect(&collector, latest(1, 8, &[]), min_age).await;
        assert_eq!(wal_ids(&wal_store).await, vec![7, 8]);
    }

    #[tokio::test]
    async fn test_collect_by_id_waits_min_age_for_wals_too_young_when_listed() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=3 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        let min_age = Duration::from_secs(60);
        let collector = build_id_collector(wal_store.clone(), clock.clone());
        let newest = wal_store.metadata(3.into()).await.unwrap().last_modified;
        clock.set(newest.timestamp_millis());

        // The listing finds WALs 1 and 2 below the boundary, but they are too
        // young.
        collect(&collector, latest(1, 2, &[]), min_age).await;
        collect(&collector, latest(1, 3, &[]), min_age).await;
        advance(&clock, 30);
        collect(&collector, latest(1, 3, &[]), min_age).await;
        assert_eq!(wal_ids(&wal_store).await, vec![1, 2, 3]);

        advance(&clock, 31);
        collect(&collector, latest(1, 3, &[]), min_age).await;
        assert_eq!(wal_ids(&wal_store).await, vec![3]);
    }

    #[tokio::test]
    async fn test_collect_by_id_keeps_checkpoint_wals_until_released() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=6 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());
        let checkpoint = WalFileRange(Bound::Excluded(2), Bound::Excluded(5));

        collect(&collector, latest(1, 1, &[]), Duration::ZERO).await;
        collect(&collector, latest(1, 2, &[]), Duration::ZERO).await;
        assert_eq!(wal_ids(&wal_store).await, vec![2, 3, 4, 5, 6]);

        advance(&clock, 1);
        collect(
            &collector,
            latest(1, 6, std::slice::from_ref(&checkpoint)),
            Duration::ZERO,
        )
        .await;
        advance(&clock, 1);
        collect(&collector, latest(1, 6, &[checkpoint]), Duration::ZERO).await;
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 6]);

        advance(&clock, 1);
        collect(&collector, latest(1, 6, &[]), Duration::ZERO).await;
        assert_eq!(wal_ids(&wal_store).await, vec![6]);
    }

    #[tokio::test]
    async fn test_collect_by_id_deletes_without_listing() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=3 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());
        collect(&collector, latest(1, 1, &[]), Duration::ZERO).await;
        collect(&collector, latest(1, 2, &[]), Duration::ZERO).await;
        assert_eq!(wal_ids(&wal_store).await, vec![2, 3]);

        // While the epoch stays the same, no writer writes a fence below the
        // replay boundary. So GC does not look at the objects. A listing keeps
        // this zero-byte object, but GC by ID deletes it.
        write_fence_wal(&wal_store, 4).await;
        write_regular_wal(&wal_store, 5).await;
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        collect(&collector, latest(1, 6, &[]), Duration::ZERO).await;
        assert!(wal_ids(&wal_store).await.is_empty());
    }

    #[tokio::test]
    async fn test_collect_by_id_dry_run_deletes_nothing() {
        let wal_store = build_wal_store();
        let clock = Arc::new(MockSystemClock::new());
        for wal_id in 1..=4 {
            write_regular_wal(&wal_store, wal_id).await;
        }
        make_all_wals_older_than(&wal_store, &clock, Duration::ZERO).await;
        let collector = build_id_collector(wal_store.clone(), clock.clone());

        for boundary in [2, 3, 4, 4] {
            collector
                .collect_with_context(latest(1, boundary, &[]), Duration::ZERO, true)
                .await
                .unwrap();
            advance(&clock, 1);
        }
        assert_eq!(wal_ids(&wal_store).await, vec![1, 2, 3, 4]);
    }
}
