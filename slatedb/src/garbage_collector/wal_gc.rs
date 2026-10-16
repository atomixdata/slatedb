use super::GcTask;
use crate::manifest::Manifest;
use crate::{
    error::SlateDBError,
    manifest::store::ManifestStore,
    wal::{WalFileRange, WalGc},
};
use chrono::{DateTime, Utc};
use log::warn;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
pub(crate) struct WalGcTask {
    manifest_store: Arc<ManifestStore>,
    wal_gc: Arc<dyn WalGc>,
    resource: &'static str,
    min_age: Duration,
    dry_run: bool,
    /// The epoch of the writer this collector runs in, if any.
    writer_epoch: Option<u64>,
}

impl std::fmt::Debug for WalGcTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalGcTask")
            .field("resource", &self.resource.to_string())
            .finish()
    }
}

impl WalGcTask {
    pub(super) fn new(
        manifest_store: Arc<ManifestStore>,
        wal_gc: Arc<dyn WalGc>,
        resource: &'static str,
        min_age: Duration,
        dry_run: bool,
        writer_epoch: Option<u64>,
    ) -> Self {
        Self {
            manifest_store,
            wal_gc,
            resource,
            min_age,
            dry_run,
            writer_epoch,
        }
    }

    fn referenced_wal_ranges(
        latest_manifest_id: u64,
        active_manifests: &BTreeMap<u64, Manifest>,
    ) -> Vec<WalFileRange> {
        active_manifests
            .iter()
            .map(|(manifest_id, manifest)| {
                if *manifest_id == latest_manifest_id {
                    // Keep the current compaction boundary and everything after it. Retaining the
                    // boundary matches the existing GC protocol and protects concurrent writers.
                    WalFileRange(
                        Bound::Included(manifest.core.replay_after_wal_id),
                        Bound::Unbounded,
                    )
                } else {
                    // A checkpoint only references WALs that must be replayed for its manifest.
                    WalFileRange(
                        Bound::Excluded(manifest.core.replay_after_wal_id),
                        Bound::Excluded(manifest.core.next_wal_sst_id),
                    )
                }
            })
            .collect()
    }
}

impl GcTask for WalGcTask {
    /// Resolve the WAL ranges referenced by the current manifest and active checkpoints, then
    /// delegate collection to the configured WAL implementation.
    async fn collect(&self, _utc_now: DateTime<Utc>) -> Result<(), SlateDBError> {
        let latest_manifest = self.manifest_store.read_latest_manifest().await?;
        // A newer writer fences the WAL IDs above the last replay boundary this
        // writer published, then moves the boundary past them. Collect only up
        // to boundaries this writer published, so those fences stay.
        if let Some(writer_epoch) = self.writer_epoch {
            let latest_epoch = latest_manifest.manifest.writer_epoch;
            if latest_epoch != writer_epoch {
                warn!(
                    "skipping {} GC: writer epoch {} was fenced by epoch {}",
                    self.resource, writer_epoch, latest_epoch
                );
                return Ok(());
            }
        }
        let active_manifests = self
            .manifest_store
            .read_referenced_manifests(latest_manifest.id, &latest_manifest.manifest)
            .await?;
        let referenced_ranges = Self::referenced_wal_ranges(latest_manifest.id, &active_manifests);

        self.wal_gc
            .collect(referenced_ranges, self.min_age, self.dry_run)
            .await
            .map_err(Into::into)
    }

    fn resource(&self) -> &str {
        self.resource
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::format::sst::SsTableFormat;
    use crate::garbage_collector::stats::GcStats;
    use crate::manifest::store::StoredManifest;
    use crate::manifest::ManifestCore;
    use crate::tablestore::TableStoreKind;
    use crate::wal::slatedb::gc::{SlateDbWalGc, WalGcMode};
    use crate::wal::slatedb::store::WalTableStore;
    use crate::wal::WalError;
    use crate::RowEntry;
    use async_trait::async_trait;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::ObjectStore;
    use slatedb_common::clock::{DefaultSystemClock, MockSystemClock};
    use slatedb_common::metrics::MetricsRecorderHelper;
    use std::sync::Mutex;
    use std::time::Duration;
    use uuid::Uuid;

    #[derive(Default)]
    struct RecordingWalGc {
        calls: Mutex<Vec<Vec<WalFileRange>>>,
    }

    impl RecordingWalGc {
        fn calls(&self) -> Vec<Vec<WalFileRange>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl WalGc for RecordingWalGc {
        async fn collect(
            &self,
            referenced_ranges: Vec<WalFileRange>,
            _min_age: Duration,
            _dry_run: bool,
        ) -> Result<(), WalError> {
            self.calls.lock().unwrap().push(referenced_ranges);
            Ok(())
        }
    }

    #[test]
    fn test_referenced_wal_ranges() {
        let mut checkpoint_core = ManifestCore::new();
        checkpoint_core.replay_after_wal_id = 2;
        checkpoint_core.next_wal_sst_id = 6;

        let mut current_core = ManifestCore::new();
        current_core.replay_after_wal_id = 5;
        current_core.next_wal_sst_id = 8;

        let active_manifests = BTreeMap::from([
            (1, Manifest::initial(checkpoint_core)),
            (2, Manifest::initial(current_core)),
        ]);

        assert_eq!(
            WalGcTask::referenced_wal_ranges(2, &active_manifests),
            vec![
                WalFileRange(Bound::Excluded(2), Bound::Excluded(6)),
                WalFileRange(Bound::Included(5), Bound::Unbounded),
            ]
        );
    }

    #[tokio::test]
    async fn test_collect_calls_wal_gc_with_referenced_ranges() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let manifest_store = Arc::new(ManifestStore::new(
            &Path::from("/test/wal-gc-ranges"),
            object_store,
        ));

        let mut checkpoint_core = ManifestCore::new();
        checkpoint_core.replay_after_wal_id = 2;
        checkpoint_core.next_wal_sst_id = 6;
        let mut stored_manifest = StoredManifest::create_new_db(
            manifest_store.clone(),
            checkpoint_core,
            Arc::new(DefaultSystemClock::new()),
        )
        .await
        .unwrap();
        let checkpoint_manifest_id = stored_manifest.id();

        let mut dirty = stored_manifest.prepare_dirty().unwrap();
        dirty.value.core.replay_after_wal_id = 5;
        dirty.value.core.next_wal_sst_id = 8;
        dirty.value.core.checkpoints.push(Checkpoint {
            id: Uuid::new_v4(),
            manifest_id: checkpoint_manifest_id,
            expire_time: None,
            create_time: Utc::now(),
            name: None,
        });
        stored_manifest.update(dirty).await.unwrap();

        let wal_gc = Arc::new(RecordingWalGc::default());
        let task = WalGcTask::new(
            manifest_store,
            wal_gc.clone(),
            "WAL",
            Duration::ZERO,
            false,
            None,
        );

        task.collect(Utc::now()).await.unwrap();

        assert_eq!(
            wal_gc.calls(),
            vec![vec![
                WalFileRange(Bound::Excluded(2), Bound::Excluded(6)),
                WalFileRange(Bound::Included(5), Bound::Unbounded),
            ]]
        );
    }

    async fn write_wal(wal_store: &WalTableStore, wal_id: u64) -> Result<(), SlateDBError> {
        let mut sst = wal_store.table_builder();
        sst.add(RowEntry::new_value(b"key", b"value", wal_id))
            .await
            .unwrap();
        let sst = sst.build().await.unwrap();
        wal_store.write_sst(wal_id.into(), &sst).await.map(|_| ())
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

    #[tokio::test]
    async fn test_collect_keeps_newer_writer_fences() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("/test/wal-gc-takeover");
        let manifest_store = Arc::new(ManifestStore::new(&path, object_store.clone()));
        let wal_store = Arc::new(WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            path,
            TableStoreKind::GC,
        ));

        // Writer 1 wrote WALs 1-5 and flushed through 3.
        for wal_id in 1..=5 {
            write_wal(&wal_store, wal_id).await.unwrap();
        }
        let mut core = ManifestCore::new();
        core.replay_after_wal_id = 3;
        let mut stored_manifest = StoredManifest::create_new_db(
            manifest_store.clone(),
            core,
            Arc::new(DefaultSystemClock::new()),
        )
        .await
        .unwrap();
        let mut dirty = stored_manifest.prepare_dirty().unwrap();
        dirty.value.writer_epoch = 1;
        stored_manifest.update(dirty).await.unwrap();

        let clock = Arc::new(MockSystemClock::new());
        clock.set(Utc::now().timestamp_millis() + 60_000);
        let wal_gc = Arc::new(
            SlateDbWalGc::new(
                wal_store.clone(),
                Arc::new(GcStats::new(&MetricsRecorderHelper::noop())),
                WalGcMode::Regular,
                None,
                clock,
            )
            .with_collect_by_id(true),
        );
        let task = WalGcTask::new(
            manifest_store,
            wal_gc,
            "WAL",
            Duration::ZERO,
            false,
            Some(1),
        );
        task.collect(Utc::now()).await.unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5]);

        // Writer 2 claims epoch 2, fences 6 and 7, writes 8 and 9 and flushes
        // through 9, while writer 1 is still running.
        let mut dirty = stored_manifest.prepare_dirty().unwrap();
        dirty.value.writer_epoch = 2;
        stored_manifest.update(dirty).await.unwrap();
        for wal_id in [6, 7] {
            wal_store.write_wal_fence(wal_id.into()).await.unwrap();
        }
        for wal_id in [8, 9] {
            write_wal(&wal_store, wal_id).await.unwrap();
        }
        let mut dirty = stored_manifest.prepare_dirty().unwrap();
        dirty.value.core.replay_after_wal_id = 9;
        stored_manifest.update(dirty).await.unwrap();

        // Writer 1's GC must not collect up to writer 2's boundary.
        task.collect(Utc::now()).await.unwrap();
        assert_eq!(wal_ids(&wal_store).await, vec![3, 4, 5, 6, 7, 8, 9]);

        // So writer 1's delayed upload of WAL 6 still finds the fence.
        assert!(matches!(
            write_wal(&wal_store, 6).await,
            Err(SlateDBError::Fenced)
        ));
    }
}
