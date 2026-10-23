use crate::format::sst::SsTableFormat;
use crate::garbage_collector::stats::GcStats;
use crate::paths::PathResolver;
use crate::tablestore::TableStoreKind;
use crate::wal::slatedb::gc::{SlateDbWalGc, WalGcMode};
use crate::wal::slatedb::iterator::excluded_range_end;
use crate::wal::slatedb::store::{WalFileId, WalTableStore};
use crate::wal::{WalAdmin, WalError, WalGc};
use crate::VersionedManifest;
use async_trait::async_trait;
use fail_parallel::{fail_point, FailPointRegistry};
use futures::StreamExt;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use slatedb_common::clock::DefaultSystemClock;
use slatedb_common::metrics::MetricsRecorderHelper;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct SlateDbWalAdmin {
    object_store: Arc<dyn ObjectStore>,
    #[cfg_attr(not(test), allow(dead_code))]
    fp_registry: Arc<FailPointRegistry>,
}

impl SlateDbWalAdmin {
    pub(crate) fn new(
        object_store: Arc<dyn ObjectStore>,
        fp_registry: Arc<FailPointRegistry>,
    ) -> Self {
        Self {
            object_store,
            fp_registry,
        }
    }

    fn replay_range(manifest: &VersionedManifest) -> Result<(u64, u64), WalError> {
        let replay_after_wal_id = manifest.replay_after_wal_id();
        let wal_id_last_seen = manifest
            .next_wal_sst_id()
            .checked_sub(1)
            .ok_or_else(Self::invalid_manifest)?;
        Ok((replay_after_wal_id, wal_id_last_seen))
    }

    fn invalid_manifest() -> WalError {
        WalError::InternalError(Arc::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "source manifest must have a positive next WAL file ID",
        )))
    }

    fn has_wal_file_ids(replay_after_wal_id: u64, wal_id_last_seen: u64) -> bool {
        wal_id_last_seen > replay_after_wal_id
    }

    async fn paths_under(&self, path: &Path) -> Result<Vec<Path>, WalError> {
        let mut objects = self.object_store.list(Some(path));
        let mut paths = Vec::new();
        while let Some(object) = objects.next().await {
            let object = object.map_err(|err| WalError::Unavailable(Arc::new(err)))?;
            paths.push(object.location);
        }
        Ok(paths)
    }
}

#[async_trait]
impl WalAdmin for SlateDbWalAdmin {
    fn garbage_collector(&self, path: &Path) -> Arc<dyn WalGc> {
        let wal_store = Arc::new(WalTableStore::new(
            self.object_store.clone(),
            SsTableFormat::default(),
            path.clone(),
            TableStoreKind::GC,
        ));
        Arc::new(SlateDbWalGc::new(
            wal_store,
            Arc::new(GcStats::new(&MetricsRecorderHelper::noop())),
            WalGcMode::Regular,
            None,
            Arc::new(DefaultSystemClock::new()),
        ))
    }

    async fn delete_wal(&self, path: &Path, dry_run: bool) -> Result<Vec<String>, WalError> {
        // Collect the paths first so listing is complete before objects are removed.
        let wal_path = PathResolver::from_root(path.clone()).wal_path();
        let paths = self.paths_under(&wal_path).await?;
        if !dry_run {
            for object_path in &paths {
                self.object_store
                    .delete(object_path)
                    .await
                    .map_err(|err| WalError::Unavailable(Arc::new(err)))?;
            }
        }
        Ok(paths.iter().map(Path::to_string).collect())
    }

    async fn is_empty(&self, path: &Path, manifest: &VersionedManifest) -> Result<bool, WalError> {
        let (replay_after_wal_id, wal_id_last_seen) = Self::replay_range(manifest)?;
        let excluded_ranges = &manifest.core().wal_excluded_ranges;
        let path_resolver = PathResolver::from_root(path.clone());
        let mut wal_id = replay_after_wal_id + 1;
        while wal_id <= wal_id_last_seen {
            let path = path_resolver.wal_sst_path(&WalFileId::from(wal_id));
            let metadata = self
                .object_store
                .head(&path)
                .await
                .map_err(|err| WalError::Unavailable(Arc::new(err)))?;

            // Native SlateDB WAL fences are zero-byte WAL objects and contain no records.
            if metadata.size > 0 {
                return Ok(false);
            }
            // A fence that starts an excluded range hides the rest of the
            // range. Orphans there hold no accepted records, and IDs there can
            // be missing.
            wal_id = match excluded_range_end(excluded_ranges, wal_id) {
                Some(end) => end.saturating_add(1),
                None => wal_id + 1,
            };
        }
        Ok(true)
    }

    async fn clone_wal(
        &self,
        from_path: &Path,
        from_manifest: VersionedManifest,
        to_path: &Path,
    ) -> Result<(u64, u64), WalError> {
        let (replay_after_wal_id, wal_id_last_seen) = Self::replay_range(&from_manifest)?;
        let excluded_ranges = &from_manifest.core().wal_excluded_ranges;
        let from_path_resolver = PathResolver::from_root(from_path.clone());
        let to_path_resolver = PathResolver::from_root(to_path.clone());

        if Self::has_wal_file_ids(replay_after_wal_id, wal_id_last_seen) {
            for wal_id in (replay_after_wal_id + 1)..=wal_id_last_seen {
                fail_point!(self.fp_registry.clone(), "copy-wal-ssts-io-error", |_| Err(
                    WalError::Unavailable(Arc::new(std::io::Error::other("oops")))
                ));

                let id = WalFileId::from(wal_id);
                let source = from_path_resolver.wal_sst_path(&id);
                let destination = to_path_resolver.wal_sst_path(&id);
                match self.object_store.as_ref().copy(&source, &destination).await {
                    Ok(()) => {}
                    // A takeover that crashed before its fills can leave IDs
                    // after its range start missing. The clone keeps the
                    // source's ranges, so it excludes these IDs too.
                    Err(object_store::Error::NotFound { .. })
                        if excluded_ranges
                            .iter()
                            .any(|range| range.start < wal_id && wal_id <= range.end) => {}
                    Err(err) => return Err(WalError::Unavailable(Arc::new(err))),
                }
            }
        }

        Ok((replay_after_wal_id, wal_id_last_seen))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Manifest, ManifestCore, WalExcludedRange};
    use bytes::Bytes;
    use object_store::memory::InMemory;

    #[tokio::test]
    async fn delete_wal_deletes_only_objects_under_the_wal_prefix() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let wal_admin =
            SlateDbWalAdmin::new(object_store.clone(), Arc::new(FailPointRegistry::new()));
        let db_path = Path::from("db");
        let wal_object = PathResolver::from_root(db_path.clone()).wal_sst_path(&WalFileId::from(1));
        let non_wal_object = db_path
            .clone()
            .join("manifest")
            .join("00000000000000000001");
        let sibling_object = Path::from("other/wal/00000000000000000002.sst");
        object_store
            .put(&wal_object, Bytes::from_static(b"wal").into())
            .await
            .unwrap();
        object_store
            .put(&non_wal_object, Bytes::from_static(b"keep").into())
            .await
            .unwrap();
        object_store
            .put(&sibling_object, Bytes::from_static(b"keep").into())
            .await
            .unwrap();

        wal_admin.delete_wal(&db_path, false).await.unwrap();

        assert!(matches!(
            object_store.head(&wal_object).await,
            Err(object_store::Error::NotFound { .. })
        ));
        assert!(object_store.head(&non_wal_object).await.is_ok());
        assert!(object_store.head(&sibling_object).await.is_ok());
    }

    /// Writes a WAL after a takeover that crashed before its fills. The
    /// manifest records the range 2 to 4. ID 2 is a fence, ID 3 is an orphan,
    /// and ID 4 is missing. ID 5 is a fence, or a data file if `data_at_5`.
    async fn wal_with_an_unfilled_range(
        object_store: &Arc<dyn ObjectStore>,
        db_path: &Path,
        data_at_5: bool,
    ) -> VersionedManifest {
        let resolver = PathResolver::from_root(db_path.clone());
        let put = |wal_id: u64, bytes: &'static [u8]| {
            let path = resolver.wal_sst_path(&WalFileId::from(wal_id));
            async move {
                object_store
                    .put(&path, Bytes::from_static(bytes).into())
                    .await
                    .unwrap();
            }
        };
        put(2, b"").await;
        put(3, b"orphan").await;
        put(5, if data_at_5 { b"data" } else { b"" }).await;

        let mut core = ManifestCore::new();
        core.replay_after_wal_id = 1;
        core.next_wal_sst_id = 6;
        core.wal_excluded_ranges = vec![WalExcludedRange { start: 2, end: 4 }];
        VersionedManifest::from_manifest(1, Manifest::initial(core))
    }

    #[tokio::test]
    async fn is_empty_skips_an_excluded_range() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let wal_admin =
            SlateDbWalAdmin::new(object_store.clone(), Arc::new(FailPointRegistry::new()));
        let db_path = Path::from("db");

        let manifest = wal_with_an_unfilled_range(&object_store, &db_path, false).await;
        assert!(wal_admin.is_empty(&db_path, &manifest).await.unwrap());

        let db_path = Path::from("db_with_data");
        let manifest = wal_with_an_unfilled_range(&object_store, &db_path, true).await;
        assert!(!wal_admin.is_empty(&db_path, &manifest).await.unwrap());
    }

    #[tokio::test]
    async fn clone_wal_copies_around_a_missing_id_in_an_excluded_range() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let wal_admin =
            SlateDbWalAdmin::new(object_store.clone(), Arc::new(FailPointRegistry::new()));
        let from_path = Path::from("source");
        let to_path = Path::from("clone");
        let manifest = wal_with_an_unfilled_range(&object_store, &from_path, false).await;

        let replay_range = wal_admin
            .clone_wal(&from_path, manifest, &to_path)
            .await
            .unwrap();

        assert_eq!(replay_range, (1, 5));
        let resolver = PathResolver::from_root(to_path);
        for (wal_id, exists) in [(2, true), (3, true), (4, false), (5, true)] {
            let head = object_store
                .head(&resolver.wal_sst_path(&WalFileId::from(wal_id)))
                .await;
            assert_eq!(head.is_ok(), exists, "WAL {wal_id}");
        }
    }

    #[test]
    fn creates_a_path_scoped_garbage_collector() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let wal_admin = SlateDbWalAdmin::new(object_store, Arc::new(FailPointRegistry::new()));

        let _collector = wal_admin.garbage_collector(&Path::from("db"));
    }
}
