//! [`SlateDbMirrorPolicy`]: the SlateDB rules for an
//! [`ObjectStoreMirror`](slatedb_mirror::ObjectStoreMirror) (RFC 0034).
//!
//! Compacted and L0 SSTs are mirrored: reads are served only from local
//! files, writes go to both stores. Everything else (WAL, manifests,
//! compaction state, untagged calls) stays remote. Manifests are observed to
//! warm the mirror and to evict SSTs that no manifest or active checkpoint
//! references anymore.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use log::warn;
use object_store::path::Path;
use object_store::{
    GetOptions, ObjectStore, ObjectStoreExt, PutMode, PutMultipartOptions, PutOptions,
};
use parking_lot::Mutex;
use slatedb_mirror::{MirrorHandle, MirrorPolicy, ReadRoute, WriteRoute};
use slatedb_txn_obj::ObjectCodec;

use crate::db_state::SstType;
use crate::flatbuffer_types::FlatBufferManifestCodec;
use crate::manifest::Manifest;
use crate::object_store_tag::ObjectStoreCallTag;
use crate::paths::PathResolver;

const MANIFEST_DIR: &str = "manifest";
const MANIFEST_SUFFIX: &str = ".manifest";
const POLICY_NAME: &str = "SlateDbMirrorPolicy";

/// Mirrors the SSTs of one SlateDB database.
///
/// The database root is taken from the first manifest the policy sees;
/// manifests under any other root are rejected. External SSTs referenced by
/// the database may live under other roots.
///
/// The mirror must have room for the whole database: compacted SST reads are
/// served only from local files, so the policy never evicts an SST that a
/// manifest still references.
#[derive(Debug, Default)]
pub struct SlateDbMirrorPolicy {
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    root: Option<Path>,
    /// The newest manifest that moved `retain` forward.
    newest: Option<u64>,
    /// SSTs that must not be evicted: those of `newest` and of its active
    /// checkpoints. Being retained doesn't mean being local.
    retain: HashSet<Path>,
    /// Decoded checkpoint manifests by ID. Manifests are immutable.
    checkpoints: HashMap<u64, Arc<Manifest>>,
}

impl SlateDbMirrorPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks `root` against the database root, recording it if this is the
    /// first manifest seen.
    fn check_root(&self, root: &Path) -> object_store::Result<()> {
        let mut state = self.state.lock();
        match &state.root {
            None => {
                state.root = Some(root.clone());
                Ok(())
            }
            Some(known) if known == root => Ok(()),
            Some(known) => Err(policy_error(format!(
                "manifest under root `{root}` but the mirror serves `{known}`"
            ))),
        }
    }
}

#[async_trait]
impl MirrorPolicy for SlateDbMirrorPolicy {
    fn read_route(&self, path: &Path, options: &GetOptions) -> object_store::Result<ReadRoute> {
        if let Some((root, _)) = parse_manifest_path(path) {
            self.check_root(&root)?;
            return Ok(ReadRoute::Observe);
        }
        Ok(match compacted_sst_tag(&options.extensions) {
            Some(tag) if tag.retry.is_some() => ReadRoute::Refetch,
            Some(_) => ReadRoute::Local,
            None => ReadRoute::Remote,
        })
    }

    fn put_route(&self, path: &Path, options: &PutOptions) -> object_store::Result<WriteRoute> {
        if let Some((root, _)) = parse_manifest_path(path) {
            self.check_root(&root)?;
            return Ok(WriteRoute::Observe);
        }
        if compacted_sst_tag(&options.extensions).is_none() {
            return Ok(WriteRoute::Remote);
        }
        if matches!(options.mode, PutMode::Update(_)) {
            return Err(policy_error(format!(
                "SST `{path}` is written with PutMode::Update, but mirrored SSTs must be immutable"
            )));
        }
        Ok(WriteRoute::Mirror)
    }

    fn put_multipart_route(
        &self,
        _path: &Path,
        options: &PutMultipartOptions,
    ) -> object_store::Result<WriteRoute> {
        Ok(match compacted_sst_tag(&options.extensions) {
            Some(_) => WriteRoute::Mirror,
            None => WriteRoute::Remote,
        })
    }

    async fn observe(
        &self,
        path: &Path,
        bytes: &Bytes,
        mirror: &MirrorHandle,
    ) -> object_store::Result<()> {
        let Some((root, id)) = parse_manifest_path(path) else {
            return Ok(());
        };
        let manifest = decode_manifest(path, bytes)?;
        let mut warm = sst_paths(&root, &manifest);

        // Under the lock: is this manifest newer, and which of its active
        // checkpoints still need decoding? Remote reads happen without it.
        let (newer, cached, missing) = {
            let state = self.state.lock();
            if state.newest.is_some_and(|newest| id <= newest) {
                (false, Vec::new(), Vec::new())
            } else {
                let mut cached = Vec::new();
                let mut missing = Vec::new();
                for checkpoint in &manifest.core.checkpoints {
                    match state.checkpoints.get(&checkpoint.manifest_id) {
                        Some(decoded) => cached.push((checkpoint.manifest_id, decoded.clone())),
                        None => missing.push(checkpoint.manifest_id),
                    }
                }
                (true, cached, missing)
            }
        };

        let mut checkpoints = cached;
        if newer {
            for checkpoint_id in missing {
                if checkpoint_id == id {
                    checkpoints.push((id, Arc::new(manifest.clone())));
                    continue;
                }
                // Read through the wrapped store so this doesn't loop back
                // into `observe`.
                let checkpoint_path = manifest_path(&root, checkpoint_id);
                match read_manifest(mirror.remote(), &checkpoint_path).await {
                    Ok(decoded) => checkpoints.push((checkpoint_id, Arc::new(decoded))),
                    // Its SSTs then aren't retained. They're only read by a
                    // reader opened on the checkpoint, which reads (and so
                    // observes) the manifest itself.
                    Err(err) => warn!(
                        "mirror policy couldn't read checkpoint manifest [path={}, error={}]",
                        checkpoint_path, err
                    ),
                }
            }
        }

        {
            let mut state = self.state.lock();
            if newer && state.newest.is_none_or(|newest| id > newest) {
                let mut retain = warm.clone();
                for (_, checkpoint) in &checkpoints {
                    retain.extend(sst_paths(&root, checkpoint));
                }
                mirror.evict(state.retain.difference(&retain).cloned());
                state.retain = retain;
                state.newest = Some(id);
                for (checkpoint_id, checkpoint) in checkpoints {
                    state.checkpoints.insert(checkpoint_id, checkpoint);
                }
                let active: HashSet<u64> = manifest
                    .core
                    .checkpoints
                    .iter()
                    .map(|checkpoint| checkpoint.manifest_id)
                    .collect();
                state
                    .checkpoints
                    .retain(|checkpoint_id, _| active.contains(checkpoint_id));
            } else {
                // An older manifest only warms SSTs that are still retained.
                warm.retain(|sst| state.retain.contains(sst));
            }
        }

        mirror.fetch(warm).await
    }
}

/// The tag of a call for a compacted (or L0) SST, if it is one.
fn compacted_sst_tag(extensions: &object_store::Extensions) -> Option<ObjectStoreCallTag> {
    ObjectStoreCallTag::from_extensions(extensions).filter(|tag| tag.sst_type == SstType::Compacted)
}

/// Splits `<root>/manifest/<id>.manifest` into its root and manifest ID.
fn parse_manifest_path(path: &Path) -> Option<(Path, u64)> {
    let parts: Vec<_> = path.parts().collect();
    let (file, rest) = parts.split_last()?;
    let (dir, root) = rest.split_last()?;
    if dir.as_ref() != MANIFEST_DIR {
        return None;
    }
    let id = file.as_ref().strip_suffix(MANIFEST_SUFFIX)?.parse().ok()?;
    Some((Path::from_iter(root.iter().cloned()), id))
}

fn manifest_path(root: &Path, id: u64) -> Path {
    root.clone()
        .join(MANIFEST_DIR)
        .join(format!("{id:020}{MANIFEST_SUFFIX}"))
}

/// The paths of every SST a manifest references, in every tree.
fn sst_paths(root: &Path, manifest: &Manifest) -> HashSet<Path> {
    let resolver = PathResolver::new_with_external_ssts(root.clone(), manifest.external_ssts());
    manifest
        .core
        .all_sst_views()
        .map(|view| resolver.sst_path(&view.sst.id))
        .collect()
}

fn decode_manifest(path: &Path, bytes: &Bytes) -> object_store::Result<Manifest> {
    FlatBufferManifestCodec {}
        .decode(bytes)
        .map_err(|err| policy_error(format!("couldn't decode manifest `{path}`: {err}")))
}

async fn read_manifest(
    store: &Arc<dyn ObjectStore>,
    path: &Path,
) -> object_store::Result<Manifest> {
    let bytes = store.get(path).await?.bytes().await?;
    decode_manifest(path, &bytes)
}

fn policy_error(message: String) -> object_store::Error {
    object_store::Error::Generic {
        store: POLICY_NAME,
        source: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FlushOptions, FlushType, Settings};
    use crate::Db;
    use object_store::memory::InMemory;
    use slatedb_mirror::ObjectStoreMirror;

    async fn open(
        remote: Arc<dyn ObjectStore>,
        local_dir: &std::path::Path,
    ) -> (Db, Arc<ObjectStoreMirror>) {
        let mirror =
            ObjectStoreMirror::builder(local_dir, remote, Arc::new(SlateDbMirrorPolicy::new()))
                .with_remote_scan_interval(None)
                .build()
                .await
                .unwrap();
        let db = Db::builder("db", mirror.clone() as Arc<dyn ObjectStore>)
            .with_settings(Settings {
                compactor_options: None,
                ..Settings::default()
            })
            .build()
            .await
            .unwrap();
        (db, mirror)
    }

    fn local_ssts(local_dir: &std::path::Path) -> usize {
        fn count(dir: &std::path::Path) -> usize {
            std::fs::read_dir(dir)
                .unwrap()
                .map(|entry| {
                    let path = entry.unwrap().path();
                    if path.is_dir() {
                        count(&path)
                    } else {
                        usize::from(path.extension().is_some_and(|ext| ext == "sst"))
                    }
                })
                .sum()
        }
        count(local_dir)
    }

    #[tokio::test]
    async fn test_db_mirrors_and_warms_compacted_ssts() {
        let remote: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let first_dir = tempfile::tempdir().unwrap();
        let (db, _mirror) = open(remote.clone(), first_dir.path()).await;
        db.put(b"key", b"value").await.unwrap();
        db.flush_with_options(FlushOptions {
            flush_type: FlushType::MemTable,
        })
        .await
        .unwrap();
        // The L0 SST was written through the mirror, so it is local.
        assert_eq!(local_ssts(first_dir.path()), 1);
        assert_eq!(
            db.get(b"key").await.unwrap().as_deref(),
            Some(&b"value"[..])
        );
        db.close().await.unwrap();

        // A new mirror starts empty and is warmed from the manifest on open,
        // so the read is served locally.
        let second_dir = tempfile::tempdir().unwrap();
        let (db, _mirror) = open(remote, second_dir.path()).await;
        assert_eq!(local_ssts(second_dir.path()), 1);
        assert_eq!(
            db.get(b"key").await.unwrap().as_deref(),
            Some(&b"value"[..])
        );
        db.close().await.unwrap();
    }

    #[test]
    fn test_parse_manifest_path() {
        assert_eq!(
            parse_manifest_path(&Path::from("db/manifest/00000000000000000042.manifest")),
            Some((Path::from("db"), 42))
        );
        assert_eq!(
            parse_manifest_path(&Path::from("a/b/manifest/00000000000000000001.manifest")),
            Some((Path::from("a/b"), 1))
        );
        assert_eq!(
            parse_manifest_path(&Path::from("manifest/00000000000000000007.manifest")),
            Some((Path::from(""), 7))
        );
        assert_eq!(
            parse_manifest_path(&Path::from("db/compacted/01ABC.sst")),
            None
        );
        assert_eq!(
            parse_manifest_path(&Path::from("db/manifest/boundary")),
            None
        );
        assert_eq!(
            manifest_path(&Path::from("db"), 42),
            Path::from("db/manifest/00000000000000000042.manifest")
        );
    }

    #[test]
    fn test_routes() {
        let policy = SlateDbMirrorPolicy::new();
        let tagged = |sst_type| {
            ObjectStoreCallTag::new(crate::object_store_tag::TableStoreKind::Main, sst_type)
        };
        let get = |tag: Option<ObjectStoreCallTag>| GetOptions {
            extensions: tag.map(Into::into).unwrap_or_default(),
            ..GetOptions::default()
        };
        let sst = Path::from("db/compacted/01ABC.sst");
        let manifest = Path::from("db/manifest/00000000000000000001.manifest");

        assert_eq!(
            policy
                .read_route(&sst, &get(Some(tagged(SstType::Compacted))))
                .unwrap(),
            ReadRoute::Local
        );
        let mut retry = tagged(SstType::Compacted);
        retry.retry = Some(crate::object_store_tag::RetryReason::CrcMismatch);
        assert_eq!(
            policy.read_route(&sst, &get(Some(retry))).unwrap(),
            ReadRoute::Refetch
        );
        assert_eq!(
            policy
                .read_route(&sst, &get(Some(tagged(SstType::Wal))))
                .unwrap(),
            ReadRoute::Remote
        );
        assert_eq!(
            policy.read_route(&sst, &get(None)).unwrap(),
            ReadRoute::Remote
        );
        assert_eq!(
            policy.read_route(&manifest, &get(None)).unwrap(),
            ReadRoute::Observe
        );

        let put = |tag: Option<ObjectStoreCallTag>, mode| PutOptions {
            mode,
            extensions: tag.map(Into::into).unwrap_or_default(),
            ..PutOptions::default()
        };
        assert_eq!(
            policy
                .put_route(
                    &sst,
                    &put(Some(tagged(SstType::Compacted)), PutMode::Create)
                )
                .unwrap(),
            WriteRoute::Mirror
        );
        assert!(policy
            .put_route(
                &sst,
                &put(
                    Some(tagged(SstType::Compacted)),
                    PutMode::Update(object_store::UpdateVersion {
                        e_tag: None,
                        version: None
                    })
                )
            )
            .is_err());
        assert_eq!(
            policy
                .put_route(&manifest, &put(None, PutMode::Create))
                .unwrap(),
            WriteRoute::Observe
        );

        // Manifests under another root are rejected.
        let other = Path::from("other/manifest/00000000000000000001.manifest");
        assert!(policy.read_route(&other, &get(None)).is_err());
    }
}
