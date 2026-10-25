//! The SlateDB policy for an [`ObjectStoreMirror`](slatedb_mirror::ObjectStoreMirror)
//! (RFC 0034): keep a local copy of every SST that the latest manifest
//! references.
//!
//! Manifests are observed on every read and write. A newer manifest replaces
//! the retained set with its own SSTs plus those of its active checkpoints,
//! evicts SSTs no longer retained, and downloads its own SSTs before the
//! caller gets the manifest. Compacted SSTs (L0 and sorted runs) are read
//! locally and written to both stores. WAL SSTs and every other object stay
//! remote.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use log::{info, warn};
use object_store::path::Path;
use object_store::{GetOptions, ObjectStoreExt, PutMode, PutMultipartOptions, PutOptions};
use parking_lot::Mutex;
use slatedb_common::clock::{DefaultSystemClock, SystemClock};
use slatedb_mirror::{MirrorHandle, MirrorPolicy, MultipartRoute, ReadRoute, WriteRoute};
use slatedb_txn_obj::ObjectCodec;

use crate::compactions_store::CompactionsStore;
use crate::compactor_state::Compactions;
use crate::flatbuffer_types::FlatBufferManifestCodec;
use crate::manifest::Manifest;
use crate::object_store_tag::{ObjectStoreCallTag, SstType};
use crate::paths::PathResolver;

const POLICY_STORE_NAME: &str = "SlateDbMirrorPolicy";
const MANIFEST_DIR: &str = "manifest";
const MANIFEST_SUFFIX: &str = ".manifest";
const DEFAULT_COMPACTIONS_POLL_INTERVAL: Duration = Duration::from_secs(5);
/// How often the policy logs the local SSTs that nothing references.
const AUDIT_INTERVAL: Duration = Duration::from_secs(60);
/// How long the policy remembers an SST that is neither local nor retained.
const HISTORY_RETENTION: Duration = Duration::from_secs(1800);

/// Keeps a local copy of every SST that the latest manifest references.
pub struct SlateDbMirrorPolicy {
    state: Mutex<PolicyState>,
    compactions_poll_interval: Option<Duration>,
    system_clock: Arc<dyn SystemClock>,
}

#[derive(Default)]
struct PolicyState {
    /// The database root, recorded from the first manifest seen.
    root: Option<Path>,
    /// The ID of the newest manifest processed.
    newest_manifest: Option<u64>,
    /// SSTs that must not be evicted: the newest manifest's own SSTs plus
    /// those of its active checkpoints.
    retain: HashSet<Path>,
    /// SSTs of the active checkpoints' manifests, by manifest ID. Manifests
    /// are immutable, so entries never go stale.
    checkpoints: HashMap<u64, Arc<HashSet<Path>>>,
    /// What the policy saw happen to each SST, for the audit log.
    history: HashMap<Path, SstHistory>,
    /// Local SSTs that the last audit found unreferenced.
    unreferenced: HashSet<Path>,
    /// Unreferenced SSTs the audit already logged.
    reported: HashSet<Path>,
}

/// What the policy saw happen to one SST.
#[derive(Default)]
struct SstHistory {
    /// How the local copy got here, and when: "recovered" from an earlier
    /// process, "written" by this process, or "prefetched" as the output of a
    /// running compaction. `None` for SSTs downloaded because a manifest
    /// referenced them.
    arrived: Option<(&'static str, DateTime<Utc>)>,
    /// When a newer manifest last retained it.
    retained: Option<DateTime<Utc>>,
    /// When the policy last evicted it.
    evicted: Option<DateTime<Utc>>,
}

impl SstHistory {
    fn last_event(&self) -> Option<DateTime<Utc>> {
        [self.arrived.map(|(_, at)| at), self.retained, self.evicted]
            .into_iter()
            .flatten()
            .max()
    }
}

impl SlateDbMirrorPolicy {
    pub fn builder() -> SlateDbMirrorPolicyBuilder {
        SlateDbMirrorPolicyBuilder {
            compactions_poll_interval: Some(DEFAULT_COMPACTIONS_POLL_INTERVAL),
            system_clock: Arc::new(DefaultSystemClock::new()),
        }
    }

    fn check_root(&self, root: &Path) -> object_store::Result<()> {
        match &self.state.lock().root {
            Some(known) if known != root => Err(policy_error(format!(
                "manifest {root} is outside the mirrored database {known}"
            ))),
            _ => Ok(()),
        }
    }

    /// Reads a checkpoint's manifest from the wrapped store and returns its
    /// SSTs. A manifest that GC already deleted retains nothing.
    async fn checkpoint_ssts(
        mirror: &MirrorHandle,
        root: &Path,
        manifest_id: u64,
    ) -> object_store::Result<HashSet<Path>> {
        let path = manifest_path(root, manifest_id);
        let bytes = match mirror.remote().get(&path).await {
            Ok(result) => result.bytes().await?,
            Err(object_store::Error::NotFound { .. }) => {
                warn!("checkpoint manifest missing, not retaining its SSTs [path={path}]");
                return Ok(HashSet::new());
            }
            Err(error) => return Err(error),
        };
        Ok(sst_paths(root, &decode_manifest(&bytes)?))
    }
}

pub struct SlateDbMirrorPolicyBuilder {
    compactions_poll_interval: Option<Duration>,
    system_clock: Arc<dyn SystemClock>,
}

impl SlateDbMirrorPolicyBuilder {
    /// Sets how often the policy reads the compaction state to prefetch the
    /// output of running compactions, including the partial output a worker
    /// needs to resume a compaction that another worker started. `None`
    /// disables prefetching, so new compaction output is downloaded when a
    /// manifest that references it is read, and a compaction that another
    /// worker left partly done cannot be resumed. The default is 5 seconds.
    pub fn with_compactions_poll_interval(mut self, interval: Option<Duration>) -> Self {
        self.compactions_poll_interval = interval;
        self
    }

    pub fn with_system_clock(mut self, system_clock: Arc<dyn SystemClock>) -> Self {
        self.system_clock = system_clock;
        self
    }

    pub fn build(self) -> Arc<SlateDbMirrorPolicy> {
        Arc::new(SlateDbMirrorPolicy {
            state: Mutex::new(PolicyState::default()),
            compactions_poll_interval: self.compactions_poll_interval,
            system_clock: self.system_clock,
        })
    }
}

#[async_trait]
impl MirrorPolicy for SlateDbMirrorPolicy {
    fn read_route(&self, path: &Path, options: &GetOptions) -> object_store::Result<ReadRoute> {
        if let Some((root, _)) = manifest_location(path) {
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
        if let Some((root, _)) = manifest_location(path) {
            self.check_root(&root)?;
            return Ok(WriteRoute::Observe);
        }
        if compacted_sst_tag(&options.extensions).is_none() {
            return Ok(WriteRoute::Remote);
        }
        // Mirrored paths must be write-once.
        if matches!(options.mode, PutMode::Update(_)) {
            return Err(policy_error(format!("conditional update of SST {path}")));
        }
        self.record_arrival(path, "written", true);
        Ok(WriteRoute::Mirror)
    }

    fn put_multipart_route(
        &self,
        path: &Path,
        options: &PutMultipartOptions,
    ) -> object_store::Result<MultipartRoute> {
        Ok(match compacted_sst_tag(&options.extensions) {
            Some(_) => {
                self.record_arrival(path, "written", true);
                MultipartRoute::Mirror
            }
            None => MultipartRoute::Remote,
        })
    }

    async fn observe(
        &self,
        path: &Path,
        bytes: &Bytes,
        mirror: &MirrorHandle,
    ) -> object_store::Result<()> {
        let Some((root, manifest_id)) = manifest_location(path) else {
            return Ok(());
        };
        let manifest = decode_manifest(bytes)?;
        let warm = sst_paths(&root, &manifest);
        let checkpoint_ids: HashSet<u64> = manifest
            .core
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.manifest_id)
            .collect();

        // Take references to the cached checkpoint manifests while holding the
        // lock, then read the missing ones without it.
        let (newer, first, mut checkpoints, missing) = {
            let mut state = self.state.lock();
            state.root.get_or_insert_with(|| root.clone());
            let first = state.newest_manifest.is_none();
            let newer = state
                .newest_manifest
                .is_none_or(|newest| manifest_id > newest);
            let mut cached = HashMap::new();
            let mut missing = Vec::new();
            if newer {
                for id in &checkpoint_ids {
                    match state.checkpoints.get(id) {
                        Some(ssts) => {
                            cached.insert(*id, Arc::clone(ssts));
                        }
                        None => missing.push(*id),
                    }
                }
            }
            (newer, first, cached, missing)
        };
        if newer {
            for id in missing {
                let ssts = Self::checkpoint_ssts(mirror, &root, id).await?;
                checkpoints.insert(id, Arc::new(ssts));
            }
        }
        // The first manifest sweeps out local copies that an earlier process
        // left behind. Running compactions' output stays for the worker that
        // resumes them.
        let running = if newer && first {
            Some(Self::running_compaction_ssts(mirror, &root).await)
        } else {
            None
        };

        let warm: Vec<Path> = {
            let mut state = self.state.lock();
            if state
                .newest_manifest
                .is_none_or(|newest| manifest_id > newest)
            {
                let mut retain = warm.clone();
                for ssts in checkpoints.values() {
                    retain.extend(ssts.iter().cloned());
                }
                let now = self.system_clock.now();
                let dropped: Vec<Path> = state.retain.difference(&retain).cloned().collect();
                for path in &dropped {
                    state.history.entry(path.clone()).or_default().evicted = Some(now);
                }
                for path in &retain {
                    state.history.entry(path.clone()).or_default().retained = Some(now);
                }
                // Evictions are issued under the lock, so they reach the
                // mirror in manifest order.
                mirror.evict(dropped);
                if state.newest_manifest.is_none() {
                    let running = running.clone().unwrap_or_default();
                    let prefix = compacted_prefix(&root);
                    let mut unused = Vec::new();
                    let mut unused_bytes = 0;
                    for (path, size) in mirror.local_objects() {
                        let history = state.history.entry(path.clone()).or_default();
                        history.arrived.get_or_insert(("recovered", now));
                        if path.as_ref().starts_with(&prefix)
                            && !retain.contains(&path)
                            && !running.contains(&path)
                        {
                            unused_bytes += size;
                            unused.push(path);
                        }
                    }
                    info!(
                        "evicting local SSTs that the first manifest does not reference [manifest_id={}, count={}, bytes={}]",
                        manifest_id,
                        unused.len(),
                        unused_bytes
                    );
                    mirror.evict(unused);
                }
                state.retain = retain;
                state.newest_manifest = Some(manifest_id);
                state.checkpoints = checkpoints;
                warm.into_iter().collect()
            } else {
                // An older manifest only warms SSTs that are still retained.
                warm.into_iter()
                    .filter(|sst| state.retain.contains(sst))
                    .collect()
            }
        };
        mirror.fetch(warm).await
    }

    async fn run(self: Arc<Self>, mirror: MirrorHandle) {
        let Some(interval) = self.compactions_poll_interval else {
            return;
        };
        let mut last_audit = self.system_clock.now();
        loop {
            self.system_clock.sleep(interval).await;
            let Some(root) = self.state.lock().root.clone() else {
                continue;
            };
            let store = CompactionsStore::new(&root, Arc::clone(mirror.remote()));
            let compactions = match store.try_read_latest_compactions().await {
                Ok(Some(compactions)) => compactions,
                Ok(None) => continue,
                Err(error) => {
                    warn!("failed to read compactions for prefetch [error={error}]");
                    continue;
                }
            };
            let running = active_compaction_ssts(&root, &compactions.compactions);
            for path in &running {
                if !mirror.contains(path) {
                    self.record_arrival(path, "prefetched", false);
                }
            }
            mirror.prefetch(running.iter().cloned());
            let now = self.system_clock.now();
            if (now - last_audit)
                .to_std()
                .is_ok_and(|age| age >= AUDIT_INTERVAL)
            {
                self.audit(&root, &mirror, &running, now);
                last_audit = now;
            }
        }
    }
}

impl SlateDbMirrorPolicy {
    /// Records how an SST's local copy arrived. `replace` overwrites an
    /// earlier arrival.
    fn record_arrival(&self, path: &Path, how: &'static str, replace: bool) {
        let now = self.system_clock.now();
        let mut state = self.state.lock();
        let history = state.history.entry(path.clone()).or_default();
        if replace || history.arrived.is_none() {
            history.arrived = Some((how, now));
        }
    }

    /// Reads the outputs of the running compactions. Returns none when the
    /// compaction state can't be read.
    async fn running_compaction_ssts(mirror: &MirrorHandle, root: &Path) -> HashSet<Path> {
        let store = CompactionsStore::new(root, Arc::clone(mirror.remote()));
        match store.try_read_latest_compactions().await {
            Ok(Some(compactions)) => active_compaction_ssts(root, &compactions.compactions),
            Ok(None) => HashSet::new(),
            Err(error) => {
                warn!("failed to read compactions for the first manifest [error={error}]");
                HashSet::new()
            }
        }
    }

    /// Logs the local SSTs that neither the retained set nor a running
    /// compaction references. An SST is logged once it stays unreferenced for
    /// two audits in a row, with what the policy saw happen to it.
    fn audit(
        &self,
        root: &Path,
        mirror: &MirrorHandle,
        running: &HashSet<Path>,
        now: DateTime<Utc>,
    ) {
        let local = mirror.local_objects();
        let prefix = compacted_prefix(root);
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let ago = |at: Option<DateTime<Utc>>| at.map(|at| (now - at).num_seconds());
        let mut unreferenced = HashSet::new();
        let (mut local_bytes, mut unreferenced_bytes) = (0, 0);
        for (path, size) in &local {
            local_bytes += size;
            if !path.as_ref().starts_with(&prefix)
                || state.retain.contains(path)
                || running.contains(path)
            {
                continue;
            }
            unreferenced_bytes += size;
            unreferenced.insert(path.clone());
            if state.unreferenced.contains(path) && state.reported.insert(path.clone()) {
                let history = state.history.get(path);
                info!(
                    "mirror audit: unreferenced local SST [path={}, bytes={}, arrived={:?}, arrived_secs_ago={:?}, retained_secs_ago={:?}, evicted_secs_ago={:?}]",
                    path,
                    size,
                    history.and_then(|h| h.arrived.map(|(how, _)| how)),
                    history.and_then(|h| ago(h.arrived.map(|(_, at)| at))),
                    history.and_then(|h| ago(h.retained)),
                    history.and_then(|h| ago(h.evicted)),
                );
            }
        }
        info!(
            "mirror audit [local_files={}, local_bytes={}, unreferenced_files={}, unreferenced_bytes={}, retained_files={}]",
            local.len(),
            local_bytes,
            unreferenced.len(),
            unreferenced_bytes,
            state.retain.len()
        );
        state.reported.retain(|path| unreferenced.contains(path));
        state.unreferenced = unreferenced;
        let local: HashSet<&Path> = local.iter().map(|(path, _)| path).collect();
        let retain = &state.retain;
        state.history.retain(|path, history| {
            local.contains(path)
                || retain.contains(path)
                || history
                    .last_event()
                    .and_then(|at| (now - at).to_std().ok())
                    .is_some_and(|age| age < HISTORY_RETENTION)
        });
    }
}

/// The prefix of the database's compacted SSTs, the only objects the mirror
/// keeps locally.
fn compacted_prefix(root: &Path) -> String {
    format!(
        "{}/",
        PathResolver::from_root(root.clone()).compacted_path()
    )
}

/// The SSTs that active compactions have written: their recorded outputs and
/// their subcompactions' partial outputs, which a worker resuming a compaction
/// that another worker started reads back.
fn active_compaction_ssts(root: &Path, compactions: &Compactions) -> HashSet<Path> {
    let resolver = PathResolver::from_root(root.clone());
    compactions
        .iter_active()
        .flat_map(|compaction| {
            compaction.output_ssts().into_iter().chain(
                compaction
                    .subcompactions()
                    .iter()
                    .flat_map(|subcompaction| subcompaction.output_ssts().iter().cloned()),
            )
        })
        .map(|sst| resolver.sst_path(&sst.id))
        .collect()
}

/// Returns the tag of a compacted SST call. WAL SSTs and untagged calls have
/// none.
fn compacted_sst_tag(extensions: &object_store::Extensions) -> Option<ObjectStoreCallTag> {
    ObjectStoreCallTag::from_extensions(extensions).filter(|tag| tag.sst_type == SstType::Compacted)
}

/// Splits `<root>/manifest/<id>.manifest` into the root and the manifest ID.
fn manifest_location(path: &Path) -> Option<(Path, u64)> {
    let parts: Vec<_> = path.parts().collect();
    let [root @ .., dir, file] = parts.as_slice() else {
        return None;
    };
    if dir.as_ref() != MANIFEST_DIR {
        return None;
    }
    let id = file.as_ref().strip_suffix(MANIFEST_SUFFIX)?.parse().ok()?;
    Some((Path::from_iter(root.iter().cloned()), id))
}

fn manifest_path(root: &Path, manifest_id: u64) -> Path {
    root.clone()
        .join(MANIFEST_DIR)
        .join(format!("{manifest_id:020}{MANIFEST_SUFFIX}"))
}

fn decode_manifest(bytes: &Bytes) -> object_store::Result<Manifest> {
    FlatBufferManifestCodec {}
        .decode(bytes)
        .map_err(|source| object_store::Error::Generic {
            store: POLICY_STORE_NAME,
            source,
        })
}

/// Every SST that a manifest references, in all trees. External SSTs resolve
/// under their database's path.
fn sst_paths(root: &Path, manifest: &Manifest) -> HashSet<Path> {
    let resolver = PathResolver::new_with_external_ssts(root.clone(), manifest.external_ssts());
    manifest
        .core
        .all_sst_views()
        .map(|view| resolver.sst_path(&view.sst.id))
        .collect()
}

fn policy_error(message: String) -> object_store::Error {
    object_store::Error::Generic {
        store: POLICY_STORE_NAME,
        source: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytes_range::BytesRange;
    use crate::compactions_store::StoredCompactions;
    use crate::compactor_state::{
        Compaction, CompactionContext, CompactionSpec, CompactionStatus, SourceId,
    };
    use crate::config::{DbReaderOptions, FlushOptions, FlushType};
    use crate::db_state::{SsTableHandle, SsTableId, SsTableInfo, SsTableView};
    use crate::error::RetryReason;
    use crate::format::sst::SST_FORMAT_VERSION_LATEST;
    use crate::manifest::store::ManifestStore;
    use crate::manifest::ManifestCore;
    use crate::object_store_tag::TableStoreKind;
    use crate::subcompaction::Subcompaction;
    use crate::{Db, DbReader, DbReaderMode};
    use object_store::memory::InMemory;
    use object_store::{ObjectStore, UpdateVersion};
    use slatedb_mirror::ObjectStoreMirror;
    use ulid::Ulid;

    fn compacted_tag() -> ObjectStoreCallTag {
        ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Compacted)
    }

    fn get_options(tag: Option<ObjectStoreCallTag>) -> GetOptions {
        GetOptions {
            extensions: tag.map(Into::into).unwrap_or_default(),
            ..GetOptions::default()
        }
    }

    fn put_options(tag: Option<ObjectStoreCallTag>, mode: PutMode) -> PutOptions {
        PutOptions {
            mode,
            extensions: tag.map(Into::into).unwrap_or_default(),
            ..PutOptions::default()
        }
    }

    #[test]
    fn test_manifest_location() {
        assert_eq!(
            manifest_location(&Path::from("db/manifest/00000000000000000007.manifest")),
            Some((Path::from("db"), 7))
        );
        assert_eq!(
            manifest_path(&Path::from("db"), 7),
            Path::from("db/manifest/00000000000000000007.manifest")
        );
        assert_eq!(manifest_location(&Path::from("db/manifest/boundary")), None);
        assert_eq!(manifest_location(&Path::from("db/compacted/x.sst")), None);
    }

    #[test]
    fn test_read_route() {
        let policy = SlateDbMirrorPolicy::builder().build();
        let sst = Path::from("db/compacted/01J.sst");
        let mut retry = compacted_tag();
        retry.retry = Some(RetryReason::CrcMismatch);
        let wal = ObjectStoreCallTag::new(TableStoreKind::Main, SstType::Wal);

        let route = |path: &Path, tag| policy.read_route(path, &get_options(tag)).unwrap();
        assert_eq!(
            route(
                &Path::from("db/manifest/00000000000000000001.manifest"),
                None
            ),
            ReadRoute::Observe
        );
        assert_eq!(route(&sst, Some(compacted_tag())), ReadRoute::Local);
        assert_eq!(route(&sst, Some(retry)), ReadRoute::Refetch);
        assert_eq!(
            route(&Path::from("db/wal/1.sst"), Some(wal)),
            ReadRoute::Remote
        );
        assert_eq!(route(&sst, None), ReadRoute::Remote);
    }

    #[test]
    fn test_put_route() {
        let policy = SlateDbMirrorPolicy::builder().build();
        let sst = Path::from("db/compacted/01J.sst");
        let route = |path: &Path, tag, mode| policy.put_route(path, &put_options(tag, mode));

        assert_eq!(
            route(
                &Path::from("db/manifest/00000000000000000001.manifest"),
                None,
                PutMode::Create
            )
            .unwrap(),
            WriteRoute::Observe
        );
        assert_eq!(
            route(&sst, Some(compacted_tag()), PutMode::Create).unwrap(),
            WriteRoute::Mirror
        );
        assert_eq!(
            route(&sst, None, PutMode::Overwrite).unwrap(),
            WriteRoute::Remote
        );
        let update = PutMode::Update(UpdateVersion {
            e_tag: None,
            version: None,
        });
        assert!(route(&sst, Some(compacted_tag()), update).is_err());
    }

    #[test]
    fn test_active_compaction_ssts() {
        let root = Path::from("db");
        let resolver = PathResolver::from_root(root.clone());
        let sst = || {
            SsTableHandle::new(
                SsTableId::from(Ulid::new()),
                SST_FORMAT_VERSION_LATEST,
                SsTableInfo::default(),
            )
        };
        let spec = || CompactionSpec::new(vec![SourceId::SortedRun(1)], 1);
        let (partial, recorded, finished) = (sst(), sst(), sst());
        // A compaction whose worker died partway through has output only in
        // its subcompactions.
        let resumable =
            Compaction::new(Ulid::new(), spec()).with_ctx(Some(CompactionContext::new(
                vec![
                    Subcompaction::new(BytesRange::from_slice(..b"m".as_slice()))
                        .with_output_ssts(vec![partial.clone()]),
                    Subcompaction::new(BytesRange::from_slice(b"m".as_slice()..)),
                ],
                None,
            )));
        let compacted = Compaction::new(Ulid::new(), spec())
            .with_status(CompactionStatus::Compacted)
            .with_output_ssts(vec![recorded.clone()]);
        let done = Compaction::new(Ulid::new(), spec())
            .with_status(CompactionStatus::Completed)
            .with_output_ssts(vec![finished]);
        let compactions = Compactions::new(1).with_compactions(vec![resumable, compacted, done]);

        let expected: HashSet<Path> = [partial, recorded]
            .iter()
            .map(|sst| resolver.sst_path(&sst.id))
            .collect();
        assert_eq!(active_compaction_ssts(&root, &compactions), expected);
    }

    async fn mirror(
        remote: Arc<dyn ObjectStore>,
        dir: &tempfile::TempDir,
    ) -> Arc<ObjectStoreMirror> {
        ObjectStoreMirror::builder(dir.path(), remote, SlateDbMirrorPolicy::builder().build())
            .with_remote_scan_interval(None)
            .build()
            .await
            .unwrap()
    }

    async fn latest_ssts(remote: Arc<dyn ObjectStore>, root: &Path) -> HashSet<Path> {
        let manifest = ManifestStore::new(root, remote)
            .read_latest_manifest()
            .await
            .unwrap();
        sst_paths(root, &manifest.manifest)
    }

    fn manifest_bytes(ids: &[Ulid]) -> Bytes {
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).l0 = ids
            .iter()
            .map(|id| {
                SsTableView::identity(SsTableHandle::new(
                    SsTableId::from(*id),
                    SST_FORMAT_VERSION_LATEST,
                    // A decoded manifest needs each SST's first key.
                    SsTableInfo {
                        first_entry: Some(Bytes::from_static(b"a")),
                        ..SsTableInfo::default()
                    },
                ))
            })
            .collect();
        FlatBufferManifestCodec {}.encode(&Manifest::initial(core))
    }

    async fn wait_until(mut done: impl FnMut() -> bool) -> bool {
        for _ in 0..200 {
            if done() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        done()
    }

    #[tokio::test]
    async fn test_observe_evicts_unreferenced_local_ssts_on_the_first_manifest() {
        let remote: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let root = Path::from("db");
        let store = Arc::new(CompactionsStore::new(&root, Arc::clone(&remote)));
        let mut compactions = StoredCompactions::create(store, 1).await.unwrap();
        // Compaction IDs and output SSTs can't be older than the compaction
        // state's clock, so the last SST is new.
        let compaction_id = Ulid::new();
        let ids: Vec<Ulid> = (1..=3)
            .map(|n| Ulid::from_parts(n, 0))
            .chain([Ulid::new()])
            .collect();
        let resolver = PathResolver::from_root(root.clone());
        let ssts: Vec<Path> = ids
            .iter()
            .map(|id| resolver.sst_path(&SsTableId::from(*id)))
            .collect();
        for sst in &ssts {
            remote
                .put(sst, Bytes::from_static(b"sst").into())
                .await
                .unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let open = |policy: Arc<SlateDbMirrorPolicy>| {
            ObjectStoreMirror::builder(dir.path(), Arc::clone(&remote), policy)
                .with_remote_scan_interval(None)
                .build()
        };

        // An earlier process downloaded all four SSTs.
        {
            let policy = SlateDbMirrorPolicy::builder()
                .with_compactions_poll_interval(None)
                .build();
            let mirror = open(policy.clone()).await.unwrap();
            policy
                .observe(
                    &manifest_path(&root, 1),
                    &manifest_bytes(&ids),
                    &mirror.handle(),
                )
                .await
                .unwrap();
        }
        // The fourth SST is the output of a running compaction.
        let output = SsTableHandle::new(
            SsTableId::from(ids[3]),
            SST_FORMAT_VERSION_LATEST,
            SsTableInfo::default(),
        );
        let running = Compaction::new(
            compaction_id,
            CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
        )
        .with_status(CompactionStatus::Compacted)
        .with_output_ssts(vec![output]);
        let mut dirty = compactions.prepare_dirty().unwrap();
        dirty.value.insert(running);
        compactions.update(dirty).await.unwrap();

        // The next process's first manifest references only the second and third.
        let policy = SlateDbMirrorPolicy::builder()
            .with_compactions_poll_interval(None)
            .build();
        let mirror = open(policy.clone()).await.unwrap();
        let handle = mirror.handle();
        assert!(ssts.iter().all(|sst| handle.contains(sst)));
        policy
            .observe(
                &manifest_path(&root, 2),
                &manifest_bytes(&ids[1..3]),
                &handle,
            )
            .await
            .unwrap();

        assert!(wait_until(|| !handle.contains(&ssts[0])).await);
        assert!(ssts[1..].iter().all(|sst| handle.contains(sst)));
    }

    #[tokio::test]
    async fn test_observe_evicts_ssts_a_newer_manifest_drops() {
        let remote: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let root = Path::from("db");
        let ids: Vec<Ulid> = (1..=3).map(|n| Ulid::from_parts(n, 0)).collect();
        let resolver = PathResolver::from_root(root.clone());
        let ssts: Vec<Path> = ids
            .iter()
            .map(|id| resolver.sst_path(&SsTableId::from(*id)))
            .collect();
        for sst in &ssts {
            remote
                .put(sst, Bytes::from_static(b"sst").into())
                .await
                .unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let policy = SlateDbMirrorPolicy::builder()
            .with_compactions_poll_interval(None)
            .build();
        let mirror = ObjectStoreMirror::builder(dir.path(), Arc::clone(&remote), policy.clone())
            .with_remote_scan_interval(None)
            .build()
            .await
            .unwrap();
        let handle = mirror.handle();

        policy
            .observe(
                &manifest_path(&root, 1),
                &manifest_bytes(&ids[..2]),
                &handle,
            )
            .await
            .unwrap();
        assert!(handle.contains(&ssts[0]) && handle.contains(&ssts[1]));

        // Manifest 2 drops the first SST and adds the third.
        policy
            .observe(
                &manifest_path(&root, 2),
                &manifest_bytes(&ids[1..]),
                &handle,
            )
            .await
            .unwrap();
        assert!(handle.contains(&ssts[2]));
        assert!(wait_until(|| !handle.contains(&ssts[0])).await);

        // A late read of manifest 1 doesn't download what manifest 2 dropped.
        policy
            .observe(
                &manifest_path(&root, 1),
                &manifest_bytes(&ids[..2]),
                &handle,
            )
            .await
            .unwrap();
        assert!(!handle.contains(&ssts[0]));
        assert!(handle.contains(&ssts[1]));
    }

    #[tokio::test]
    async fn test_observe_downloads_manifest_ssts_for_reader() {
        let remote: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let root = Path::from("db");
        let writer_dir = tempfile::tempdir().unwrap();
        let reader_dir = tempfile::tempdir().unwrap();
        let writer_mirror = mirror(Arc::clone(&remote), &writer_dir).await;
        let db = Db::builder(root.clone(), writer_mirror.clone() as Arc<dyn ObjectStore>)
            .build()
            .await
            .unwrap();
        let flush = FlushOptions {
            flush_type: FlushType::MemTable,
        };
        db.put(b"a", b"1").await.unwrap();
        db.flush_with_options(flush.clone()).await.unwrap();
        let first = latest_ssts(Arc::clone(&remote), &root).await;
        assert!(!first.is_empty());
        // The writer mirrors its own SSTs as it uploads them.
        assert!(first.iter().all(|sst| writer_mirror.handle().contains(sst)));

        let reader_mirror = mirror(Arc::clone(&remote), &reader_dir).await;
        let reader = DbReader::builder(root.clone(), reader_mirror.clone() as Arc<dyn ObjectStore>)
            .with_reader_mode(DbReaderMode::FollowLatest)
            .with_options(DbReaderOptions {
                manifest_poll_interval: Duration::from_millis(50),
                skip_wal_replay: true,
                ..DbReaderOptions::default()
            })
            .build()
            .await
            .unwrap();
        // Opening the reader reads the manifest, which downloads its SSTs.
        assert!(first.iter().all(|sst| reader_mirror.handle().contains(sst)));

        db.put(b"b", b"2").await.unwrap();
        db.flush_with_options(flush).await.unwrap();
        let second = latest_ssts(Arc::clone(&remote), &root).await;
        assert!(second.len() > first.len());
        // The reader's next poll downloads the new SST.
        assert!(
            wait_until(|| second
                .iter()
                .all(|sst| reader_mirror.handle().contains(sst)))
            .await
        );

        reader.close().await.unwrap();
        db.close().await.unwrap();
    }
}
