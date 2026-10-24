//! State shared by the mirror, its handles, and its background tasks.

use std::collections::HashMap;
use std::future::Future;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io;
use std::path::{Path as StdPath, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use futures::future;
use futures::stream::{self, StreamExt};
use log::warn;
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, GetResultPayload, ObjectStore, ObjectStoreExt, PutPayload,
};
use parking_lot::{Mutex, RwLock};
use slatedb_common::clock::SystemClock;
use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;

use crate::download::Downloads;
use crate::error::MIRROR_STORE_NAME;
use crate::layout::{LocalName, LocalObject, PrefixMap};
use crate::ordering::{OpKind, PathOrdering};
use crate::retry::Retry;
use crate::{MirrorError, MirrorPolicy};
use slatedb_common::vfs::{Vfs, VfsLock};

pub(crate) fn local_error(source: io::Error) -> MirrorError {
    MirrorError::Local { source }
}

/// Removes `path`, ignoring `NotFound` and logging other failures.
pub(crate) async fn remove_quietly(vfs: &dyn Vfs, path: &StdPath) {
    match vfs.remove(path).await {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => warn!(
            "failed to remove mirror file [path={}, error={}]",
            path.display(),
            err
        ),
    }
}

/// The number of shards in [`LocalEntries`].
const LOCAL_ENTRY_SHARDS: usize = 64;

/// One shard of [`LocalEntries`].
type LocalEntryShard = RwLock<HashMap<Path, Arc<LocalObject>>>;

/// Every complete local copy, by object path.
///
/// Every local read looks up its path here. The map is split into shards with
/// a read-write lock each, so lookups of different paths do not wait for one
/// lock.
pub(crate) struct LocalEntries {
    shards: Box<[LocalEntryShard]>,
}

impl LocalEntries {
    fn new(entries: HashMap<Path, Arc<LocalObject>>) -> Self {
        let local_entries = Self {
            shards: (0..LOCAL_ENTRY_SHARDS)
                .map(|_| RwLock::new(HashMap::new()))
                .collect(),
        };
        for (path, object) in entries {
            local_entries.insert(path, object);
        }
        local_entries
    }

    fn shard(&self, path: &Path) -> &LocalEntryShard {
        let mut hasher = DefaultHasher::new();
        path.hash(&mut hasher);
        &self.shards[(hasher.finish() % self.shards.len() as u64) as usize]
    }

    pub(crate) fn get(&self, path: &Path) -> Option<Arc<LocalObject>> {
        self.shard(path).read().get(path).cloned()
    }

    pub(crate) fn contains(&self, path: &Path) -> bool {
        self.shard(path).read().contains_key(path)
    }

    pub(crate) fn insert(&self, path: Path, object: Arc<LocalObject>) {
        self.shard(&path).write().insert(path, object);
    }

    pub(crate) fn remove(&self, path: &Path) -> Option<Arc<LocalObject>> {
        self.shard(path).write().remove(path)
    }

    /// Returns every local copy. A copy that is installed or removed during
    /// the call can be in the result or not.
    pub(crate) fn values(&self) -> Vec<Arc<LocalObject>> {
        self.shards
            .iter()
            .flat_map(|shard| shard.read().values().cloned().collect::<Vec<_>>())
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn clear(&self) {
        for shard in self.shards.iter() {
            shard.write().clear();
        }
    }
}

pub(crate) struct Inner {
    pub(crate) root: PathBuf,
    pub(crate) remote: Arc<dyn ObjectStore>,
    pub(crate) policy: Arc<dyn MirrorPolicy>,
    pub(crate) vfs: Arc<dyn Vfs>,
    pub(crate) clock: Arc<dyn SystemClock>,
    pub(crate) retry: Retry,
    pub(crate) runtime: Handle,
    /// Cancelled when the `ObjectStoreMirror` is dropped. Stops background
    /// tasks.
    pub(crate) shutdown: CancellationToken,
    /// Every complete local copy, by object path.
    pub(crate) entries: LocalEntries,
    pub(crate) prefixes: Mutex<PrefixMap>,
    pub(crate) ordering: PathOrdering,
    pub(crate) downloads: Downloads,
    temp_counter: AtomicU64,
    _lock: Box<dyn VfsLock>,
}

impl Inner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        root: PathBuf,
        remote: Arc<dyn ObjectStore>,
        policy: Arc<dyn MirrorPolicy>,
        vfs: Arc<dyn Vfs>,
        clock: Arc<dyn SystemClock>,
        retry: Retry,
        runtime: Handle,
        entries: HashMap<Path, Arc<LocalObject>>,
        prefixes: PrefixMap,
        download_concurrency: usize,
        lock: Box<dyn VfsLock>,
    ) -> Self {
        Self {
            root,
            remote,
            policy,
            vfs,
            clock,
            retry,
            runtime,
            shutdown: CancellationToken::new(),
            entries: LocalEntries::new(entries),
            prefixes: Mutex::new(prefixes),
            ordering: PathOrdering::default(),
            downloads: Downloads::new(download_concurrency),
            temp_counter: AtomicU64::new(0),
            _lock: lock,
        }
    }

    pub(crate) fn file(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    pub(crate) fn local(&self, path: &Path) -> Option<Arc<LocalObject>> {
        self.entries.get(path)
    }

    /// Returns a guard for a new temporary data file for `name`.
    pub(crate) fn temp_file(&self, name: &LocalName) -> TempFile {
        self.temp(name.temp(self.temp_counter.fetch_add(1, Ordering::Relaxed)))
    }

    fn temp(&self, file_name: String) -> TempFile {
        TempFile {
            vfs: Arc::clone(&self.vfs),
            runtime: self.runtime.clone(),
            path: self.file(&file_name),
            armed: true,
        }
    }

    /// Runs `future` in the background until it finishes or the mirror is
    /// dropped.
    pub(crate) fn spawn<F>(&self, future: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let shutdown = self.shutdown.clone();
        self.runtime.spawn(async move {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {}
                _ = future => {}
            }
        });
    }

    /// Writes `payload` to a new file at `path` and flushes it.
    pub(crate) async fn write_payload(
        &self,
        path: &StdPath,
        payload: &PutPayload,
    ) -> io::Result<()> {
        let mut writer = self.vfs.create(path).await?;
        for chunk in payload {
            writer.write(chunk.clone()).await?;
        }
        writer.finish().await
    }

    /// Publishes a complete temporary file as the local copy of `path`,
    /// replacing any existing copy. Writes `.meta` first, then renames the data
    /// file into place, so a crash never leaves a data file without its
    /// metadata. `temp` is removed on failure.
    pub(crate) async fn install(
        &self,
        path: &Path,
        name: &LocalName,
        temp: TempFile,
        object: LocalObject,
    ) -> Result<Arc<LocalObject>, MirrorError> {
        let result = self.install_files(name, &temp, &object).await;
        if let Err(err) = result {
            temp.remove().await;
            return Err(err);
        }
        temp.disarm();
        let object = Arc::new(object);
        self.entries.insert(path.clone(), Arc::clone(&object));
        Ok(object)
    }

    async fn install_files(
        &self,
        name: &LocalName,
        temp: &TempFile,
        object: &LocalObject,
    ) -> Result<(), MirrorError> {
        self.prefixes.lock().check_or_insert(name)?;

        let meta = PutPayload::from(Bytes::from(object.to_meta_bytes()?));
        let meta_temp =
            self.temp(name.meta_temp(self.temp_counter.fetch_add(1, Ordering::Relaxed)));
        let written = match self.write_payload(&meta_temp.path, &meta).await {
            Ok(()) => {
                self.vfs
                    .rename(&meta_temp.path, &self.file(&name.meta()))
                    .await
            }
            Err(err) => Err(err),
        };
        if let Err(err) = written {
            meta_temp.remove().await;
            return Err(local_error(err));
        }
        meta_temp.disarm();

        self.vfs
            .rename(&temp.path, &self.file(&name.object_file_name))
            .await
            .map_err(local_error)
    }

    /// Removes the local copy of `path`, if there is one. The caller must hold
    /// `path`'s turn in the ordering.
    pub(crate) async fn remove_local(&self, path: &Path) -> Result<(), MirrorError> {
        if self.entries.remove(path).is_none() {
            return Ok(());
        }
        self.remove_files(path).await
    }

    /// Removes `path`'s data and `.meta` files if they exist, whether or not
    /// the in-memory map has them. The caller must hold `path`'s turn in the
    /// ordering and have removed it from the map.
    async fn remove_files(&self, path: &Path) -> Result<(), MirrorError> {
        let name = LocalName::new(path);
        // Data first, so a crash never leaves a data file without its
        // metadata.
        for file in [&name.object_file_name, &name.meta()] {
            match self.vfs.remove(&self.file(file)).await {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(local_error(err)),
            }
        }
        Ok(())
    }

    /// Deletes `path` remotely and removes its local copy. Both run in
    /// parallel under one turn in the ordering, so the remote delete can't
    /// overtake an earlier write to the same path. Neither side's failure
    /// stops the other.
    pub(crate) async fn delete(&self, path: &Path) -> object_store::Result<()> {
        let ticket = self
            .ordering
            .register(path, OpKind::Delete)
            .expect("deletes are always queued");
        ticket.ready().await;
        let (remote, local) = future::join(self.remote.delete(path), self.remove_local(path)).await;
        drop(ticket);
        remote?;
        local?;
        Ok(())
    }

    /// Like [`Self::delete`] for several paths, but deletes them from the remote
    /// store with one `delete_stream` call, so a store with a bulk delete API
    /// deletes them with few requests. Returns one result per path, in order. A
    /// path that the remote call did not delete gets an error. `paths` must not
    /// repeat a path, because a second ticket for a path waits for the first.
    pub(crate) async fn delete_batch(&self, paths: Vec<Path>) -> Vec<object_store::Result<Path>> {
        let tickets: Vec<_> = paths
            .iter()
            .map(|path| {
                self.ordering
                    .register(path, OpKind::Delete)
                    .expect("deletes are always queued")
            })
            .collect();
        future::join_all(tickets.iter().map(|ticket| ticket.ready())).await;
        let remote = self
            .remote
            .delete_stream(stream::iter(paths.clone().into_iter().map(Ok)).boxed())
            .collect::<Vec<_>>();
        let local = future::join_all(paths.iter().map(|path| self.remove_local(path)));
        let (remote, local) = future::join(remote, local).await;
        drop(tickets);
        let mut deleted = std::collections::HashSet::with_capacity(paths.len());
        for result in remote {
            match result {
                Ok(path) => {
                    deleted.insert(path);
                }
                Err(object_store::Error::NotFound { path, .. }) => {
                    deleted.insert(Path::from(path));
                }
                Err(error) => warn!("bulk delete failed [error={error}]"),
            }
        }
        paths
            .into_iter()
            .zip(local)
            .map(|(path, local)| {
                if !deleted.contains(&path) {
                    return Err(object_store::Error::Generic {
                        store: MIRROR_STORE_NAME,
                        source: format!("the bulk delete did not delete {path}").into(),
                    });
                }
                local?;
                Ok(path)
            })
            .collect()
    }

    /// Like [`Self::delete`]'s local half, but removes the files even if the
    /// in-memory map doesn't have `path`. Used by the remote scan, which finds
    /// paths on disk.
    pub(crate) async fn purge_local(&self, path: &Path) -> Result<(), MirrorError> {
        let ticket = self
            .ordering
            .register(path, OpKind::Delete)
            .expect("deletes are always queued");
        ticket.ready().await;
        self.entries.remove(path);
        self.remove_files(path).await
    }

    /// Serves a GET or HEAD from the local copy described by `object`.
    pub(crate) async fn read_local(
        &self,
        path: &Path,
        object: &LocalObject,
        options: &GetOptions,
    ) -> object_store::Result<GetResult> {
        let meta = &object.meta;
        if options
            .version
            .as_ref()
            .is_some_and(|version| meta.version.as_ref() != Some(version))
        {
            // The local copy is some other version.
            return Err(MirrorError::NotLocal { path: path.clone() }.into());
        }
        options.check_preconditions(meta)?;

        let (range, bytes) = if options.head {
            (0..meta.size, Bytes::new())
        } else {
            let range =
                match &options.range {
                    Some(range) => range.as_range(meta.size).map_err(|source| {
                        object_store::Error::Generic {
                            store: MIRROR_STORE_NAME,
                            source: Box::new(source),
                        }
                    })?,
                    None => 0..meta.size,
                };
            let bytes = if range.is_empty() {
                Bytes::new()
            } else {
                let name = LocalName::new(path);
                self.vfs
                    .read_range(&self.file(&name.object_file_name), range.clone())
                    .await
                    .map_err(|err| match err.kind() {
                        // Evicted since we looked it up.
                        io::ErrorKind::NotFound => MirrorError::NotLocal { path: path.clone() },
                        _ => local_error(err),
                    })?
            };
            (range, bytes)
        };

        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::once(async move { Ok(bytes) }).boxed()),
            meta: meta.clone(),
            range,
            attributes: object.attributes.clone(),
            extensions: Default::default(),
        })
    }
}

/// A temporary file that is removed unless it's renamed into place.
///
/// Dropping an armed guard removes the file in the background, which covers
/// futures that are cancelled mid-write.
#[derive(Debug)]
pub(crate) struct TempFile {
    vfs: Arc<dyn Vfs>,
    runtime: Handle,
    pub(crate) path: PathBuf,
    armed: bool,
}

impl TempFile {
    /// Removes the file now.
    pub(crate) async fn remove(mut self) {
        self.armed = false;
        remove_quietly(self.vfs.as_ref(), &self.path).await;
    }

    /// Keeps the file. Call after it has been renamed away.
    pub(crate) fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if self.armed {
            let vfs = Arc::clone(&self.vfs);
            let path = std::mem::take(&mut self.path);
            self.runtime
                .spawn(async move { remove_quietly(vfs.as_ref(), &path).await });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::Arc;

    use chrono::DateTime;
    use object_store::path::Path;
    use object_store::{Attributes, ObjectMeta};

    use super::LocalEntries;
    use crate::layout::LocalObject;

    fn object(path: &Path) -> Arc<LocalObject> {
        Arc::new(LocalObject {
            meta: ObjectMeta {
                location: path.clone(),
                last_modified: DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
                size: 1,
                e_tag: None,
                version: None,
            },
            attributes: Attributes::new(),
        })
    }

    #[test]
    fn should_find_and_remove_entries_in_every_shard() {
        let paths: Vec<_> = (0..256)
            .map(|i| Path::from(format!("db/compacted/{i:04}.sst")))
            .collect();
        let entries = LocalEntries::new(
            paths[..128]
                .iter()
                .map(|path| (path.clone(), object(path)))
                .collect::<HashMap<_, _>>(),
        );
        for path in &paths[128..] {
            entries.insert(path.clone(), object(path));
        }
        for path in &paths {
            assert!(entries.contains(path));
            assert_eq!(&entries.get(path).unwrap().meta.location, path);
        }
        assert_eq!(entries.values().len(), paths.len());

        for path in paths.iter().step_by(2) {
            assert!(entries.remove(path).is_some());
            assert!(entries.remove(path).is_none());
        }
        let remaining: BTreeSet<_> = entries
            .values()
            .into_iter()
            .map(|object| object.meta.location.clone())
            .collect();
        let expected: BTreeSet<_> = paths.iter().skip(1).step_by(2).cloned().collect();
        assert_eq!(remaining, expected);
        assert!(!entries.contains(&paths[0]));
        assert!(entries.get(&paths[0]).is_none());
    }
}
