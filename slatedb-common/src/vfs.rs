//! A small asynchronous local filesystem abstraction.
//!
//! Components that do their local I/O through [`Vfs`] can swap in other
//! implementations (io_uring, simulation). [`StdVfs`] is backed by the
//! standard filesystem.

use std::fmt::Debug;
use std::fs::TryLockError;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use crate::file_handle_cache::{read_exact_at_offset, FileHandleCache};
use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::AsyncWriteExt;
use tokio::sync::oneshot;

/// A regular file in a [`Vfs::list`] result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfsEntry {
    /// The file name, without its directory.
    pub name: String,
    /// The file size in bytes.
    pub size: u64,
}

/// Asynchronous local filesystem operations.
///
/// A `Vfs` doesn't cache, evict, or know about object stores. It only moves
/// bytes to and from local files.
#[async_trait]
pub trait Vfs: Debug + Send + Sync + 'static {
    /// Creates `path` and any missing parent directories.
    async fn create_dir_all(&self, path: &Path) -> io::Result<()>;

    /// Takes an exclusive lock on `path`, creating the file if it doesn't
    /// exist. The lock is held until the returned guard is dropped, and must
    /// be released if the process dies. Fails with
    /// [`io::ErrorKind::WouldBlock`] if someone else holds it. The file is left
    /// in place when the lock is released.
    async fn lock(&self, path: &Path) -> io::Result<Box<dyn VfsLock>>;

    /// Lists the regular files directly in `dir`. Subdirectories and names
    /// that aren't valid UTF-8 are skipped.
    async fn list(&self, dir: &Path) -> io::Result<Vec<VfsEntry>>;

    /// Reads `range` from `path`. Fails with [`io::ErrorKind::UnexpectedEof`]
    /// if the file ends before `range.end`.
    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<Bytes>;

    /// Reads all of `path`.
    async fn read(&self, path: &Path) -> io::Result<Bytes>;

    /// Creates a new file at `path` and returns a writer for it. Fails with
    /// [`io::ErrorKind::AlreadyExists`] if `path` exists.
    async fn create(&self, path: &Path) -> io::Result<Box<dyn VfsWriter>>;

    /// Renames `from` to `to`, atomically replacing `to` if it exists.
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;

    /// Removes the file at `path`. Fails with [`io::ErrorKind::NotFound`] if it
    /// doesn't exist.
    async fn remove(&self, path: &Path) -> io::Result<()>;
}

/// Writes a file created by [`Vfs::create`], in order.
#[async_trait]
pub trait VfsWriter: Debug + Send {
    /// Appends `bytes` to the file.
    async fn write(&mut self, bytes: Bytes) -> io::Result<()>;

    /// Flushes buffered writes and waits for pending writes to complete.
    /// Call this once, after the last write.
    /// This does not guarantee that the contents survive a machine crash.
    async fn finish(&mut self) -> io::Result<()>;
}

/// An exclusive lock taken by [`Vfs::lock`]. Dropping it releases the lock.
pub trait VfsLock: Debug + Send + Sync {}

/// A [`Vfs`] backed by the standard filesystem.
///
/// Range reads reuse open files. By default, each read runs in one blocking
/// task. [`StdVfs::with_read_threads`] runs them on dedicated threads instead.
/// Clones share the cache of open files and the read threads.
/// The cache retains at most 1,000 open files by default.
/// Writes are flushed before publication, without syncing them to disk.
#[derive(Debug, Clone)]
pub struct StdVfs {
    file_handle_cache: FileHandleCache,
    read_pool: Option<Arc<ReadPool>>,
}

impl Default for StdVfs {
    fn default() -> Self {
        Self {
            file_handle_cache: FileHandleCache::new(1000),
            read_pool: None,
        }
    }
}

impl StdVfs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of open files retained by the cache.
    ///
    /// # Panics
    ///
    /// Panics if `max_open_file_handles` is zero.
    pub fn with_max_open_file_handles(mut self, max_open_file_handles: usize) -> Self {
        self.file_handle_cache = FileHandleCache::new(max_open_file_handles);
        self
    }

    /// Runs range reads on `threads` dedicated threads, each with its own
    /// queue. The threads stop when the last clone is dropped.
    ///
    /// Fails if `threads` is zero or a thread cannot start.
    pub fn with_read_threads(mut self, threads: usize) -> io::Result<Self> {
        self.read_pool = Some(Arc::new(ReadPool::new(threads)?));
        Ok(self)
    }
}

/// Reads `len` bytes at `offset` from `path`, through `file_handle_cache`.
fn read_at(
    file_handle_cache: &FileHandleCache,
    path: &Path,
    offset: u64,
    len: usize,
) -> io::Result<Bytes> {
    let handle = file_handle_cache.try_get_or_open(path)?;
    let mut buf = vec![0; len];
    read_exact_at_offset(&handle, &mut buf, offset)?;
    Ok(Bytes::from(buf))
}

/// A range read for a [`ReadPool`] thread.
struct ReadJob {
    file_handle_cache: FileHandleCache,
    path: PathBuf,
    offset: u64,
    len: usize,
    reply: oneshot::Sender<io::Result<Bytes>>,
}

/// Dedicated threads for range reads, each with its own queue.
///
/// `spawn_blocking` allocates a task for each read. It also takes the lock of
/// the blocking pool to queue the task, and the worker thread takes it again
/// to dequeue it. This pool sends each read to the next queue in turn, with no
/// shared lock. A slow read delays the reads queued behind it on that thread.
#[derive(Debug)]
struct ReadPool {
    queues: Vec<mpsc::Sender<ReadJob>>,
    next: AtomicUsize,
}

impl ReadPool {
    // Read threads must not run on the async runtime.
    #[allow(clippy::disallowed_types)]
    fn new(threads: usize) -> io::Result<Self> {
        if threads == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read pool needs at least one thread",
            ));
        }
        let mut queues = Vec::with_capacity(threads);
        for i in 0..threads {
            let (sender, receiver) = mpsc::channel::<ReadJob>();
            std::thread::Builder::new()
                .name(format!("slatedb-vfs-read-{i}"))
                .spawn(move || {
                    // Stops when the pool drops its sender.
                    while let Ok(job) = receiver.recv() {
                        let result =
                            read_at(&job.file_handle_cache, &job.path, job.offset, job.len);
                        // The caller can stop waiting for the result.
                        let _ = job.reply.send(result);
                    }
                })?;
            queues.push(sender);
        }
        Ok(Self {
            queues,
            next: AtomicUsize::new(0),
        })
    }

    async fn read(
        &self,
        file_handle_cache: FileHandleCache,
        path: PathBuf,
        offset: u64,
        len: usize,
    ) -> io::Result<Bytes> {
        let stopped = || io::Error::other("read pool thread stopped");
        let (reply, result) = oneshot::channel();
        let queue = self.next.fetch_add(1, Ordering::Relaxed) % self.queues.len();
        self.queues[queue]
            .send(ReadJob {
                file_handle_cache,
                path,
                offset,
                len,
                reply,
            })
            .map_err(|_| stopped())?;
        result.await.map_err(|_| stopped())?
    }
}

#[async_trait]
impl Vfs for StdVfs {
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        tokio::fs::create_dir_all(path).await
    }

    async fn lock(&self, path: &Path) -> io::Result<Box<dyn VfsLock>> {
        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .await?
            .into_std()
            .await;
        match file.try_lock() {
            Ok(()) => Ok(Box::new(StdVfsLock { _file: file })),
            Err(TryLockError::WouldBlock) => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("`{}` is locked by another process", path.display()),
            )),
            Err(TryLockError::Error(err)) => Err(err),
        }
    }

    async fn list(&self, dir: &Path) -> io::Result<Vec<VfsEntry>> {
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(dir).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            let metadata = match entry.metadata().await {
                Ok(metadata) => metadata,
                // Removed between read_dir and metadata.
                Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            };
            if metadata.is_file() {
                entries.push(VfsEntry {
                    name,
                    size: metadata.len(),
                });
            }
        }
        Ok(entries)
    }

    async fn read_range(&self, path: &Path, range: Range<u64>) -> io::Result<Bytes> {
        let len = usize::try_from(range.end.saturating_sub(range.start))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "range is too large"))?;
        let path = path.to_path_buf();
        let file_handle_cache = self.file_handle_cache.clone();
        if let Some(read_pool) = &self.read_pool {
            return read_pool
                .read(file_handle_cache, path, range.start, len)
                .await;
        }
        #[allow(clippy::disallowed_methods)]
        tokio::task::spawn_blocking(move || read_at(&file_handle_cache, &path, range.start, len))
            .await?
    }

    async fn read(&self, path: &Path) -> io::Result<Bytes> {
        tokio::fs::read(path).await.map(Bytes::from)
    }

    async fn create(&self, path: &Path) -> io::Result<Box<dyn VfsWriter>> {
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .await?;
        Ok(Box::new(StdVfsWriter { file }))
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let from = from.to_path_buf();
        let to = to.to_path_buf();
        let file_handle_cache = self.file_handle_cache.clone();
        #[allow(clippy::disallowed_methods)]
        tokio::task::spawn_blocking(move || {
            std::fs::rename(&from, &to)?;
            // Keep invalidation in this task so cancellation cannot skip it.
            file_handle_cache.invalidate(&from);
            file_handle_cache.invalidate(&to);
            Ok(())
        })
        .await?
    }

    async fn remove(&self, path: &Path) -> io::Result<()> {
        let path = path.to_path_buf();
        let file_handle_cache = self.file_handle_cache.clone();
        #[allow(clippy::disallowed_methods)]
        tokio::task::spawn_blocking(move || {
            let result = std::fs::remove_file(&path);
            file_handle_cache.invalidate(&path);
            result
        })
        .await?
    }
}

#[derive(Debug)]
struct StdVfsLock {
    // The OS releases the lock when the file is closed.
    _file: std::fs::File,
}

impl VfsLock for StdVfsLock {}

#[derive(Debug)]
struct StdVfsWriter {
    file: tokio::fs::File,
}

#[async_trait]
impl VfsWriter for StdVfsWriter {
    async fn write(&mut self, bytes: Bytes) -> io::Result<()> {
        self.file.write_all(&bytes).await
    }

    async fn finish(&mut self) -> io::Result<()> {
        self.file.flush().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn should_write_read_and_list_files() {
        let dir = tempfile::tempdir().unwrap();
        let vfs = StdVfs::new();
        let path = dir.path().join("a");

        let mut writer = vfs.create(&path).await.unwrap();
        writer.write(Bytes::from_static(b"hello ")).await.unwrap();
        writer.write(Bytes::from_static(b"world")).await.unwrap();
        writer.finish().await.unwrap();

        assert_eq!(vfs.read(&path).await.unwrap(), "hello world");
        assert_eq!(vfs.read_range(&path, 6..11).await.unwrap(), "world");
        let err = vfs.read_range(&path, 6..12).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);

        vfs.create_dir_all(&dir.path().join("sub")).await.unwrap();
        assert_eq!(
            vfs.list(dir.path()).await.unwrap(),
            vec![VfsEntry {
                name: "a".to_string(),
                size: 11
            }]
        );
    }

    #[tokio::test]
    async fn should_not_create_over_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let vfs = StdVfs::new();
        let path = dir.path().join("a");
        vfs.create(&path).await.unwrap();
        let err = vfs.create(&path).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
    }

    #[tokio::test]
    async fn should_rename_over_and_remove_files() {
        let dir = tempfile::tempdir().unwrap();
        let vfs = StdVfs::new();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        for (path, contents) in [(&a, "new"), (&b, "old")] {
            let mut writer = vfs.create(path).await.unwrap();
            writer.write(Bytes::from(contents)).await.unwrap();
            writer.finish().await.unwrap();
        }

        // Keep the old file linked so the cache needs explicit invalidation.
        std::fs::hard_link(&b, dir.path().join("old-link")).unwrap();
        assert_eq!(vfs.read_range(&a, 0..3).await.unwrap(), "new");
        assert_eq!(vfs.read_range(&b, 0..3).await.unwrap(), "old");
        let reader = vfs.clone();

        vfs.rename(&a, &b).await.unwrap();
        assert_eq!(reader.read_range(&b, 0..3).await.unwrap(), "new");
        let err = reader.read_range(&a, 0..3).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);

        let mut writer = vfs.create(&a).await.unwrap();
        writer.write(Bytes::from_static(b"fresh")).await.unwrap();
        writer.finish().await.unwrap();
        assert_eq!(reader.read_range(&a, 0..5).await.unwrap(), "fresh");

        std::fs::hard_link(&b, dir.path().join("new-link")).unwrap();
        vfs.remove(&b).await.unwrap();
        let err = reader.read_range(&b, 0..3).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        let err = vfs.remove(&b).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn should_read_ranges_concurrently_through_shared_cache() {
        read_ranges_concurrently(StdVfs::new().with_max_open_file_handles(2)).await;
    }

    #[tokio::test]
    async fn should_read_ranges_concurrently_on_read_threads() {
        let vfs = StdVfs::new()
            .with_max_open_file_handles(2)
            .with_read_threads(3)
            .unwrap();
        read_ranges_concurrently(vfs).await;
    }

    #[test]
    fn should_reject_zero_read_threads() {
        let err = StdVfs::new().with_read_threads(0).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    async fn read_ranges_concurrently(vfs: StdVfs) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"0123456789").unwrap();
        let missing = vfs.read_range(&dir.path().join("missing"), 0..1).await;
        assert_eq!(missing.unwrap_err().kind(), io::ErrorKind::NotFound);
        let mut tasks = tokio::task::JoinSet::new();
        for offset in 0..10 {
            let vfs = vfs.clone();
            let path = path.clone();
            tasks.spawn(async move {
                for _ in 0..100 {
                    let bytes = vfs.read_range(&path, offset..offset + 1).await.unwrap();
                    assert_eq!(bytes[0], b'0' + offset as u8);
                }
            });
        }
        while let Some(result) = tasks.join_next().await {
            result.unwrap();
        }
        assert_eq!(vfs.file_handle_cache.len(), 1);

        assert!(vfs.read_range(&path, 10..10).await.unwrap().is_empty());
        let err = vfs.read_range(&path, 9..11).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn should_hold_lock_until_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let vfs = StdVfs::new();
        let path = dir.path().join("LOCK");

        let lock = vfs.lock(&path).await.unwrap();
        let err = vfs.lock(&path).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);

        drop(lock);
        vfs.lock(&path).await.unwrap();
        assert!(path.exists());
    }
}
