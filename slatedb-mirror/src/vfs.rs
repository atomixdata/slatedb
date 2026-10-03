//! Local filesystem access for the mirror.
//!
//! The mirror does all of its local I/O through [`Vfs`]. [`StdVfs`] is the
//! default. Other implementations (io_uring, simulation) can be passed to
//! `ObjectStoreMirrorBuilder::with_vfs`.

use std::fmt::Debug;
use std::fs::TryLockError;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::io::AsyncWriteExt;

/// A regular file in a [`Vfs::list`] result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfsEntry {
    /// The file name, without its directory.
    pub name: String,
    /// The file size in bytes.
    pub size: u64,
}

/// The local filesystem operations the mirror needs.
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

    /// Flushes buffered writes and syncs the file's contents to disk. The
    /// mirror calls this once, before renaming the file into place.
    async fn finish(&mut self) -> io::Result<()>;
}

/// An exclusive lock taken by [`Vfs::lock`]. Dropping it releases the lock.
pub trait VfsLock: Debug + Send + Sync {}

/// The default for [`StdVfs::with_max_open_files`].
pub const DEFAULT_MAX_OPEN_FILES: usize = 4096;

/// A [`Vfs`] backed by the standard filesystem through `tokio::fs`.
///
/// Range reads keep files open in a bounded cache, so a read of a file that is
/// already open is one positional read on the blocking pool, without an open
/// or a seek.
#[derive(Clone)]
pub struct StdVfs {
    handles: Arc<quick_cache::sync::Cache<PathBuf, Arc<std::fs::File>>>,
}

impl Debug for StdVfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StdVfs")
            .field("open_files", &self.handles.len())
            .finish()
    }
}

impl Default for StdVfs {
    fn default() -> Self {
        Self::new()
    }
}

impl StdVfs {
    pub fn new() -> Self {
        Self::with_max_open_files(DEFAULT_MAX_OPEN_FILES)
    }

    /// Keeps at most `max_open_files` files open for range reads.
    pub fn with_max_open_files(max_open_files: usize) -> Self {
        Self {
            handles: Arc::new(quick_cache::sync::Cache::new(max_open_files.max(1))),
        }
    }

    /// Returns an open handle for `path`, opening and caching it on a miss.
    ///
    /// A handle cached for a file that is then replaced or removed would keep
    /// reading the old contents, so `rename` and `remove` invalidate the paths
    /// they change. If one of them runs between the `open` and the `insert`
    /// below, its invalidation comes too early; the `fstat` after the insert
    /// catches that case and drops the entry.
    fn open_cached(&self, path: &Path) -> io::Result<Arc<std::fs::File>> {
        if let Some(file) = self.handles.get(path) {
            return Ok(file);
        }
        let file = Arc::new(std::fs::File::open(path)?);
        self.handles.insert(path.to_path_buf(), file.clone());
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if file.metadata().map_or(true, |m| m.nlink() == 0) {
                self.handles.remove(path);
            }
        }
        Ok(file)
    }

    fn invalidate(&self, path: &Path) {
        self.handles.remove(path);
    }
}

/// Reads exactly `buf.len()` bytes at `offset` without moving the file
/// cursor, so readers can share one handle.
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut read = 0;
        while read < buf.len() {
            let n = file.seek_read(&mut buf[read..], offset + read as u64)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            read += n;
        }
        Ok(())
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
        let vfs = self.clone();
        let path = path.to_path_buf();
        // One trip to the blocking pool per read, like `tokio::fs`, but with
        // the open skipped for cached files and no separate seek.
        #[allow(clippy::disallowed_methods)]
        tokio::task::spawn_blocking(move || {
            let file = vfs.open_cached(&path)?;
            let mut buf = vec![0; len];
            read_exact_at(&file, &mut buf, range.start)?;
            Ok(Bytes::from(buf))
        })
        .await
        .map_err(io::Error::other)?
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
        let result = tokio::fs::rename(from, to).await;
        self.invalidate(from);
        self.invalidate(to);
        result
    }

    async fn remove(&self, path: &Path) -> io::Result<()> {
        let result = tokio::fs::remove_file(path).await;
        self.invalidate(path);
        result
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
        self.file.flush().await?;
        self.file.sync_all().await
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

        vfs.rename(&a, &b).await.unwrap();
        assert_eq!(vfs.read(&b).await.unwrap(), "new");

        vfs.remove(&b).await.unwrap();
        let err = vfs.remove(&b).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn should_not_read_replaced_or_removed_files_through_cached_handles() {
        let dir = tempfile::tempdir().unwrap();
        let vfs = StdVfs::new();
        let (tmp, path) = (dir.path().join("tmp"), dir.path().join("a"));
        let write = |path: PathBuf, contents: &'static str| {
            let vfs = vfs.clone();
            async move {
                let mut writer = vfs.create(&path).await.unwrap();
                writer.write(Bytes::from(contents)).await.unwrap();
                writer.finish().await.unwrap();
            }
        };

        write(path.clone(), "old").await;
        assert_eq!(vfs.read_range(&path, 0..3).await.unwrap(), "old");

        write(tmp.clone(), "new").await;
        vfs.rename(&tmp, &path).await.unwrap();
        assert_eq!(vfs.read_range(&path, 0..3).await.unwrap(), "new");

        vfs.remove(&path).await.unwrap();
        let err = vfs.read_range(&path, 0..3).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
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
