//! A cache of open files and a helper for positional reads.

use std::sync::Arc;

/// A cache of open file descriptors, keyed by filesystem path.
///
/// Uses a sharded concurrent cache for O(1) lookup and eviction. A single
/// mutex over an LRU list cannot carry this path: every lookup mutates the
/// recency list, so readers serialize even on hits, and at tens of thousands
/// of reads per second the handoff cost dominates the read itself.
///
/// Each read specifies its own offset, so threads can share a handle without
/// a lock for each file. On Unix, these reads do not move the file cursor.
#[derive(Clone)]
pub struct FileHandleCache {
    inner: Arc<quick_cache::sync::Cache<std::path::PathBuf, Arc<std::fs::File>>>,
}

impl std::fmt::Debug for FileHandleCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileHandleCache")
            .field("len", &self.inner.len())
            .field("cap", &self.inner.capacity())
            .finish()
    }
}

impl FileHandleCache {
    /// Creates a cache that retains at most `max_handles` open files.
    ///
    /// # Panics
    ///
    /// Panics if `max_handles` is zero.
    pub fn new(max_handles: usize) -> Self {
        assert!(max_handles > 0, "max_handles must be > 0");
        Self {
            inner: Arc::new(quick_cache::sync::Cache::new(max_handles)),
        }
    }

    /// Returns the number of handles retained by the cache.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether the cache retains no handles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of handles the cache can retain.
    pub fn capacity(&self) -> u64 {
        self.inner.capacity()
    }

    /// Look up a cached file handle, or open the file and cache it.
    /// Returns `Ok(None)` if the file does not exist on disk.
    /// The returned `Arc` keeps the file open after its cache entry is evicted.
    pub fn get_or_open(
        &self,
        path: &std::path::Path,
    ) -> Result<Option<Arc<std::fs::File>>, std::io::Error> {
        match self.try_get_or_open(path) {
            Ok(handle) => Ok(Some(handle)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// Look up a cached file handle, or open the file and cache it.
    /// Unlike [`Self::get_or_open`], returns the error from opening the file
    /// as-is, including `NotFound`.
    /// The returned `Arc` keeps the file open after its cache entry is evicted.
    ///
    /// A hit runs no syscall. The cache does not check that a cached handle
    /// still points to the file at `path`. Code that deletes or replaces a
    /// cached file must call [`Self::invalidate`] after it.
    ///
    /// The `open` on a miss runs after the shard lock is released. If it ran
    /// with the lock held, every reader of that shard would wait for it.
    pub fn try_get_or_open(
        &self,
        path: &std::path::Path,
    ) -> Result<Arc<std::fs::File>, std::io::Error> {
        if let Some(handle) = self.inner.get(path) {
            return Ok(handle);
        }

        let handle = Arc::new(std::fs::File::open(path)?);

        // Two threads missing on the same path both open it and both insert;
        // one entry wins and the other's descriptor closes when its last
        // reference drops. Both callers hold a valid handle either way, so
        // the race costs a redundant open rather than correctness.
        self.inner.insert(path.to_path_buf(), handle.clone());

        // A delete or a replace can run between the `open` and the `insert`.
        // Its `invalidate` then runs before the `insert`, so the cache keeps a
        // handle to an unlinked file.
        // The fstat below runs after the `insert`. If the file is unlinked,
        // remove the entry. If a later unlink runs, its `invalidate` runs after
        // the `insert` and removes the entry.
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let unlinked = handle.metadata().map_or(true, |m| m.nlink() == 0);
            if unlinked {
                self.invalidate(path);
            }
        }

        Ok(handle)
    }

    /// Remove a cached handle, e.g. after eviction or after a write replaces
    /// the file (since the cached fd would still reference the old inode).
    pub fn invalidate(&self, path: &std::path::Path) {
        self.inner.remove(path);
    }
}

/// Reads exactly `buf.len()` bytes at `offset`.
///
/// Concurrent readers can share the file because each read specifies its offset.
/// On Unix, reads do not move the file cursor. On Windows, they do.
pub fn read_exact_at_offset(
    file: &std::fs::File,
    buf: &mut [u8],
    offset: u64,
) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut bytes_read = 0;
        while bytes_read < buf.len() {
            let n = file.seek_read(&mut buf[bytes_read..], offset + bytes_read as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            bytes_read += n;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::io::Seek;

    #[test]
    fn should_reuse_handles_across_clones() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        let cache = FileHandleCache::new(2);
        assert!(cache.get_or_open(&path).unwrap().is_none());
        assert!(cache.is_empty());

        std::fs::write(&path, b"contents").unwrap();
        let first = cache.get_or_open(&path).unwrap().unwrap();
        let second = cache.clone().get_or_open(&path).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn should_return_not_found_error_from_try_get_or_open() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        let cache = FileHandleCache::new(1);
        let err = cache.try_get_or_open(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(cache.is_empty());
    }

    #[test]
    fn should_bound_handles_and_keep_active_readers() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FileHandleCache::new(2);
        let paths = ["a", "b", "c"].map(|name| {
            let path = dir.path().join(name);
            std::fs::write(&path, name).unwrap();
            path
        });
        let handles = paths
            .iter()
            .map(|path| cache.get_or_open(path).unwrap().unwrap())
            .collect::<Vec<_>>();
        assert!(cache.len() <= 2);

        // A handle that the cache evicted still reads its file.
        for (handle, name) in handles.iter().zip(["a", "b", "c"]) {
            let mut buf = [0];
            read_exact_at_offset(handle, &mut buf, 0).unwrap();
            assert_eq!(&buf, name.as_bytes());
        }
    }

    #[test]
    fn should_read_concurrently_at_independent_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"0123456789").unwrap();
        let cache = FileHandleCache::new(1);
        let handle = cache.get_or_open(&path).unwrap().unwrap();

        std::thread::scope(|scope| {
            for offset in 0..10 {
                let handle = &handle;
                scope.spawn(move || {
                    for _ in 0..100 {
                        let mut buf = [0];
                        read_exact_at_offset(handle, &mut buf, offset).unwrap();
                        assert_eq!(buf[0], b'0' + offset as u8);
                    }
                });
            }
        });
        #[cfg(unix)]
        assert_eq!((&*handle).stream_position().unwrap(), 0);

        let err = read_exact_at_offset(&handle, &mut [0; 2], 9).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn should_reopen_after_invalidation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"old").unwrap();
        let cache = FileHandleCache::new(1);
        let old = cache.get_or_open(&path).unwrap().unwrap();

        std::fs::rename(&path, dir.path().join("moved")).unwrap();
        std::fs::write(&path, b"new").unwrap();
        cache.invalidate(&path);
        assert!(cache.is_empty());
        let new = cache.get_or_open(&path).unwrap().unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        let mut buf = [0; 3];
        read_exact_at_offset(&new, &mut buf, 0).unwrap();
        assert_eq!(&buf, b"new");
    }

    #[test]
    fn should_open_a_replaced_or_removed_file_after_invalidation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        let replacement = dir.path().join("replacement");
        std::fs::write(&path, b"old").unwrap();
        let cache = FileHandleCache::new(1);
        let old = cache.get_or_open(&path).unwrap().unwrap();

        // Code that replaces a cached file invalidates it, as the mirror VFS
        // does after a rename.
        std::fs::write(&replacement, b"new").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        cache.invalidate(&path);
        let new = cache.get_or_open(&path).unwrap().unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        let mut buf = [0; 3];
        read_exact_at_offset(&new, &mut buf, 0).unwrap();
        assert_eq!(&buf, b"new");

        std::fs::remove_file(&path).unwrap();
        cache.invalidate(&path);
        assert!(cache.get_or_open(&path).unwrap().is_none());
        assert!(cache.is_empty());
    }
}
