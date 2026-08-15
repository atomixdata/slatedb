use crate::cached_object_store::stats::CachedObjectStoreStats;
use crate::rand::DbRand;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use log::{debug, warn};
use object_store::path::Path;
use object_store::{Attributes, ObjectMeta};
use rand::{distr::Alphanumeric, Rng};
use slatedb_common::clock::SystemClock;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::io::Write;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OnceCell};
use walkdir::WalkDir;

use crate::cached_object_store::storage::{
    LocalCacheEntry, LocalCacheHead, LocalCacheStorage, LocalCacheTee,
};
use crate::utils::format_bytes_si;

/// Suffix attached to in-flight tee files so a startup sweep can reliably
/// distinguish them from committed cache files. Visible cache files use the
/// fixed `_part*` / `_head` naming; tee files insert `.tmp-<rand>` before
/// any final rename.
const TEE_TMP_INFIX: &str = ".tmp-";

/// `O_DIRECT` requires offset, length, and buffer to be aligned. ext4
/// and xfs both demand 4KB on common Linux installs.
const O_DIRECT_ALIGN: usize = 4096;

/// Heap allocation aligned to [`O_DIRECT_ALIGN`]. Used for both the
/// read path (round range to alignment, slice on completion) and the
/// write path (copy aligned-length payloads in before pwriting).
struct AlignedBuf {
    ptr: *mut u8,
    capacity: usize,
    layout: std::alloc::Layout,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    fn new(capacity: usize) -> std::io::Result<Self> {
        let aligned_cap = capacity.div_ceil(O_DIRECT_ALIGN) * O_DIRECT_ALIGN;
        let layout = std::alloc::Layout::from_size_align(aligned_cap, O_DIRECT_ALIGN)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        // SAFETY: aligned_cap >= O_DIRECT_ALIGN > 0
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(std::io::Error::other("aligned alloc failed"));
        }
        Ok(Self {
            ptr,
            capacity: aligned_cap,
            layout,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr was alloc'd with `capacity` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr was alloc'd with `capacity` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: ptr came from `alloc_zeroed` with this exact layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// Write a `Bytes` payload to `file`, optionally pacing the write so a single
/// large SST part doesn't saturate the device. Reads from foreground requests
/// can otherwise queue behind a sustained 100MB+ write at compaction time and
/// inflate `r_await`.
///
/// Pacing is controlled via env vars and only kicks in when both are set:
/// - `SLATEDB_CACHE_WRITE_CHUNK_BYTES`: chunk size (e.g. 8388608 = 8MB).
/// - `SLATEDB_CACHE_WRITE_PAUSE_US`: microseconds to sleep between chunks.
///
/// Unset / zero on either side preserves the original `write_all` semantics
/// (one syscall, no pause). Reading the env on every call is cheap relative
/// to a 100MB+ disk write and lets the bencher tune without a rebuild.
fn write_all_paced(file: &mut std::fs::File, bytes: &[u8]) -> std::io::Result<()> {
    let chunk_bytes = std::env::var("SLATEDB_CACHE_WRITE_CHUNK_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let pause_us = std::env::var("SLATEDB_CACHE_WRITE_PAUSE_US")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    if chunk_bytes == 0 || pause_us == 0 || bytes.len() <= chunk_bytes {
        return file.write_all(bytes);
    }

    let pause = Duration::from_micros(pause_us);
    let mut offset = 0;
    while offset < bytes.len() {
        let end = (offset + chunk_bytes).min(bytes.len());
        file.write_all(&bytes[offset..end])?;
        offset = end;
        if offset < bytes.len() {
            std::thread::sleep(pause);
        }
    }
    Ok(())
}

/// A cached file handle node. Callers that obtain an `Arc<CachedFileHandle>`
/// keep the underlying fd alive even after the entry is evicted from the cache.
#[derive(Debug)]
pub(crate) struct CachedFileHandle {
    file: std::fs::File,
}

impl CachedFileHandle {
    pub(crate) fn file(&self) -> &std::fs::File {
        &self.file
    }
}

/// A bounded LRU cache of open file descriptors, keyed by filesystem
/// path.
///
/// Backed by `lru::LruCache` under a `parking_lot::Mutex`. The mutex
/// is held only long enough to bump the recency entry / evict / insert,
/// not across the actual file open or any I/O. Once a caller has the
/// `Arc<CachedFileHandle>` it can read concurrently with other holders.
///
/// Eviction matters under O_DIRECT: every cached part keeps an open
/// fd, and over a long run with a large keyspace the unbounded
/// scc-backed version exceeded `RLIMIT_NOFILE`. The cap here is the
/// hard ceiling on simultaneously held fds; tune via
/// `ObjectStoreCacheOptions::max_open_file_handles`.
///
/// Individual file reads use positional I/O (`pread` / `read_exact_at`)
/// which does not touch the file cursor, so multiple threads can read from
/// the same `Arc<CachedFileHandle>` concurrently without any per-file
/// locking.
#[derive(Clone)]
pub(crate) struct FileHandleCache {
    inner: Arc<parking_lot::Mutex<lru::LruCache<std::path::PathBuf, Arc<CachedFileHandle>>>>,
    /// Capacity reported in Debug; the LruCache also stores it but
    /// surfacing it without locking is convenient.
    capacity: usize,
    /// When set, file handles for reads are opened with `O_DIRECT |
    /// O_NOATIME` so reads bypass the page cache. Reads then need
    /// 4KB-aligned offsets, lengths, and buffers; the read paths
    /// handle this by reading aligned ranges and slicing.
    direct_io: bool,
}

impl std::fmt::Debug for FileHandleCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileHandleCache")
            .field("len", &self.inner.lock().len())
            .field("capacity", &self.capacity)
            .field("direct_io", &self.direct_io)
            .finish()
    }
}

impl FileHandleCache {
    fn new(max_handles: usize, direct_io: bool) -> Self {
        let capacity = max_handles.max(1);
        let cap = std::num::NonZeroUsize::new(capacity).expect("capacity is at least 1");
        Self {
            inner: Arc::new(parking_lot::Mutex::new(lru::LruCache::new(cap))),
            capacity,
            direct_io,
        }
    }

    /// Look up a cached file handle, or open the file and cache it.
    /// Returns `Ok(None)` if the file does not exist on disk.
    ///
    /// When `direct_io` is true the file is opened with `O_DIRECT |
    /// O_NOATIME`. Callers that read non-aligned files (e.g. heads) must
    /// pass `direct_io = false` even when the storage is configured with
    /// `O_DIRECT` overall.
    fn get_or_open(
        &self,
        path: &std::path::Path,
        direct_io: bool,
    ) -> Result<Option<Arc<CachedFileHandle>>, std::io::Error> {
        // Hot path: bump LRU recency and clone the Arc. Drop the lock
        // before doing any I/O on the returned handle.
        {
            let mut guard = self.inner.lock();
            if let Some(entry) = guard.get(path) {
                let entry = entry.clone();
                drop(guard);
                if Self::is_valid(&entry, path) {
                    return Ok(Some(entry));
                }
                // Stale (file replaced or unlinked). Remove and fall
                // through to reopen.
                self.inner.lock().pop(path);
            }
        }

        let mut open = std::fs::OpenOptions::new();
        open.read(true);
        #[cfg(target_os = "linux")]
        if direct_io {
            use std::os::unix::fs::OpenOptionsExt;
            open.custom_flags(libc::O_DIRECT | libc::O_NOATIME);
        }
        let file = match open.open(path) {
            Ok(f) => f,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };

        // Hint the kernel that we will read randomly inside this file. SST
        // reads pull small block ranges out of large parts; the default
        // readahead is wasted I/O. Best-effort: ignore failures. With
        // O_DIRECT this is a no-op on the page-cache side, but the hint
        // still applies to readahead semantics on some kernels.
        Self::set_random_advise(&file);

        let handle = Arc::new(CachedFileHandle { file });

        // Race-tolerant insert: if another thread inserted concurrently
        // we use theirs and drop our freshly opened fd. The LRU's
        // `put` returns the previous value when the key already
        // exists, but for fd-cache semantics we want the first writer
        // to win so concurrent readers all share the same Arc.
        let mut guard = self.inner.lock();
        if let Some(existing) = guard.get(path) {
            return Ok(Some(existing.clone()));
        }
        guard.put(path.to_path_buf(), handle.clone());
        Ok(Some(handle))
    }

    /// Check whether a cached file descriptor still refers to a live file.
    ///
    /// On Unix an unlinked file keeps its data accessible through open fds,
    /// but `fstat` will report `nlink == 0`. This single in-kernel syscall is
    /// much cheaper than a full `open` and lets us detect deleted or replaced
    /// files without a TOCTOU-prone path `stat`.
    ///
    /// On non-Unix platforms (e.g. Windows), we fall back to checking whether
    /// the path still exists on disk.
    #[cfg(unix)]
    fn is_valid(handle: &CachedFileHandle, _path: &std::path::Path) -> bool {
        use std::os::unix::fs::MetadataExt;
        handle.file().metadata().is_ok_and(|m| m.nlink() > 0)
    }

    #[cfg(not(unix))]
    fn is_valid(_handle: &CachedFileHandle, path: &std::path::Path) -> bool {
        path.exists()
    }

    /// Hint the kernel that we will read randomly inside this file. SST
    /// reads pull small block ranges out of large parts; the default
    /// readahead window wastes I/O fetching adjacent blocks we don't need.
    /// Best-effort: failures are ignored silently. Linux uses
    /// `posix_fadvise(POSIX_FADV_RANDOM)`; macOS uses `fcntl(F_RDAHEAD, 0)`.
    #[cfg(target_os = "linux")]
    fn set_random_advise(file: &std::fs::File) {
        use std::os::unix::io::AsRawFd;
        // SAFETY: posix_fadvise only inspects the fd; it does not transfer
        // ownership and is safe to call on any open file.
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM);
        }
    }

    #[cfg(target_os = "macos")]
    fn set_random_advise(file: &std::fs::File) {
        use std::os::unix::io::AsRawFd;
        // F_RDAHEAD with arg 0 disables sequential prefetching for the file.
        // SAFETY: fcntl with F_RDAHEAD only sets a per-fd flag.
        unsafe {
            libc::fcntl(file.as_raw_fd(), libc::F_RDAHEAD, 0);
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn set_random_advise(_file: &std::fs::File) {}

    /// Remove a cached handle, e.g. after eviction or after a write replaces
    /// the file (since the cached fd would still reference the old inode).
    fn invalidate(&self, path: &std::path::Path) {
        self.inner.lock().pop(path);
    }
}

/// Cross-platform positional read. Reads exactly `buf.len()` bytes at `offset`
/// without altering the file cursor, allowing concurrent readers on the same fd.
fn read_exact_at_offset(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
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

#[derive(Debug)]
pub struct FsCacheStorage {
    root_folder: std::path::PathBuf,
    evictor: Option<Arc<FsCacheEvictor>>,
    rand: Arc<DbRand>,
    file_handle_cache: FileHandleCache,
    /// When set, opens write tmp files with `O_DIRECT` (only when the
    /// payload is 4KB-aligned) and read fds via `FileHandleCache` are
    /// opened with `O_DIRECT | O_NOATIME`. Heads + tail short writes
    /// fall back to buffered.
    direct_io: bool,
}

impl FsCacheStorage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        root_folder: std::path::PathBuf,
        max_cache_size_bytes: Option<usize>,
        scan_interval: Option<Duration>,
        stats: Arc<CachedObjectStoreStats>,
        system_clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
        max_open_file_handles: usize,
        direct_io: bool,
    ) -> Self {
        let file_handle_cache = FileHandleCache::new(max_open_file_handles, direct_io);
        let evictor = max_cache_size_bytes.map(|max_cache_size_bytes| {
            Arc::new(FsCacheEvictor::new(
                root_folder.clone(),
                max_cache_size_bytes,
                scan_interval,
                stats,
                system_clock,
                rand.clone(),
                file_handle_cache.clone(),
            ))
        });

        Self {
            root_folder,
            evictor,
            rand,
            file_handle_cache,
            direct_io,
        }
    }
}

#[async_trait::async_trait]
impl LocalCacheStorage for FsCacheStorage {
    fn entry(
        &self,
        location: &object_store::path::Path,
        part_size: usize,
    ) -> Box<dyn LocalCacheEntry> {
        Box::new(FsCacheEntry {
            root_folder: self.root_folder.clone(),
            location: location.clone(),
            evictor: self.evictor.clone(),
            part_size,
            rand: self.rand.clone(),
            file_handle_cache: self.file_handle_cache.clone(),
            direct_io: self.direct_io,
        })
    }

    async fn start_evictor(&self) {
        if let Some(evictor) = &self.evictor {
            evictor.start().await
        }
    }

    fn begin_tee(&self, location: &Path, part_size: usize) -> Option<Box<dyn LocalCacheTee>> {
        Some(Box::new(FsCacheTee::new(
            self.root_folder.clone(),
            location.clone(),
            part_size,
            self.evictor.clone(),
            self.rand.clone(),
            self.file_handle_cache.clone(),
        )))
    }

    async fn remove(&self, location: &Path) -> object_store::Result<()> {
        let dir = self.root_folder.join(location.to_string());

        // Enumerate cached part/head files in a single blocking task, capture
        // their sizes for evictor accounting, then delete the directory in one
        // shot. This avoids N spawn_blocking trips for big SSTs.
        let dir_for_blocking = dir.clone();
        #[allow(clippy::disallowed_methods)]
        let entries = tokio::task::spawn_blocking(
            move || -> std::io::Result<Vec<(std::path::PathBuf, u64)>> {
                let read_dir = match std::fs::read_dir(&dir_for_blocking) {
                    Ok(rd) => rd,
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
                    Err(err) => return Err(err),
                };
                let mut out = Vec::new();
                for entry in read_dir {
                    let entry = entry?;
                    let metadata = entry.metadata()?;
                    if metadata.is_file() {
                        out.push((entry.path(), metadata.len()));
                    }
                }
                // Best-effort directory removal. If a concurrent writer recreates
                // the directory after we listed it, remove_dir_all may still
                // succeed; if it fails NotFound we treat that as success.
                match std::fs::remove_dir_all(&dir_for_blocking) {
                    Ok(()) => Ok(out),
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(out),
                    Err(err) => Err(err),
                }
            },
        )
        .await
        .map_err(wrap_io_err)?
        .map_err(wrap_io_err)?;

        if entries.is_empty() {
            return Ok(());
        }

        // Invalidate any cached file handles in one lock acquisition per path.
        // The lock is internal to FileHandleCache; we already hold no other
        // locks at this point.
        for (path, _) in entries.iter() {
            self.file_handle_cache.invalidate(path);
        }

        // Batch-update the evictor accounting in a single critical section.
        if let Some(evictor) = &self.evictor {
            evictor.forget_entries(entries).await;
        }

        Ok(())
    }
}

impl Display for FsCacheStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FsCacheStorage({})", self.root_folder.display())
    }
}

#[derive(Debug)]
pub(crate) struct FsCacheEntry {
    root_folder: std::path::PathBuf,
    location: Path,
    part_size: usize,
    evictor: Option<Arc<FsCacheEvictor>>,
    rand: Arc<DbRand>,
    file_handle_cache: FileHandleCache,
    /// Inherited from the storage. When true, atomic_write opens with
    /// `O_DIRECT` and copies into an aligned buffer (only when payload
    /// is 4KB-aligned), and reads use aligned buffers + range slicing.
    direct_io: bool,
}

impl FsCacheEntry {
    async fn atomic_write(&self, path: std::path::PathBuf, buf: Bytes) -> object_store::Result<()> {
        let tmp_path = path.with_extension(format!("_tmp{}", self.make_rand_suffix()));

        // Notify the evictor of this cache entry. The evictor's mpsc
        // channel is shared with the much-higher-frequency read-access
        // tracking (every `read_part` call); under load the channel
        // can fill, in which case `track_entry_accessed` returns
        // false. Previously this silently skipped the file write,
        // which manifested as readers falling through to S3 because
        // a fraction of head/part files were never written despite
        // the SST being published in the manifest.
        //
        // The fix: tracking failure must NOT skip the write. Eviction
        // accounting can be slightly off (the evictor will pick it up
        // on the next periodic scan); a missing cache file cannot.
        if let Some(evictor) = &self.evictor {
            evictor
                .track_entry_accessed(path.clone(), buf.len(), true)
                .await;
        }

        // Spawn a blocking task and do synchronous I/O rather than use the tokio async apis.
        // Under the hood, on linux systems , tokio itself spawns a blocking task for each call to
        // drive i/o since it hasn't yet adopted the native fully async i/o api (io_uring). Each
        // blocking task adds overhead, so its better to just batch all the calls into a single
        // blocking task.
        // see https://github.com/slatedb/slatedb/pull/1342
        let invalidate_path = path.clone();
        let direct_io = self.direct_io;
        #[allow(clippy::disallowed_methods)]
        tokio::task::spawn_blocking(move || {
            let tmp_path = tmp_path.as_path();
            // ensure the parent folder exists
            if let Some(folder_path) = tmp_path.parent() {
                std::fs::create_dir_all(folder_path).map_err(wrap_io_err)?;
            }

            // Apply O_DIRECT only when the payload length is 4KB-
            // aligned. Heads (~287 bytes) and any tail-end short
            // writes don't satisfy the alignment constraint and stay
            // buffered. The aligned-payload path also needs a
            // 4KB-aligned source buffer; we copy the bytes into an
            // AlignedBuf before writing.
            let aligned_payload = direct_io && !buf.is_empty() && buf.len() % O_DIRECT_ALIGN == 0;
            let mut open = std::fs::OpenOptions::new();
            open.write(true).create(true).truncate(true);
            #[cfg(target_os = "linux")]
            if aligned_payload {
                use std::os::unix::fs::OpenOptionsExt;
                open.custom_flags(libc::O_DIRECT);
            }
            let mut file = open.open(tmp_path).map_err(wrap_io_err)?;
            if aligned_payload {
                let mut aligned = AlignedBuf::new(buf.len()).map_err(wrap_io_err)?;
                aligned.as_mut_slice()[..buf.len()].copy_from_slice(&buf);
                file.write_all(&aligned.as_slice()[..buf.len()])
                    .map_err(wrap_io_err)?;
            } else {
                write_all_paced(&mut file, &buf).map_err(wrap_io_err)?;
            }
            // No fsync. The cache is reconstructable from the upstream
            // object store; a power loss may leave torn or zero-length
            // files, which the read path treats as a miss and refetches.
            // Skipping fsync removes a per-part barrier that dominates the
            // write hot path on local SSDs.
            std::fs::rename(tmp_path, path).map_err(wrap_io_err)
        })
        .await?
        .map_err(wrap_io_err)?;

        // The rename replaced the file at `path`, so any previously cached
        // handle now points to the old (unlinked) inode. Invalidate it so
        // the next read opens the new file.
        self.file_handle_cache.invalidate(&invalidate_path);

        Ok(())
    }

    // every origin file will be split into multiple parts, and all the parts will be saved in the same
    // folder. the part file name is expected to be in the format of `_part{part_size}-{part_number}`,
    // examples: /tmp/mydata.csv/_part1mb-000000001
    pub(crate) fn make_part_path(
        root_folder: std::path::PathBuf,
        location: &Path,
        part_number: usize,
        part_size: usize,
    ) -> std::path::PathBuf {
        // containing the part size in the file name, allows user change the part size on
        // the fly, without the need to invalidate the cache.
        let part_size_name = if part_size.is_multiple_of(1024 * 1024) {
            format!("{}mb", part_size / (1024 * 1024))
        } else {
            format!("{}kb", part_size / 1024)
        };
        let suffix = format!("_part{}-{:09}", part_size_name, part_number);
        let mut path = root_folder.join(location.to_string());
        path.push(suffix);
        path
    }

    fn make_head_path(root_folder: std::path::PathBuf, location: &Path) -> std::path::PathBuf {
        let suffix = "_head".to_string();
        let mut path = root_folder.join(location.to_string());
        path.push(suffix);
        path
    }

    fn make_rand_suffix(&self) -> String {
        let mut rng = self.rand.rng();
        (0..24).map(|_| rng.sample(Alphanumeric) as char).collect()
    }
}

#[async_trait::async_trait]
impl LocalCacheEntry for FsCacheEntry {
    async fn save_part(&self, part_number: usize, buf: Bytes) -> object_store::Result<()> {
        let part_path = Self::make_part_path(
            self.root_folder.clone(),
            &self.location,
            part_number,
            self.part_size,
        );

        self.atomic_write(part_path, buf).await
    }

    async fn read_part(
        &self,
        part_number: usize,
        range_in_part: Range<usize>,
    ) -> object_store::Result<Option<Bytes>> {
        let part_path = Self::make_part_path(
            self.root_folder.clone(),
            &self.location,
            part_number,
            self.part_size,
        );

        // Spawn a blocking task and do synchronous I/O rather than use the tokio async apis.
        // Under the hood, on linux systems , tokio itself spawns a blocking task for each call to
        // drive i/o since it hasn't yet adopted the native fully async i/o api (io_uring). Each
        // blocking task adds overhead, so its better to just batch all the calls into a single
        // blocking task.
        // see https://github.com/slatedb/slatedb/pull/1342
        let file_cache = self.file_handle_cache.clone();
        let this_part_path = part_path.clone();
        let direct_io = self.direct_io;
        #[allow(clippy::disallowed_methods)]
        let result = tokio::task::spawn_blocking(move || {
            let file = match file_cache.get_or_open(&this_part_path, direct_io) {
                Ok(Some(f)) => f,
                Ok(None) => return Ok(None),
                Err(err) => return Err(wrap_io_err(err)),
            };

            if direct_io {
                // O_DIRECT path: round the requested range out to 4KB
                // boundaries, read aligned bytes into an aligned buf,
                // then slice the result back to the caller's range.
                let req_off = range_in_part.start as u64;
                let req_len = range_in_part.len();
                let aligned_off = (req_off / O_DIRECT_ALIGN as u64) * O_DIRECT_ALIGN as u64;
                let head_pad = (req_off - aligned_off) as usize;
                let aligned_len = (head_pad + req_len).div_ceil(O_DIRECT_ALIGN) * O_DIRECT_ALIGN;
                let mut aligned = AlignedBuf::new(aligned_len).map_err(wrap_io_err)?;
                read_exact_at_offset(
                    file.file(),
                    &mut aligned.as_mut_slice()[..aligned_len],
                    aligned_off,
                )
                .map_err(wrap_io_err)?;
                let slice = &aligned.as_slice()[head_pad..head_pad + req_len];
                return Ok(Some(Bytes::copy_from_slice(slice)));
            }

            // Buffered path: positional I/O (pread) — no seek required, and safe for
            // concurrent readers sharing the same Arc<File>.
            let mut buffer = vec![0; range_in_part.len()];
            read_exact_at_offset(file.file(), &mut buffer, range_in_part.start as u64)
                .map_err(wrap_io_err)?;
            Ok(Some(Bytes::from(buffer)))
        })
        .await
        .map_err(wrap_io_err)??;

        // track the part access for evictor
        if result.is_some() {
            if let Some(evictor) = &self.evictor {
                evictor
                    .track_entry_accessed(part_path, self.part_size, false)
                    .await;
            }
        }

        Ok(result)
    }

    async fn cached_parts(
        &self,
    ) -> object_store::Result<Vec<crate::cached_object_store::storage::PartID>> {
        let head_path = Self::make_head_path(self.root_folder.clone(), &self.location);
        let directory_path = match head_path.parent() {
            Some(directory_path) => directory_path.to_path_buf(),
            None => return Ok(vec![]),
        };

        #[allow(clippy::disallowed_methods)]
        tokio::task::spawn_blocking(move || {
            let target_prefix = "_part";

            let entries = match std::fs::read_dir(&directory_path) {
                Ok(entries) => entries,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
                Err(err) => return Err(wrap_io_err(err)),
            };

            let mut part_file_names = vec![];
            for entry in entries {
                let entry = entry.map_err(wrap_io_err)?;
                let metadata = entry.metadata().map_err(wrap_io_err)?;
                if metadata.is_dir() {
                    continue;
                }
                let file_name = entry.file_name();
                let file_name_str = file_name.to_string_lossy();
                if file_name_str.starts_with(target_prefix) {
                    part_file_names.push(file_name_str.to_string());
                }
            }

            // not cached at all
            if part_file_names.is_empty() {
                return Ok(vec![]);
            }

            // sort the paths in alphabetical order
            part_file_names.sort();

            // retrieve the part numbers from the paths
            let mut part_numbers = Vec::with_capacity(part_file_names.len());
            for part_file_name in part_file_names.iter() {
                let part_number = part_file_name
                    .split('-')
                    .next_back()
                    .and_then(|part_number| part_number.parse::<usize>().ok());
                if let Some(part_number) = part_number {
                    part_numbers.push(part_number);
                }
            }

            Ok(part_numbers)
        })
        .await
        .map_err(wrap_io_err)?
    }

    async fn save_head(&self, head: (&ObjectMeta, &Attributes)) -> object_store::Result<()> {
        // if the meta file exists and not corrupted, do nothing
        match self.read_head().await {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {}
            Err(_) => {
                // TODO: add a warning
            }
        }

        let head: LocalCacheHead = head.into();
        let buf: Bytes = serde_json::to_vec(&head).map_err(wrap_io_err)?.into();

        let meta_path = Self::make_head_path(self.root_folder.clone(), &self.location);

        self.atomic_write(meta_path, buf).await
    }

    async fn read_head(&self) -> object_store::Result<Option<(ObjectMeta, Attributes)>> {
        let head_path = Self::make_head_path(self.root_folder.clone(), &self.location);

        // Spawn a blocking task and do synchronous I/O rather than use the tokio async apis.
        // Under the hood, on linux systems , tokio itself spawns a blocking task for each call to
        // drive i/o since it hasn't yet adopted the native fully async i/o api (io_uring). Each
        // blocking task adds overhead, so its better to just batch all the calls into a single
        // blocking task.
        let file_cache = self.file_handle_cache.clone();
        let this_head_path = head_path.clone();
        #[allow(clippy::disallowed_methods)]
        let result = tokio::task::spawn_blocking(move || {
            // Heads are short JSON (~287 bytes), not 4KB-aligned, and
            // are written buffered. Always open them buffered for read.
            let file = match file_cache.get_or_open(&this_head_path, false) {
                Ok(Some(f)) => f,
                Ok(None) => return Ok(None),
                Err(err) => return Err(wrap_io_err(err)),
            };

            let metadata = file.file().metadata().map_err(wrap_io_err)?;
            let head_size_bytes = metadata.len() as usize;

            // Use positional read from offset 0 to read the entire file.
            let mut buffer = vec![0u8; head_size_bytes];
            read_exact_at_offset(file.file(), &mut buffer, 0).map_err(wrap_io_err)?;

            let content = String::from_utf8(buffer).map_err(|e| {
                wrap_io_err(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            })?;

            let head: LocalCacheHead = serde_json::from_str(&content).map_err(wrap_io_err)?;
            Ok(Some((head.meta(), head.attributes(), head_size_bytes)))
        })
        .await
        .map_err(wrap_io_err)??;

        if let Some((meta, attributes, head_size_bytes)) = result {
            // track the head access for evictor
            if let Some(evictor) = &self.evictor {
                let head_path = Self::make_head_path(self.root_folder.clone(), &self.location);
                evictor
                    .track_entry_accessed(head_path, head_size_bytes, false)
                    .await;
            }
            Ok(Some((meta, attributes)))
        } else {
            Ok(None)
        }
    }
}

/// A streaming tee writer that funnels SST bytes into the on-disk cache as
/// they are produced. Writes go to per-part temp files via a dedicated worker
/// task so the producer (the SST writer) only pays the cost of an mpsc send
/// per part. On commit, all temp files are renamed to their final names; on
/// drop without commit, temp files are cleaned up best-effort.
///
/// Bottleneck notes:
///
/// - A single worker task per tee keeps writes to disk in order without
///   blocking the producer. The mpsc has a small fixed capacity (4) which
///   bounds buffered memory at ~4 * part_size while still smoothing over
///   short bursts.
/// - The producer copies bytes into a part-sized `BytesMut` once; this memcpy
///   would happen on disk write anyway. There is no extra clone.
/// - Renames at commit time happen in a single `spawn_blocking` task to keep
///   syscall overhead off the runtime.
pub(crate) struct FsCacheTee {
    root_folder: std::path::PathBuf,
    location: Path,
    part_size: usize,
    rand: Arc<DbRand>,
    file_handle_cache: FileHandleCache,
    evictor: Option<Arc<FsCacheEvictor>>,

    /// Buffer for the in-progress part. Drained when full.
    part_buf: bytes::BytesMut,
    /// 0-based index of the next part to be written.
    next_part_number: usize,
    /// Cumulative bytes accepted via `extend` so far. Used to populate the
    /// committed head's `size` field, which must match the upstream object's
    /// size for `read_head` to return a sensible value.
    total_bytes: u64,
    /// Tracks every temp file we have asked the worker to write, paired with
    /// the final name to rename to on commit. Always in flush order.
    pending: Vec<(std::path::PathBuf, std::path::PathBuf)>,
    /// Channel to the worker. Dropping closes the channel and the worker
    /// drains and exits.
    tx: Option<tokio::sync::mpsc::Sender<TeeMsg>>,
    /// Worker join handle, awaited at commit / drop time.
    worker: Option<tokio::task::JoinHandle<object_store::Result<()>>>,
    /// Set once any extend / flush hits an error. After this point the tee
    /// is poisoned: extends are silently dropped and commit cleans up.
    poisoned: bool,
}

impl std::fmt::Debug for FsCacheTee {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsCacheTee")
            .field("location", &self.location.as_ref())
            .field("part_size", &self.part_size)
            .field("next_part_number", &self.next_part_number)
            .field("pending", &self.pending.len())
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

enum TeeMsg {
    Write {
        tmp_path: std::path::PathBuf,
        bytes: Bytes,
    },
}

impl FsCacheTee {
    fn new(
        root_folder: std::path::PathBuf,
        location: Path,
        part_size: usize,
        evictor: Option<Arc<FsCacheEvictor>>,
        rand: Arc<DbRand>,
        file_handle_cache: FileHandleCache,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel::<TeeMsg>(4);
        let worker = tokio::spawn(Self::run_worker(rx));
        Self {
            root_folder,
            location,
            part_size,
            rand,
            file_handle_cache,
            evictor,
            // Grow lazily. `BytesMut::with_capacity(part_size)` would
            // allocate the full part up front, which for large part sizes
            // (256MB) wastes memory if the SST is smaller than one part.
            // The amortized growth cost is dominated by the disk write.
            part_buf: bytes::BytesMut::new(),
            next_part_number: 0,
            total_bytes: 0,
            pending: Vec::new(),
            tx: Some(tx),
            worker: Some(worker),
            poisoned: false,
        }
    }

    async fn run_worker(mut rx: tokio::sync::mpsc::Receiver<TeeMsg>) -> object_store::Result<()> {
        while let Some(msg) = rx.recv().await {
            match msg {
                TeeMsg::Write { tmp_path, bytes } => {
                    if let Err(e) = Self::write_tmp_file(tmp_path, bytes).await {
                        // Drain the rest of the channel without writing so
                        // the producer doesn't block on an unread channel.
                        while rx.recv().await.is_some() {}
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    async fn write_tmp_file(
        tmp_path: std::path::PathBuf,
        bytes: Bytes,
    ) -> object_store::Result<()> {
        #[allow(clippy::disallowed_methods)]
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            if let Some(parent) = tmp_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            write_all_paced(&mut file, &bytes)?;
            // No fsync. The tee is best-effort: on a crash before the
            // rename-into-place barrier the orphan tmp files are swept by
            // scan_entries on next start. On a crash after rename, a torn
            // or short final file simply produces a cache miss on read.
            Ok(())
        })
        .await
        .map_err(wrap_io_err)?
        .map_err(wrap_io_err)
    }

    fn make_rand_suffix(&self) -> String {
        let mut rng = self.rand.rng();
        (0..16).map(|_| rng.sample(Alphanumeric) as char).collect()
    }

    /// Reserve the next part: returns (final_path, tmp_path). The final_path
    /// matches what `FsCacheEntry::read_part` expects so a successful rename
    /// makes the part discoverable. The tmp file lives in the same folder
    /// (so rename is atomic) with a `.tmp-<rand>` infix that the startup
    /// sweep recognizes.
    fn reserve_part_paths(&self, part_number: usize) -> (std::path::PathBuf, std::path::PathBuf) {
        let final_path = FsCacheEntry::make_part_path(
            self.root_folder.clone(),
            &self.location,
            part_number,
            self.part_size,
        );
        let final_name = final_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let tmp_name = format!("{}{}{}", final_name, TEE_TMP_INFIX, self.make_rand_suffix());
        let mut tmp = final_path.clone();
        tmp.set_file_name(tmp_name);
        (final_path, tmp)
    }

    fn reserve_head_paths(&self) -> (std::path::PathBuf, std::path::PathBuf) {
        let final_path = {
            let mut p = self.root_folder.join(self.location.to_string());
            p.push("_head");
            p
        };
        let tmp_name = format!("_head{}{}", TEE_TMP_INFIX, self.make_rand_suffix());
        let mut tmp = final_path.clone();
        tmp.set_file_name(tmp_name);
        (final_path, tmp)
    }

    /// Send one buffered part to the worker. Caller must reset `part_buf`
    /// afterwards.
    async fn dispatch_part(&mut self, payload: Bytes) -> object_store::Result<()> {
        let part_number = self.next_part_number;
        self.next_part_number += 1;
        let (final_path, tmp_path) = self.reserve_part_paths(part_number);
        self.pending.push((tmp_path.clone(), final_path));

        let tx = match self.tx.as_ref() {
            Some(tx) => tx,
            None => return Ok(()),
        };
        // Send is awaited; if the worker is slow, the producer waits. With
        // capacity 4 this caps memory at ~4 * part_size in flight. If the
        // channel is closed (worker died), we mark poisoned and stop.
        if tx
            .send(TeeMsg::Write {
                tmp_path,
                bytes: payload,
            })
            .await
            .is_err()
        {
            self.poisoned = true;
        }
        Ok(())
    }

    /// Best-effort cleanup of any temp files we have registered. Awaits the
    /// worker first if a join handle is provided, so we don't race with an
    /// in-flight write that would resurrect a tmp file after we deleted it.
    fn schedule_cleanup(
        pending: Vec<(std::path::PathBuf, std::path::PathBuf)>,
        worker: Option<tokio::task::JoinHandle<object_store::Result<()>>>,
    ) {
        if pending.is_empty() && worker.is_none() {
            return;
        }
        #[allow(clippy::disallowed_methods)]
        tokio::spawn(async move {
            // Drain the worker first so any tmp file it was about to create
            // exists on disk before we try to delete it.
            if let Some(handle) = worker {
                let _ = handle.await;
            }
            if pending.is_empty() {
                return;
            }
            #[allow(clippy::disallowed_methods)]
            let _ = tokio::task::spawn_blocking(move || {
                for (tmp_path, _final_path) in pending {
                    let _ = std::fs::remove_file(&tmp_path);
                }
            })
            .await;
        });
    }
}

#[async_trait::async_trait]
impl LocalCacheTee for FsCacheTee {
    async fn extend(&mut self, buf: &[u8]) -> object_store::Result<()> {
        if self.poisoned {
            return Ok(());
        }
        self.total_bytes += buf.len() as u64;
        let mut remaining = buf;
        while !remaining.is_empty() {
            let need = self.part_size - self.part_buf.len();
            let take = remaining.len().min(need);
            self.part_buf.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            if self.part_buf.len() == self.part_size {
                let payload =
                    std::mem::replace(&mut self.part_buf, bytes::BytesMut::new()).freeze();
                self.dispatch_part(payload).await?;
                if self.poisoned {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    async fn commit(
        mut self: Box<Self>,
        meta: &ObjectMeta,
        attrs: &Attributes,
    ) -> object_store::Result<()> {
        // Flush the tail.
        if !self.poisoned && !self.part_buf.is_empty() {
            let payload = std::mem::replace(&mut self.part_buf, bytes::BytesMut::new()).freeze();
            self.dispatch_part(payload).await?;
        }

        // Build a head snapshot using the upstream meta but with the size we
        // actually wrote to disk. Some callers may pass a stub meta; this is
        // the canonical source of truth for cache reads.
        let head_meta = ObjectMeta {
            location: meta.location.clone(),
            last_modified: meta.last_modified,
            size: self.total_bytes,
            e_tag: meta.e_tag.clone(),
            version: meta.version.clone(),
        };
        let head: LocalCacheHead = (&head_meta, attrs).into();
        let head_bytes = match serde_json::to_vec(&head) {
            Ok(v) => Bytes::from(v),
            Err(e) => {
                // Can't write a head; cleanup and bail.
                let pending = std::mem::take(&mut self.pending);
                drop(self.tx.take());
                let worker = self.worker.take();
                Self::schedule_cleanup(pending, worker);
                return Err(wrap_io_err(e));
            }
        };

        // Reserve and dispatch the head.
        let (head_final, head_tmp) = self.reserve_head_paths();
        self.pending.push((head_tmp.clone(), head_final.clone()));
        let head_idx = self.pending.len() - 1;
        if let Some(tx) = self.tx.as_ref() {
            if tx
                .send(TeeMsg::Write {
                    tmp_path: head_tmp.clone(),
                    bytes: head_bytes,
                })
                .await
                .is_err()
            {
                self.poisoned = true;
            }
        }

        // Drop the sender so the worker exits, then await its result.
        drop(self.tx.take());
        let worker_result = match self.worker.take() {
            Some(handle) => handle.await.unwrap_or_else(|e| Err(wrap_io_err(e))),
            None => Ok(()),
        };

        if self.poisoned || worker_result.is_err() {
            let pending = std::mem::take(&mut self.pending);
            // Worker has already exited (we awaited it above); pass None.
            Self::schedule_cleanup(pending, None);
            return worker_result;
        }

        // All temp files are on disk (not fsynced; the cache is best-effort
        // and reconstructable from upstream). Atomically rename in
        // a single blocking task. Order matters: rename all parts before the
        // head, so that once read_head() returns Some(...) every part is
        // visible.
        let pending = std::mem::take(&mut self.pending);
        let evictor = self.evictor.clone();
        let file_handle_cache = self.file_handle_cache.clone();

        #[allow(clippy::disallowed_methods)]
        let rename_result = tokio::task::spawn_blocking(
            move || -> std::io::Result<Vec<(std::path::PathBuf, u64)>> {
                // Reorder so the head rename is last.
                let mut parts: Vec<(std::path::PathBuf, std::path::PathBuf)> = pending;
                // The head was pushed at head_idx; move it to the end.
                if head_idx < parts.len() {
                    let head_pair = parts.remove(head_idx);
                    parts.push(head_pair);
                }
                let mut renamed = Vec::with_capacity(parts.len());
                for (tmp, final_) in parts {
                    std::fs::rename(&tmp, &final_)?;
                    let size = std::fs::metadata(&final_).map(|m| m.len()).unwrap_or(0);
                    renamed.push((final_, size));
                }
                Ok(renamed)
            },
        )
        .await
        .map_err(wrap_io_err)?
        .map_err(wrap_io_err);

        let renamed = match rename_result {
            Ok(r) => r,
            Err(e) => {
                // Best-effort: any unrenamed temp files will be picked up by
                // the next startup sweep. Final files that did rename remain
                // and will be admitted by the evictor's periodic scan.
                return Err(e);
            }
        };

        // Invalidate any stale file handles (the rename replaces inodes).
        for (path, _) in renamed.iter() {
            file_handle_cache.invalidate(path);
        }

        // Register sizes with the evictor in one batch so the new entries
        // are accounted for and trigger eviction if we just pushed past the
        // limit. We use track_entry_accessed in evict=true mode for the
        // last entry so a single eviction sweep runs after all parts.
        if let Some(evictor) = evictor {
            let count = renamed.len();
            for (i, (path, size)) in renamed.into_iter().enumerate() {
                let trigger_evict = i + 1 == count;
                evictor
                    .track_entry_accessed(path, size as usize, trigger_evict)
                    .await;
            }
        }

        Ok(())
    }
}

impl Drop for FsCacheTee {
    fn drop(&mut self) {
        // Producer dropped without committing. Take whatever we registered
        // and clean it up off-thread, after the worker has fully drained
        // (so we don't race with an in-flight write that would recreate a
        // tmp file we just unlinked).
        let pending = std::mem::take(&mut self.pending);
        // Closing the channel lets the worker drain and exit.
        drop(self.tx.take());
        let worker = self.worker.take();
        Self::schedule_cleanup(pending, worker);
    }
}

type FsCacheEvictorWork = (std::path::PathBuf, usize, bool);
// Minimum time between aggregated "evictor queue is full" warnings.
const QUEUE_FULL_LOG_INTERVAL_MS: i64 = 30_000;

/// FsCacheEvictor evicts the cache entries when the cache size exceeds the limit. it is expected to
/// run in the background to avoid blocking the caller, and it'll be triggered whenever a new cache entry
/// is added.
#[derive(Debug)]
struct FsCacheEvictor {
    root_folder: std::path::PathBuf,
    max_cache_size_bytes: usize,
    scan_interval: Option<Duration>,
    tx: tokio::sync::mpsc::Sender<FsCacheEvictorWork>,
    rx: Mutex<Option<tokio::sync::mpsc::Receiver<FsCacheEvictorWork>>>,
    started: AtomicBool,
    queue_full_count: AtomicU64,
    last_queue_full_log_ms: AtomicI64,
    background_evict_handle: OnceCell<tokio::task::JoinHandle<()>>,
    background_scan_handle: OnceCell<tokio::task::JoinHandle<()>>,
    inner: OnceCell<Arc<FsCacheEvictorInner>>,
    stats: Arc<CachedObjectStoreStats>,
    system_clock: Arc<dyn SystemClock>,
    rand: Arc<DbRand>,
    file_handle_cache: FileHandleCache,
}

impl FsCacheEvictor {
    fn new(
        root_folder: std::path::PathBuf,
        max_cache_size_bytes: usize,
        scan_interval: Option<Duration>,
        stats: Arc<CachedObjectStoreStats>,
        system_clock: Arc<dyn SystemClock>,
        rand: Arc<DbRand>,
        file_handle_cache: FileHandleCache,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        Self {
            root_folder,
            scan_interval,
            max_cache_size_bytes,
            tx,
            rx: Mutex::new(Some(rx)),
            started: AtomicBool::new(false),
            queue_full_count: AtomicU64::new(0),
            last_queue_full_log_ms: AtomicI64::new(i64::MIN),
            background_evict_handle: OnceCell::new(),
            background_scan_handle: OnceCell::new(),
            inner: OnceCell::new(),
            stats,
            system_clock,
            rand,
            file_handle_cache,
        }
    }

    async fn start(&self) {
        let inner = Arc::new(FsCacheEvictorInner::new(
            self.root_folder.clone(),
            self.max_cache_size_bytes,
            self.stats.clone(),
            self.rand.clone(),
            self.file_handle_cache.clone(),
        ));

        let guard = self.rx.lock();
        let rx = guard.await.take().expect("evictor already started");

        // Make the inner state reachable for direct callers (e.g. forget_entries
        // from invalidate()) before flipping `started` so observers don't see a
        // started evictor without a populated inner.
        self.inner.set(inner.clone()).ok();

        self.started.store(true, Ordering::Release);

        // scan the cache folder (defaults as every 1 hour) to keep the in-memory cache_entries eventually
        // consistent with the cache folder.
        self.background_scan_handle
            .set(tokio::spawn(Self::background_scan(
                inner.clone(),
                self.scan_interval,
                self.system_clock.clone(),
            )))
            .ok();

        // start the background evictor task, it'll be triggered whenever a new cache entry is added
        self.background_evict_handle
            .set(tokio::spawn(Self::background_evict(
                inner,
                rx,
                self.system_clock.clone(),
            )))
            .ok();
    }

    fn started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }

    async fn background_evict(
        inner: Arc<FsCacheEvictorInner>,
        mut rx: tokio::sync::mpsc::Receiver<FsCacheEvictorWork>,
        system_clock: Arc<dyn SystemClock>,
    ) {
        loop {
            match rx.recv().await {
                Some((path, bytes, evict)) => {
                    inner
                        .track_entry_accessed(path, bytes, system_clock.now(), evict)
                        .await;
                }
                None => return,
            }
        }
    }

    async fn background_scan(
        inner: Arc<FsCacheEvictorInner>,
        scan_interval: Option<Duration>,
        system_clock: Arc<dyn SystemClock>,
    ) {
        inner.clone().scan_entries(true).await;

        if let Some(scan_interval) = scan_interval {
            loop {
                system_clock.clone().sleep(scan_interval).await;
                inner.clone().scan_entries(true).await;
            }
        }
    }

    /// Drop accounting for a batch of cache entries that were just removed
    /// from disk. Bypasses the mpsc channel (which is reserved for foreground
    /// access events) and updates state in a single critical section, so a
    /// large SST with many parts costs one lock acquisition instead of N.
    ///
    /// Lazy initialization note: when called before `start()`, the in-memory
    /// state is empty (the background scan hasn't populated it yet), so most
    /// entries will be missing and we just no-op for them. The on-disk files
    /// are already gone, and the next scan won't pick them up. Safe.
    async fn forget_entries(&self, entries: Vec<(std::path::PathBuf, u64)>) {
        if entries.is_empty() {
            return;
        }
        // Delegate straight to the inner state holder. We need to grab a stable
        // Arc<FsCacheEvictorInner>; we don't have one stored here because
        // start() owns it. We could add one, but for now: only act if started.
        // Pre-start, the in-memory accounting is empty so there is nothing to
        // forget; we only need to delete files (already done by the caller).
        if !self.started() {
            return;
        }
        if let Some(inner) = self.inner.get() {
            inner.forget_entries(entries).await;
        }
    }

    // Allow send() here because we should never see a closed channel. The evictor owns both the
    // sender and receiver. It doesn't close the channel, and both sender and receiver are dropped
    // when the evictor is dropped.
    #[allow(clippy::disallowed_methods)]
    async fn track_entry_accessed(
        &self,
        path: std::path::PathBuf,
        bytes: usize,
        evict: bool,
    ) -> bool {
        if !self.started() {
            return true;
        }

        match self.tx.try_send((path, bytes, evict)) {
            Ok(()) => true,
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                self.queue_full_count.fetch_add(1, Ordering::AcqRel);
                let now_ms = self.system_clock.now().timestamp_millis();
                let last_log_ms = self.last_queue_full_log_ms.load(Ordering::Acquire);
                if now_ms.saturating_sub(last_log_ms) >= QUEUE_FULL_LOG_INTERVAL_MS
                    && self
                        .last_queue_full_log_ms
                        .compare_exchange(last_log_ms, now_ms, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                {
                    let queue_full_count = self.queue_full_count.swap(0, Ordering::AcqRel);
                    warn!(
                        "evictor queue skipped cache write/access event because it was full {} times in the last 30s",
                        queue_full_count,
                    );
                }
                false
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => false,
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    access_time: DateTime<Utc>,
    size_bytes: usize,
    key_index: usize,
}

#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<std::path::PathBuf, CacheEntry>,
    keys: Vec<std::path::PathBuf>,
}

/// FsCacheEvictorInner manages the cache entries in `CacheState`, and evict the cache entries
/// when the cache size exceeds the limit. it uses a pick-of-2 strategy to approximate LRU, and evict
/// the older file when the cache size exceeds the limit.
///
/// On start up, FsCacheEvictorInner will scan the cache folder to load the cache files into the in-memory
/// trie cache_entries. This loading process is interleaved with the maybe_evict is being called, so the
/// cache entries should be wrapped with Mutex<_>.
#[derive(Debug)]
struct FsCacheEvictorInner {
    root_folder: std::path::PathBuf,
    max_cache_size_bytes: usize,
    track_lock: Mutex<()>,
    cache_state: Mutex<CacheState>,
    cache_size_bytes: AtomicU64,
    stats: Arc<CachedObjectStoreStats>,
    rand: Arc<DbRand>,
    file_handle_cache: FileHandleCache,
}

impl FsCacheEvictorInner {
    fn new(
        root_folder: std::path::PathBuf,
        max_cache_size_bytes: usize,
        stats: Arc<CachedObjectStoreStats>,
        rand: Arc<DbRand>,
        file_handle_cache: FileHandleCache,
    ) -> Self {
        Self {
            root_folder,
            max_cache_size_bytes,
            track_lock: Mutex::new(()),
            cache_state: Mutex::new(CacheState::default()),
            cache_size_bytes: AtomicU64::new(0_u64),
            stats,
            rand,
            file_handle_cache,
        }
    }

    // scan the cache folder, and load the cache entries into memory.
    // this function is only called on start up, and it's expected to run interleavely with
    // maybe_evict is being called.
    async fn scan_entries(self: Arc<Self>, evict: bool) {
        let root_folder = self.root_folder.clone();

        // Walk the cache folder once. While we're at it, sweep any orphan tee
        // tmp files (named "<original>.tmp-<rand>") left behind by an aborted
        // pre-warm write or a process crash. Doing this inside the scan adds
        // zero extra syscalls beyond `unlink` per orphan, and runs every
        // scan_interval so transient crashes self-heal.
        #[allow(clippy::disallowed_methods)]
        let paths = tokio::task::spawn_blocking(move || {
            let mut keep = Vec::new();
            for entry in WalkDir::new(&root_folder)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                if !entry.file_type().is_file() {
                    continue;
                }
                let path = entry.path().to_path_buf();
                let is_orphan = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.contains(TEE_TMP_INFIX));
                if is_orphan {
                    let _ = std::fs::remove_file(&path);
                    continue;
                }
                keep.push(path);
            }
            keep
        })
        .await
        .unwrap_or_default();

        for path in paths {
            let metadata = match tokio::fs::metadata(&path).await {
                Ok(metadata) => metadata,
                Err(err) => {
                    if err.kind() != std::io::ErrorKind::NotFound {
                        warn!(
                            "evictor failed to get the metadata of the cache file [path={:?}, error={}]",
                            path, err
                        );
                    }

                    continue;
                }
            };
            #[allow(clippy::disallowed_types)]
            let atime = metadata
                .accessed()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .into();
            let bytes = metadata.len() as usize;

            self.track_entry_accessed(path, bytes, atime, evict).await;
        }
    }

    /// track the cache entry access, and evict the cache files when the cache size exceeds the limit if evict is true,
    /// return the bytes of the evicted files. please note that track_entry_accessed might be called concurrently from
    /// the rescanner and evictor tasks, it's expected to be wrapped with a lock to ensure the serial execution.
    async fn track_entry_accessed(
        &self,
        path: std::path::PathBuf,
        bytes: usize,
        accessed_time: DateTime<Utc>,
        evict: bool,
    ) -> usize {
        let _track_guard = self.track_lock.lock().await;

        let entry_count = {
            let mut cache_state = self.cache_state.lock().await;

            match cache_state.entries.get_mut(&path) {
                Some(entry) => {
                    entry.access_time = accessed_time;
                }
                None => {
                    let key_index = cache_state.keys.len();
                    cache_state.keys.push(path.clone());
                    cache_state.entries.insert(
                        path.clone(),
                        CacheEntry {
                            access_time: accessed_time,
                            size_bytes: bytes,
                            key_index,
                        },
                    );
                    self.cache_size_bytes
                        .fetch_add(bytes as u64, Ordering::SeqCst);
                }
            }
            cache_state.entries.len()
        };

        self.stats.object_store_cache_keys.set(entry_count as i64);
        self.stats
            .object_store_cache_bytes
            .set(self.cache_size_bytes.load(Ordering::Relaxed) as i64);

        // if the cache size is still below the limit, do nothing
        if self.cache_size_bytes.load(Ordering::Relaxed) <= self.max_cache_size_bytes as u64 {
            return 0;
        }
        // TODO: check the disk space ratio here, if the disk space is low, also triggers evict.

        // The maximum byte size the cache will take up on disk. If a write would cause the
        // cache to exceed this threshold, entries are evicted using an 2-random strategy until
        // the cache reaches 90% of `max_cache_size_bytes`.
        //
        // It's ok to call evict after inserting the new entry, because we will evict entries with eailer `accessed_time`.
        // This ensures that the newly added entry will not be evicted immediately.
        let evicted_bytes: usize = if evict
            && self.cache_size_bytes.load(Ordering::Relaxed) > self.max_cache_size_bytes as u64
        {
            // We sacrifice floating-point precision error to prevent possible overflow(i.e. self.max_cache_size_bytes * 9 / 10).
            let target_size = ((self.max_cache_size_bytes as f64) * 0.9) as u64;
            self.evict_to_target_size(target_size).await
        } else {
            0
        };

        evicted_bytes
    }

    /// Drop accounting for a batch of cache entries that have already been
    /// removed from disk by an external caller (e.g. invalidation after a
    /// remote DELETE). Takes the cache_state lock once for the whole batch
    /// and updates metrics in one pass.
    async fn forget_entries(&self, entries: Vec<(std::path::PathBuf, u64)>) {
        if entries.is_empty() {
            return;
        }
        let _track_guard = self.track_lock.lock().await;

        let (entry_count, total_bytes, removed_count) = {
            let mut cache_state = self.cache_state.lock().await;
            let mut total: u64 = 0;
            let mut removed_count: u64 = 0;
            for (path, _size_hint) in entries.iter() {
                if let Some(removed) = cache_state.entries.remove(path) {
                    cache_state.keys.swap_remove(removed.key_index);
                    if removed.key_index < cache_state.keys.len() {
                        let swapped_key = cache_state.keys[removed.key_index].clone();
                        if let Some(swapped) = cache_state.entries.get_mut(&swapped_key) {
                            swapped.key_index = removed.key_index;
                        }
                    }
                    self.cache_size_bytes
                        .fetch_sub(removed.size_bytes as u64, Ordering::SeqCst);
                    total += removed.size_bytes as u64;
                    removed_count += 1;
                }
            }
            (cache_state.entries.len(), total, removed_count)
        };

        self.stats.object_store_cache_keys.set(entry_count as i64);
        self.stats
            .object_store_cache_bytes
            .set(self.cache_size_bytes.load(Ordering::Relaxed) as i64);
        if removed_count > 0 {
            self.stats
                .object_store_cache_evicted_bytes
                .increment(total_bytes);
            self.stats
                .object_store_cache_evicted_keys
                .increment(removed_count);
        }
    }

    /// Evict cache entries until the cache size is below the target size.
    /// This method acquires the lock once to pick all eviction targets, then releases the lock
    /// to delete files, and finally acquires the lock again to update the state.
    async fn evict_to_target_size(&self, target_size: u64) -> usize {
        let picked_targets = self.pick_evict_targets(target_size).await;

        if picked_targets.is_empty() {
            if self.cache_size_bytes.load(Ordering::Relaxed) > target_size {
                warn!(
                    "cache_size_bytes still exceeds max_cache_size_bytes but no more entries can be evicted(cache_size_bytes={}, max_cache_size_bytes={})",
                    self.cache_size_bytes.load(Ordering::Relaxed),
                    self.max_cache_size_bytes
                );
            }
            return 0;
        }

        let mut deleted_targets: Vec<(std::path::PathBuf, usize)> =
            Vec::with_capacity(picked_targets.len());
        for (target, target_bytes) in picked_targets {
            // if the file is not found, still try to remove it from the cache_entries, and decrease the cache_size_bytes.
            // this might happen when the file is removed by other processes, or due to a race between the background
            // scan (which collects paths then processes them) and eviction deleting files in between.
            match tokio::fs::remove_file(&target).await {
                Ok(()) => {
                    debug!(
                        "evictor evicted cache file [path={:?}, bytes={}]",
                        target,
                        format_bytes_si(target_bytes as u64)
                    );
                    deleted_targets.push((target.clone(), target_bytes));
                    self.file_handle_cache.invalidate(&target);
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    // File already gone, still need to clean up state
                    deleted_targets.push((target.clone(), target_bytes));
                    self.file_handle_cache.invalidate(&target);
                }
                Err(err) => {
                    warn!("evictor failed to remove the cache file [error={}]", err);
                    // Skip this file, don't add to deleted_targets
                }
            }
        }

        if deleted_targets.is_empty() {
            return 0;
        }

        let (entry_count, total_evicted_bytes) = {
            let mut cache_state = self.cache_state.lock().await;
            let mut total_bytes: usize = 0;

            for (target, target_bytes) in deleted_targets.iter() {
                if let Some(removed) = cache_state.entries.remove(target) {
                    cache_state.keys.swap_remove(removed.key_index);
                    if removed.key_index < cache_state.keys.len() {
                        let swapped_key = cache_state.keys[removed.key_index].clone();
                        if let Some(swapped) = cache_state.entries.get_mut(&swapped_key) {
                            swapped.key_index = removed.key_index;
                        }
                    }
                    self.cache_size_bytes
                        .fetch_sub(*target_bytes as u64, Ordering::SeqCst);
                    total_bytes += target_bytes;
                }
            }

            (cache_state.entries.len(), total_bytes)
        };

        // Sync the metrics after eviction
        self.stats
            .object_store_cache_evicted_bytes
            .increment(total_evicted_bytes as u64);
        self.stats
            .object_store_cache_evicted_keys
            .increment(deleted_targets.len() as u64);
        self.stats.object_store_cache_keys.set(entry_count as i64);
        self.stats
            .object_store_cache_bytes
            .set(self.cache_size_bytes.load(Ordering::Relaxed) as i64);

        total_evicted_bytes
    }

    /// Pick multiple eviction targets in a single pass using pick-of-2 strategy, which is an approximation
    //  of LRU, it randomized pick two files, compare their last access time, and choose the older one to evict.
    ///
    /// Returns a list of (path, size_bytes) tuples to evict.
    async fn pick_evict_targets(&self, target_size: u64) -> Vec<(std::path::PathBuf, usize)> {
        let cache_state = self.cache_state.lock().await;

        if cache_state.keys.len() < 2 {
            return vec![];
        }

        let mut targets = Vec::new();
        // Track the simulated cache size during eviction but do not modify the actual cache size until
        // after files are deleted.
        let mut simulated_size = self.cache_size_bytes.load(Ordering::Relaxed);
        // Track which indices have been selected for eviction
        let mut picked_indices: HashSet<usize> = HashSet::new();

        let mut rng = self.rand.rng();

        while simulated_size > target_size {
            // Need at least 2 non-evicted entries to pick from
            let available_count = cache_state.keys.len() - picked_indices.len();
            if available_count < 2 {
                break;
            }

            // Pick two random indices that haven't been evicted yet
            let idx0 = match self.pick_random_available_index(
                &mut rng,
                &cache_state.keys,
                &picked_indices,
                None,
            ) {
                Some(idx) => idx,
                None => break,
            };
            let idx1 = match self.pick_random_available_index(
                &mut rng,
                &cache_state.keys,
                &picked_indices,
                Some(idx0),
            ) {
                Some(idx) => idx,
                None => break,
            };

            let path0 = &cache_state.keys[idx0];
            let path1 = &cache_state.keys[idx1];

            let entry0 = match cache_state.entries.get(path0) {
                Some(e) => e,
                None => break,
            };
            let entry1 = match cache_state.entries.get(path1) {
                Some(e) => e,
                None => break,
            };

            let (chosen_idx, chosen_path, chosen_bytes) =
                if entry0.access_time <= entry1.access_time {
                    (idx0, path0.clone(), entry0.size_bytes)
                } else {
                    (idx1, path1.clone(), entry1.size_bytes)
                };

            picked_indices.insert(chosen_idx);
            simulated_size = simulated_size.saturating_sub(chosen_bytes as u64);
            targets.push((chosen_path, chosen_bytes));
        }

        targets
    }

    // Pick a random index from keys that hasn't been chosen yet and optionally excludes
    // a specific index. Returns None if no available index exists.
    fn pick_random_available_index(
        &self,
        rng: &mut impl rand::Rng,
        keys: &[std::path::PathBuf],
        picked: &HashSet<usize>,
        exclude_idx: Option<usize>,
    ) -> Option<usize> {
        let excluded_not_picked = exclude_idx.is_some_and(|idx| !picked.contains(&idx));
        let available_count = keys
            .len()
            .saturating_sub(picked.len())
            .saturating_sub(usize::from(excluded_not_picked));

        if available_count == 0 {
            return None;
        }

        loop {
            let idx = rng.random_range(0..keys.len());
            if !picked.contains(&idx) && Some(idx) != exclude_idx {
                return Some(idx);
            }
        }
    }
}

fn wrap_io_err(err: impl std::error::Error + Send + Sync + 'static) -> object_store::Error {
    object_store::Error::Generic {
        store: "cached_object_store",
        source: Box::new(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cached_object_store::stats::{CACHE_BYTES, CACHE_KEYS, EVICTED_BYTES, EVICTED_KEYS};
    use crate::test_utils::gen_rand_bytes;
    use filetime::FileTime;
    use slatedb_common::clock::DefaultSystemClock;
    use slatedb_common::metrics::{lookup_metric, DefaultMetricsRecorder, MetricsRecorderHelper};
    use std::{io::Write, sync::atomic::Ordering, time::SystemTime};

    fn gen_rand_file(
        folder_path: &std::path::Path,
        file_name: &str,
        n: usize,
    ) -> std::path::PathBuf {
        let file_path = folder_path.join(file_name);
        let bytes = gen_rand_bytes(n);
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(&bytes).unwrap();
        file_path
    }

    #[tokio::test]
    async fn test_evictor() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_evictor_")
            .tempdir()
            .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();

        let evictor = FsCacheEvictorInner::new(
            temp_dir.path().to_path_buf(),
            1024 * 2,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DbRand::default()),
            FileHandleCache::new(1000, false),
        );

        let path0 = gen_rand_file(temp_dir.path(), "file0", 1024);
        let evicted = evictor
            .track_entry_accessed(path0, 1024, DefaultSystemClock::default().now(), true)
            .await;
        assert_eq!(evicted, 0);

        let path1 = gen_rand_file(temp_dir.path(), "file1", 1024);
        let evicted = evictor
            .track_entry_accessed(path1, 1024, DefaultSystemClock::default().now(), true)
            .await;
        assert_eq!(evicted, 0);

        let path2 = gen_rand_file(temp_dir.path(), "file2", 1024);
        let evicted = evictor
            .track_entry_accessed(path2, 1024, DefaultSystemClock::default().now(), true)
            .await;
        assert_eq!(evicted, 2048);

        let file_paths = walkdir::WalkDir::new(temp_dir.path())
            .into_iter()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(file_paths.len(), 2); // the folder file "." is also counted
    }

    #[tokio::test]
    async fn test_evictor_track_entry_accessed_backpressure() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_evictor_backpressure_")
            .tempdir()
            .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();

        let evictor = FsCacheEvictor::new(
            temp_dir.path().to_path_buf(),
            1024,
            None,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            FileHandleCache::new(1000, false),
        );

        // Simulate started state without a running receiver so the channel can fill.
        evictor.started.store(true, Ordering::Release);

        for idx in 0..100 {
            let accepted = evictor
                .track_entry_accessed(std::path::PathBuf::from(format!("file{idx}")), 1, true)
                .await;
            assert!(accepted);
        }

        let accepted = evictor
            .track_entry_accessed(std::path::PathBuf::from("overflow"), 1, true)
            .await;
        assert!(!accepted);
    }

    #[tokio::test]
    async fn test_evictor_pick() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_evictor_")
            .tempdir()
            .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let evictor = Arc::new(FsCacheEvictorInner::new(
            temp_dir.path().to_path_buf(),
            1024 * 2,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DbRand::default()),
            FileHandleCache::new(1000, false),
        ));

        let path0 = gen_rand_file(temp_dir.path(), "file0", 1024);
        gen_rand_file(temp_dir.path(), "file1", 1025);

        filetime::set_file_atime(&path0, FileTime::from_system_time(SystemTime::UNIX_EPOCH))
            .unwrap();

        evictor.clone().scan_entries(false).await;

        let targets = evictor.pick_evict_targets(1025).await;

        assert_eq!(targets.len(), 1);
        let (target_path, size) = &targets[0];
        assert_eq!(*target_path, path0);
        assert_eq!(*size, 1024);
    }

    #[tokio::test]
    async fn test_evictor_rescan() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_evictor_")
            .tempdir()
            .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();

        let evictor = Arc::new(FsCacheEvictorInner::new(
            temp_dir.path().to_path_buf(),
            1024 * 2,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DbRand::default()),
            FileHandleCache::new(1000, false),
        ));

        gen_rand_file(temp_dir.path(), "file0", 1024);
        gen_rand_file(temp_dir.path(), "file1", 1025);

        // rescan two times, the cache size should be 2049 unchanged
        evictor.clone().scan_entries(false).await;
        let cache_size_bytes = evictor.cache_size_bytes.load(Ordering::SeqCst);
        assert_eq!(cache_size_bytes, 2049);
        evictor.clone().scan_entries(false).await;
        let cache_size_bytes = evictor.cache_size_bytes.load(Ordering::SeqCst);
        assert_eq!(cache_size_bytes, 2049);
    }

    #[rstest::rstest]
    // Basic case: 2 keys, nothing picked, no exclusion
    #[case(&[0, 1], &[], None, &[0, 1])]
    // Basic case: 2 keys, nothing picked, exclude index 0
    #[case(&[0, 1], &[], Some(0), &[1])]
    // Basic case: 2 keys, nothing picked, exclude index 1
    #[case(&[0, 1], &[], Some(1), &[0])]
    // 3 keys, index 0 picked, no exclusion -> can return 1 or 2
    #[case(&[0, 1, 2], &[0], None, &[1, 2])]
    // 3 keys, index 0 picked, exclude index 1 -> must return 2
    #[case(&[0, 1, 2], &[0], Some(1), &[2])]
    // 4 keys, indices 0,1 picked, no exclusion -> can return 2 or 3
    #[case(&[0, 1, 2, 3], &[0, 1], None, &[2, 3])]
    // 4 keys, indices 0,1 picked, exclude 2 -> must return 3
    #[case(&[0, 1, 2, 3], &[0, 1], Some(2), &[3])]
    // Corner case: exclude_idx is already in picked (redundant exclusion) -
    // index 0 is both picked and excluded
    #[case(&[0, 1], &[0], Some(0), &[1])]
    // Corner case: no available index (all picked or excluded)
    #[case(&[0, 1], &[0], Some(1), &[])]
    fn test_pick_random_available_index(
        #[case] key_indices: &[usize],
        #[case] picked_indices: &[usize],
        #[case] exclude_idx: Option<usize>,
        #[case] expected_possible: &[usize],
    ) {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_pick_")
            .tempdir()
            .unwrap();
        let recorder = slatedb_common::metrics::MetricsRecorderHelper::noop();
        let evictor = FsCacheEvictorInner::new(
            temp_dir.path().to_path_buf(),
            1024,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DbRand::default()),
            FileHandleCache::new(1000, false),
        );

        let keys: Vec<std::path::PathBuf> = key_indices
            .iter()
            .map(|i| std::path::PathBuf::from(format!("file{}", i)))
            .collect();
        let picked: HashSet<usize> = picked_indices.iter().copied().collect();

        let mut rng = evictor.rand.rng();

        if expected_possible.is_empty() {
            // Should return None when no available index exists
            let result = evictor.pick_random_available_index(&mut rng, &keys, &picked, exclude_idx);
            assert!(
                result.is_none(),
                "pick_random_available_index should return None, got {:?}",
                result
            );
        } else {
            for _ in 0..100 {
                let result =
                    evictor.pick_random_available_index(&mut rng, &keys, &picked, exclude_idx);
                assert!(
                    result.is_some_and(|r| expected_possible.contains(&r)),
                    "pick_random_available_index returned {:?}, expected one of {:?}",
                    result,
                    expected_possible
                );
            }
        }
    }

    #[tokio::test]
    async fn test_should_record_cache_and_eviction_metrics() {
        // given:
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_metrics_")
            .tempdir()
            .unwrap();
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let helper = MetricsRecorderHelper::new(recorder.clone(), Default::default());

        let evictor = FsCacheEvictorInner::new(
            temp_dir.path().to_path_buf(),
            1024 * 2,
            Arc::new(CachedObjectStoreStats::new(&helper)),
            Arc::new(DbRand::default()),
            FileHandleCache::new(1000, false),
        );

        // when: add two entries (within capacity)
        let path0 = gen_rand_file(temp_dir.path(), "file0", 1024);
        evictor
            .track_entry_accessed(path0, 1024, DefaultSystemClock::default().now(), true)
            .await;

        let path1 = gen_rand_file(temp_dir.path(), "file1", 1024);
        evictor
            .track_entry_accessed(path1, 1024, DefaultSystemClock::default().now(), true)
            .await;

        // then: cache_keys and cache_bytes are set
        assert_eq!(lookup_metric(&recorder, CACHE_KEYS), Some(2));
        assert_eq!(lookup_metric(&recorder, CACHE_BYTES), Some(2048));

        // when: add a third entry that triggers eviction
        let path2 = gen_rand_file(temp_dir.path(), "file2", 1024);
        evictor
            .track_entry_accessed(path2, 1024, DefaultSystemClock::default().now(), true)
            .await;

        // then: evicted_keys and evicted_bytes are recorded
        let evicted_keys = lookup_metric(&recorder, EVICTED_KEYS).unwrap();
        let evicted_bytes = lookup_metric(&recorder, EVICTED_BYTES).unwrap();
        assert!(
            evicted_keys >= 1,
            "expected evicted_keys >= 1, got {evicted_keys}"
        );
        assert!(
            evicted_bytes >= 1024,
            "expected evicted_bytes >= 1024, got {evicted_bytes}"
        );

        // cache_keys should be updated after eviction
        let keys = lookup_metric(&recorder, CACHE_KEYS).unwrap();
        assert!(keys >= 1, "expected cache_keys >= 1, got {keys}");
    }

    #[tokio::test]
    async fn test_remove_drops_parts_head_and_evictor_accounting() {
        // Spin up an FsCacheStorage with the evictor started so the on-disk
        // accounting is real. Save a few parts and a head, then remove() and
        // assert files, the evictor's cache_size_bytes, and the metrics all go
        // back to a clean slate.
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_remove_")
            .tempdir()
            .unwrap();
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let helper = MetricsRecorderHelper::new(recorder.clone(), Default::default());
        let stats = Arc::new(CachedObjectStoreStats::new(&helper));
        let storage = FsCacheStorage::new(
            temp_dir.path().to_path_buf(),
            Some(1024 * 1024),
            None,
            stats.clone(),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        );
        storage.start_evictor().await;

        let location = Path::from("compacted/x.sst");
        let part_size = 1024;
        let entry = storage.entry(&location, part_size);
        for part_number in 0..3 {
            entry
                .save_part(part_number, Bytes::from(vec![0u8; part_size]))
                .await
                .unwrap();
        }
        let now = Utc::now();
        let meta = ObjectMeta {
            location: location.clone(),
            last_modified: now,
            size: (part_size * 3) as u64,
            e_tag: None,
            version: None,
        };
        let attrs = Attributes::new();
        entry.save_head((&meta, &attrs)).await.unwrap();

        // Sanity: parts and head exist.
        assert_eq!(entry.cached_parts().await.unwrap().len(), 3);
        assert!(entry.read_head().await.unwrap().is_some());

        // Drive the evictor channel so the in-memory state has caught up to
        // the writes before we measure. The save_part path enqueues
        // track_entry_accessed events through a bounded mpsc channel; we
        // poll for the accounting to reflect the writes instead of using a
        // fixed sleep, which is flaky under parallel test load.
        let inner = storage
            .evictor
            .as_ref()
            .unwrap()
            .inner
            .get()
            .unwrap()
            .clone();
        let mut cache_size_before = 0u64;
        for _ in 0..50 {
            cache_size_before = inner.cache_size_bytes.load(Ordering::SeqCst);
            if cache_size_before > 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            cache_size_before > 0,
            "evictor should have tracked the saved parts within 1s"
        );

        storage.remove(&location).await.unwrap();

        // Files should be gone.
        let leftover: Vec<_> = walkdir::WalkDir::new(temp_dir.path())
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .collect();
        assert!(
            leftover.is_empty(),
            "expected all cache files to be removed, found {:?}",
            leftover
        );

        // Evictor accounting should be back to zero for these entries.
        let cache_size_after = inner.cache_size_bytes.load(Ordering::SeqCst);
        assert_eq!(cache_size_after, 0);
    }

    #[tokio::test]
    async fn test_tee_commit_populates_parts_and_head() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_tee_commit_")
            .tempdir()
            .unwrap();
        let recorder = MetricsRecorderHelper::noop();
        let storage = FsCacheStorage::new(
            temp_dir.path().to_path_buf(),
            Some(1024 * 1024),
            None,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        );
        storage.start_evictor().await;

        let location = Path::from("compacted/tee.sst");
        let part_size = 1024;
        let payload: Vec<u8> = (0..2500u32).map(|i| (i % 251) as u8).collect();

        let mut tee = storage.begin_tee(&location, part_size).expect("tee");
        // Push in irregular chunks to exercise buffering across part
        // boundaries.
        for chunk in payload.chunks(300) {
            tee.extend(chunk).await.unwrap();
        }
        let meta = ObjectMeta {
            location: location.clone(),
            last_modified: Utc::now(),
            size: payload.len() as u64,
            e_tag: None,
            version: None,
        };
        tee.commit(&meta, &Attributes::new()).await.unwrap();

        // Read every byte back via the cache entry.
        let entry = storage.entry(&location, part_size);
        let mut got = Vec::with_capacity(payload.len());
        let mut part_number = 0;
        while got.len() < payload.len() {
            let remaining = payload.len() - got.len();
            let want = remaining.min(part_size);
            let chunk = entry
                .read_part(part_number, 0..want)
                .await
                .unwrap()
                .expect("part should be cached after commit");
            got.extend_from_slice(&chunk);
            part_number += 1;
        }
        assert_eq!(got, payload);

        let head = entry.read_head().await.unwrap().expect("head should exist");
        assert_eq!(head.0.size, payload.len() as u64);
    }

    #[tokio::test]
    async fn test_tee_drop_without_commit_leaves_no_temp_files() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_tee_abort_")
            .tempdir()
            .unwrap();
        let recorder = MetricsRecorderHelper::noop();
        let storage = FsCacheStorage::new(
            temp_dir.path().to_path_buf(),
            Some(1024 * 1024),
            None,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        );
        storage.start_evictor().await;

        let location = Path::from("compacted/abort.sst");
        let part_size = 1024;
        {
            let mut tee = storage.begin_tee(&location, part_size).expect("tee");
            // Write enough to flush at least one part to a tmp file.
            tee.extend(&vec![0xab; 2048]).await.unwrap();
            // Drop without commit. The Drop impl schedules cleanup off-thread.
        }

        // Give the cleanup task a moment to run. Cleanup is fire-and-forget;
        // the test waits long enough for the spawned blocking task to finish
        // a small handful of unlinks.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let leftover: Vec<_> = walkdir::WalkDir::new(temp_dir.path())
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .collect();
        assert!(
            leftover.is_empty(),
            "expected no cache files after drop without commit, found {:?}",
            leftover
                .iter()
                .map(|e| e.path().to_path_buf())
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_scan_sweeps_orphan_tee_tmp_files() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_tmp_sweep_")
            .tempdir()
            .unwrap();
        // Pre-seed the directory with a mix of legitimate cache files and
        // orphan tmp files, simulating a process that crashed mid-tee.
        let object_dir = temp_dir.path().join("compacted/orphan.sst");
        std::fs::create_dir_all(&object_dir).unwrap();
        let legit = object_dir.join("_part1kb-000000000");
        std::fs::write(&legit, b"keep-me").unwrap();
        let orphan_part = object_dir.join("_part1kb-000000000.tmp-abcdef");
        std::fs::write(&orphan_part, b"discard-me").unwrap();
        let orphan_head = object_dir.join("_head.tmp-xyz");
        std::fs::write(&orphan_head, b"discard-me-too").unwrap();

        let recorder = MetricsRecorderHelper::noop();
        let inner = Arc::new(FsCacheEvictorInner::new(
            temp_dir.path().to_path_buf(),
            1024 * 1024,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DbRand::default()),
            FileHandleCache::new(1000, false),
        ));

        inner.scan_entries(false).await;

        assert!(legit.exists(), "legit cache file must survive sweep");
        assert!(!orphan_part.exists(), "orphan part tmp must be swept");
        assert!(!orphan_head.exists(), "orphan head tmp must be swept");
    }

    #[tokio::test]
    async fn test_remove_missing_location_is_ok() {
        let temp_dir = tempfile::Builder::new()
            .prefix("objstore_cache_test_remove_missing_")
            .tempdir()
            .unwrap();
        let recorder = MetricsRecorderHelper::noop();
        let storage = FsCacheStorage::new(
            temp_dir.path().to_path_buf(),
            Some(1024),
            None,
            Arc::new(CachedObjectStoreStats::new(&recorder)),
            Arc::new(DefaultSystemClock::new()),
            Arc::new(DbRand::default()),
            1000,
            false, // direct_io
        );
        storage.start_evictor().await;

        // Removing a never-cached location must be a clean no-op.
        storage.remove(&Path::from("never/seen.sst")).await.unwrap();
    }
}
