//! `io_uring`-backed implementation of [`LocalCacheStorage`].
//!
//! Routes every cache read and write through a dedicated OS thread that owns
//! a single `io_uring` instance, instead of dispatching each `pread`/`write`
//! through `tokio::task::spawn_blocking`. The blocking pool is shared by
//! every other consumer in the process; on a busy machine its threads land
//! anywhere the kernel will schedule them, including cores running
//! compaction. That contention is what causes the residual reader-throughput
//! dips even when the working set is fully page-cached. Pinning all cache
//! I/O onto a dedicated worker removes that scheduler-level coupling.
//!
//! Linux-only. The module is only compiled in on `target_os = "linux"`; the
//! constructor on other platforms is wired so that `use_io_uring` silently
//! falls back to [`super::FsCacheStorage`].

use crate::cached_object_store::storage::{
    LocalCacheEntry, LocalCacheHead, LocalCacheStorage, LocalCacheTee, PartID,
};
use bytes::Bytes;
use crossbeam_channel::{Receiver as CbReceiver, Sender as CbSender};
use io_uring::{opcode, types, IoUring};
use object_store::path::Path;
use object_store::{Attributes, ObjectMeta};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::sync::oneshot;
use tracing::{info, warn};

/// `O_DIRECT` alignment requirement on common Linux filesystems. Underlying
/// device may demand 512, but ext4/xfs typically want 4 KB.
const O_DIRECT_ALIGN: usize = 4096;
/// SQ depth. Generous; a few hundred concurrent ops is plausible during a
/// burst (compactor draining input + foreground readers).
const URING_SQ_ENTRIES: u32 = 1024;
/// Max requests pulled from the channel before we submit. Keeps each
/// `submit_and_wait` cycle bounded even under sustained load.
const SUBMIT_BATCH_SIZE: usize = 64;

fn wrap_io_err<E>(err: E) -> object_store::Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    object_store::Error::Generic {
        store: "cached_object_store_uring",
        source: Box::new(err),
    }
}

fn make_part_path(
    root_folder: std::path::PathBuf,
    location: &Path,
    part_number: usize,
    part_size: usize,
) -> std::path::PathBuf {
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
    let mut path = root_folder.join(location.to_string());
    path.push("_head");
    path
}

/// 4 KB-aligned heap buffer for `O_DIRECT` reads. `Bytes::from_owner` keeps
/// the alignment alive for the lifetime of the slice we hand back.
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
        // SAFETY: layout has nonzero size (aligned_cap >= O_DIRECT_ALIGN > 0)
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
        // SAFETY: ptr was alloc'd with `capacity` bytes, alignment is 4 KB.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }
}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
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

/// One in-flight io_uring op. Owns the storage backing its SQE so the kernel
/// does not see a freed buffer; dropped on completion.
enum InFlight {
    Read {
        // Buffer the kernel writes into. Stays alive until completion.
        buf: Vec<u8>,
        // O_DIRECT path uses an aligned buf; we slice on completion.
        aligned: Option<AlignedBuf>,
        // Original requested range inside `buf` / aligned slice.
        requested_offset_in_buf: usize,
        requested_len: usize,
        sender: oneshot::Sender<object_store::Result<Bytes>>,
        // Worker-side timestamp at SQE submission. Used to record
        // observed read latency into the adaptive pacing controller
        // when the CQE arrives.
        submitted_at: std::time::Instant,
    },
    Write {
        // Bytes the kernel reads from. Either a Bytes payload (buffered) or
        // an aligned heap buffer (O_DIRECT). Either way, stays alive until
        // completion.
        _buf: WriteBuf,
        // Owned File handle: dropped (closed) before the rename so the
        // metadata is durable from the kernel's POV.
        file: Option<File>,
        // After the write completes successfully, rename `tmp_path` →
        // `final_path`. If they're equal, rename is a no-op.
        tmp_path: std::path::PathBuf,
        final_path: std::path::PathBuf,
        sender: oneshot::Sender<object_store::Result<()>>,
    },
    /// One chunk of a paced write. The actual byte buffer + open file +
    /// sender live in the [`PacedWrite`] entry keyed by `paced_write_id`
    /// in the worker's `paced_writes` map. This variant is just a
    /// bookmark: when the chunk's CQE arrives the worker looks up the
    /// paced write, marks the chunk completed, and either schedules
    /// the next chunk after `pause` or finalizes the write.
    PacedChunk {
        paced_write_id: u64,
        chunk_len: usize,
    },
}

/// State for a paced write that is interleaved with normal worker ops
/// instead of blocking the worker for its full duration. At any time
/// there's at most one chunk in flight per paced write — submission
/// pauses for `pause` between chunk completions. The worker keeps
/// these alive in `paced_writes: HashMap<u64, PacedWrite>` and uses
/// `delayed_chunks` to dispatch the next chunk at the right time.
struct PacedWrite {
    file: Option<File>,
    raw_fd: RawFd,
    tmp_path: std::path::PathBuf,
    final_path: std::path::PathBuf,
    /// Owns the bytes for the kernel for the entire write. Per-chunk
    /// raw pointer offsets reference into this buffer (buffered path
    /// only; O_DIRECT pacing copies into `aligned_in_flight`).
    bytes: Bytes,
    total_len: usize,
    chunk_bytes: usize,
    /// Offset of the next chunk to submit. When `next_offset_to_submit
    /// == total_len`, all chunks are submitted; we still need the last
    /// CQE before finalizing.
    next_offset_to_submit: usize,
    /// Whether this write was opened with `O_DIRECT`. Drives the per-chunk
    /// aligned-buffer copy + ftruncate-on-finalize tail handling. When
    /// `false`, chunks use raw pointers into `bytes` (buffered path).
    direct_io: bool,
    /// The currently-in-flight chunk's aligned buffer for the O_DIRECT
    /// path. Kept here so it outlives submission and stays valid until
    /// the CQE arrives. None on the buffered path (we use `bytes`
    /// directly), and None between chunks. There is at most one chunk
    /// in flight per paced write by design.
    aligned_in_flight: Option<AlignedBuf>,
    /// True when the last chunk was rounded up beyond `total_len` to
    /// satisfy O_DIRECT alignment. After all chunks complete, we
    /// `ftruncate` the file back down to `total_len` so reads see the
    /// real size, not the zero-padded tail.
    needs_truncate: bool,
    /// Sender for the user's awaited `Result<()>`. Filled with Ok on
    /// successful rename, Err on any chunk failure.
    sender: oneshot::Sender<object_store::Result<()>>,
    failed: Option<object_store::Error>,
}

/// Adaptive pacing controller. Tracks recent read latencies and
/// scales the inter-chunk pause around the configured base value so
/// writes back off when foreground reads tail-spike and speed up
/// when the device is idle. Roughly mirrors RocksDB's `auto_tuned`
/// rate limiter, but driven by observed read p99 instead of write
/// throughput.
struct PacingController {
    /// Base pause between chunks; the multiplier scales around this.
    base_pause: std::time::Duration,
    /// Target foreground read p99 in microseconds. Pause scales up
    /// when measured p99 exceeds this, down when below.
    target_read_p99_us: u64,
    /// Sliding window of recent read completions: `(completion_time,
    /// latency_us)`. Trimmed on insert to drop entries older than
    /// `window`.
    read_window: std::collections::VecDeque<(std::time::Instant, u64)>,
    window: std::time::Duration,
}

impl PacingController {
    fn new(base_pause: std::time::Duration, target_read_p99_us: u64) -> Self {
        Self {
            base_pause,
            target_read_p99_us,
            read_window: std::collections::VecDeque::with_capacity(2048),
            window: std::time::Duration::from_secs(1),
        }
    }

    fn record_read(&mut self, latency_us: u64) {
        let now = std::time::Instant::now();
        self.read_window.push_back((now, latency_us));
        let cutoff = now.checked_sub(self.window);
        if let Some(cutoff) = cutoff {
            while let Some((t, _)) = self.read_window.front() {
                if *t < cutoff {
                    self.read_window.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    fn current_p99_us(&self) -> u64 {
        if self.read_window.is_empty() {
            return 0;
        }
        let mut latencies: Vec<u64> = self.read_window.iter().map(|(_, l)| *l).collect();
        latencies.sort_unstable();
        let idx = ((latencies.len() * 99) / 100).min(latencies.len() - 1);
        latencies[idx]
    }

    /// How long to wait before the next chunk. Multiplier is clamped
    /// to `[0.25, 4.0]` so the rate can't run away in either direction.
    /// Without enough samples (< 32) we fall back to the base.
    fn next_pause(&self) -> std::time::Duration {
        if self.read_window.len() < 32 || self.target_read_p99_us == 0 {
            return self.base_pause;
        }
        let p99 = self.current_p99_us();
        if p99 == 0 {
            return self.base_pause;
        }
        let ratio = p99 as f64 / self.target_read_p99_us as f64;
        let multiplier = ratio.clamp(0.25, 4.0);
        let micros = (self.base_pause.as_micros() as f64 * multiplier) as u64;
        std::time::Duration::from_micros(micros)
    }
}

// The held buffer is never read on the Rust side, but the kernel reads from
// it for the duration of the I/O — the variant exists only to keep the
// allocation alive until completion.
#[allow(dead_code)]
enum WriteBuf {
    Bytes(Bytes),
    Aligned(AlignedBuf),
}

/// Channel message sent from tokio tasks to the worker thread.
enum WorkerOp {
    /// Read `len` bytes at `offset` from `path`. Resolves the path through
    /// the worker's file-handle cache; opens lazily on first miss. If
    /// `direct_io` is set on the storage, the open happens with `O_DIRECT`.
    Read {
        path: std::path::PathBuf,
        offset: u64,
        len: usize,
        sender: oneshot::Sender<object_store::Result<Bytes>>,
    },
    /// Write `bytes` in full to a fresh file at `tmp_path`, then rename it
    /// atomically to `final_path`. The worker creates parent dirs as
    /// needed. No fsync — cache is reconstructable from upstream.
    AtomicWrite {
        tmp_path: std::path::PathBuf,
        final_path: std::path::PathBuf,
        bytes: Bytes,
        sender: oneshot::Sender<object_store::Result<()>>,
    },
    /// Drop the cached fd for `path` (called after rename / delete so the
    /// next read opens the new inode).
    InvalidateFd { path: std::path::PathBuf },
    /// Insert a pre-opened fd into the worker's fd_cache for `path`.
    /// Used by the tee-commit path to hand the worker already-opened
    /// fds for newly-written part / head files, so the first
    /// foreground read avoids a synchronous `open()` on the worker
    /// thread (which has been observed taking 50+ ms under load, head-
    /// of-line blocking subsequent pread SQEs).
    SeedFdCache {
        path: std::path::PathBuf,
        file: Arc<File>,
    },
    /// Remove the cache directory for a location (recursive).
    RemoveDir {
        dir: std::path::PathBuf,
        sender: oneshot::Sender<object_store::Result<()>>,
    },
    /// Stop the worker and join.
    Shutdown,
}

#[derive(Debug)]
struct WorkerHandle {
    tx: CbSender<WorkerOp>,
    join: parking_lot::Mutex<Option<JoinHandle<()>>>,
}

/// Parse an inclusive CPU range `"start-end"`, or a single core `"n"`.
fn parse_cpu_range(spec: &str) -> Option<(usize, usize)> {
    let spec = spec.trim();
    if let Some((a, b)) = spec.split_once('-') {
        let start = a.trim().parse().ok()?;
        let end = b.trim().parse().ok()?;
        return (start <= end).then_some((start, end));
    }
    let n = spec.parse().ok()?;
    Some((n, n))
}

/// Restrict the calling thread to the inclusive CPU range `cpus` via
/// `sched_setaffinity`. The io_uring worker then floats across that CPU
/// set under the kernel scheduler.
fn set_uring_affinity(cpus: (usize, usize)) {
    let (start, end) = cpus;
    // SAFETY: `set` is zeroed before use; `sched_setaffinity` with pid 0
    // targets the calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for cpu in start..=end {
            libc::CPU_SET(cpu, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            warn!(
                "failed to set slatedb-uring affinity to CPUs {}-{}: {}",
                start,
                end,
                std::io::Error::last_os_error()
            );
        }
    }
}

/// Read an inclusive CPU range from env var `var`. Returns `None` if
/// unset; logs and returns `None` if set but malformed.
fn cpu_range_from_env(var: &str) -> Option<(usize, usize)> {
    let spec = std::env::var(var).ok()?;
    match parse_cpu_range(&spec) {
        Some(range) => Some(range),
        None => {
            warn!("ignoring malformed {}={:?}", var, spec);
            None
        }
    }
}

/// Read a positive worker count from env var `var`, defaulting to 1.
fn worker_count_from_env(var: &str) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1)
}

impl WorkerHandle {
    /// Spawn a worker thread. `kind` is `"r"` or `"w"` and only feeds the
    /// thread name; `cpus`, when set, pins the thread to that CPU range.
    fn spawn(
        direct_io: bool,
        kind: &'static str,
        worker_idx: usize,
        cpus: Option<(usize, usize)>,
    ) -> std::io::Result<Self> {
        let (tx, rx) = crossbeam_channel::unbounded::<WorkerOp>();
        let join = std::thread::Builder::new()
            .name(format!("slatedb-uring-{}{}", kind, worker_idx))
            .spawn(move || {
                if let Some(cpus) = cpus {
                    set_uring_affinity(cpus);
                }
                if let Err(e) = run_worker(rx, direct_io, worker_idx) {
                    warn!(
                        "slatedb-uring worker {}{} exited with error: {:?}",
                        kind, worker_idx, e
                    );
                }
            })?;
        Ok(Self {
            tx,
            join: parking_lot::Mutex::new(Some(join)),
        })
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(WorkerOp::Shutdown);
        if let Some(j) = self.join.lock().take() {
            let _ = j.join();
        }
    }
}

/// Worker thread main loop. Owns the io_uring, the file-handle cache, and
/// the in-flight map.
fn run_worker(rx: CbReceiver<WorkerOp>, direct_io: bool, worker_idx: usize) -> std::io::Result<()> {
    let mut ring = IoUring::new(URING_SQ_ENTRIES)?;
    let mut fd_cache: HashMap<std::path::PathBuf, Arc<File>> = HashMap::new();
    let mut pending: HashMap<u64, InFlight> = HashMap::new();
    let mut next_user_data: u64 = 1;

    // Paced write state. Each paced write owns its file + bytes here.
    // `pending` carries one `InFlight::PacedChunk` per in-flight chunk
    // pointing back to this map. `delayed_chunks` is a small set of
    // (ready_at, paced_write_id) pairs the worker checks each loop
    // iteration before submitting; this lets paced writes share the
    // worker with reads instead of blocking it.
    let mut paced_writes: HashMap<u64, PacedWrite> = HashMap::new();
    let mut next_paced_id: u64 = 1;
    let mut delayed_chunks: Vec<(std::time::Instant, u64)> = Vec::new();

    // Cache-write pacing. Reads env once at worker start. When set, big
    // AtomicWrite ops get split into `chunk_bytes`-sized SQEs with
    // `pause` between each chunk's completion and the next chunk's
    // submission. Per-chunk submission is interleaved with normal
    // worker ops so foreground reads aren't held up.
    let chunk_bytes: usize = std::env::var("SLATEDB_CACHE_WRITE_CHUNK_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let pause_us: u64 = std::env::var("SLATEDB_CACHE_WRITE_PAUSE_US")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let pacing_enabled = chunk_bytes > 0 && pause_us > 0;
    // Adaptive: target foreground read p99 in microseconds. The pause
    // scales around `pause_us` based on observed p99: faster when reads
    // are quick, slower when they tail-spike. 0 disables adaptation.
    // Default 1ms — buffered reads served from page cache are <µs;
    // O_DIRECT reads land on NVMe at ~100µs each. Even a couple of
    // hundred µs already signals device queue contention worth
    // backing off.
    let target_read_p99_us: u64 = std::env::var("SLATEDB_CACHE_TARGET_READ_P99_US")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000);
    let mut pacing_controller = PacingController::new(
        std::time::Duration::from_micros(pause_us),
        target_read_p99_us,
    );
    if pacing_enabled && worker_idx == 0 {
        info!(
            "io_uring write pacing enabled [chunk_bytes={}, base_pause_us={}, target_read_p99_us={}]",
            chunk_bytes, pause_us, target_read_p99_us
        );
    }

    loop {
        // Drain channel into SQ. Block-recv only when there's truly
        // nothing to do — including no paced writes between chunks.
        // CRITICAL: include paced_writes in the guard. If paced writes
        // exist but pending is empty (between chunks during the
        // inter-chunk pause), blocking recv would sleep until a new op
        // arrives and miss the chunk's deadline. The idle-bridge sleep
        // below handles waiting in that case.
        let mut batched = 0usize;
        loop {
            if batched >= SUBMIT_BATCH_SIZE {
                break;
            }
            let op = if pending.is_empty() && batched == 0 && paced_writes.is_empty() {
                match rx.recv() {
                    Ok(op) => op,
                    Err(_) => return Ok(()), // sender closed
                }
            } else {
                match rx.try_recv() {
                    Ok(op) => op,
                    Err(crossbeam_channel::TryRecvError::Empty) => break,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(()),
                }
            };

            match op {
                WorkerOp::Shutdown => return Ok(()),
                WorkerOp::InvalidateFd { path } => {
                    fd_cache.remove(&path);
                }
                WorkerOp::SeedFdCache { path, file } => {
                    // The opener (running on tokio::spawn_blocking)
                    // already paid for the syscall; we just insert.
                    // This unconditionally overwrites any prior entry
                    // for `path` because the tee-commit rename
                    // produced a new inode; a stale entry here would
                    // serve old bytes.
                    fd_cache.insert(path, file);
                }
                WorkerOp::RemoveDir { dir, sender } => {
                    let res = std::fs::remove_dir_all(&dir).map_err(wrap_io_err);
                    // Drop any cached fds under this dir.
                    fd_cache.retain(|p, _| !p.starts_with(&dir));
                    let _ = sender.send(res);
                }
                WorkerOp::Read {
                    path,
                    offset,
                    len,
                    sender,
                } => {
                    let fd = match get_or_open_for_read(&mut fd_cache, &path, direct_io) {
                        Ok(arc) => arc,
                        Err(e) => {
                            let _ = sender.send(Err(wrap_io_err(e)));
                            continue;
                        }
                    };
                    let user_data = next_user_data;
                    next_user_data = next_user_data.wrapping_add(1);
                    if direct_io {
                        // Round to 4KB on both ends; we'll slice on completion.
                        let aligned_offset =
                            (offset / O_DIRECT_ALIGN as u64) * O_DIRECT_ALIGN as u64;
                        let head_pad = (offset - aligned_offset) as usize;
                        let aligned_len =
                            (head_pad + len).div_ceil(O_DIRECT_ALIGN) * O_DIRECT_ALIGN;
                        let mut aligned = match AlignedBuf::new(aligned_len) {
                            Ok(b) => b,
                            Err(e) => {
                                let _ = sender.send(Err(wrap_io_err(e)));
                                continue;
                            }
                        };
                        let ptr = aligned.as_mut_slice().as_mut_ptr();
                        let entry =
                            opcode::Read::new(types::Fd(fd.as_raw_fd()), ptr, aligned_len as u32)
                                .offset(aligned_offset)
                                .build()
                                .user_data(user_data);
                        // SAFETY: pending map keeps the buffer alive until we
                        // drain the matching CQE.
                        unsafe {
                            if ring.submission().push(&entry).is_err() {
                                let _ = sender
                                    .send(Err(wrap_io_err(std::io::Error::other("uring SQ full"))));
                                continue;
                            }
                        }
                        pending.insert(
                            user_data,
                            InFlight::Read {
                                buf: Vec::new(),
                                aligned: Some(aligned),
                                requested_offset_in_buf: head_pad,
                                requested_len: len,
                                sender,
                                submitted_at: std::time::Instant::now(),
                            },
                        );
                    } else {
                        let mut buf = vec![0u8; len];
                        let ptr = buf.as_mut_ptr();
                        let entry = opcode::Read::new(types::Fd(fd.as_raw_fd()), ptr, len as u32)
                            .offset(offset)
                            .build()
                            .user_data(user_data);
                        // SAFETY: see comment above.
                        unsafe {
                            if ring.submission().push(&entry).is_err() {
                                let _ = sender
                                    .send(Err(wrap_io_err(std::io::Error::other("uring SQ full"))));
                                continue;
                            }
                        }
                        pending.insert(
                            user_data,
                            InFlight::Read {
                                buf,
                                aligned: None,
                                requested_offset_in_buf: 0,
                                requested_len: len,
                                sender,
                                submitted_at: std::time::Instant::now(),
                            },
                        );
                    }
                    batched += 1;
                }
                WorkerOp::AtomicWrite {
                    tmp_path,
                    final_path,
                    bytes,
                    sender,
                } => {
                    // Cooperative paced fast-path: when pacing is enabled
                    // and the payload is bigger than chunk_bytes, register
                    // a `PacedWrite` and queue its first chunk for
                    // immediate submission. Subsequent chunks are
                    // scheduled in `delayed_chunks` after each completion.
                    // The worker's main loop continues to drain the read
                    // channel between chunks so foreground reads aren't
                    // held up.
                    //
                    // O_DIRECT-aware: when `direct_io` is set, open the
                    // file with O_DIRECT, pre-allocate the full extent
                    // with `fallocate`, and per chunk copy bytes into an
                    // aligned buffer rounded up to the FS block size.
                    // The tail chunk is the only one that may be padded;
                    // we `ftruncate` to the real length after the last
                    // chunk completes. `chunk_bytes` is required to be a
                    // multiple of `O_DIRECT_ALIGN` (1 MiB satisfies this)
                    // so non-tail chunks need no padding at all.
                    //
                    // Why all this matters: with O_DIRECT writes the
                    // bytes never enter the page cache, so the
                    // `POSIX_FADV_DONTNEED` calls the compactor issues
                    // after a compaction become trivial no-ops — no
                    // per-file open + scan + drop sweep on the io_uring
                    // worker, which used to block foreground reads for
                    // multi-second stretches at compaction completion.
                    if pacing_enabled && bytes.len() > chunk_bytes {
                        if let Some(parent) = tmp_path.parent() {
                            if let Err(e) = std::fs::create_dir_all(parent) {
                                let _ = sender.send(Err(wrap_io_err(e)));
                                continue;
                            }
                        }
                        // O_DIRECT requires `chunk_bytes` to be a multiple
                        // of the FS block size. The pacing knob is set by
                        // the operator; reject misconfiguration loudly so
                        // a future "chunk_bytes=1023" doesn't silently
                        // corrupt or fail every write.
                        let direct_io_paced =
                            direct_io && chunk_bytes.is_multiple_of(O_DIRECT_ALIGN);
                        if direct_io && !direct_io_paced {
                            warn!(
                                "io_uring paced write: chunk_bytes={} is not a multiple of \
                                 O_DIRECT_ALIGN={}; falling back to buffered for this write",
                                chunk_bytes, O_DIRECT_ALIGN
                            );
                        }
                        let mut open = OpenOptions::new();
                        open.create(true).truncate(true).write(true);
                        if direct_io_paced {
                            open.custom_flags(libc::O_DIRECT);
                        }
                        let file = match open.open(&tmp_path) {
                            Ok(f) => f,
                            Err(e) => {
                                let _ = sender.send(Err(wrap_io_err(e)));
                                continue;
                            }
                        };
                        let raw_fd = file.as_raw_fd();
                        // Preallocate the full extent in one shot. The
                        // FS now knows the final size before any pwrites
                        // arrive, so it allocates one contiguous extent
                        // instead of growing the file chunk-by-chunk.
                        // This keeps the on-disk layout sequential at
                        // the LBA level, friendly to NVMe controllers.
                        // Best-effort: on filesystems that don't support
                        // fallocate, the writes still succeed - they'll
                        // just allocate as they go.
                        // SAFETY: raw_fd is valid; len is the total
                        // payload size.
                        let preallocated =
                            unsafe { libc::fallocate(raw_fd, 0, 0, bytes.len() as libc::off_t) };
                        if preallocated != 0 {
                            // Not fatal; log once and move on.
                            log::debug!(
                                "io_uring paced write: fallocate failed [path={}, errno={}]; \
                                 writes will extend the file on demand",
                                tmp_path.display(),
                                std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                            );
                        }
                        let id = next_paced_id;
                        next_paced_id = next_paced_id.wrapping_add(1);
                        paced_writes.insert(
                            id,
                            PacedWrite {
                                file: Some(file),
                                raw_fd,
                                tmp_path,
                                final_path,
                                total_len: bytes.len(),
                                bytes,
                                chunk_bytes,
                                next_offset_to_submit: 0,
                                direct_io: direct_io_paced,
                                aligned_in_flight: None,
                                needs_truncate: false,
                                sender,
                                failed: None,
                            },
                        );
                        // First chunk: submit immediately on the next
                        // delayed_chunks pass below.
                        delayed_chunks.push((std::time::Instant::now(), id));
                        continue;
                    }

                    // Open the tmp file synchronously (creates parent dirs
                    // first). Opens are infrequent; the win from io_uring is
                    // on the actual write itself.
                    if let Some(parent) = tmp_path.parent() {
                        if let Err(e) = std::fs::create_dir_all(parent) {
                            let _ = sender.send(Err(wrap_io_err(e)));
                            continue;
                        }
                    }
                    let mut open = OpenOptions::new();
                    open.create(true).truncate(true).write(true);
                    let aligned_payload = direct_io && bytes.len() % O_DIRECT_ALIGN == 0;
                    if aligned_payload {
                        // Only apply O_DIRECT to writes whose payload length
                        // satisfies alignment. Heads (and any tail-end short
                        // writes) stay buffered.
                        open.custom_flags(libc::O_DIRECT);
                    }
                    let file = match open.open(&tmp_path) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ = sender.send(Err(wrap_io_err(e)));
                            continue;
                        }
                    };
                    let raw = file.as_raw_fd();
                    let user_data = next_user_data;
                    next_user_data = next_user_data.wrapping_add(1);

                    let (write_buf, ptr, len) = if aligned_payload {
                        let mut aligned = match AlignedBuf::new(bytes.len()) {
                            Ok(b) => b,
                            Err(e) => {
                                let _ = sender.send(Err(wrap_io_err(e)));
                                continue;
                            }
                        };
                        aligned.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
                        let p = aligned.as_ref().as_ptr();
                        let l = bytes.len() as u32;
                        (WriteBuf::Aligned(aligned), p, l)
                    } else {
                        let p = bytes.as_ptr();
                        let l = bytes.len() as u32;
                        (WriteBuf::Bytes(bytes.clone()), p, l)
                    };

                    let entry = opcode::Write::new(types::Fd(raw), ptr, len)
                        .offset(0)
                        .build()
                        .user_data(user_data);
                    // SAFETY: write_buf lives in `pending` until the matching
                    // CQE is drained, so the kernel sees a stable buffer for
                    // the entirety of the I/O.
                    unsafe {
                        if ring.submission().push(&entry).is_err() {
                            let _ = sender
                                .send(Err(wrap_io_err(std::io::Error::other("uring SQ full"))));
                            continue;
                        }
                    }
                    pending.insert(
                        user_data,
                        InFlight::Write {
                            _buf: write_buf,
                            file: Some(file),
                            tmp_path,
                            final_path,
                            sender,
                        },
                    );
                    batched += 1;
                }
            }
        }

        // Submit any paced-write chunks whose pause has expired. We do
        // this after draining the channel and before submit_and_wait
        // so the same submit() call carries them along with whatever
        // foreground reads we just queued.
        let now = std::time::Instant::now();
        let mut still_delayed = Vec::with_capacity(delayed_chunks.len());
        for (deadline, id) in delayed_chunks.drain(..) {
            if deadline > now {
                still_delayed.push((deadline, id));
                continue;
            }
            let pw = match paced_writes.get_mut(&id) {
                Some(p) => p,
                None => continue, // already finalized / canceled
            };
            // If a previous chunk failed, finalize without submitting more.
            if pw.failed.is_some() {
                continue;
            }
            let chunk_start = pw.next_offset_to_submit;
            let chunk_end = (chunk_start + pw.chunk_bytes).min(pw.total_len);
            let chunk_len = chunk_end - chunk_start;
            // Choose the source pointer + submitted length. Two paths:
            //
            // - Buffered (`direct_io == false`): point straight into
            //   `pw.bytes` and submit `chunk_len`. The kernel can deal
            //   with arbitrary lengths and offsets here.
            //
            // - O_DIRECT (`direct_io == true`): copy `chunk_len` bytes
            //   into an aligned buffer of length `round_up(chunk_len,
            //   O_DIRECT_ALIGN)`. AlignedBuf::new already rounds up and
            //   zero-fills, so the tail bytes past `chunk_len` are
            //   zero. We submit the rounded-up length; the FS records
            //   the file size at `chunk_start + write_len`, which may
            //   overshoot `pw.total_len` on the final chunk. The
            //   finalize path runs `ftruncate(total_len)` to chop it.
            let chunk_user_data = next_user_data;
            next_user_data = next_user_data.wrapping_add(1);

            let (chunk_ptr, submit_len) = if pw.direct_io {
                let needs_pad = !chunk_len.is_multiple_of(O_DIRECT_ALIGN);
                let aligned = match AlignedBuf::new(chunk_len) {
                    Ok(b) => b,
                    Err(e) => {
                        pw.failed = Some(wrap_io_err(e));
                        // Drain by pretending the chunk submitted and
                        // completed with error: jump to finalize on the
                        // next pass. Push into the new vec being built
                        // (`still_delayed`) rather than `delayed_chunks`,
                        // which is currently being drained.
                        pw.next_offset_to_submit = pw.total_len;
                        still_delayed.push((std::time::Instant::now(), id));
                        next_user_data = next_user_data.wrapping_sub(1);
                        continue;
                    }
                };
                let mut aligned = aligned;
                // Copy the real bytes; the rest of the aligned buffer
                // is already zero from alloc_zeroed.
                aligned.as_mut_slice()[..chunk_len]
                    .copy_from_slice(&pw.bytes[chunk_start..chunk_end]);
                let ptr = aligned.as_ref().as_ptr();
                let submit_len = aligned.as_ref().len();
                if needs_pad {
                    pw.needs_truncate = true;
                }
                pw.aligned_in_flight = Some(aligned);
                (ptr, submit_len)
            } else {
                // SAFETY: `pw.bytes` is held in `paced_writes` until
                // the chunk's CQE arrives (we don't remove the
                // PacedWrite before then), so this raw pointer + len is
                // valid for the kernel's exclusive read.
                let ptr = unsafe { pw.bytes.as_ptr().add(chunk_start) };
                (ptr, chunk_len)
            };

            let entry = opcode::Write::new(types::Fd(pw.raw_fd), chunk_ptr, submit_len as u32)
                .offset(chunk_start as u64)
                .build()
                .user_data(chunk_user_data);
            // SAFETY: see comment above re. buffer lifetime.
            unsafe {
                if ring.submission().push(&entry).is_err() {
                    // Defer; try again next tick. Release the aligned
                    // buffer (if any) since the kernel never saw it -
                    // we'll allocate fresh on the retry.
                    pw.aligned_in_flight = None;
                    still_delayed.push((deadline, id));
                    next_user_data = next_user_data.wrapping_sub(1);
                    continue;
                }
            }
            pw.next_offset_to_submit = chunk_end;
            // `chunk_len` here is the *logical* length the CQE result
            // should match (the kernel returns the number of bytes the
            // write covered, which equals `submit_len` on success). We
            // pass `submit_len` to the completion handler so the
            // short-write check uses the same number the kernel sees.
            pending.insert(
                chunk_user_data,
                InFlight::PacedChunk {
                    paced_write_id: id,
                    chunk_len: submit_len,
                },
            );
            batched += 1;
        }
        delayed_chunks = still_delayed;

        // Submit + wait for at least one completion if anything is in flight.
        if batched > 0 || !pending.is_empty() {
            let want = if pending.is_empty() { 0 } else { 1 };
            ring.submit_and_wait(want)?;
        }

        // Drain CQ.
        let mut cq = ring.completion();
        cq.sync();
        for cqe in cq.by_ref() {
            let user_data = cqe.user_data();
            let result = cqe.result();
            // Paced chunks need bespoke handling: schedule the next
            // chunk after `pause` on success, or finalize the whole
            // write on the last chunk / on error.
            match pending.remove(&user_data) {
                Some(InFlight::PacedChunk {
                    paced_write_id,
                    chunk_len,
                }) => {
                    handle_paced_chunk_completion(
                        paced_write_id,
                        chunk_len,
                        result,
                        &mut paced_writes,
                        &mut delayed_chunks,
                        &pacing_controller,
                    );
                    continue;
                }
                Some(other) => {
                    // Reinsert and dispatch through the usual match below.
                    pending.insert(user_data, other);
                }
                None => {
                    warn!(
                        "slatedb-uring: completion for unknown user_data={}",
                        user_data
                    );
                    continue;
                }
            }
            match pending.remove(&user_data) {
                Some(InFlight::Read {
                    buf,
                    aligned,
                    requested_offset_in_buf,
                    requested_len,
                    sender,
                    submitted_at,
                }) => {
                    // Feed observed read latency into the adaptive
                    // pacing controller before doing anything else with
                    // the buffer; pacing wants prompt signal even if
                    // the result is an error.
                    let lat_us = submitted_at.elapsed().as_micros().min(u64::MAX as u128) as u64;
                    pacing_controller.record_read(lat_us);
                    if result < 0 {
                        let _ = sender
                            .send(Err(wrap_io_err(std::io::Error::from_raw_os_error(-result))));
                    } else {
                        let n = result as usize;
                        // For O_DIRECT path, slice from aligned buf.
                        if let Some(aligned) = aligned {
                            let avail = n.saturating_sub(requested_offset_in_buf);
                            let take = avail.min(requested_len);
                            let slice = &aligned.as_ref()
                                [requested_offset_in_buf..requested_offset_in_buf + take];
                            let bytes = Bytes::copy_from_slice(slice);
                            let _ = sender.send(Ok(bytes));
                        } else {
                            // Buffered path: trim to actual short read length.
                            let take = n.min(requested_len);
                            let mut out = buf;
                            out.truncate(take);
                            let _ = sender.send(Ok(Bytes::from(out)));
                        }
                    }
                }
                Some(InFlight::Write {
                    _buf,
                    mut file,
                    tmp_path,
                    final_path,
                    sender,
                }) => {
                    // Drop the underlying File first (closes the fd) so the
                    // rename observes a fully-flushed metadata state. The
                    // _buf goes with it (the kernel is no longer reading).
                    drop(file.take());
                    let send_result = if result < 0 {
                        let _ = std::fs::remove_file(&tmp_path);
                        Err(wrap_io_err(std::io::Error::from_raw_os_error(-result)))
                    } else if tmp_path == final_path {
                        // Tee path: write to tmp only, rename happens later
                        // in commit().
                        Ok(())
                    } else {
                        std::fs::rename(&tmp_path, &final_path).map_err(wrap_io_err)
                    };
                    let _ = sender.send(send_result);
                }
                Some(InFlight::PacedChunk { paced_write_id, .. }) => {
                    // Should be unreachable: the dispatch above handles
                    // PacedChunk and `continue`s before we reach this
                    // match again. Guard anyway in case of refactor.
                    warn!(
                        "slatedb-uring: PacedChunk reached secondary dispatch (id={})",
                        paced_write_id
                    );
                }
                None => {
                    // Lost cqe? Should not happen.
                    warn!(
                        "slatedb-uring: completion for unknown user_data={}",
                        user_data
                    );
                }
            }
        }

        // Idle bridge: when the only outstanding work is a paced write
        // currently in its inter-chunk pause (pending empty,
        // paced_writes non-empty), sleep until the next chunk's
        // deadline so we don't busy-loop. Wake early if a foreground
        // op arrives. CRITICAL: peek with `rx.len()` — never
        // `try_recv`, which would consume the op and strand its
        // sender.
        if !paced_writes.is_empty() && pending.is_empty() {
            let next_deadline = delayed_chunks
                .iter()
                .map(|(d, _)| *d)
                .min()
                .unwrap_or_else(|| std::time::Instant::now() + std::time::Duration::from_millis(1));
            let now = std::time::Instant::now();
            if next_deadline > now {
                let sleep_total = (next_deadline - now).min(std::time::Duration::from_millis(50));
                let sleep_deadline = now + sleep_total;
                while std::time::Instant::now() < sleep_deadline {
                    if rx.len() > 0 {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(500));
                }
            }
        }
    }
}

/// Handle a CQE for one chunk of a paced write. Updates the matching
/// `PacedWrite` state and either schedules the next chunk after the
/// adaptive pause (from the controller) or finalizes the write (close
/// + rename + signal the user oneshot). On error, finalizes early
/// with the error.
fn handle_paced_chunk_completion(
    paced_write_id: u64,
    chunk_len: usize,
    cqe_result: i32,
    paced_writes: &mut HashMap<u64, PacedWrite>,
    delayed_chunks: &mut Vec<(std::time::Instant, u64)>,
    pacing_controller: &PacingController,
) {
    let Some(pw) = paced_writes.get_mut(&paced_write_id) else {
        return;
    };
    // The kernel is done with the just-completed chunk's buffer; release
    // it now. Subsequent chunks allocate fresh in the submission path.
    pw.aligned_in_flight = None;
    if cqe_result < 0 {
        pw.failed = Some(wrap_io_err(std::io::Error::from_raw_os_error(-cqe_result)));
    } else {
        let written = cqe_result as usize;
        if written != chunk_len {
            pw.failed = Some(wrap_io_err(std::io::Error::other(format!(
                "uring short write: {} of {}",
                written, chunk_len
            ))));
        }
    }

    // Still have more chunks to submit and no failure: schedule next chunk
    // after the adaptive pause. Recomputed each chunk so transient read
    // p99 spikes immediately throttle subsequent chunks.
    if pw.failed.is_none() && pw.next_offset_to_submit < pw.total_len {
        let deadline = std::time::Instant::now() + pacing_controller.next_pause();
        delayed_chunks.push((deadline, paced_write_id));
        return;
    }

    // Finalize: either we just completed the last chunk, or we hit an
    // error and want to bail out. Either way drop the file + rename +
    // signal sender.
    let mut pw = paced_writes.remove(&paced_write_id).expect("just had it");
    // If the last O_DIRECT chunk was padded, the file's logical size
    // overshoots `total_len`. Bring it back down before rename so
    // readers see the right size (`cached_head` derives `size` from
    // the disk file's metadata in some paths). Done while we still
    // hold the fd to avoid an open/close round-trip.
    if pw.needs_truncate && pw.failed.is_none() {
        if let Some(file) = pw.file.as_ref() {
            // SAFETY: file is owned and the fd is valid for the duration
            // of this call.
            let rc = unsafe { libc::ftruncate(file.as_raw_fd(), pw.total_len as libc::off_t) };
            if rc != 0 {
                pw.failed = Some(wrap_io_err(std::io::Error::last_os_error()));
            }
        }
    }
    drop(pw.file.take()); // close fd before rename
    let result = if let Some(err) = pw.failed.take() {
        let _ = std::fs::remove_file(&pw.tmp_path);
        Err(err)
    } else if pw.tmp_path == pw.final_path {
        // Tee path: rename happens later in commit().
        Ok(())
    } else {
        std::fs::rename(&pw.tmp_path, &pw.final_path).map_err(wrap_io_err)
    };
    let _ = pw.sender.send(result);
}

fn get_or_open_for_read(
    fd_cache: &mut HashMap<std::path::PathBuf, Arc<File>>,
    path: &std::path::Path,
    direct_io: bool,
) -> std::io::Result<Arc<File>> {
    if let Some(f) = fd_cache.get(path) {
        // Validate by stat: if the inode has been unlinked (rename replaced
        // it), reopen so we read from the new inode.
        if is_handle_valid(f) {
            return Ok(f.clone());
        }
        fd_cache.remove(path);
    }
    let mut open = OpenOptions::new();
    open.read(true);
    if direct_io {
        open.custom_flags(libc::O_DIRECT | libc::O_NOATIME);
    }
    // Time the open syscall. Runs synchronously on the io_uring worker
    // thread: any time spent here blocks subsequent foreground pread
    // SQEs queued behind it. A 50 ms+ open under load means the FS
    // metadata layer is contended (typically by parallel compaction
    // tee writes / rename batches hitting the same inode allocator).
    // Pair with `slow advance_block: next_iter ...` to confirm the
    // first-block-of-fresh-SST slowness is the open and not the read.
    const SLOW_OPEN_THRESHOLD: std::time::Duration = std::time::Duration::from_millis(20);
    let open_start = std::time::Instant::now();
    let file = open.open(path)?;
    let open_elapsed = open_start.elapsed();
    if open_elapsed > SLOW_OPEN_THRESHOLD {
        log::warn!(
            "slow io_uring worker open: path={} took={:?} direct_io={}",
            path.display(),
            open_elapsed,
            direct_io,
        );
    }
    let arc = Arc::new(file);
    fd_cache.insert(path.to_path_buf(), arc.clone());
    Ok(arc)
}

fn is_handle_valid(file: &File) -> bool {
    // Mirror the storage_fs.rs trick: a renamed-over inode has nlink == 0
    // because the dir entry was unlinked by `rename()`.
    use std::os::unix::fs::MetadataExt;
    file.metadata().map(|m| m.nlink() > 0).unwrap_or(false)
}

// -----------------------------------------------------------------------------
// Public types implementing the LocalCacheStorage / Entry / Tee traits.
// -----------------------------------------------------------------------------

/// io_uring-backed `LocalCacheStorage`. Owns two fixed-size pools of
/// dedicated I/O worker threads - a read pool and a write pool - each
/// worker owning its own `IoUring` + fd cache + pending map. Splitting
/// reads from writes keeps foreground read SQEs from queueing behind
/// background write SQEs in the same submission ring, and lets the two
/// classes pin to different CPUs. Within a pool, operations are sharded
/// by hashing the location path: every operation for a given SST always
/// lands on the same worker so its fd cache stays consistent and
/// per-file ordering (e.g. tee chunk submission) is preserved without
/// cross-worker locking. Pool sizes from `SLATEDB_URING_READ_WORKERS` /
/// `SLATEDB_URING_WRITE_WORKERS` (default 1 each); CPU affinity from
/// `SLATEDB_URING_READ_CPUS` / `SLATEDB_URING_WRITE_CPUS`.
#[derive(Debug)]
pub(crate) struct IoUringCacheStorage {
    root_folder: std::path::PathBuf,
    direct_io: bool,
    read_workers: Vec<Arc<WorkerHandle>>,
    write_workers: Vec<Arc<WorkerHandle>>,
    /// Monotonically increasing tee id, used to namespace temp filenames.
    tee_seq: AtomicU64,
}

impl std::fmt::Display for IoUringCacheStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IoUringCacheStorage(root={}, direct_io={}, read_workers={}, write_workers={})",
            self.root_folder.display(),
            self.direct_io,
            self.read_workers.len(),
            self.write_workers.len(),
        )
    }
}

impl IoUringCacheStorage {
    /// Try to construct. Returns Err if the kernel rejects `io_uring_setup`
    /// (ENOSYS on too-old kernels, EPERM with seccomp policies, etc.).
    pub(crate) fn try_new(
        root_folder: std::path::PathBuf,
        direct_io: bool,
    ) -> std::io::Result<Self> {
        let read_cpus = cpu_range_from_env("SLATEDB_URING_READ_CPUS");
        let write_cpus = cpu_range_from_env("SLATEDB_URING_WRITE_CPUS");
        let n_read = worker_count_from_env("SLATEDB_URING_READ_WORKERS");
        let n_write = worker_count_from_env("SLATEDB_URING_WRITE_WORKERS");
        let mut read_workers = Vec::with_capacity(n_read);
        for i in 0..n_read {
            read_workers.push(Arc::new(WorkerHandle::spawn(direct_io, "r", i, read_cpus)?));
        }
        let mut write_workers = Vec::with_capacity(n_write);
        for i in 0..n_write {
            write_workers.push(Arc::new(WorkerHandle::spawn(
                direct_io, "w", i, write_cpus,
            )?));
        }
        info!(
            "using io_uring cache storage [root={}, direct_io={}, \
             read_workers={}, write_workers={}]",
            root_folder.display(),
            direct_io,
            n_read,
            n_write
        );
        Ok(Self {
            root_folder,
            direct_io,
            read_workers,
            write_workers,
            tee_seq: AtomicU64::new(0),
        })
    }

    /// Pick the worker that owns `location` within `workers`. Same
    /// location always maps to the same worker for the lifetime of the
    /// storage so the fd cache stays consistent and per-file ordering is
    /// preserved.
    fn worker_in(workers: &[Arc<WorkerHandle>], location: &Path) -> Arc<WorkerHandle> {
        if workers.len() == 1 {
            return workers[0].clone();
        }
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        location.to_string().hash(&mut h);
        let idx = (h.finish() as usize) % workers.len();
        workers[idx].clone()
    }

    /// Read worker owning `location` (serves `Read` and fd-cache ops).
    fn read_worker(&self, location: &Path) -> Arc<WorkerHandle> {
        Self::worker_in(&self.read_workers, location)
    }

    /// Write worker owning `location` (serves `AtomicWrite` / `RemoveDir`).
    fn write_worker(&self, location: &Path) -> Arc<WorkerHandle> {
        Self::worker_in(&self.write_workers, location)
    }
}

#[async_trait::async_trait]
impl LocalCacheStorage for IoUringCacheStorage {
    fn entry(&self, location: &Path, part_size: usize) -> Box<dyn LocalCacheEntry> {
        Box::new(IoUringCacheEntry {
            root_folder: self.root_folder.clone(),
            read_worker: self.read_worker(location),
            write_worker: self.write_worker(location),
            location: location.clone(),
            part_size,
        })
    }

    async fn start_evictor(&self) {
        // Eviction is not implemented for the io_uring backend yet. The
        // backend is opt-in and used in benchmarks where the cache cap is
        // disabled (max_cache_size_bytes = None), so eviction would be a
        // no-op anyway. Adding it later means wiring the same scan/track
        // logic onto a worker timer; out of scope for the first cut.
    }

    async fn remove(&self, location: &Path) -> object_store::Result<()> {
        let dir = self.root_folder.join(location.to_string());
        let worker = self.write_worker(location);
        let (sender, recv) = oneshot::channel();
        worker
            .tx
            .send(WorkerOp::RemoveDir { dir, sender })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        recv.await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?
    }

    fn begin_tee(&self, location: &Path, part_size: usize) -> Option<Box<dyn LocalCacheTee>> {
        let tee_id = self.tee_seq.fetch_add(1, Ordering::Relaxed);
        Some(Box::new(IoUringCacheTee {
            root_folder: self.root_folder.clone(),
            read_worker: self.read_worker(location),
            write_worker: self.write_worker(location),
            location: location.clone(),
            part_size,
            tee_id,
            part_buf: bytes::BytesMut::new(),
            next_part_number: 0,
            poisoned: false,
            pending_renames: Vec::new(),
        }))
    }
}

#[derive(Debug)]
struct IoUringCacheEntry {
    root_folder: std::path::PathBuf,
    location: Path,
    part_size: usize,
    read_worker: Arc<WorkerHandle>,
    write_worker: Arc<WorkerHandle>,
}

impl IoUringCacheEntry {
    fn make_rand_suffix(&self) -> String {
        // Same shape as storage_fs: `_tmp` + 16 alpha chars. We're inside
        // the I/O hot path; use rand::thread_rng() to avoid threading a
        // shared rng through the storage.
        use rand::distr::Alphanumeric;
        use rand::Rng;
        let mut rng = rand::rng();
        (0..16).map(|_| rng.sample(Alphanumeric) as char).collect()
    }
}

#[async_trait::async_trait]
impl LocalCacheEntry for IoUringCacheEntry {
    async fn save_part(&self, part_number: PartID, buf: Bytes) -> object_store::Result<()> {
        let final_path = make_part_path(
            self.root_folder.clone(),
            &self.location,
            part_number,
            self.part_size,
        );
        let tmp_path = final_path.with_extension(format!("_tmp{}", self.make_rand_suffix()));
        let (sender, recv) = oneshot::channel();
        self.write_worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path,
                final_path: final_path.clone(),
                bytes: buf,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let result = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        // Invalidate any cached fd at final_path so the next read sees the
        // new inode. The fd cache lives on the read worker.
        let _ = self
            .read_worker
            .tx
            .send(WorkerOp::InvalidateFd { path: final_path });
        result
    }

    async fn read_part(
        &self,
        part_number: PartID,
        range_in_part: std::ops::Range<usize>,
    ) -> object_store::Result<Option<Bytes>> {
        let path = make_part_path(
            self.root_folder.clone(),
            &self.location,
            part_number,
            self.part_size,
        );
        let len = range_in_part.end - range_in_part.start;
        let offset = range_in_part.start as u64;
        let (sender, recv) = oneshot::channel();
        if self
            .read_worker
            .tx
            .send(WorkerOp::Read {
                path,
                offset,
                len,
                sender,
            })
            .is_err()
        {
            return Err(wrap_io_err(std::io::Error::other(
                "uring worker channel closed",
            )));
        }
        match recv.await {
            Ok(Ok(bytes)) => Ok(Some(bytes)),
            Ok(Err(e)) => match e {
                object_store::Error::Generic { ref source, .. } => {
                    // Translate ENOENT to None (cache miss), as FsCacheStorage does.
                    if let Some(io) = source.downcast_ref::<std::io::Error>() {
                        if io.kind() == std::io::ErrorKind::NotFound {
                            return Ok(None);
                        }
                    }
                    Err(e)
                }
                _ => Err(e),
            },
            Err(e) => Err(wrap_io_err(std::io::Error::other(format!(
                "oneshot recv: {e}"
            )))),
        }
    }

    async fn cached_parts(&self) -> object_store::Result<Vec<PartID>> {
        // Cold path; sync read_dir is fine.
        let dir = self.root_folder.join(self.location.to_string());
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(wrap_io_err(e)),
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(rest) = name_str.strip_prefix("_part") {
                if let Some(idx) = rest.rfind('-') {
                    if let Ok(n) = rest[idx + 1..].parse::<usize>() {
                        out.push(n);
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    async fn save_head(&self, head: (&ObjectMeta, &Attributes)) -> object_store::Result<()> {
        let head_path = make_head_path(self.root_folder.clone(), &self.location);
        // If the head already exists, skip — same semantics as FsCacheStorage.
        if head_path.exists() {
            return Ok(());
        }
        let head_struct: LocalCacheHead = head.into();
        let buf: Bytes = serde_json::to_vec(&head_struct)
            .map_err(wrap_io_err)?
            .into();
        let tmp_path = head_path.with_extension(format!("_tmp{}", self.make_rand_suffix()));
        let (sender, recv) = oneshot::channel();
        self.write_worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path,
                final_path: head_path.clone(),
                bytes: buf,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let result = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        let _ = self
            .read_worker
            .tx
            .send(WorkerOp::InvalidateFd { path: head_path });
        result
    }

    async fn read_head(&self) -> object_store::Result<Option<(ObjectMeta, Attributes)>> {
        let path = make_head_path(self.root_folder.clone(), &self.location);
        let metadata = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(wrap_io_err(e)),
        };
        let len = metadata.len() as usize;
        if len == 0 {
            return Ok(None);
        }
        let (sender, recv) = oneshot::channel();
        self.read_worker
            .tx
            .send(WorkerOp::Read {
                path,
                offset: 0,
                len,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let bytes = match recv.await {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => return Err(e),
            Err(e) => {
                return Err(wrap_io_err(std::io::Error::other(format!(
                    "oneshot recv: {e}"
                ))))
            }
        };
        let head: LocalCacheHead = match serde_json::from_slice(&bytes) {
            Ok(h) => h,
            Err(e) => return Err(wrap_io_err(e)),
        };
        // Note: we don't have the head's stat for "self.direct_io"-backed
        // path. Pass through meta + attrs directly.
        Ok(Some((head.meta(), head.attributes())))
    }
}

#[derive(Debug)]
struct IoUringCacheTee {
    root_folder: std::path::PathBuf,
    location: Path,
    part_size: usize,
    read_worker: Arc<WorkerHandle>,
    write_worker: Arc<WorkerHandle>,
    tee_id: u64,
    part_buf: bytes::BytesMut,
    next_part_number: usize,
    poisoned: bool,
    /// Tmp path → final path pairs collected from each `extend` flush.
    /// Renames happen in `commit` after upstream confirms the upload.
    pending_renames: Vec<(std::path::PathBuf, std::path::PathBuf)>,
}

impl IoUringCacheTee {
    async fn dispatch_part(&mut self, payload: Bytes) -> object_store::Result<()> {
        let final_path = make_part_path(
            self.root_folder.clone(),
            &self.location,
            self.next_part_number,
            self.part_size,
        );
        let tmp_path =
            final_path.with_extension(format!("_tee{}-{}", self.tee_id, self.next_part_number));
        self.next_part_number += 1;
        // Write the temp file via worker. We reuse the AtomicWrite path
        // because rename-on-write is exactly what we want for the FINAL
        // commit — but here we want temp-only with no rename. We do an
        // AtomicWrite with final_path == tmp_path: the write hits tmp_path,
        // the "rename" is a no-op (same source/dest).
        //
        // Simpler: write to tmp_path directly. We model this as a WriteThen
        // op: AtomicWrite where final_path equals tmp_path skips the rename
        // because std::fs::rename(a, a) is OK on Linux (no-op).
        let (sender, recv) = oneshot::channel();
        self.write_worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path: tmp_path.clone(),
                final_path: tmp_path.clone(),
                bytes: payload,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let result = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        if let Err(e) = result {
            self.poisoned = true;
            return Err(e);
        }
        self.pending_renames.push((tmp_path, final_path));
        Ok(())
    }
}

#[async_trait::async_trait]
impl LocalCacheTee for IoUringCacheTee {
    async fn extend(&mut self, buf: &[u8]) -> object_store::Result<()> {
        if self.poisoned {
            return Ok(());
        }
        let mut remaining = buf;
        while !remaining.is_empty() {
            let need = self.part_size - self.part_buf.len();
            let take = remaining.len().min(need);
            self.part_buf.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            if self.part_buf.len() == self.part_size {
                let payload = std::mem::take(&mut self.part_buf).freeze();
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
        // Flush tail.
        if !self.poisoned && !self.part_buf.is_empty() {
            let payload = std::mem::take(&mut self.part_buf).freeze();
            self.dispatch_part(payload).await?;
        }
        if self.poisoned {
            // Best-effort cleanup of any temp files we did write.
            for (tmp, _) in self.pending_renames.drain(..) {
                let _ = std::fs::remove_file(&tmp);
            }
            return Err(wrap_io_err(std::io::Error::other("tee poisoned")));
        }

        // Build head bytes + write head temp file.
        let head_struct: LocalCacheHead = (meta, attrs).into();
        let head_bytes: Bytes = serde_json::to_vec(&head_struct)
            .map_err(wrap_io_err)?
            .into();
        let head_final = make_head_path(self.root_folder.clone(), &self.location);
        let head_tmp = head_final.with_extension(format!("_tee{}-head", self.tee_id));
        let (sender, recv) = oneshot::channel();
        self.write_worker
            .tx
            .send(WorkerOp::AtomicWrite {
                tmp_path: head_tmp.clone(),
                final_path: head_tmp.clone(),
                bytes: head_bytes,
                sender,
            })
            .map_err(|_| wrap_io_err(std::io::Error::other("uring worker channel closed")))?;
        let head_write = recv
            .await
            .map_err(|e| wrap_io_err(std::io::Error::other(format!("oneshot recv: {e}"))))?;
        if let Err(e) = head_write {
            for (tmp, _) in self.pending_renames.drain(..) {
                let _ = std::fs::remove_file(&tmp);
            }
            let _ = std::fs::remove_file(&head_tmp);
            return Err(e);
        }

        // All temp files are on disk; rename parts first, head last. Done
        // synchronously because rename is metadata-only and we want the
        // ordering invariant: once head is visible, every part is visible.
        let pending_renames = std::mem::take(&mut self.pending_renames);
        let head_pair = (head_tmp, head_final);
        let renames = pending_renames.clone();
        let renames_count = renames.len();
        let head_pair_for_blocking = head_pair.clone();
        let result = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            for (tmp, final_) in renames {
                std::fs::rename(&tmp, &final_)?;
            }
            std::fs::rename(&head_pair_for_blocking.0, &head_pair_for_blocking.1)?;
            Ok(())
        })
        .await
        .map_err(wrap_io_err)?
        .map_err(wrap_io_err);

        if let Err(e) = &result {
            warn!(
                "io_uring tee commit rename failed: {:?} (renames_attempted={})",
                e, renames_count
            );
        }

        // Seed the worker's fd_cache with already-opened fds for every
        // newly-renamed file. Done on `spawn_blocking` so the open()
        // syscalls stay off the io_uring worker thread; the worker
        // just receives `Arc<File>` payloads and stuffs them into its
        // local map. The next foreground read for any of these paths
        // skips the synchronous open and goes straight to pread.
        //
        // We seed unconditionally (overwriting any prior entry) because
        // the rename produced a new inode; an old cached fd would now
        // point at an unlinked inode. The fd_cache's existing
        // `is_handle_valid` check would have caught that on first read,
        // but pre-seeding avoids the open in the success path too. The
        // fd cache lives on the read worker, so seed it there.
        let worker_tx = self.read_worker.tx.clone();
        let head_path = head_pair.1.clone();
        let part_paths: Vec<std::path::PathBuf> = pending_renames
            .iter()
            .map(|(_, final_)| final_.clone())
            .collect();
        // Open in a single blocking task so we make at most one trip
        // to the spawn_blocking pool per commit; the writes are
        // expected to amortize over the size of the SST.
        tokio::task::spawn_blocking(move || {
            let mut open = OpenOptions::new();
            open.read(true)
                .custom_flags(libc::O_DIRECT | libc::O_NOATIME);
            for path in part_paths.into_iter().chain(std::iter::once(head_path)) {
                match open.open(&path) {
                    Ok(f) => {
                        let _ = worker_tx.send(WorkerOp::SeedFdCache {
                            path,
                            file: Arc::new(f),
                        });
                    }
                    Err(e) => {
                        // Best-effort: a failed seed just means the
                        // first reader will pay the open cost. Log so
                        // it's visible if it ever becomes systematic.
                        warn!(
                            "io_uring tee commit: failed to pre-open {} for fd seed: {:?}",
                            path.display(),
                            e
                        );
                    }
                }
            }
        });

        result
    }
}
