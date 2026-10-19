use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

use self::stats::{WalBufferStats, WalStage};
use crate::dispatcher::{MessageHandler, MessageHandlerExecutor, MessageTickerDef};
use crate::error::SlateDBError;
use crate::types::RowEntry;
use crate::utils::SafeSender;
use crate::utils::{format_bytes_si, WatchableOnceCellReader};
use crate::wal;
use crate::wal::{FlushResultFuture, WalError, WalEvent, WalStatus, WalWriter};
use async_trait::async_trait;
use futures::{stream::BoxStream, FutureExt, StreamExt};
use log::{error, trace};
use slatedb_common::metrics::MetricsRecorderHelper;
use std::panic::AssertUnwindSafe;
use tokio::task::JoinHandle;
use tokio::{runtime::Handle, sync::oneshot};
use tracing::instrument;

use super::store::WalTableStore;

pub(crate) const WAL_BUFFER_TASK_NAME: &str = "wal_writer";

/// Reads the monotonic clock for the pipeline batch wait and the diagnostic
/// stage timers. Neither decides which data is durable, so they do not go
/// through the `SystemClock` that deterministic simulation tests control.
#[allow(clippy::disallowed_methods)]
fn monotonic_now() -> Instant {
    Instant::now()
}

/// [`SlateDbWalWriter`] buffers write operations in memory before flushing them to persistent storage.
/// The flush operation only targets Remote storage right now, later we can add an option to flush to local
/// storage.
///
/// It maintains a `current_wal` buffer for active writes and a queue of immutable WALs pending flush.
///
/// By default, it offers a best-effort durability guarantee based on:
///
/// - `max_wal_size`: Flushes when `max_wal_size` bytes is exceeded
/// - `max_flush_interval`: Flushes after `max_flush_interval` elapses, if set
/// - `max_wal_flushes_before_l0_flush`: Requests a memtable flush when this many WAL files have
///   been flushed since the latest memtable replay point
///
/// For strict durability requirements on synchronous writes, use [`SlateDbWalWriter::flush()`] to explicitly
/// trigger a flush operation and await the result. This will flush ALL the in memory WALs (including the
/// current WAL) to remote storage.
///
/// The manager is thread-safe and can be safely shared across multiple threads.
///
/// Please note:
///
/// - the size limit (`max_wal_size`) is a soft threshold. WAL entries within a single write batch are
///   guaranteed to be written atomically to the same WAL file.
/// - Fatal errors during flush operations are stored internally and propagated to all subsequent
///   operations. The manager becomes unusable after encountering a fatal error.
pub(crate) struct SlateDbWalWriter {
    inner: Arc<parking_lot::RwLock<SlateDbWalWriterInner>>,
    stats: Arc<WalBufferStats>,
    table_store: Arc<WalTableStore>,
    max_wal_bytes_size: usize,
    max_wal_flushes_before_l0_flush: u64,
    /// The largest flush_epoch for which a size-triggered flush request has been
    /// sent. Compared against `flush_epoch` in the inner struct to avoid sending
    /// redundant flush requests for the same WAL.
    last_flush_requested_epoch: AtomicU64,
    /// task executor for the background worker.
    task_executor: Arc<MessageHandlerExecutor>,
}

struct SlateDbWalWriterInner {
    current_wal: WalBuffer,
    /// When the current WAL is ready to be flushed, it'll be moved to the `immutable_wals`.
    /// The flusher will try flush all the immutable wals to remote storage.
    immutable_wals: VecDeque<(u64, Arc<WalBuffer>)>,
    /// The next wal id that will be generated
    next_wal_id: u64,
    /// Monotonically increasing epoch incremented each time the current WAL is
    /// frozen. Used with `last_flush_requested_epoch` to deduplicate size-triggered
    /// flush requests.
    flush_epoch: u64,
    /// The flusher will update the last_flushed_wal_id and last_flushed_seq when the flush is done.
    last_flushed_wal_id: u64,
    /// The last seq that was flushed to the WAL. This value will be None until the first flush.
    last_flushed_seq: Option<u64>,
    /// Set to Some with error reason if the flush task has exited
    flush_task_exited_reason: Option<WalError>,
    /// The channel to send the flush work to the background worker.
    flush_tx: SafeSender<WalFlushWork>,
    /// The status as of its last computation, for observers that find the
    /// lock taken.
    published: Arc<PublishedWalStatus>,
}

/// A [`WalStatus`] as of its last computation, readable without the WAL lock.
///
/// The writer holds the WAL lock exclusively while it appends a write group,
/// and every write checks the WAL status for backpressure. Reading the status
/// through the lock would stall each write that arrives during an append.
#[derive(Default)]
struct PublishedWalStatus {
    estimated_bytes: AtomicUsize,
    last_flushed_wal_id: AtomicU64,
    /// The last flushed sequence number plus one, or zero before the first
    /// flush.
    last_flushed_seq_plus_one: AtomicU64,
    buffered_wal_entries_count: AtomicUsize,
    closed: AtomicBool,
}

impl PublishedWalStatus {
    fn store(&self, status: &WalStatus) {
        self.estimated_bytes
            .store(status.estimated_bytes, Ordering::Relaxed);
        self.last_flushed_wal_id
            .store(status.last_flushed_wal_id, Ordering::Relaxed);
        let seq_plus_one = status.last_flushed_seq.map_or(0, |seq| {
            seq.checked_add(1)
                .expect("sequence numbers stay below u64::MAX")
        });
        self.last_flushed_seq_plus_one
            .store(seq_plus_one, Ordering::Relaxed);
        self.buffered_wal_entries_count
            .store(status.buffered_wal_entries_count, Ordering::Relaxed);
        self.closed
            .store(status.closed_reason.is_some(), Ordering::Relaxed);
    }

    /// Returns the published status, or `None` once the WAL has closed: the
    /// reason it closed is only available under the lock.
    fn load(&self) -> Option<WalStatus> {
        if self.closed.load(Ordering::Relaxed) {
            return None;
        }
        Some(WalStatus {
            closed_reason: None,
            estimated_bytes: self.estimated_bytes.load(Ordering::Relaxed),
            last_flushed_wal_id: self.last_flushed_wal_id.load(Ordering::Relaxed),
            last_flushed_seq: self
                .last_flushed_seq_plus_one
                .load(Ordering::Relaxed)
                .checked_sub(1),
            buffered_wal_entries_count: self.buffered_wal_entries_count.load(Ordering::Relaxed),
        })
    }
}

/// Stores entries to the write-ahead log (WAL) in memory.
///
/// In contrast to the [`KVTable`], the `WalBuffer` does not sort the entries according to the key,
/// but keeps the order in which the entries were added.
/// The assumption is that the entries are added in order of the sequence number to the WAL.
/// Ordering by sequence number is sufficient for replaying entries from the WAL in case of a
/// failure.
/// Since the `WalBuffer` does not maintain the order by key it saves some CPU cycles compared to
/// a [`KVTable`].
struct WalBuffer {
    /// queue for the entries
    entries: VecDeque<RowEntry>,
    /// the sequence number of the most recent addition to this WAL buffer
    last_seq: u64,
    /// size of the entries that has been added to the WAL buffer in bytes
    entries_size: usize,
    /// When the buffer was created. The diagnostic timers measure from here.
    created: Instant,
    /// The number of appends to this buffer.
    appends: u64,
    /// The sum of the append times, measured from `created`.
    append_offsets: Duration,
    /// When the writer froze this buffer for upload.
    frozen_at: Option<Instant>,
}

/// An iterator over entries in a WalBuffer.
struct WalBufferIterator {
    /// The entries to iterate over (owned).
    entries: std::vec::IntoIter<RowEntry>,
}

impl SlateDbWalWriter {
    pub(crate) async fn start_new(
        closed_result_reader: WatchableOnceCellReader<Result<(), SlateDBError>>,
        recorder: &MetricsRecorderHelper,
        last_flushed_wal_id: u64,
        table_store: Arc<WalTableStore>,
        max_wal_bytes_size: usize,
        max_wal_flushes_before_l0_flush: u64,
        max_flush_interval: Option<Duration>,
        group_commit: bool,
        max_inflight_flushes: usize,
        task_executor: Arc<MessageHandlerExecutor>,
    ) -> Result<Self, SlateDBError> {
        let current_wal = WalBuffer::new();
        let immutable_wals = VecDeque::new();
        let (flush_tx, flush_rx) = SafeSender::unbounded_channel(closed_result_reader);
        let inner = SlateDbWalWriterInner {
            current_wal,
            immutable_wals,
            flush_epoch: 1,
            last_flushed_wal_id,
            next_wal_id: last_flushed_wal_id + 1,
            last_flushed_seq: None,
            flush_task_exited_reason: None,
            flush_tx,
            published: Arc::default(),
        };
        // Publishes the initial status.
        inner.compute_status(&table_store);
        let inner = Arc::new(parking_lot::RwLock::new(inner));
        let stats = Arc::new(WalBufferStats::new(recorder));
        let wal_flush_handler = WalFlushHandler {
            max_flush_interval,
            group_commit,
            max_inflight_flushes: max_inflight_flushes.max(1),
            inner: inner.clone(),
            table_store: table_store.clone(),
            stats: stats.clone(),
            listener: None,
            inflight: VecDeque::new(),
            waiters: Vec::new(),
            deferred_flush: false,
            current_wal_since: monotonic_now(),
            batch_tick_scheduled: false,
        };
        task_executor.add_handler(
            WAL_BUFFER_TASK_NAME.to_string(),
            Box::new(wal_flush_handler),
            flush_rx,
            &Handle::current(),
        )?;
        Ok(Self {
            inner,
            stats,
            table_store,
            max_wal_bytes_size,
            max_wal_flushes_before_l0_flush,
            last_flush_requested_epoch: AtomicU64::new(0),
            task_executor,
        })
    }

    /// Check if we need to flush the wal with considering max_wal_size. the checking over `max_wal_size`
    /// is not very strict, we have to ensure a write batch into a single WAL file.
    ///
    /// It's the caller's duty to call `maybe_trigger_flush` after calling `append`.
    fn maybe_trigger_flush(&self) -> Result<(), WalError> {
        let (need_flush, flush_epoch) = {
            let inner = self.inner.read();
            // checks the size of the current wal
            let (need_flush, flush_epoch) =
                inner.needs_flush(&self.table_store, self.max_wal_bytes_size);
            (need_flush, flush_epoch)
        };
        if need_flush {
            // Only send a flush request if one hasn't already been sent for this epoch.
            // compare_exchange ensures only one writer wins per epoch.
            let last = self.last_flush_requested_epoch.load(Ordering::Relaxed);
            if last < flush_epoch
                && self
                    .last_flush_requested_epoch
                    .compare_exchange(last, flush_epoch, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                self.send_flush_request(None)?;
            }
        }

        let status = self.status()?;
        self.stats
            .estimated_bytes
            .set(status.estimated_bytes as i64);
        Ok(())
    }

    /// Send a flush request to the background flush worker.
    fn send_flush_request(
        &self,
        result_tx: Option<oneshot::Sender<Result<(), WalError>>>,
    ) -> Result<(), WalError> {
        self.stats.flush_requests.increment(1);
        self.inner
            .read()
            .send_flush_msg(WalFlushWork::Flush { result_tx })
    }
}

#[async_trait]
impl WalWriter for SlateDbWalWriter {
    fn status(&self) -> Result<WalStatus, WalStatus> {
        self.inner.read().status(&self.table_store)
    }

    /// Append row entries to the current WAL. Returns a watcher for durability notification.
    async fn append(&mut self, entries: &[RowEntry]) -> Result<(), WalError> {
        self.inner.write().append(entries)?;
        self.maybe_trigger_flush()?;
        Ok(())
    }

    fn should_flush_memtable(&self, replay_after_wal_id: u64) -> bool {
        let last_flushed_wal_id = self.inner.read().last_flushed_wal_id;
        let Some(wal_id_gap) = last_flushed_wal_id.checked_sub(replay_after_wal_id) else {
            return false;
        };
        wal_id_gap >= self.max_wal_flushes_before_l0_flush
    }

    fn observer(&self) -> Box<dyn wal::WalObserver> {
        Box::new(SlateDbWalObserver {
            published: self.inner.read().published.clone(),
            inner: self.inner.clone(),
            table_store: self.table_store.clone(),
        })
    }

    async fn flush(&mut self) -> Result<FlushResultFuture, WalError> {
        let (result_tx, result_rx) = oneshot::channel();
        self.send_flush_request(Some(result_tx))?;
        Ok(async {
            result_rx
                .await
                .unwrap_or_else(|e| Err(WalError::InternalError(Arc::new(e))))
        }
        .boxed())
    }

    async fn close(&mut self) -> Result<(), WalError> {
        if let Some(result) = self
            .task_executor
            .shutdown_or_deregister_task(WAL_BUFFER_TASK_NAME)
            .await
        {
            return Ok(result?);
        };
        self.inner
            .write()
            .mark_closed(WalError::Closed, &self.table_store);
        Ok(())
    }
}

impl SlateDbWalWriterInner {
    fn check_exited(&self) -> Result<(), WalError> {
        match self.flush_task_exited_reason.as_ref() {
            Some(err) => Err(err.clone()),
            None => Ok(()),
        }
    }

    fn send_flush_msg(&self, msg: WalFlushWork) -> Result<(), WalError> {
        self.check_exited()?;
        // TODO: there is a small window here where the dispatcher closes `flush_rx` before
        //       calling cleanup. In this case we may have exited with a different error than
        //       a clean close. To fix this we'd need some pre-cleanup hook the dispatcher can
        //       call to propagate the error
        self.flush_tx.send(msg).map_err(|_e| WalError::Closed)
    }

    fn append(&mut self, entries: &[RowEntry]) -> Result<(), WalError> {
        // TODO: validate the seq number is always increasing.
        self.check_exited()?;
        for entry in entries {
            self.current_wal.append(entry.clone());
        }
        self.current_wal.note_append();
        Ok(())
    }

    fn needs_flush(&self, table_store: &WalTableStore, max_wal_bytes_size: usize) -> (bool, u64) {
        // check the size of the current wal
        let current_wal_size =
            table_store.estimate_encoded_size(self.current_wal.len(), self.current_wal.size());
        trace!(
            "checking flush trigger [current_wal_size={}, max_wal_bytes_size={}]",
            format_bytes_si(current_wal_size as u64),
            format_bytes_si(max_wal_bytes_size as u64),
        );
        let need_flush = current_wal_size >= max_wal_bytes_size;
        (need_flush, self.flush_epoch)
    }

    /// The last buffered seq that is not yet durable, if any. A `flush()`
    /// caller is done once `last_flushed_seq` reaches it.
    fn unflushed_target_seq(&self) -> Option<u64> {
        let last = self.current_wal.last_seq().or_else(|| {
            self.immutable_wals
                .back()
                .and_then(|(_, wal)| wal.last_seq())
        })?;
        if self.last_flushed_seq.is_some_and(|flushed| flushed >= last) {
            None
        } else {
            Some(last)
        }
    }

    /// The immutable WAL at `index`, in flush order.
    fn immutable_wal(&self, index: usize) -> Option<(u64, Arc<WalBuffer>)> {
        let (wal_id, wal) = self.immutable_wals.get(index)?;
        assert!(*wal_id > self.last_flushed_wal_id);
        Some((*wal_id, wal.clone()))
    }

    /// Returns the total size of all unflushed WALs in bytes.
    fn estimated_bytes(&self, table_store: &WalTableStore) -> usize {
        let current_wal_size =
            table_store.estimate_encoded_size(self.current_wal.len(), self.current_wal.size());
        let imm_wal_size = self
            .immutable_wals
            .iter()
            .map(|(_, wal)| table_store.estimate_encoded_size(wal.len(), wal.size()))
            .sum::<usize>();
        current_wal_size + imm_wal_size
    }

    fn status(&self, table_store: &WalTableStore) -> Result<WalStatus, WalStatus> {
        let status = self.compute_status(table_store);
        if status.closed_reason.is_none() {
            Ok(status)
        } else {
            Err(status)
        }
    }

    fn compute_status(&self, table_store: &WalTableStore) -> WalStatus {
        let flushing_wal_entries_count = self
            .immutable_wals
            .iter()
            .map(|(_, wal)| wal.len())
            .sum::<usize>();
        let buffered_wal_entries_count = self.current_wal.len() + flushing_wal_entries_count;
        let status = WalStatus {
            closed_reason: self.flush_task_exited_reason.clone(),
            estimated_bytes: self.estimated_bytes(table_store),
            last_flushed_wal_id: self.last_flushed_wal_id,
            last_flushed_seq: self.last_flushed_seq,
            buffered_wal_entries_count,
        };
        self.published.store(&status);
        status
    }

    fn mark_closed(&mut self, reason: WalError, table_store: &WalTableStore) -> WalStatus {
        self.flush_task_exited_reason = Some(reason);
        self.freeze_current_wal();
        self.immutable_wals.clear();
        self.compute_status(table_store)
    }

    fn freeze_current_wal(&mut self) {
        if self.current_wal.is_empty() {
            return;
        }
        let next_wal_id = self.next_wal_id;
        self.next_wal_id += 1;
        let mut current_wal = std::mem::replace(&mut self.current_wal, WalBuffer::new());
        current_wal.frozen_at = Some(monotonic_now());
        self.flush_epoch += 1;
        self.immutable_wals
            .push_back((next_wal_id, Arc::new(current_wal)));
    }

    fn record_flushed_wal(&mut self, flushed_wal_id: u64, flushed_wal: &Arc<WalBuffer>) {
        let (front_wal_id, front_wal_buffer) = self
            .immutable_wals
            .pop_front()
            .expect("no immutable wals found to pop");
        assert_eq!(front_wal_id, flushed_wal_id);
        assert!(Arc::ptr_eq(&front_wal_buffer, flushed_wal));
        assert_eq!(
            flushed_wal_id,
            self.last_flushed_wal_id + 1,
            "flushed wal id {} not next wal id after previous flushed {}",
            flushed_wal_id,
            self.last_flushed_wal_id
        );
        self.last_flushed_wal_id = flushed_wal_id;
        if let Some(seq) = flushed_wal.last_seq() {
            if let Some(last_flushed_seq) = self.last_flushed_seq {
                assert!(seq > last_flushed_seq);
            }
            self.last_flushed_seq = Some(seq);
        }
    }
}

impl WalBuffer {
    /// Creates a new empty `WalBuffer`.
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            last_seq: 0,
            entries_size: 0,
            created: monotonic_now(),
            appends: 0,
            append_offsets: Duration::ZERO,
            frozen_at: None,
        }
    }

    /// Records the time of an append for the diagnostic timers.
    fn note_append(&mut self) {
        self.appends += 1;
        self.append_offsets += self.created.elapsed();
    }

    /// The average time from an append to the freeze of this buffer.
    fn freeze_wait(&self) -> Option<Duration> {
        let frozen_at = self.frozen_at?;
        let appends = u32::try_from(self.appends).ok().filter(|n| *n > 0)?;
        Some(frozen_at.saturating_duration_since(self.created + self.append_offsets / appends))
    }

    fn append(&mut self, entry: RowEntry) {
        self.last_seq = entry.seq;
        self.entries_size += entry.estimated_size();
        self.entries.push_back(entry);
    }

    /// Returns an iterator to iterate over the entries.
    fn iter(&self) -> WalBufferIterator {
        WalBufferIterator::new(self)
    }

    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of entries in the buffer.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the size of entries in bytes.
    fn size(&self) -> usize {
        self.entries_size
    }

    /// Returns the last sequence number written to this buffer.
    fn last_seq(&self) -> Option<u64> {
        if self.last_seq == 0 {
            None
        } else {
            Some(self.last_seq)
        }
    }
}

impl WalBufferIterator {
    /// Creates a new iterator from a WalBuffer.
    /// This clones all entries from the buffer.
    pub(crate) fn new(wal_buffer: &WalBuffer) -> Self {
        let entries = wal_buffer.entries.iter().cloned().collect::<Vec<_>>();
        Self {
            entries: entries.into_iter(),
        }
    }

    pub(crate) fn next(&mut self) -> Option<RowEntry> {
        self.entries.next()
    }
}

enum WalFlushWork {
    Flush {
        result_tx: Option<oneshot::Sender<Result<(), WalError>>>,
    },
    /// An upload started by the handler finished.
    Flushed {
        wal_id: u64,
        result: Result<(), SlateDBError>,
        /// When the upload finished.
        finished: Instant,
    },
    /// The flush timer fired, or a deferred `flush()` request's batch wait
    /// ended: freeze the current WAL if a slot is free.
    Tick,
    /// An upload task panicked. The handler re-raises the panic so the
    /// dispatcher reports it like a panic in the handler itself.
    Panicked {
        payload: Box<dyn std::any::Any + Send>,
    },
    Subscribe {
        listener: wal::WalStatusListener,
    },
}

impl Debug for WalFlushWork {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WalFlushWork::Flush { .. } => f.write_str("Flush"),
            WalFlushWork::Flushed { wal_id, .. } => write!(f, "Flushed({wal_id})"),
            WalFlushWork::Tick => f.write_str("Tick"),
            WalFlushWork::Panicked { .. } => f.write_str("Panicked"),
            WalFlushWork::Subscribe { .. } => f.write_str("Subscribe"),
        }
    }
}

/// A WAL upload in progress, in WAL order.
struct InflightFlush {
    wal_id: u64,
    wal: Arc<WalBuffer>,
    task: JoinHandle<Result<(), SlateDBError>>,
    done: bool,
    /// When the upload finished.
    finished: Option<Instant>,
}

struct WalFlushHandler {
    max_flush_interval: Option<Duration>,
    /// Flush again as soon as a flush finishes if writes are buffered.
    group_commit: bool,
    /// How many WAL uploads may run at once.
    max_inflight_flushes: usize,
    inner: Arc<parking_lot::RwLock<SlateDbWalWriterInner>>,
    table_store: Arc<WalTableStore>,
    stats: Arc<WalBufferStats>,
    listener: Option<wal::WalStatusListener>,
    /// Uploads in WAL order. Durability is recorded from the front, so a
    /// later WAL that finishes first waits for the earlier ones.
    inflight: VecDeque<InflightFlush>,
    /// `flush()` callers with the seq each one waits for.
    waiters: Vec<(u64, oneshot::Sender<Result<(), WalError>>)>,
    /// A flush request arrived while every slot was busy; freeze the current
    /// WAL when a slot frees up even without group commit.
    deferred_flush: bool,
    /// When the current WAL started collecting entries.
    current_wal_since: Instant,
    /// A delayed tick is already scheduled for a deferred `flush()` request.
    batch_tick_scheduled: bool,
}

/// With several uploads in flight, a `flush()` request freezes the current
/// WAL only once it is this old; a tick does it otherwise. Serialized uploads
/// batch naturally because a request waits for the upload in flight, but
/// with free slots every request would start its own small upload.
const PIPELINE_BATCH_WAIT: Duration = Duration::from_millis(1);

impl WalFlushHandler {
    /// Starts uploads until `max_inflight_flushes` are running. Freezes the
    /// current WAL for upload when `freeze_current` is set and a slot is free,
    /// so writes keep batching while every slot is busy.
    fn start_flushes(&mut self, freeze_current: bool) {
        loop {
            if self.inflight.len() >= self.max_inflight_flushes {
                if freeze_current && !self.inner.read().current_wal.is_empty() {
                    self.deferred_flush = true;
                }
                return;
            }
            let next = {
                let mut inner = self.inner.write();
                let in_flight = self.inflight.len();
                if inner.immutable_wals.len() <= in_flight {
                    if !freeze_current || inner.current_wal.is_empty() {
                        return;
                    }
                    inner.freeze_current_wal();
                    self.current_wal_since = monotonic_now();
                }
                inner.immutable_wal(in_flight)
            };
            let Some((wal_id, wal)) = next else {
                return;
            };
            self.spawn_flush(wal_id, wal);
        }
    }

    /// How much longer a `flush()` request should let the current WAL batch
    /// before freezing it. Zero when uploads are serialized, when the WAL is
    /// old enough, or when nothing is buffered.
    fn batch_wait_remaining(&self) -> Duration {
        if self.max_inflight_flushes <= 1 || self.inner.read().current_wal.is_empty() {
            return Duration::ZERO;
        }
        PIPELINE_BATCH_WAIT.saturating_sub(self.current_wal_since.elapsed())
    }

    // The batch wait is a short monotonic timer, like `monotonic_now`.
    #[allow(clippy::disallowed_methods)]
    fn schedule_batch_tick(&mut self, wait: Duration) {
        if self.batch_tick_scheduled {
            return;
        }
        self.batch_tick_scheduled = true;
        let inner = self.inner.clone();
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            let _ = inner.read().send_flush_msg(WalFlushWork::Tick);
        });
    }

    fn spawn_flush(&mut self, wal_id: u64, wal: Arc<WalBuffer>) {
        let table_store = self.table_store.clone();
        let stats = self.stats.clone();
        let inner = self.inner.clone();
        let upload = wal.clone();
        let task = tokio::spawn(async move {
            let started = monotonic_now();
            let upload_result =
                AssertUnwindSafe(flush_one_wal(&table_store, &stats, wal_id, &upload))
                    .catch_unwind()
                    .await;
            let finished = monotonic_now();
            let appends = upload.appends;
            if let Some(freeze_wait) = upload.freeze_wait() {
                stats.record_stage(WalStage::Freeze, freeze_wait, appends);
            }
            if let Some(frozen_at) = upload.frozen_at {
                stats.record_stage(
                    WalStage::Slot,
                    started.saturating_duration_since(frozen_at),
                    appends,
                );
            }
            stats.record_stage(WalStage::Upload, finished - started, appends);
            let upload = upload_result;
            let result = match upload {
                Ok(result) => result,
                Err(payload) => {
                    let _ = inner
                        .read()
                        .send_flush_msg(WalFlushWork::Panicked { payload });
                    return Err(SlateDBError::BackgroundTaskPanic(format!(
                        "WAL flush task panicked [wal_id={wal_id}]"
                    )));
                }
            };
            if result.is_err() {
                error!("failed to flush WAL [wal_id={}]", wal_id);
            }
            // The handler records durability in WAL order on this message.
            // A closed channel means the writer is shutting down, and
            // `cleanup` then collects the result from the task itself.
            let _ = inner.read().send_flush_msg(WalFlushWork::Flushed {
                wal_id,
                result: result.clone(),
                finished,
            });
            result
        });
        self.inflight.push_back(InflightFlush {
            wal_id,
            wal,
            task,
            done: false,
            finished: None,
        });
    }

    /// Records every finished upload at the front of the queue, in WAL order.
    fn record_flushed(&mut self) {
        while self.inflight.front().is_some_and(|flush| flush.done) {
            let flushed = self.inflight.pop_front().expect("checked above");
            if let Some(finished) = flushed.finished {
                self.stats
                    .record_stage(WalStage::Order, finished.elapsed(), flushed.wal.appends);
            }
            let status = {
                let mut inner = self.inner.write();
                inner.record_flushed_wal(flushed.wal_id, &flushed.wal);
                inner.compute_status(&self.table_store)
            };
            trace!(
                "recorded durable WAL [wal_id={}, last_flushed_seq={:?}]",
                flushed.wal_id,
                status.last_flushed_seq
            );
            drop(flushed);
            self.notify_listener(WalEvent::WalFlushed(status));
        }
    }

    fn resolve_waiters(&mut self) {
        let Some(last_flushed_seq) = self.inner.read().last_flushed_seq else {
            return;
        };
        trace!(
            "resolving waiters [last_flushed_seq={}, waiters={:?}]",
            last_flushed_seq,
            self.waiters.iter().map(|(seq, _)| *seq).collect::<Vec<_>>()
        );
        let (done, pending): (Vec<_>, Vec<_>) = self
            .waiters
            .drain(..)
            .partition(|(seq, _)| *seq <= last_flushed_seq);
        self.waiters = pending;
        for (_, result_tx) in done {
            let _ = result_tx.send(Ok(()));
        }
    }

    fn notify_listener(&self, event: WalEvent) {
        if let Some(l) = self.listener.as_ref() {
            (*l)(event);
        }
    }
}

#[instrument(level = "trace", skip_all, err(level = tracing::Level::DEBUG))]
async fn flush_one_wal(
    table_store: &WalTableStore,
    stats: &WalBufferStats,
    wal_id: u64,
    wal: &WalBuffer,
) -> Result<(), SlateDBError> {
    stats.flushes.increment(1);

    let mut sst_builder = table_store.table_builder();
    let mut iter = wal.iter();
    while let Some(entry) = iter.next() {
        sst_builder.add(entry).await?;
    }

    let encoded_sst = sst_builder.build().await?;
    let written_bytes = encoded_sst.remaining_len() as u64;
    table_store.write_sst(wal_id.into(), &encoded_sst).await?;
    stats.flush_bytes.increment(written_bytes);
    Ok(())
}

#[async_trait]
impl MessageHandler<WalFlushWork> for WalFlushHandler {
    fn tickers(&mut self) -> Vec<MessageTickerDef<WalFlushWork>> {
        if let Some(max_flush_interval) = self.max_flush_interval {
            return vec![MessageTickerDef::new(
                max_flush_interval,
                Box::new(|| WalFlushWork::Tick),
            )];
        }
        vec![]
    }

    async fn handle(&mut self, message: WalFlushWork) -> Result<(), SlateDBError> {
        match message {
            WalFlushWork::Flush { result_tx } => {
                trace!(
                    "flush request [waiter={}, inflight={}, deferred={}]",
                    result_tx.is_some(),
                    self.inflight.len(),
                    self.deferred_flush
                );
                let explicit = result_tx.is_some();
                if let Some(result_tx) = result_tx {
                    let target = self.inner.read().unflushed_target_seq();
                    trace!("flush request target [seq={:?}]", target);
                    match target {
                        Some(seq) => self.waiters.push((seq, result_tx)),
                        None => {
                            let _ = result_tx.send(Ok(()));
                        }
                    }
                }
                let batch_wait = self.batch_wait_remaining();
                if explicit && batch_wait > Duration::ZERO {
                    self.schedule_batch_tick(batch_wait);
                } else {
                    self.start_flushes(true);
                }
                Ok(())
            }
            WalFlushWork::Tick => {
                self.batch_tick_scheduled = false;
                self.start_flushes(true);
                Ok(())
            }
            WalFlushWork::Flushed {
                wal_id,
                result,
                finished,
            } => {
                trace!(
                    "upload finished [wal_id={}, ok={}, inflight={:?}]",
                    wal_id,
                    result.is_ok(),
                    self.inflight
                        .iter()
                        .map(|f| (f.wal_id, f.done))
                        .collect::<Vec<_>>()
                );
                result?;
                let flush = self
                    .inflight
                    .iter_mut()
                    .find(|flush| flush.wal_id == wal_id)
                    .expect("completion for a WAL that is not in flight");
                flush.done = true;
                flush.finished = Some(finished);
                self.record_flushed();
                self.resolve_waiters();
                // With group commit, keep flushing while writes are buffered
                // rather than waiting for the next tick or request. A request
                // that found every slot busy is honoured now either way.
                let freeze = self.group_commit || self.deferred_flush;
                self.deferred_flush = false;
                self.start_flushes(freeze);
                Ok(())
            }
            WalFlushWork::Panicked { payload } => std::panic::resume_unwind(payload),
            WalFlushWork::Subscribe { listener } => {
                // TODO: support multiple listeners. For now, the db listener is the only one
                assert!(self.listener.is_none());
                self.listener = Some(listener);
                Ok(())
            }
        }
    }

    async fn cleanup(
        &mut self,
        mut messages: BoxStream<'async_trait, WalFlushWork>,
        result: Result<(), SlateDBError>,
    ) -> Result<(), SlateDBError> {
        // Let the uploads already running finish and record them in order, so
        // a clean close loses nothing that was handed to the object store.
        let mut result = result;
        while let Some(mut flush) = self.inflight.pop_front() {
            let upload = match (&mut flush.task).await {
                Ok(upload) => upload,
                Err(e) => Err(SlateDBError::BackgroundTaskPanic(format!(
                    "WAL flush task failed: {e}"
                ))),
            };
            match (&result, upload) {
                (Ok(()), Ok(())) => {
                    self.inflight.push_front(flush);
                    self.inflight.front_mut().expect("just pushed").done = true;
                    self.record_flushed();
                }
                (Ok(()), Err(e)) => result = Err(e),
                (Err(_), _) => {}
            }
        }
        self.resolve_waiters();

        let error = result
            .clone()
            .err()
            .map(WalError::from)
            .unwrap_or(WalError::Closed);

        let final_status = self
            .inner
            .write()
            .mark_closed(error.clone(), &self.table_store);
        self.notify_listener(WalEvent::WalClosed(final_status.clone()));

        for (_, result_tx) in self.waiters.drain(..) {
            let _ = result_tx.send(Err(error.clone()));
        }

        // drain remaining messages
        while let Some(msg) = messages.next().await {
            match msg {
                WalFlushWork::Flush { result_tx } => {
                    if let Some(result_tx) = result_tx {
                        let _ = result_tx.send(Err(error.clone()));
                    }
                }
                WalFlushWork::Flushed { .. }
                | WalFlushWork::Tick
                | WalFlushWork::Panicked { .. } => {}
                WalFlushWork::Subscribe { listener } => {
                    (*listener)(WalEvent::WalClosed(final_status.clone()))
                }
            }
        }
        Ok(())
    }
}

/// Interface for getting information about the current state of the Wal
#[derive(Clone)]
struct SlateDbWalObserver {
    inner: Arc<parking_lot::RwLock<SlateDbWalWriterInner>>,
    table_store: Arc<WalTableStore>,
    published: Arc<PublishedWalStatus>,
}

impl wal::WalObserver for SlateDbWalObserver {
    /// Gets information about the Wal buffer's current state, as of its last
    /// computation if the writer holds the lock.
    fn status(&self) -> Result<WalStatus, WalStatus> {
        if let Some(inner) = self.inner.try_read() {
            return inner.status(self.table_store.as_ref());
        }
        match self.published.load() {
            Some(status) => Ok(status),
            None => self.inner.read().status(self.table_store.as_ref()),
        }
    }

    fn subscribe(&self, listener: wal::WalStatusListener) -> Result<(), WalError> {
        self.inner
            .read()
            .send_flush_msg(WalFlushWork::Subscribe { listener })
    }
}

pub mod stats {
    use slatedb_common::metrics::{CounterFn, GaugeFn, HistogramFn, MetricsRecorderHelper};
    use std::sync::Arc;
    use std::time::Duration;

    macro_rules! wal_stat_name {
        ($suffix:expr) => {
            concat!("slatedb.wal.", $suffix)
        };
    }

    pub const WAL_BUFFER_FLUSHES: &str = wal_stat_name!("wal_buffer_flushes");
    pub const WAL_BUFFER_FLUSH_REQUESTS: &str = wal_stat_name!("wal_buffer_flush_requests");
    pub const WAL_BUFFER_ESTIMATED_BYTES: &str = wal_stat_name!("wal_buffer_estimated_bytes");
    pub const WAL_FLUSH_BYTES: &str = wal_stat_name!("wal_flush_bytes");
    pub const WAL_DIAG_FREEZE_WAIT: &str = wal_stat_name!("wal_diag_freeze_wait_seconds");
    pub const WAL_DIAG_SLOT_WAIT: &str = wal_stat_name!("wal_diag_slot_wait_seconds");
    pub const WAL_DIAG_UPLOAD: &str = wal_stat_name!("wal_diag_upload_seconds");
    pub const WAL_DIAG_ORDER_WAIT: &str = wal_stat_name!("wal_diag_order_wait_seconds");
    pub const WAL_DIAG_FREEZE_WAIT_US: &str = wal_stat_name!("wal_diag_freeze_wait_us");
    pub const WAL_DIAG_SLOT_WAIT_US: &str = wal_stat_name!("wal_diag_slot_wait_us");
    pub const WAL_DIAG_UPLOAD_US: &str = wal_stat_name!("wal_diag_upload_us");
    pub const WAL_DIAG_ORDER_WAIT_US: &str = wal_stat_name!("wal_diag_order_wait_us");
    pub const WAL_DIAG_APPENDS: &str = wal_stat_name!("wal_diag_appends");

    use slatedb_common::metrics::FINE_LATENCY_BOUNDARIES as DIAG_BOUNDARIES;

    /// A stage that a write goes through between its append and its durability.
    #[derive(Clone, Copy)]
    pub(super) enum WalStage {
        /// From the append to the freeze of its WAL.
        Freeze,
        /// From the freeze to the start of the upload.
        Slot,
        /// The upload.
        Upload,
        /// From the end of the upload to the record of durability in WAL order.
        Order,
    }

    /// A diagnostic timer: one histogram observation per WAL, and a counter of
    /// microseconds weighted by the appends in the WAL.
    pub(super) struct StageTimer {
        per_wal: Arc<dyn HistogramFn>,
        weighted_us: Arc<dyn CounterFn>,
    }

    impl StageTimer {
        fn new(recorder: &MetricsRecorderHelper, histogram: &str, counter: &str) -> Self {
            Self {
                per_wal: recorder.histogram(histogram, DIAG_BOUNDARIES).register(),
                weighted_us: recorder.counter(counter).register(),
            }
        }
    }

    pub(super) struct WalBufferStats {
        pub(super) estimated_bytes: Arc<dyn GaugeFn>,
        pub(super) flushes: Arc<dyn CounterFn>,
        pub(super) flush_requests: Arc<dyn CounterFn>,
        pub(super) flush_bytes: Arc<dyn CounterFn>,
        freeze: StageTimer,
        slot: StageTimer,
        upload: StageTimer,
        order: StageTimer,
        appends: Arc<dyn CounterFn>,
    }

    impl WalBufferStats {
        pub(super) fn new(recorder: &MetricsRecorderHelper) -> Self {
            Self {
                estimated_bytes: recorder.gauge(WAL_BUFFER_ESTIMATED_BYTES).register(),
                flushes: recorder.counter(WAL_BUFFER_FLUSHES).register(),
                flush_requests: recorder.counter(WAL_BUFFER_FLUSH_REQUESTS).register(),
                flush_bytes: recorder.counter(WAL_FLUSH_BYTES).register(),
                freeze: StageTimer::new(recorder, WAL_DIAG_FREEZE_WAIT, WAL_DIAG_FREEZE_WAIT_US),
                slot: StageTimer::new(recorder, WAL_DIAG_SLOT_WAIT, WAL_DIAG_SLOT_WAIT_US),
                upload: StageTimer::new(recorder, WAL_DIAG_UPLOAD, WAL_DIAG_UPLOAD_US),
                order: StageTimer::new(recorder, WAL_DIAG_ORDER_WAIT, WAL_DIAG_ORDER_WAIT_US),
                appends: recorder.counter(WAL_DIAG_APPENDS).register(),
            }
        }

        /// Records how long the appends of one WAL spent in `stage`.
        pub(super) fn record_stage(&self, stage: WalStage, duration: Duration, appends: u64) {
            let timer = match stage {
                WalStage::Freeze => &self.freeze,
                WalStage::Slot => &self.slot,
                WalStage::Upload => &self.upload,
                WalStage::Order => &self.order,
            };
            timer.per_wal.record(duration.as_secs_f64());
            let micros = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
            timer.weighted_us.increment(micros.saturating_mul(appends));
            if matches!(stage, WalStage::Upload) {
                self.appends.increment(appends);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db_status::{ClosedResultWriter, DbStatusManager};
    use crate::format::sst::SsTableFormat;
    use crate::object_store_tag::TableStoreKind;
    use crate::oracle::DbOracle;
    use crate::test_utils::GatedObjectStore;
    use crate::types::{RowEntry, ValueDeletable};
    use crate::wal::slatedb::sst_iterator::{WalSstIterator, WalSstIteratorOptions};
    use bytes::Bytes;
    use object_store::{memory::InMemory, path::Path, ObjectStore};
    use slatedb_common::clock::DefaultSystemClock;
    use slatedb_common::metrics::{
        lookup_metric, DefaultMetricsRecorder, MetricLevel, MetricsRecorderHelper,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn make_entry(key: &str, value: &str, seq: u64, create_ts: Option<i64>) -> RowEntry {
        RowEntry::new(
            Bytes::from(key.to_string()),
            ValueDeletable::Value(Bytes::from(value.to_string())),
            seq,
            create_ts,
            None,
        )
    }

    #[test]
    fn test_new_buffer_initial_state() {
        let buffer = WalBuffer::new();

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.size(), 0);
        assert_eq!(buffer.last_seq(), None);
    }

    #[test]
    fn test_append_single_entry() {
        let mut buffer = WalBuffer::new();
        let entry = make_entry("key1", "value1", 42, Some(1000));
        let expected_size = entry.estimated_size();

        buffer.append(entry);

        assert!(!buffer.is_empty());
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.size(), expected_size);
        assert_eq!(buffer.last_seq(), Some(42));
    }

    #[test]
    fn test_append_multiple_entries() {
        let mut buffer = WalBuffer::new();

        let entry1 = make_entry("key1", "value1", 10, Some(100));
        let entry2 = make_entry("key2", "value2", 20, Some(200));
        let entry3 = make_entry("key3", "value3", 30, Some(300));
        let entry4 = make_entry("key4", "value4", 40, None);

        let size1 = entry1.estimated_size();
        let size2 = entry2.estimated_size();
        let size3 = entry3.estimated_size();
        let size4 = entry4.estimated_size();

        buffer.append(entry1);
        buffer.append(entry2);
        buffer.append(entry3);
        buffer.append(entry4);

        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.size(), size1 + size2 + size3 + size4);
        assert_eq!(buffer.last_seq(), Some(40));
    }

    #[test]
    fn test_iter() {
        let mut buffer = WalBuffer::new();
        let mut iter = buffer.iter();
        assert!(iter.next().is_none());

        let entry1 = make_entry("key1", "value1", 1, Some(100));
        let entry2 = make_entry("key2", "value2", 2, Some(200));
        let entry3 = make_entry("key3", "value3", 3, Some(300));

        buffer.append(entry1.clone());
        buffer.append(entry2.clone());
        buffer.append(entry3.clone());

        let mut iter = buffer.iter();
        let read1 = iter.next().unwrap();
        assert_eq!(read1.key, entry1.key);
        assert_eq!(read1.seq, entry1.seq);

        let read2 = iter.next().unwrap();
        assert_eq!(read2.key, entry2.key);
        assert_eq!(read2.seq, entry2.seq);

        let read3 = iter.next().unwrap();
        assert_eq!(read3.key, entry3.key);
        assert_eq!(read3.seq, entry3.seq);

        assert!(iter.next().is_none());

        let mut iter = buffer.iter();
        buffer.append(make_entry("key4", "value4", 4, None));

        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 3);

        let mut iter = buffer.iter();
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 4);
    }

    #[test]
    fn test_large_entry_size() {
        let mut buffer = WalBuffer::new();

        let large_key = "k".repeat(10_000);
        let large_value = "v".repeat(100_000);
        let entry = RowEntry::new(
            Bytes::from(large_key),
            ValueDeletable::Value(Bytes::from(large_value)),
            1,
            None,
            None,
        );
        let expected_size = entry.estimated_size();

        buffer.append(entry);

        assert_eq!(buffer.size(), expected_size);
        assert!(buffer.size() > 100_000);
    }

    async fn setup_wal_buffer() -> (
        SlateDbWalWriter,
        Arc<WalTableStore>,
        Arc<DbStatusManager>,
        Arc<DefaultMetricsRecorder>,
    ) {
        setup_wal_buffer_with_flush_interval(Duration::from_millis(10)).await
    }

    async fn setup_wal_buffer_with_flush_interval(
        flush_interval: Duration,
    ) -> (
        SlateDbWalWriter,
        Arc<WalTableStore>,
        Arc<DbStatusManager>,
        Arc<DefaultMetricsRecorder>,
    ) {
        setup_wal_buffer_with_args(flush_interval, Arc::new(|_status| {})).await
    }

    async fn setup_wal_buffer_with_args(
        flush_interval: Duration,
        listener: wal::WalStatusListener,
    ) -> (
        SlateDbWalWriter,
        Arc<WalTableStore>,
        Arc<DbStatusManager>,
        Arc<DefaultMetricsRecorder>,
    ) {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        setup_wal_buffer_with_store(object_store, flush_interval, listener, false, 1).await
    }

    async fn setup_wal_buffer_with_store(
        object_store: Arc<dyn ObjectStore>,
        flush_interval: Duration,
        listener: wal::WalStatusListener,
        group_commit: bool,
        max_inflight_flushes: usize,
    ) -> (
        SlateDbWalWriter,
        Arc<WalTableStore>,
        Arc<DbStatusManager>,
        Arc<DefaultMetricsRecorder>,
    ) {
        let table_store = Arc::new(WalTableStore::new(
            object_store,
            SsTableFormat::default(),
            Path::from("/root"),
            TableStoreKind::Main,
        ));
        let system_clock = Arc::new(DefaultSystemClock::new());
        let status_manager = Arc::new(DbStatusManager::new(0));
        let oracle = Arc::new(DbOracle::new(0, 0, 0, status_manager.clone()));
        let recorder = Arc::new(DefaultMetricsRecorder::new());
        let helper = MetricsRecorderHelper::new(recorder.clone(), MetricLevel::default());
        let task_executor = Arc::new(MessageHandlerExecutor::new(
            status_manager.clone(),
            system_clock.clone(),
        ));
        let wal_buffer = SlateDbWalWriter::start_new(
            status_manager.result_reader(),
            &helper,
            0, // recent_flushed_wal_id
            table_store.clone(),
            1000,                 // max_wal_bytes_size
            4096,                 // max_wal_flushes_before_l0_flush
            Some(flush_interval), // max_flush_interval
            group_commit,
            max_inflight_flushes,
            task_executor.clone(),
        )
        .await
        .unwrap();
        let observer = wal_buffer.observer();
        observer
            .subscribe(Arc::new(move |status| {
                (*listener)(status.clone());
                let WalEvent::WalFlushed(status) = status else {
                    return;
                };
                oracle.advance_durable_seq(status.last_flushed_seq.unwrap_or(0))
            }))
            .unwrap();
        task_executor
            .monitor_on(&Handle::current())
            .expect("failed to monitor executor");
        (wal_buffer, table_store, status_manager, recorder)
    }

    #[tokio::test]
    async fn test_observer_status_while_writer_holds_lock() {
        let (mut wal_buffer, _, _, _) = setup_wal_buffer().await;
        wal_buffer
            .append(&[make_entry("key1", "value1", 1, None)])
            .await
            .unwrap();
        let observer = wal_buffer.observer();
        let expected = observer.status().unwrap();

        // Hold the lock as an append does; the observer must not wait for it.
        let guard = wal_buffer.inner.write();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || tx.send(observer.status()).unwrap());
        let status = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("status waited for the WAL lock")
            .unwrap();
        drop(guard);

        assert_eq!(status.estimated_bytes, expected.estimated_bytes);
        assert_eq!(status.last_flushed_wal_id, expected.last_flushed_wal_id);
        assert_eq!(status.last_flushed_seq, expected.last_flushed_seq);
        assert_eq!(
            status.buffered_wal_entries_count,
            expected.buffered_wal_entries_count
        );
        assert!(expected.buffered_wal_entries_count > 0);
    }

    #[tokio::test]
    async fn test_should_flush_memtable_at_max_wal_gap() {
        let (wal_buffer, _, _, _) = setup_wal_buffer().await;
        let max_wal_gap = wal_buffer.max_wal_flushes_before_l0_flush;
        let last_flushed_wal_id = max_wal_gap + 10;
        wal_buffer.inner.write().last_flushed_wal_id = last_flushed_wal_id;

        assert!(!wal_buffer.should_flush_memtable(last_flushed_wal_id - max_wal_gap + 1));
        assert!(wal_buffer.should_flush_memtable(last_flushed_wal_id - max_wal_gap));
        assert!(wal_buffer.should_flush_memtable(last_flushed_wal_id - max_wal_gap - 1));
        assert!(!wal_buffer.should_flush_memtable(last_flushed_wal_id + 1));
    }

    #[tokio::test]
    async fn test_basic_append_and_flush_operations() {
        let (mut wal_buffer, table_store, _, _) = setup_wal_buffer().await;

        // Append some entries
        let entry1 = make_entry("key1", "value1", 1, None);
        let entry2 = make_entry("key2", "value2", 2, None);

        wal_buffer
            .append(std::slice::from_ref(&entry1))
            .await
            .unwrap();
        wal_buffer
            .append(std::slice::from_ref(&entry2))
            .await
            .unwrap();

        // Flush the buffer
        wal_buffer.flush().await.unwrap().await.unwrap();

        // Verify entries were written to storage
        let table = table_store.open_sst(1.into()).await.unwrap();
        let mut iter =
            WalSstIterator::new(table, table_store.clone(), WalSstIteratorOptions::default())
                .await
                .unwrap();

        let read_entry1 = iter.next().await.unwrap().unwrap();
        assert_eq!(read_entry1.key, entry1.key);
        assert_eq!(read_entry1.value, entry1.value);
        assert_eq!(read_entry1.seq, entry1.seq);

        let read_entry2 = iter.next().await.unwrap().unwrap();
        assert_eq!(read_entry2.key, entry2.key);
        assert_eq!(read_entry2.value, entry2.value);
        assert_eq!(read_entry2.seq, entry2.seq);

        assert!(iter.next().await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_size_based_flush_triggering() {
        // Append an oversized entry to trigger a flush, then wait until its sequence is durable.
        let (mut wal_buffer, _, status_manager, _) =
            setup_wal_buffer_with_flush_interval(Duration::MAX).await;
        let seq = 1;
        let value = "v".repeat(wal_buffer.max_wal_bytes_size);
        wal_buffer
            .append(&[make_entry("key", &value, seq, None)])
            .await
            .unwrap();
        status_manager
            .subscribe()
            .wait_for(|status| status.durable_seq >= seq)
            .await
            .unwrap();

        assert_eq!(wal_buffer.status().unwrap().last_flushed_wal_id, 1);
    }

    #[tokio::test]
    async fn test_immutable_wal_reclaim() {
        let (mut wal_buffer, _, _, _) = setup_wal_buffer().await;

        // Append entries to create multiple WALs
        for i in 0..100 {
            let seq = i + 1;
            let entry = make_entry(&format!("key{}", i), &format!("value{}", i), seq, None);
            wal_buffer.append(&[entry]).await.unwrap();
            wal_buffer.flush().await.unwrap().await.unwrap();
        }
        assert_eq!(wal_buffer.status().unwrap().last_flushed_wal_id, 100);
        assert_eq!(wal_buffer.inner.read().immutable_wals.len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_maybe_trigger_flush_spams_flush_requests() {
        let (mut wal_buffer, _, _, recorder) =
            setup_wal_buffer_with_flush_interval(Duration::MAX).await;

        // Simulate many writers each appending a small entry and calling
        // maybe_trigger_flush, just like the real write path does. Under
        // load the WAL stays over the size threshold across many calls,
        // and each one would send a redundant flush request without the fix.
        let num_writes: u64 = 100;
        for seq in 1..=num_writes {
            let entry = make_entry(&format!("key{}", seq), &format!("value{}", seq), seq, None);
            wal_buffer.append(&[entry]).await.unwrap();
            wal_buffer.maybe_trigger_flush().unwrap();
        }

        let size_triggered_requests =
            lookup_metric(&recorder, stats::WAL_BUFFER_FLUSH_REQUESTS).unwrap();

        // Explicitly flush to drain everything, including any partial current WAL.
        wal_buffer.flush().await.unwrap().await.unwrap();

        let actual_flushes = lookup_metric(&recorder, stats::WAL_BUFFER_FLUSHES).unwrap();

        // With the flush_requested flag, the number of size-triggered requests
        // should be bounded by the number of WALs, not by the number of writes.
        // Before the fix this was ~41 requests for 2 flushes (with 100 writes).
        assert!(
            actual_flushes >= 1,
            "expected at least one flush but got {}",
            actual_flushes,
        );
        assert!(
            size_triggered_requests <= actual_flushes,
            "size_triggered_requests ({}) should not exceed actual_flushes ({})",
            size_triggered_requests,
            actual_flushes,
        );
    }

    fn recording_listener() -> (wal::WalStatusListener, Arc<Mutex<Vec<WalEvent>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let recorder = events.clone();
        let listener = Arc::new(move |event| {
            recorder.lock().unwrap().push(event);
        });
        (listener, events)
    }

    #[tokio::test]
    async fn test_listener_notified_when_flush_task_flushes_wal() {
        // given:
        let (listener, events) = recording_listener();
        let (mut wal_buffer, _, _, _) = setup_wal_buffer_with_args(Duration::MAX, listener).await;

        // when: Append an entry and explicitly flush it, driving the background flush task.
        wal_buffer
            .append(&[make_entry("key1", "value1", 1, None)])
            .await
            .unwrap();
        wal_buffer.flush().await.unwrap().await.unwrap();

        // then: the listener should have been notified that wal 1 was flushed.
        let recorded = events.lock().unwrap().clone();
        let mut flushed: Vec<_> = recorded
            .iter()
            .filter_map(|e| {
                let WalEvent::WalFlushed(status) = e else {
                    return None;
                };
                Some(status)
            })
            .collect();
        assert_eq!(flushed.len(), 1);
        let status = flushed.pop().unwrap();
        assert_eq!(status.last_flushed_wal_id, 1);
        assert_eq!(status.last_flushed_seq, Some(1));
    }

    /// Yields until the flush handler has frozen `count` WALs.
    async fn wait_for_frozen_wals(wal_buffer: &SlateDbWalWriter, count: usize) {
        while wal_buffer.inner.read().immutable_wals.len() < count {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_flush_pipelines_uploads_and_records_in_order() {
        let gated = Arc::new(GatedObjectStore::new(Arc::new(InMemory::new())));
        let (mut wal_buffer, table_store, _, _) = setup_wal_buffer_with_store(
            gated.clone(),
            Duration::from_secs(3600),
            Arc::new(|_status| {}),
            true,
            3,
        )
        .await;
        gated.put_opts_gate.close();

        // Three requests fill the three slots, so each one freezes its own
        // WAL. Each request is handled before the next entry is appended, or
        // the handler would batch the entries into one WAL.
        let mut results = Vec::new();
        for seq in 1..=3 {
            let entry = make_entry(&format!("key{seq}"), "value", seq, None);
            wal_buffer
                .append(std::slice::from_ref(&entry))
                .await
                .unwrap();
            results.push(wal_buffer.flush().await.unwrap());
            wait_for_frozen_wals(&wal_buffer, seq as usize).await;
        }
        gated.put_opts_gate.wait_for_arrivals(3).await;
        {
            let inner = wal_buffer.inner.read();
            assert_eq!(inner.immutable_wals.len(), 3);
            assert_eq!(inner.last_flushed_wal_id, 0);
        }

        // With every slot busy, a fourth request batches instead of freezing.
        let entry = make_entry("key4", "value", 4, None);
        wal_buffer
            .append(std::slice::from_ref(&entry))
            .await
            .unwrap();
        results.push(wal_buffer.flush().await.unwrap());
        tokio::task::yield_now().await;
        {
            let inner = wal_buffer.inner.read();
            assert_eq!(inner.immutable_wals.len(), 3);
            assert_eq!(inner.current_wal.len(), 1);
        }

        gated.put_opts_gate.release();
        for result in results {
            result.await.unwrap();
        }
        {
            let inner = wal_buffer.inner.read();
            assert_eq!(inner.last_flushed_wal_id, 4);
            assert_eq!(inner.last_flushed_seq, Some(4));
            assert!(inner.immutable_wals.is_empty());
        }
        for wal_id in 1..=4u64 {
            let table = table_store.open_sst(wal_id.into()).await.unwrap();
            let mut iter =
                WalSstIterator::new(table, table_store.clone(), WalSstIteratorOptions::default())
                    .await
                    .unwrap();
            let entry = iter.next().await.unwrap().unwrap();
            assert_eq!(entry.seq, wal_id);
            assert!(iter.next().await.unwrap().is_none());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_flush_records_later_wal_only_after_earlier_wal() {
        let gated = Arc::new(GatedObjectStore::new(Arc::new(InMemory::new())));
        let (mut wal_buffer, _, _, _) = setup_wal_buffer_with_store(
            gated.clone(),
            Duration::from_secs(3600),
            Arc::new(|_status| {}),
            true,
            2,
        )
        .await;
        gated.put_opts_gate.close();

        let mut results = Vec::new();
        for seq in 1..=2 {
            let entry = make_entry(&format!("key{seq}"), "value", seq, None);
            wal_buffer
                .append(std::slice::from_ref(&entry))
                .await
                .unwrap();
            results.push(wal_buffer.flush().await.unwrap());
            wait_for_frozen_wals(&wal_buffer, seq as usize).await;
        }
        gated.put_opts_gate.wait_for_arrivals(2).await;

        // Let one upload through. Whichever it was, WAL 2 cannot be durable
        // before WAL 1, so at most WAL 1 is recorded.
        gated.put_opts_gate.admit(1);
        let second = results.pop().expect("two results");
        let second = tokio::time::timeout(Duration::from_millis(100), second).await;
        assert!(second.is_err(), "WAL 2 must not be durable before WAL 1");
        assert!(wal_buffer.inner.read().last_flushed_wal_id <= 1);

        gated.put_opts_gate.release();
        for result in results {
            result.await.unwrap();
        }
        let mut second = wal_buffer.flush().await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), &mut second)
            .await
            .expect("flush completes")
            .unwrap();
        assert_eq!(wal_buffer.inner.read().last_flushed_wal_id, 2);
    }
}
