use std::collections::{BTreeSet, VecDeque};
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use self::stats::WalBufferStats;
use crate::dispatcher::{MessageHandler, MessageHandlerExecutor, MessageTickerDef};
use crate::error::SlateDBError;
use crate::types::RowEntry;
use crate::utils::SafeSender;
use crate::utils::{format_bytes_si, WatchableOnceCellReader};
use crate::wal;
use crate::wal::{FlushResultFuture, WalError, WalEvent, WalStatus, WalWriter};
use async_trait::async_trait;
use futures::{stream::BoxStream, FutureExt, StreamExt};
use log::{error, trace, warn};
use slatedb_common::metrics::MetricsRecorderHelper;
use tokio::{runtime::Handle, sync::oneshot};
use tracing::instrument;

use super::store::WalTableStore;

pub(crate) const WAL_BUFFER_TASK_NAME: &str = "wal_writer";
const WAL_UPLOAD_TASK_NAME: &str = "wal_uploader";

type FlushReplySender = oneshot::Sender<Result<(), WalError>>;

/// [`SlateDbWalWriter`] buffers write operations in memory before flushing them to persistent storage.
/// The flush operation only targets Remote storage right now, later we can add an option to flush to local
/// storage.
///
/// It maintains a `current_wal` buffer for active writes and a queue of frozen
/// WALs that are not yet published.
///
/// By default, it offers a best-effort durability guarantee based on:
///
/// - `max_wal_size`: Requests a flush when `max_wal_size` bytes is exceeded
/// - `max_flush_interval`: Requests a flush after `max_flush_interval` elapses,
///   if set
/// - `max_wal_flushes_before_l0_flush`: Requests a memtable flush when this
///   many WAL files have been published since the latest memtable replay point
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
    /// Task executor for the coordinator and the upload workers.
    task_executor: Arc<MessageHandlerExecutor>,
}

struct SlateDbWalWriterInner {
    current_wal: WalBuffer,
    /// Frozen WALs in file ID order, from freeze until publication. Upload
    /// workers can finish them out of order, but they leave this queue in
    /// order.
    immutable_wals: VecDeque<(u64, Arc<WalBuffer>)>,
    /// The next wal id that will be generated
    next_wal_id: u64,
    /// Monotonically increasing epoch incremented each time the current WAL is
    /// frozen. Used with `last_flush_requested_epoch` to deduplicate size-triggered
    /// flush requests.
    flush_epoch: u64,
    /// The last published WAL file ID. Every WAL file up to and including this
    /// ID is durable. The coordinator advances it in WAL order.
    last_flushed_wal_id: u64,
    /// The last sequence number in the published WAL files. None until the
    /// first file is published.
    last_flushed_seq: Option<u64>,
    /// Set to Some with the error reason if the coordinator has exited.
    flush_task_exited_reason: Option<WalError>,
    /// The channel to the coordinator. It carries flush requests, upload
    /// results, and subscriptions.
    flush_tx: SafeSender<WalFlushWork>,
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
        max_inflight_uploads: usize,
        max_wal_flushes_before_l0_flush: u64,
        max_flush_interval: Option<Duration>,
        task_executor: Arc<MessageHandlerExecutor>,
    ) -> Result<Self, SlateDBError> {
        assert!(max_inflight_uploads > 0);
        let current_wal = WalBuffer::new();
        let immutable_wals = VecDeque::new();
        let (flush_tx, flush_rx) = SafeSender::unbounded_channel(closed_result_reader.clone());
        let inner = SlateDbWalWriterInner {
            current_wal,
            immutable_wals,
            flush_epoch: 1,
            last_flushed_wal_id,
            next_wal_id: last_flushed_wal_id + 1,
            last_flushed_seq: None,
            flush_task_exited_reason: None,
            flush_tx,
        };
        let inner = Arc::new(parking_lot::RwLock::new(inner));
        let stats = Arc::new(WalBufferStats::new(recorder));
        let (upload_tx, upload_rx) = SafeSender::unbounded_channel(closed_result_reader);
        let handlers = (0..max_inflight_uploads)
            .map(|_| {
                Box::new(WalUploadHandler {
                    table_store: table_store.clone(),
                    stats: stats.clone(),
                    completion_tx: inner.read().flush_tx.clone(),
                }) as Box<dyn MessageHandler<WalUploadWork>>
            })
            .collect();
        task_executor.add_handlers(
            WAL_UPLOAD_TASK_NAME.to_string(),
            handlers,
            upload_rx,
            &Handle::current(),
        )?;
        let wal_flush_handler = WalFlushHandler {
            max_flush_interval,
            inner: inner.clone(),
            table_store: table_store.clone(),
            stats: stats.clone(),
            listener: None,
            upload_tx,
            max_inflight_uploads,
            unpublished_uploads: BTreeSet::new(),
            flush_requested: false,
            queued_flushes: Vec::new(),
            ongoing_flushes: VecDeque::new(),
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

    /// Enqueues a flush request for the WAL coordinator.
    ///
    /// A successful return confirms that the request was queued. If supplied,
    /// `reply_tx` receives the reply after the covered writes are durable or
    /// the writer fails.
    fn send_flush_request(&self, reply_tx: Option<FlushReplySender>) -> Result<(), WalError> {
        self.stats.flush_requests.increment(1);
        self.inner
            .read()
            .send_flush_msg(WalFlushWork::RequestFlush { reply_tx })
    }
}

#[async_trait]
impl WalWriter for SlateDbWalWriter {
    fn status(&self) -> Result<WalStatus, WalStatus> {
        self.inner.read().status(&self.table_store)
    }

    /// Appends row entries to the current WAL, and requests a flush when the
    /// size limit is reached.
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
            inner: self.inner.clone(),
            table_store: self.table_store.clone(),
        })
    }

    /// Queues a flush request and returns a future for its reply.
    ///
    /// The reply covers every batch appended before this request and can
    /// include later batches. Dropping the future does not cancel the request.
    async fn flush(&mut self) -> Result<FlushResultFuture, WalError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_flush_request(Some(reply_tx))?;
        Ok(async {
            reply_rx
                .await
                .unwrap_or_else(|e| Err(WalError::InternalError(Arc::new(e))))
        }
        .boxed())
    }

    async fn close(&mut self) -> Result<(), WalError> {
        // Stop scheduling, fail pending flush requests, and then join every
        // upload worker. An upload in progress can still finish, but it is not
        // published.
        let coordinator_result = self
            .task_executor
            .shutdown_or_deregister_task(WAL_BUFFER_TASK_NAME)
            .await;
        let upload_result = self
            .task_executor
            .shutdown_or_deregister_task(WAL_UPLOAD_TASK_NAME)
            .await;
        if coordinator_result.is_none() {
            self.inner
                .write()
                .mark_closed(WalError::Closed, &self.table_store);
        }
        coordinator_result.unwrap_or(Ok(()))?;
        upload_result.unwrap_or(Ok(()))?;
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

    /// Returns the estimated size in bytes of the current WAL and the frozen
    /// WALs that are not yet published.
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
        WalStatus {
            closed_reason: self.flush_task_exited_reason.clone(),
            estimated_bytes: self.estimated_bytes(table_store),
            last_flushed_wal_id: self.last_flushed_wal_id,
            last_flushed_seq: self.last_flushed_seq,
            buffered_wal_entries_count,
        }
    }

    fn mark_closed(&mut self, reason: WalError, table_store: &WalTableStore) -> WalStatus {
        self.flush_task_exited_reason = Some(reason);
        self.freeze_current_wal();
        self.immutable_wals.clear();
        self.compute_status(table_store)
    }

    /// Freezes the current WAL and returns its file ID and buffer.
    ///
    /// Returns None when the current buffer is empty.
    fn freeze_current_wal(&mut self) -> Option<(u64, Arc<WalBuffer>)> {
        if self.current_wal.is_empty() {
            return None;
        }
        let next_wal_id = self.next_wal_id;
        self.next_wal_id += 1;
        let current_wal = std::mem::replace(&mut self.current_wal, WalBuffer::new());
        self.flush_epoch += 1;
        let wal = Arc::new(current_wal);
        self.immutable_wals.push_back((next_wal_id, wal.clone()));
        Some((next_wal_id, wal))
    }

    /// Records the oldest frozen WAL as published, and returns its buffer.
    ///
    /// Panics if `flushed_wal_id` is not the oldest frozen WAL.
    fn record_flushed_wal(&mut self, flushed_wal_id: u64) -> Arc<WalBuffer> {
        let (front_wal_id, flushed_wal) = self
            .immutable_wals
            .pop_front()
            .expect("no immutable wals found to pop");
        assert_eq!(front_wal_id, flushed_wal_id);
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
        flushed_wal
    }
}

impl WalBuffer {
    /// Creates a new empty `WalBuffer`.
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            last_seq: 0,
            entries_size: 0,
        }
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

/// Carries requests and upload results for the WAL coordinator.
enum WalFlushWork {
    /// Reports the storage result of one WAL upload.
    ///
    /// The coordinator decides when to publish the file.
    UploadComplete {
        wal_id: u64,
        result: Result<(), SlateDBError>,
    },
    /// Requests a flush without waiting for storage.
    ///
    /// An optional sender receives the reply after the covered writes are
    /// durable or an error occurs.
    RequestFlush {
        reply_tx: Option<FlushReplySender>,
    },
    Subscribe {
        listener: wal::WalStatusListener,
    },
}

impl Debug for WalFlushWork {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WalFlushWork::UploadComplete { wal_id, result } => f
                .debug_struct("UploadComplete")
                .field("wal_id", wal_id)
                .field("result", result)
                .finish(),
            WalFlushWork::RequestFlush { .. } => f.write_str("RequestFlush"),
            WalFlushWork::Subscribe { .. } => f.write_str("Subscribe"),
        }
    }
}

/// Schedules uploads for flush requests and publishes uploaded files in order.
struct WalFlushHandler {
    max_flush_interval: Option<Duration>,
    inner: Arc<parking_lot::RwLock<SlateDbWalWriterInner>>,
    table_store: Arc<WalTableStore>,
    stats: Arc<WalBufferStats>,
    listener: Option<wal::WalStatusListener>,
    upload_tx: SafeSender<WalUploadWork>,
    /// Maximum number of issued files that are not yet published.
    max_inflight_uploads: usize,
    /// IDs of uploaded files that wait for an earlier upload before
    /// publication.
    unpublished_uploads: BTreeSet<u64>,
    /// Set when any flush request arrives. Cleared when scheduling freezes the
    /// current buffer.
    flush_requested: bool,
    /// Flush requests waiting to be scheduled.
    queued_flushes: Vec<FlushReplySender>,
    /// Flush requests already scheduled, waiting for a reply.
    ///
    /// Each entry stores the last WAL file ID the request needs. The
    /// coordinator replies when that file is published.
    ongoing_flushes: VecDeque<(u64, FlushReplySender)>,
}

impl WalFlushHandler {
    /// Freezes the current buffer and starts its upload when a flush is
    /// requested.
    ///
    /// Returns without waiting for storage. If every upload slot is in use, the
    /// request remains queued. Returns an error if the upload worker cannot
    /// accept the work.
    ///
    /// Callers must reply to requests that are already durable after this call.
    #[instrument(level = "trace", skip_all, err(level = tracing::Level::DEBUG))]
    fn schedule_flush(&mut self) -> Result<(), SlateDBError> {
        if !self.flush_requested {
            return Ok(());
        }
        let (boundary, next) = {
            let mut inner = self.inner.write();
            // If every upload slot is in use, leave the request pending.
            let inflight_uploads = inner.next_wal_id - 1 - inner.last_flushed_wal_id;
            if inflight_uploads >= self.max_inflight_uploads as u64 {
                self.stats.inflight_uploads_full.increment(1);
                return Ok(());
            }
            let next = inner.freeze_current_wal();
            if next.is_some() {
                self.stats
                    .inflight_uploads
                    .set((inflight_uploads + 1) as i64);
            }
            (inner.next_wal_id - 1, next)
        };
        self.flush_requested = false;

        // Assign reply boundaries when data freezes.
        self.ongoing_flushes.extend(
            self.queued_flushes
                .drain(..)
                .map(|reply_tx| (boundary, reply_tx)),
        );
        if let Some((wal_id, wal)) = next {
            self.upload_tx.send(WalUploadWork { wal_id, wal })?;
        }
        Ok(())
    }

    /// Records an upload result, and publishes every uploaded WAL whose older
    /// WALs are all published.
    fn complete_upload(
        &mut self,
        wal_id: u64,
        result: Result<(), SlateDBError>,
    ) -> Result<(), SlateDBError> {
        {
            let inner = self.inner.read();
            assert!(
                inner.last_flushed_wal_id < wal_id && wal_id < inner.next_wal_id,
                "completion for an unknown WAL upload"
            );
        }
        assert!(
            !self.unpublished_uploads.contains(&wal_id),
            "duplicate WAL upload completion"
        );
        if let Err(error) = result {
            error!("failed to upload WAL [wal_id={}]", wal_id);
            return Err(error);
        }
        self.unpublished_uploads.insert(wal_id);
        loop {
            let (wal_id, status, wal) = {
                let mut inner = self.inner.write();
                let wal_id = inner.last_flushed_wal_id + 1;
                if !self.unpublished_uploads.remove(&wal_id) {
                    break;
                }
                let wal = inner.record_flushed_wal(wal_id);
                self.stats
                    .inflight_uploads
                    .set((inner.next_wal_id - 1 - inner.last_flushed_wal_id) as i64);
                (wal_id, inner.compute_status(&self.table_store), wal)
            };
            if Arc::strong_count(&wal) > 1 {
                warn!("outstanding references to wal id {} after flushing", wal_id);
            }
            drop(wal);
            self.notify_listener(WalEvent::WalFlushed(status));
        }
        Ok(())
    }

    /// Replies to flush requests whose files are published.
    ///
    /// Call it after every scheduling attempt. An upload completion can publish
    /// files, and scheduling with an empty buffer can give a request a file
    /// that is already published. No later upload completes for such a file,
    /// so an idle flush would wait forever without this reply.
    fn reply_to_durable_flushes(&mut self) {
        let last_flushed_wal_id = self.inner.read().last_flushed_wal_id;
        // Boundaries never decrease from front to back. Reply from the front
        // until a request needs a file that is not yet published. No later
        // request can be ready, because it needs the same or a later file.
        while self
            .ongoing_flushes
            .front()
            .is_some_and(|(boundary, _)| *boundary <= last_flushed_wal_id)
        {
            let (_, reply_tx) = self.ongoing_flushes.pop_front().expect("pending flush");
            let _ = reply_tx.send(Ok(()));
        }
    }

    fn notify_listener(&self, event: WalEvent) {
        if let Some(l) = self.listener.as_ref() {
            (*l)(event);
        }
    }
}

#[async_trait]
impl MessageHandler<WalFlushWork> for WalFlushHandler {
    fn tickers(&mut self) -> Vec<MessageTickerDef<WalFlushWork>> {
        if let Some(max_flush_interval) = self.max_flush_interval {
            return vec![MessageTickerDef::new(
                max_flush_interval,
                Box::new(|| WalFlushWork::RequestFlush { reply_tx: None }),
            )];
        }
        vec![]
    }

    async fn handle(&mut self, message: WalFlushWork) -> Result<(), SlateDBError> {
        match message {
            WalFlushWork::RequestFlush { reply_tx } => {
                if let Some(reply_tx) = reply_tx {
                    self.queued_flushes.push(reply_tx);
                }
                self.flush_requested = true;
                let scheduling_result = self.schedule_flush();
                self.reply_to_durable_flushes();
                scheduling_result
            }
            WalFlushWork::UploadComplete { wal_id, result } => {
                self.complete_upload(wal_id, result)?;
                let scheduling_result = self.schedule_flush();
                self.reply_to_durable_flushes();
                scheduling_result
            }
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

        for reply_tx in self.queued_flushes.drain(..) {
            let _ = reply_tx.send(Err(error.clone()));
        }
        for (_, reply_tx) in self.ongoing_flushes.drain(..) {
            let _ = reply_tx.send(Err(error.clone()));
        }

        // drain remaining messages
        while let Some(msg) = messages.next().await {
            match msg {
                WalFlushWork::UploadComplete { .. } => {}
                WalFlushWork::RequestFlush { reply_tx } => {
                    if let Some(reply_tx) = reply_tx {
                        let _ = reply_tx.send(Err(error.clone()));
                    }
                }
                WalFlushWork::Subscribe { listener } => {
                    (*listener)(WalEvent::WalClosed(final_status.clone()))
                }
            }
        }
        Ok(())
    }
}

/// Carries a frozen WAL buffer and its assigned file ID.
struct WalUploadWork {
    wal_id: u64,
    wal: Arc<WalBuffer>,
}

impl Debug for WalUploadWork {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalUploadWork")
            .field("wal_id", &self.wal_id)
            .finish()
    }
}

/// Encodes and uploads frozen WAL files.
///
/// Reports each result to the coordinator. An upload does not publish its
/// file. The coordinator publishes files in order.
struct WalUploadHandler {
    table_store: Arc<WalTableStore>,
    stats: Arc<WalBufferStats>,
    completion_tx: SafeSender<WalFlushWork>,
}

impl WalUploadHandler {
    /// Writes one frozen WAL buffer to object storage.
    ///
    /// A successful return means that the file is stored. The coordinator
    /// publishes it separately.
    #[instrument(level = "trace", skip_all, err(level = tracing::Level::DEBUG))]
    async fn do_flush_one_wal(&self, wal_id: u64, wal: Arc<WalBuffer>) -> Result<(), SlateDBError> {
        self.stats.flushes.increment(1);

        let mut sst_builder = self.table_store.table_builder();
        let mut iter = wal.iter();
        while let Some(entry) = iter.next() {
            sst_builder.add(entry).await?;
        }

        let encoded_sst = sst_builder.build().await?;
        let written_bytes = encoded_sst.remaining_len() as u64;
        self.table_store
            .write_sst(wal_id.into(), &encoded_sst)
            .await?;
        self.stats.flush_bytes.increment(written_bytes);
        Ok(())
    }
}

#[async_trait]
impl MessageHandler<WalUploadWork> for WalUploadHandler {
    async fn handle(&mut self, work: WalUploadWork) -> Result<(), SlateDBError> {
        let result = self.do_flush_one_wal(work.wal_id, work.wal).await;
        let completion = WalFlushWork::UploadComplete {
            wal_id: work.wal_id,
            result,
        };
        // The coordinator owns upload error handling. A failed send means that
        // the coordinator already exited, and every flush request for this
        // file already received an error reply.
        let _ = self.completion_tx.send(completion);
        Ok(())
    }

    async fn cleanup(
        &mut self,
        _messages: BoxStream<'async_trait, WalUploadWork>,
        _result: Result<(), SlateDBError>,
    ) -> Result<(), SlateDBError> {
        Ok(())
    }
}

/// Interface for getting information about the current state of the Wal
#[derive(Clone)]
struct SlateDbWalObserver {
    inner: Arc<parking_lot::RwLock<SlateDbWalWriterInner>>,
    table_store: Arc<WalTableStore>,
}

impl wal::WalObserver for SlateDbWalObserver {
    /// Gets information about the Wal buffer's current state
    fn status(&self) -> Result<WalStatus, WalStatus> {
        self.inner.read().status(self.table_store.as_ref())
    }

    fn subscribe(&self, listener: wal::WalStatusListener) -> Result<(), WalError> {
        self.inner
            .read()
            .send_flush_msg(WalFlushWork::Subscribe { listener })
    }
}

pub mod stats {
    use slatedb_common::metrics::{CounterFn, GaugeFn, MetricsRecorderHelper};
    use std::sync::Arc;

    macro_rules! wal_stat_name {
        ($suffix:expr) => {
            concat!("slatedb.wal.", $suffix)
        };
    }

    pub const WAL_BUFFER_FLUSHES: &str = wal_stat_name!("wal_buffer_flushes");
    pub const WAL_BUFFER_FLUSH_REQUESTS: &str = wal_stat_name!("wal_buffer_flush_requests");
    pub const WAL_BUFFER_ESTIMATED_BYTES: &str = wal_stat_name!("wal_buffer_estimated_bytes");
    pub const WAL_FLUSH_BYTES: &str = wal_stat_name!("wal_flush_bytes");
    /// Frozen WAL files that are issued for upload but not yet published.
    pub const WAL_INFLIGHT_UPLOADS: &str = wal_stat_name!("wal_inflight_uploads");
    /// Flush attempts that waited because every upload slot was in use.
    pub const WAL_INFLIGHT_UPLOADS_FULL: &str = wal_stat_name!("wal_inflight_uploads_full");

    pub(super) struct WalBufferStats {
        pub(super) estimated_bytes: Arc<dyn GaugeFn>,
        pub(super) flushes: Arc<dyn CounterFn>,
        pub(super) flush_requests: Arc<dyn CounterFn>,
        pub(super) flush_bytes: Arc<dyn CounterFn>,
        pub(super) inflight_uploads: Arc<dyn GaugeFn>,
        pub(super) inflight_uploads_full: Arc<dyn CounterFn>,
    }

    impl WalBufferStats {
        pub(super) fn new(recorder: &MetricsRecorderHelper) -> Self {
            Self {
                estimated_bytes: recorder.gauge(WAL_BUFFER_ESTIMATED_BYTES).register(),
                flushes: recorder.counter(WAL_BUFFER_FLUSHES).register(),
                flush_requests: recorder.counter(WAL_BUFFER_FLUSH_REQUESTS).register(),
                flush_bytes: recorder.counter(WAL_FLUSH_BYTES).register(),
                inflight_uploads: recorder.gauge(WAL_INFLIGHT_UPLOADS).register(),
                inflight_uploads_full: recorder.counter(WAL_INFLIGHT_UPLOADS_FULL).register(),
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
            1,                    // max_inflight_uploads
            4096,                 // max_wal_flushes_before_l0_flush
            Some(flush_interval), // max_flush_interval
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
}
