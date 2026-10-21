use crate::dispatcher::MessageHandlerExecutor;
use crate::error::SlateDBError;
use crate::utils::{build_concurrent, WatchableOnceCellReader};
use crate::wal::slatedb::iterator::{
    SlateDbWalIterator, SlateDbWalIteratorOptions, WalExcludedRanges, WalIteratorEndBound,
};
use crate::wal::slatedb::writer::SlateDbWalWriter;
use crate::wal::{WalError, WriterInitResult, WriterManifest};
use crate::{wal, Settings};
use async_trait::async_trait;
use fail_parallel::{fail_point_send, FailPointTx};
use log::warn;
use slatedb_common::metrics::MetricsRecorderHelper;
use std::sync::Arc;
use std::time::Duration;

use super::store::WalTableStore;

#[derive(Clone, Copy)]
pub(crate) struct SlateDbWalWriterInitOptions {
    max_wal_bytes_size: usize,
    max_inflight_wal_uploads: usize,
    max_wal_flushes_before_l0_flush: u64,
    max_flush_interval: Option<Duration>,
}

impl SlateDbWalWriterInitOptions {
    pub(crate) fn max_inflight_wal_uploads(&self) -> usize {
        self.max_inflight_wal_uploads
    }
}

impl From<&Settings> for SlateDbWalWriterInitOptions {
    fn from(settings: &Settings) -> Self {
        Self {
            max_wal_bytes_size: settings.l0_sst_size_bytes,
            max_inflight_wal_uploads: usable_max_inflight_wal_uploads(
                settings.max_inflight_wal_uploads,
            ),
            max_wal_flushes_before_l0_flush: settings.max_wal_flushes_before_l0_flush,
            max_flush_interval: settings.flush_interval,
        }
    }
}

/// Returns the maximum number of in-flight WAL uploads that this build can use.
///
/// Production uploads stay serial until the feature is enabled. Only tests and
/// `bench-internal` builds use a configured value above 1.
fn usable_max_inflight_wal_uploads(configured: usize) -> usize {
    if cfg!(any(test, feature = "bench-internal")) {
        return configured;
    }
    if configured > 1 {
        warn!(
            "ignoring max_inflight_wal_uploads without the bench-internal feature [max_inflight_wal_uploads={}]",
            configured
        );
    }
    1
}

pub(crate) struct SlateDbWalWriterInit {
    closed_result_reader: WatchableOnceCellReader<Result<(), SlateDBError>>,
    recorder: MetricsRecorderHelper,
    table_store: Arc<WalTableStore>,
    max_wal_bytes_size: usize,
    max_inflight_wal_uploads: usize,
    max_wal_flushes_before_l0_flush: u64,
    max_flush_interval: Option<Duration>,
    task_executor: Arc<MessageHandlerExecutor>,
    #[cfg_attr(not(test), allow(dead_code))]
    fp_tx: FailPointTx,
}

impl SlateDbWalWriterInit {
    pub(crate) fn new(
        closed_result_reader: WatchableOnceCellReader<Result<(), SlateDBError>>,
        recorder: MetricsRecorderHelper,
        table_store: Arc<WalTableStore>,
        options: SlateDbWalWriterInitOptions,
        task_executor: Arc<MessageHandlerExecutor>,
        fp_tx: FailPointTx,
    ) -> Self {
        Self {
            closed_result_reader,
            recorder,
            table_store,
            max_wal_bytes_size: options.max_wal_bytes_size,
            max_inflight_wal_uploads: options.max_inflight_wal_uploads,
            max_wal_flushes_before_l0_flush: options.max_wal_flushes_before_l0_flush,
            max_flush_interval: options.max_flush_interval,
            task_executor,
            fp_tx,
        }
    }
}

#[async_trait]
impl wal::WriterInit for SlateDbWalWriterInit {
    async fn fence_and_init(
        &self,
        writer_manifest: &mut WriterManifest,
    ) -> Result<WriterInitResult, WalError> {
        let manifest = writer_manifest.manifest();
        let replay_after_wal_id = manifest.core().replay_after_wal_id;
        let window = manifest.core().wal_excluded_range_max_size;
        let iterator_options = SlateDbWalIteratorOptions::default();
        let mut frontier = replay_after_wal_id + 1;

        // Each pass takes over at the current frontier g. The fence create at
        // g fails if an older writer's upload landed there after the walk.
        // That file is accepted history and the older writer can acknowledge
        // it, so the next pass walks past it to a new frontier. The older
        // writer has at most `window` uploads left, so the loop is finite.
        let mut _attempt = 0;
        loop {
            _attempt += 1;
            // This writer holds the epoch, so only it records new ranges, and
            // the replay boundary cannot advance.
            let excluded_ranges =
                WalExcludedRanges::known(&writer_manifest.manifest().core().wal_excluded_ranges);
            frontier = self
                .table_store
                .walk_to_frontier(frontier.into(), &excluded_ranges)
                .await?
                .value();
            fail_point_send!(self.fp_tx, "WalkToFrontier");

            let end_wal_id = frontier + window - 1;
            // Exclude the IDs that an older writer can still upload into: at
            // most `window` IDs, starting at the frontier. Record the range
            // before the fence, so a reader that sees the fence finds the
            // range. With a window of 1, g is the only ID an older writer can
            // still upload to, so the fence alone excludes it.
            if window > 1 {
                writer_manifest
                    .exclude_wal_range(frontier, end_wal_id)
                    .await?;
            }
            let wrote_fence = match self.table_store.write_wal_fence(frontier.into()).await {
                Ok(()) => true,
                Err(SlateDBError::Fenced) => false,
                Err(err) => return Err(err.into()),
            };
            fail_point_send!(self.fp_tx, format!("{}:{}", "WriteWalFence", _attempt));

            if wrote_fence {
                // Fence the other missing IDs, so an older upload fails at its
                // create. Orphans stay: the range excludes them.
                build_concurrent(
                    frontier + 1..=end_wal_id,
                    iterator_options.max_fetch_tasks,
                    |wal_id| async move {
                        match self.table_store.write_wal_fence(wal_id.into()).await {
                            Ok(()) | Err(SlateDBError::Fenced) => Ok(None::<()>),
                            Err(err) => Err(err),
                        }
                    },
                )
                .await?;
                // Replay ends at the fence at g. With this takeover's range, it
                // reports the IDs through `end_wal_id` as consumed, and the
                // writer starts after them.
                let replay_iterator = SlateDbWalIterator::range(
                    replay_after_wal_id + 1,
                    WalIteratorEndBound::Exclusive(frontier + 1),
                    iterator_options,
                    self.table_store.clone(),
                )?
                .with_excluded_ranges(WalExcludedRanges::known(
                    &writer_manifest.manifest().core().wal_excluded_ranges,
                ));
                let wal_writer = SlateDbWalWriter::start_new(
                    self.closed_result_reader.clone(),
                    &self.recorder,
                    end_wal_id,
                    self.table_store.clone(),
                    self.max_wal_bytes_size,
                    self.max_inflight_wal_uploads,
                    self.max_wal_flushes_before_l0_flush,
                    self.max_flush_interval,
                    self.task_executor.clone(),
                )
                .await?;
                let result = WriterInitResult {
                    replay_iterator: Box::new(replay_iterator),
                    wal_writer: Box::new(wal_writer),
                };
                return Ok(result);
            }
            // If a newer writer took the frontier, the refresh returns
            // `Fenced`. Otherwise an older upload took it, and the next pass
            // walks on to a new frontier.
            writer_manifest.refresh().await?;
            fail_point_send!(self.fp_tx, format!("{}:{}", "RefreshManifest", _attempt));
        }
    }
}
