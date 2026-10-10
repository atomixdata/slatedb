use crate::dispatcher::MessageHandlerExecutor;
use crate::error::SlateDBError;
use crate::manifest::Manifest;
use crate::utils::WatchableOnceCellReader;
use crate::wal::slatedb::iterator::{
    SlateDbWalIterator, SlateDbWalIteratorOptions, WalIteratorEndBound,
};
use crate::wal::slatedb::writer::SlateDbWalWriter;
use crate::wal::{WalError, WriterInitResult, WriterManifest};
use crate::{wal, Settings};
use async_trait::async_trait;
use fail_parallel::{fail_point_send, FailPointTx};
use slatedb_common::metrics::MetricsRecorderHelper;
use std::sync::Arc;
use std::time::Duration;

use super::store::{WalFileId, WalTableStore};

#[derive(Clone, Copy)]
pub(crate) struct SlateDbWalWriterInitOptions {
    max_wal_bytes_size: usize,
    max_wal_flushes_before_l0_flush: u64,
    max_flush_interval: Option<Duration>,
    group_commit: bool,
    max_inflight_flushes: usize,
}

impl From<&Settings> for SlateDbWalWriterInitOptions {
    fn from(settings: &Settings) -> Self {
        Self {
            max_wal_bytes_size: settings
                .wal_buffer_bytes
                .unwrap_or(settings.l0_sst_size_bytes),
            max_wal_flushes_before_l0_flush: settings.max_wal_flushes_before_l0_flush,
            max_flush_interval: settings.flush_interval,
            group_commit: settings.wal_group_commit,
            max_inflight_flushes: settings.wal_max_inflight_flushes.max(1),
        }
    }
}

pub(crate) struct SlateDbWalWriterInit {
    closed_result_reader: WatchableOnceCellReader<Result<(), SlateDBError>>,
    recorder: MetricsRecorderHelper,
    table_store: Arc<WalTableStore>,
    max_wal_bytes_size: usize,
    max_wal_flushes_before_l0_flush: u64,
    max_flush_interval: Option<Duration>,
    group_commit: bool,
    max_inflight_flushes: usize,
    empty_wal_id: u64,
    task_executor: Arc<MessageHandlerExecutor>,
    #[cfg_attr(not(test), allow(dead_code))]
    fp_tx: FailPointTx,
}

impl SlateDbWalWriterInit {
    pub(crate) async fn load(
        closed_result_reader: WatchableOnceCellReader<Result<(), SlateDBError>>,
        recorder: MetricsRecorderHelper,
        table_store: Arc<WalTableStore>,
        options: SlateDbWalWriterInitOptions,
        manifest: &Manifest,
        task_executor: Arc<MessageHandlerExecutor>,
        fp_tx: FailPointTx,
    ) -> Result<Self, SlateDBError> {
        let empty_wal_id = table_store
            .next_wal_sst_id(manifest.core.replay_after_wal_id.into())
            .await?
            .value();
        fail_point_send!(fp_tx, "LoadEmptyWalId");
        Ok(Self {
            closed_result_reader,
            recorder,
            table_store,
            max_wal_bytes_size: options.max_wal_bytes_size,
            max_wal_flushes_before_l0_flush: options.max_wal_flushes_before_l0_flush,
            max_flush_interval: options.max_flush_interval,
            group_commit: options.group_commit,
            max_inflight_flushes: options.max_inflight_flushes,
            empty_wal_id,
            task_executor,
            fp_tx,
        })
    }
}

#[async_trait]
impl wal::WriterInit for SlateDbWalWriterInit {
    async fn fence_and_init(
        &self,
        writer_manifest: &mut WriterManifest,
    ) -> Result<WriterInitResult, WalError> {
        let mut empty_wal_id = self.empty_wal_id;
        let mut manifest = writer_manifest.manifest();
        // verify that the empty_wal_id we computed is still valid. Its possible that between
        // computing empty_wal_id and fencing the manifest, the fenced writer advanced the gc
        // boundary (replay_after_wal_id)
        if empty_wal_id <= manifest.core().replay_after_wal_id {
            // the wal gc boundary advanced because the old writer finished a flush - recompute
            // the next wal id
            empty_wal_id = self
                .table_store
                .next_wal_sst_id(manifest.core().replay_after_wal_id.into())
                .await?
                .value();
            writer_manifest.refresh().await?;
            manifest = writer_manifest.manifest();
            fail_point_send!(self.fp_tx, "ReloadEmptyWalId");
            // at this point we still hold the epoch, so it should not be possible for the barrier
            // to have advanced past the computed empty_wal_id
            assert!(empty_wal_id > manifest.core().replay_after_wal_id);
        }
        let manifest = manifest.clone();
        // this writer is the only writer that could have written replay_after_wal_id,
        // so it should not be possible for it to have advanced past the fencing wal.
        // older writers would have failed with a stale epoch
        let replay_after_wal_id = manifest.core().replay_after_wal_id;
        assert!(empty_wal_id > replay_after_wal_id);

        // Uploads overlap, so a crash or a fenced writer can leave WALs past a
        // missing id. Durability is recorded in WAL order, so nothing past
        // the first hole was ever acknowledged and the hole ends the log. The
        // old writer may still land uploads for up to `window` ids past its
        // last durable one, so every id in that window gets a fence and
        // anything it already landed there is dropped.
        let window = self.max_inflight_flushes as u64;
        let mut fence_start = (replay_after_wal_id + 1).max(empty_wal_id.saturating_sub(window));
        while fence_start < empty_wal_id && self.table_store.wal_exists(fence_start.into()).await? {
            fence_start += 1;
        }

        let mut _attempt = 0;
        let mut wal_id = fence_start;
        while wal_id < fence_start + window {
            _attempt += 1;
            let wrote_fence = match self.table_store.write_wal_fence(wal_id.into()).await {
                Ok(()) => true,
                Err(SlateDBError::Fenced) => false,
                Err(err) => return Err(err.into()),
            };
            fail_point_send!(self.fp_tx, format!("{}:{}", "WriteWalFence", _attempt));

            if wrote_fence {
                wal_id += 1;
                continue;
            }
            // Refresh validates that we own the latest epoch still.
            writer_manifest.refresh().await?;
            fail_point_send!(self.fp_tx, format!("{}:{}", "RefreshManifest", _attempt));
            if wal_id == fence_start {
                // The old writer landed the next contiguous WAL before the
                // fence, so the log extends by one. Try the next wal ID.
                fence_start += 1;
                wal_id = fence_start;
            } else {
                // Landed past the fence by a writer that can no longer
                // acknowledge it. Drop it and fence the id.
                self.table_store.delete_sst(wal_id.into()).await?;
            }
        }
        for orphan in self
            .table_store
            .list_wal_ssts(WalFileId::from(fence_start + window)..)
            .await?
        {
            self.table_store.delete_sst(orphan.id).await?;
        }

        // Replay runs through the fences, which are empty, so the manifest's
        // next WAL id lands after the window like the writer's.
        let replay_iterator = SlateDbWalIterator::range(
            replay_after_wal_id + 1,
            WalIteratorEndBound::Exclusive(fence_start + window),
            SlateDbWalIteratorOptions::default(),
            self.table_store.clone(),
        )?;
        let wal_writer = SlateDbWalWriter::start_new(
            self.closed_result_reader.clone(),
            &self.recorder,
            fence_start + window - 1,
            self.table_store.clone(),
            self.max_wal_bytes_size,
            self.max_wal_flushes_before_l0_flush,
            self.max_flush_interval,
            self.group_commit,
            self.max_inflight_flushes,
            self.task_executor.clone(),
        )
        .await?;
        let result = WriterInitResult {
            replay_iterator: Box::new(replay_iterator),
            wal_writer: Box::new(wal_writer),
        };
        Ok(result)
    }
}
