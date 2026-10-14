//! Diagnostic timers for the write path: backpressure, the wait for the
//! single batch writer, and the steps the writer takes for each batch.

use slatedb_common::metrics::{
    CounterFn, HistogramFn, MetricsRecorderHelper, FINE_LATENCY_BOUNDARIES,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::Instant;

macro_rules! write_diag_name {
    ($suffix:expr) => {
        concat!("slatedb.db.write_diag_", $suffix)
    };
}

pub(crate) const WRITE_DIAG_BACKPRESSURE: &str = write_diag_name!("backpressure_seconds");
pub(crate) const WRITE_DIAG_QUEUE_WAIT: &str = write_diag_name!("queue_wait_seconds");
pub(crate) const WRITE_DIAG_SERVICE: &str = write_diag_name!("service_seconds");
pub(crate) const WRITE_DIAG_TOTAL: &str = write_diag_name!("total_seconds");
pub(crate) const WRITE_DIAG_STEP_US: &str = write_diag_name!("step_us");
pub(crate) const WRITE_DIAG_BUSY_US: &str = write_diag_name!("busy_us");

/// Reads the monotonic clock for the diagnostic timers. They never decide
/// what is written, so they bypass the `SystemClock` that DST controls.
#[allow(clippy::disallowed_methods)]
pub(crate) fn monotonic_now() -> Instant {
    Instant::now()
}

/// A step the batch writer takes for one write batch.
#[derive(Clone, Copy)]
pub(crate) enum WriteStep {
    /// Clock read, sequence number and conflict check.
    Seq,
    /// Converting the batch to row entries.
    Extract,
    /// Appending the entries to the WAL buffer.
    WalAppend,
    /// Inserting the entries into the memtable.
    Memtable,
    /// Recording the commit with the transaction manager.
    TxnTrack,
    /// Recording the memtable sequence and the freeze check.
    Freeze,
}

const STEPS: [(WriteStep, &str); 6] = [
    (WriteStep::Seq, "seq"),
    (WriteStep::Extract, "extract"),
    (WriteStep::WalAppend, "wal_append"),
    (WriteStep::Memtable, "memtable"),
    (WriteStep::TxnTrack, "txn_track"),
    (WriteStep::Freeze, "freeze"),
];

pub(crate) struct WriteDiagStats {
    pub(crate) backpressure: Arc<dyn HistogramFn>,
    pub(crate) queue_wait: Arc<dyn HistogramFn>,
    pub(crate) service: Arc<dyn HistogramFn>,
    pub(crate) total: Arc<dyn HistogramFn>,
    steps: Vec<Arc<dyn CounterFn>>,
    busy_write_us: Arc<dyn CounterFn>,
    busy_flush_us: Arc<dyn CounterFn>,
}

impl WriteDiagStats {
    pub(crate) fn new(recorder: &MetricsRecorderHelper) -> Self {
        let histogram = |name: &str| recorder.histogram(name, FINE_LATENCY_BOUNDARIES).register();
        let busy = |kind: &str| {
            recorder
                .counter(WRITE_DIAG_BUSY_US)
                .labels(&[("kind", kind)])
                .register()
        };
        Self {
            backpressure: histogram(WRITE_DIAG_BACKPRESSURE),
            queue_wait: histogram(WRITE_DIAG_QUEUE_WAIT),
            service: histogram(WRITE_DIAG_SERVICE),
            total: histogram(WRITE_DIAG_TOTAL),
            steps: STEPS
                .iter()
                .map(|(_, name)| {
                    recorder
                        .counter(WRITE_DIAG_STEP_US)
                        .labels(&[("step", name)])
                        .register()
                })
                .collect(),
            busy_write_us: busy("write"),
            busy_flush_us: busy("flush"),
        }
    }

    /// Adds the time since `since` to `step` and returns the current time.
    pub(crate) fn step(&self, step: WriteStep, since: Instant) -> Instant {
        let now = monotonic_now();
        self.steps[step as usize].increment(micros(now - since));
        now
    }

    /// Records the time the batch writer spent on one message.
    pub(crate) fn busy(&self, write: bool, duration: Duration) {
        let counter = if write {
            &self.busy_write_us
        } else {
            &self.busy_flush_us
        };
        counter.increment(micros(duration));
    }
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
