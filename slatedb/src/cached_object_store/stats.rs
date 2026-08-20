use slatedb_common::metrics::{
    CounterFn, GaugeFn, HistogramFn, MetricsRecorderHelper, LATENCY_BOUNDARIES,
};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

macro_rules! oscache_stat_name {
    ($suffix:expr) => {
        concat!("slatedb.object_store_cache.", $suffix)
    };
}

pub const PART_HIT_COUNT: &str = oscache_stat_name!("part_hit_count");
pub const PART_DISK_READ_DURATION: &str = oscache_stat_name!("part_disk_read_duration");
pub const PART_READ_QUEUE_DURATION: &str = oscache_stat_name!("part_read_queue_duration");
pub const PART_READ_EXEC_DURATION: &str = oscache_stat_name!("part_read_exec_duration");
pub const PART_READ_PREAD_DURATION: &str = oscache_stat_name!("part_read_pread_duration");
pub const PART_ACCESS_COUNT: &str = oscache_stat_name!("part_access_count");
pub const CACHE_KEYS: &str = oscache_stat_name!("cache_keys");
pub const CACHE_BYTES: &str = oscache_stat_name!("cache_bytes");
pub const EVICTED_KEYS: &str = oscache_stat_name!("evicted_keys");
pub const EVICTED_BYTES: &str = oscache_stat_name!("evicted_bytes");

#[derive(Clone)]
pub struct CachedObjectStoreStats {
    pub(super) object_store_cache_part_hits: Arc<dyn CounterFn>,
    /// Latency of serving a part from the local disk cache, covering only
    /// reads that hit. A miss falls through to the object store and is
    /// timed by the object store's own request metrics instead.
    pub(super) object_store_cache_part_disk_read_duration: Arc<dyn HistogramFn>,
    /// How long a part read waits for a thread after being handed to
    /// `spawn_blocking`, measured from the submit to the first instruction of
    /// the closure. Nonzero means the blocking pool, not the disk, is what the
    /// read is waiting on.
    pub(super) object_store_cache_part_read_queue_duration: Arc<dyn HistogramFn>,
    /// How long the closure itself runs once it has a thread: the handle
    /// lookup plus the read. Together with the queue duration this accounts
    /// for the whole of `part_disk_read_duration`.
    pub(super) object_store_cache_part_read_exec_duration: Arc<dyn HistogramFn>,
    /// Time in the positional read syscall alone, with no cache lookup or
    /// scheduling included. This is the floor: what the device actually costs.
    pub(super) object_store_cache_part_read_pread_duration: Arc<dyn HistogramFn>,
    pub(super) object_store_cache_part_access: Arc<dyn CounterFn>,
    pub(super) object_store_cache_keys: Arc<dyn GaugeFn>,
    pub(super) object_store_cache_bytes: Arc<dyn GaugeFn>,
    pub(super) object_store_cache_evicted_keys: Arc<dyn CounterFn>,
    pub(super) object_store_cache_evicted_bytes: Arc<dyn CounterFn>,
}

impl Debug for CachedObjectStoreStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedObjectStoreStats")
            .field("object_store_cache_part_hits", &"<counter>")
            .field("object_store_cache_part_disk_read_duration", &"<histogram>")
            .field(
                "object_store_cache_part_read_queue_duration",
                &"<histogram>",
            )
            .field("object_store_cache_part_read_exec_duration", &"<histogram>")
            .field(
                "object_store_cache_part_read_pread_duration",
                &"<histogram>",
            )
            .field("object_store_cache_part_access", &"<counter>")
            .field("object_store_cache_keys", &"<gauge>")
            .field("object_store_cache_bytes", &"<gauge>")
            .field("object_store_cache_evicted_keys", &"<counter>")
            .field("object_store_cache_evicted_bytes", &"<counter>")
            .finish()
    }
}

impl CachedObjectStoreStats {
    pub(crate) fn new(recorder: &MetricsRecorderHelper) -> Self {
        Self {
            object_store_cache_part_hits: recorder.counter(PART_HIT_COUNT).register(),
            object_store_cache_part_disk_read_duration: recorder
                .histogram(PART_DISK_READ_DURATION, LATENCY_BOUNDARIES)
                .description("Latency of a disk cache part read that hit, in seconds")
                .register(),
            object_store_cache_part_read_queue_duration: recorder
                .histogram(PART_READ_QUEUE_DURATION, LATENCY_BOUNDARIES)
                .description("Time a part read waits for a blocking thread, in seconds")
                .register(),
            object_store_cache_part_read_exec_duration: recorder
                .histogram(PART_READ_EXEC_DURATION, LATENCY_BOUNDARIES)
                .description("Time a part read runs once on a blocking thread, in seconds")
                .register(),
            object_store_cache_part_read_pread_duration: recorder
                .histogram(PART_READ_PREAD_DURATION, LATENCY_BOUNDARIES)
                .description("Time in the positional read syscall alone, in seconds")
                .register(),
            object_store_cache_part_access: recorder.counter(PART_ACCESS_COUNT).register(),
            object_store_cache_keys: recorder.gauge(CACHE_KEYS).register(),
            object_store_cache_bytes: recorder.gauge(CACHE_BYTES).register(),
            object_store_cache_evicted_keys: recorder.counter(EVICTED_KEYS).register(),
            object_store_cache_evicted_bytes: recorder.counter(EVICTED_BYTES).register(),
        }
    }
}
