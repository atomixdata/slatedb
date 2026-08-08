use slatedb_common::metrics::{CounterFn, GaugeFn, HistogramFn, MetricsRecorderHelper};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

macro_rules! oscache_stat_name {
    ($suffix:expr) => {
        concat!("slatedb.object_store_cache.", $suffix)
    };
}

pub const PART_HIT_COUNT: &str = oscache_stat_name!("part_hit_count");
pub const PART_ACCESS_COUNT: &str = oscache_stat_name!("part_access_count");
pub const HEAD_HIT_COUNT: &str = oscache_stat_name!("head_hit_count");
pub const HEAD_ACCESS_COUNT: &str = oscache_stat_name!("head_access_count");
pub const UPSTREAM_GET_REQUESTS: &str = oscache_stat_name!("upstream_get_requests");
pub const UPSTREAM_GET_BYTES: &str = oscache_stat_name!("upstream_get_bytes");
pub const UPSTREAM_PUT_REQUESTS: &str = oscache_stat_name!("upstream_put_requests");
pub const UPSTREAM_PUT_BYTES: &str = oscache_stat_name!("upstream_put_bytes");
pub const UPSTREAM_MULTIPART_BYTES: &str = oscache_stat_name!("upstream_multipart_bytes");
pub const UPSTREAM_PUT_RATE: &str = oscache_stat_name!("upstream_put_rate_bytes_per_second");
pub const UPSTREAM_GET_LATENCY: &str = oscache_stat_name!("upstream_get_latency_seconds");

/// Latency buckets for upstream reads, out to 8s. Cache misses go to
/// the object store, whose tail runs to seconds; a request waiting on
/// one waits with it, and single-flight makes every request wanting the
/// same object wait too. Buckets must reach far enough to see that.
const GET_LATENCY_BOUNDARIES: &[f64] = &[
    5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1.0, 2.0, 4.0, 8.0,
];

/// Per-part upload rate buckets, 1 MB/s to 3.2 GB/s. Wide because a
/// part that transmits at line rate is the interesting case: it is what
/// bursts past switch buffers and drops other traffic.
const PUT_RATE_BOUNDARIES: &[f64] = &[1e6, 1e7, 2.5e7, 5e7, 1e8, 2e8, 4e8, 8e8, 1.6e9, 3.2e9];
pub const CACHE_KEYS: &str = oscache_stat_name!("cache_keys");
pub const CACHE_BYTES: &str = oscache_stat_name!("cache_bytes");
pub const EVICTED_KEYS: &str = oscache_stat_name!("evicted_keys");
pub const EVICTED_BYTES: &str = oscache_stat_name!("evicted_bytes");

#[derive(Clone)]
pub struct CachedObjectStoreStats {
    pub(super) object_store_cache_part_hits: Arc<dyn CounterFn>,
    pub(super) object_store_cache_part_access: Arc<dyn CounterFn>,
    /// Head lookups served by the in-memory head cache. Misses that fall
    /// through to the on-disk head or upstream still count as accesses.
    pub(super) object_store_cache_head_hits: Arc<dyn CounterFn>,
    pub(super) object_store_cache_head_access: Arc<dyn CounterFn>,
    /// Requests and bytes that actually reached the upstream object store
    /// (cache misses, prefetches, and uploads). Counted at the cache's
    /// upstream boundary, so disk-cache hits never increment these.
    pub(super) object_store_cache_upstream_get_requests: Arc<dyn CounterFn>,
    pub(super) object_store_cache_upstream_get_bytes: Arc<dyn CounterFn>,
    pub(super) object_store_cache_upstream_put_requests: Arc<dyn CounterFn>,
    pub(super) object_store_cache_upstream_put_bytes: Arc<dyn CounterFn>,
    /// Subset of `upstream_put_bytes` uploaded as multipart parts.
    /// Paired with the multipart_part latency histogram it gives the
    /// per-stream upload rate.
    pub(super) object_store_cache_upstream_multipart_bytes: Arc<dyn CounterFn>,
    /// Upload rate of each individual multipart part, in bytes per
    /// second, measured across the upstream call alone.
    pub(super) object_store_cache_upstream_put_rate: Arc<dyn HistogramFn>,
    /// Wall time of each read that reached the object store.
    pub(super) object_store_cache_upstream_get_latency: Arc<dyn HistogramFn>,
    pub(super) object_store_cache_keys: Arc<dyn GaugeFn>,
    pub(super) object_store_cache_bytes: Arc<dyn GaugeFn>,
    pub(super) object_store_cache_evicted_keys: Arc<dyn CounterFn>,
    pub(super) object_store_cache_evicted_bytes: Arc<dyn CounterFn>,
}

impl Debug for CachedObjectStoreStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedObjectStoreStats")
            .field("object_store_cache_part_hits", &"<counter>")
            .field("object_store_cache_part_access", &"<counter>")
            .field("object_store_cache_head_hits", &"<counter>")
            .field("object_store_cache_head_access", &"<counter>")
            .field("object_store_cache_upstream_get_requests", &"<counter>")
            .field("object_store_cache_upstream_get_bytes", &"<counter>")
            .field("object_store_cache_upstream_put_requests", &"<counter>")
            .field("object_store_cache_upstream_put_bytes", &"<counter>")
            .field("object_store_cache_upstream_multipart_bytes", &"<counter>")
            .field("object_store_cache_upstream_put_rate", &"<histogram>")
            .field("object_store_cache_upstream_get_latency", &"<histogram>")
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
            object_store_cache_part_access: recorder.counter(PART_ACCESS_COUNT).register(),
            object_store_cache_head_hits: recorder.counter(HEAD_HIT_COUNT).register(),
            object_store_cache_head_access: recorder.counter(HEAD_ACCESS_COUNT).register(),
            object_store_cache_upstream_get_requests: recorder
                .counter(UPSTREAM_GET_REQUESTS)
                .register(),
            object_store_cache_upstream_get_bytes: recorder.counter(UPSTREAM_GET_BYTES).register(),
            object_store_cache_upstream_put_requests: recorder
                .counter(UPSTREAM_PUT_REQUESTS)
                .register(),
            object_store_cache_upstream_put_bytes: recorder.counter(UPSTREAM_PUT_BYTES).register(),
            object_store_cache_upstream_multipart_bytes: recorder
                .counter(UPSTREAM_MULTIPART_BYTES)
                .register(),
            object_store_cache_upstream_put_rate: recorder
                .histogram(UPSTREAM_PUT_RATE, PUT_RATE_BOUNDARIES)
                .description("Upload rate of each multipart part in bytes per second")
                .register(),
            object_store_cache_upstream_get_latency: recorder
                .histogram(UPSTREAM_GET_LATENCY, GET_LATENCY_BOUNDARIES)
                .description("Latency of reads served from the object store rather than the cache")
                .register(),
            object_store_cache_keys: recorder.gauge(CACHE_KEYS).register(),
            object_store_cache_bytes: recorder.gauge(CACHE_BYTES).register(),
            object_store_cache_evicted_keys: recorder.counter(EVICTED_KEYS).register(),
            object_store_cache_evicted_bytes: recorder.counter(EVICTED_BYTES).register(),
        }
    }
}
