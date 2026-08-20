//! # Foyer Cache
//!
//! This module provides an implementation of an in-memory cache using the Foyer library.
//! The cache is designed to store and retrieve cached blocks, indexes, and filters
//! associated with SSTable IDs.
//!
//! ## Features
//!
//! - **Asynchronous Operations**: Utilizes Foyer's `Cache` to perform cache operations asynchronously.
//! - **Custom Weigher**: Implements a custom weigher to account for the size of cached blocks.
//! - **Flexible Configuration**: Allows customization of cache parameters such as maximum capacity.
//!
//! ## Examples
//!
//!
//! ```
//! use slatedb::{Db, Error};
//! use slatedb::db_cache::foyer::FoyerCache;
//! use slatedb::object_store::memory::InMemory;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Error> {
//!     let object_store = Arc::new(InMemory::new());
//!     let db = Db::builder("test_db", object_store)
//!         .with_db_cache(Arc::new(FoyerCache::new()))
//!         .build()
//!         .await?;
//!     Ok(())
//! }
//! ```
//!

// `Instant` is intentionally used here for monotonic elapsed-time measurement.
// SlateDB's clock abstraction is for wall-clock timestamps, not request timing.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use crate::db_cache::{CacheLoader, CachedEntry, CachedKey, DbCache, DEFAULT_MAX_CAPACITY};
use crate::error::SlateDBError;
use async_trait::async_trait;
use slatedb_common::metrics::{
    HistogramFn, MetricsRecorder, MetricsRecorderHelper, LATENCY_BOUNDARIES,
};
use std::sync::Arc;
use std::time::Instant;
use sysinfo::{CpuRefreshKind, System};

/// Which entry a [`FoyerCache`] evicts when it is full.
///
/// Declared here rather than re-exporting foyer's config types, so the
/// choice is part of SlateDB's own API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FoyerEvictionPolicy {
    /// Evict the least recently used entry.
    #[default]
    Lru,
    /// Evict in insertion order, ignoring reads.
    ///
    /// A read is only a lookup: there is no recency list to update under
    /// the shard lock, which makes hits cheaper than under [`Self::Lru`].
    /// Worth choosing for a cache sized to hold its whole working set,
    /// where the policy never actually has to pick a victim.
    Fifo,
}

/// The options for the Foyer cache.
#[derive(Clone, Copy, Debug)]
pub struct FoyerCacheOptions {
    pub max_capacity: u64,
    pub shards: usize,
    pub eviction_policy: FoyerEvictionPolicy,
}

impl Default for FoyerCacheOptions {
    fn default() -> Self {
        Self {
            max_capacity: DEFAULT_MAX_CAPACITY,
            shards: {
                let mut sys = System::new();
                sys.refresh_cpu_specifics(CpuRefreshKind::nothing());
                sys.cpus().len()
            },
            eviction_policy: FoyerEvictionPolicy::default(),
        }
    }
}

/// Name of the histogram timing a cache lookup that does not load.
pub const CACHE_GET_DURATION: &str = "slatedb.db_cache.get_duration";
/// Name of the histogram timing a lookup that may load through the loader.
pub const CACHE_FETCH_DURATION: &str = "slatedb.db_cache.fetch_duration";

/// Latency histograms for [`FoyerCache`].
///
/// Separating the two paths matters under concurrency: `get` is a pure
/// in-memory lookup, while `fetch` may run the loader or, when another task
/// is already loading the same key, wait on that task's result. Time that
/// shows up only in `fetch` is time spent coalescing, not reading.
#[derive(Clone)]
struct FoyerCacheStats {
    get_duration: Arc<dyn HistogramFn>,
    fetch_duration: Arc<dyn HistogramFn>,
}

impl FoyerCacheStats {
    fn new(recorder: &MetricsRecorderHelper) -> Self {
        Self {
            get_duration: recorder
                .histogram(CACHE_GET_DURATION, LATENCY_BOUNDARIES)
                .description("Latency of an in-memory cache lookup, in seconds")
                .register(),
            fetch_duration: recorder
                .histogram(CACHE_FETCH_DURATION, LATENCY_BOUNDARIES)
                .description(
                    "Latency of a cache lookup including loading or waiting on \
                     a concurrent load, in seconds",
                )
                .register(),
        }
    }
}

/// A cache implementation using the Foyer library.
///
/// This struct wraps a Foyer cache, providing an in-memory caching solution
/// for storing and retrieving cached blocks associated with SSTable IDs.
///
/// # Fields
///
/// * `inner` - The underlying Foyer cache instance, which maps `CachedKey`
///   keys to `CachedEntry` values.
///
/// # Notes
///
/// The cache is configured based on the provided `FoyerCacheOptions`,
/// including settings for the maximum capacity of the cache.
/// It uses a custom weigher to account for the size of cached blocks.
pub struct FoyerCache {
    inner: foyer::Cache<CachedKey, CachedEntry>,
    stats: FoyerCacheStats,
}

impl FoyerCache {
    pub fn new() -> Self {
        Self::new_with_opts(FoyerCacheOptions::default())
    }

    pub fn new_with_opts(options: FoyerCacheOptions) -> Self {
        Self::new_with_opts_and_recorder(options, MetricsRecorderHelper::noop())
    }

    /// Builds a cache that reports its latencies to `recorder`.
    pub fn with_recorder(options: FoyerCacheOptions, recorder: Arc<dyn MetricsRecorder>) -> Self {
        Self::new_with_opts_and_recorder(
            options,
            MetricsRecorderHelper::new(recorder, Default::default()),
        )
    }

    fn new_with_opts_and_recorder(
        options: FoyerCacheOptions,
        recorder: MetricsRecorderHelper,
    ) -> Self {
        let builder = foyer::CacheBuilder::new(options.max_capacity as _)
            .with_weighter(|_, v: &CachedEntry| v.size())
            .with_shards(options.shards);
        let builder = match options.eviction_policy {
            FoyerEvictionPolicy::Lru => builder.with_eviction_config(foyer::LruConfig::default()),
            FoyerEvictionPolicy::Fifo => builder.with_eviction_config(foyer::FifoConfig::default()),
        };
        Self {
            inner: builder.build(),
            stats: FoyerCacheStats::new(&recorder),
        }
    }

    /// Times an in-memory lookup with no loader involved.
    fn timed_get(&self, key: &CachedKey) -> Option<CachedEntry> {
        let started = Instant::now();
        let entry = self.inner.get(key).map(|entry| entry.value().clone());
        self.stats
            .get_duration
            .record(started.elapsed().as_secs_f64());
        entry
    }
}

impl Default for FoyerCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DbCache for FoyerCache {
    async fn get_block(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.timed_get(key))
    }

    async fn get_index(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.timed_get(key))
    }

    async fn get_filter(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.timed_get(key))
    }

    async fn get_stats(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.timed_get(key))
    }

    async fn insert(&self, key: CachedKey, value: CachedEntry) {
        self.inner.insert(key, value);
    }

    async fn remove(&self, key: &CachedKey) {
        self.inner.remove(key);
    }

    fn entry_count(&self) -> u64 {
        // foyer cache doesn't support an entry count estimate
        0
    }

    async fn fetch_block(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_index(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.load_inline(key, loader).await
    }

    async fn fetch_filter(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.load_inline(key, loader).await
    }

    async fn fetch_stats(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
    }
}

impl FoyerCache {
    /// Loads on the calling task, without deduplicating concurrent loads.
    ///
    /// Deduplication is not free: foyer spawns a task to run the load and
    /// hands every other caller a waiter, so a miss costs a spawn and a wake
    /// on top of the load. That is a good trade for entries that are read
    /// once and evicted, and a poor one for indexes and filters, which are
    /// meant to stay resident. Their misses are rare enough that collapsing
    /// concurrent ones saves little, while the scheduling hops are paid on
    /// every miss.
    async fn load_inline(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        let started = Instant::now();
        let entry = match self.timed_get(&key) {
            Some(entry) => entry,
            None => {
                // Concurrent misses for the same key each load it and the
                // last insert wins. The duplicated work is one read of an
                // entry that is about to be resident anyway.
                let entry = loader().await?;
                self.inner.insert(key, entry.clone());
                entry
            }
        };
        self.stats
            .fetch_duration
            .record(started.elapsed().as_secs_f64());
        Ok(entry)
    }

    /// Use foyer's `Cache::get_or_fetch`, which deduplicates concurrent loads for the same key.
    ///
    /// Loader errors round-trip via anyhow's source chain on the foyer error. Foyer wraps them
    /// as `ErrorKind::External` (see foyer-memory's raw.rs). We don't try to recover the original
    /// `crate::Error` value: foyer's broadcast path makes one-to-one recovery impossible for
    /// concurrent waiters, so all error returns are normalized to `SlateDBError::FoyerError`
    /// with the original chained as a source.
    async fn dedup_fetch(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        let started = Instant::now();
        let fetch = self
            .inner
            .get_or_fetch(&key, move || async move { loader().await });
        let result = fetch.await;
        self.stats
            .fetch_duration
            .record(started.elapsed().as_secs_f64());
        match result {
            Ok(entry) => Ok(entry.value().clone()),
            Err(err) => Err(SlateDBError::FoyerError(Arc::new(err)).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::block::Block;
    use bytes::Bytes;
    use rstest::rstest;

    fn block_entry() -> CachedEntry {
        CachedEntry::with_block(Arc::new(Block {
            data: Bytes::from_static(b"block"),
            offsets: vec![0],
        }))
    }

    /// Both policies serve what they store; the option only decides which
    /// entry leaves once the cache is full.
    #[rstest]
    #[case(FoyerEvictionPolicy::Lru)]
    #[case(FoyerEvictionPolicy::Fifo)]
    #[tokio::test]
    async fn test_new_with_opts_eviction_policy(#[case] eviction_policy: FoyerEvictionPolicy) {
        let cache = FoyerCache::new_with_opts(FoyerCacheOptions {
            eviction_policy,
            shards: 2,
            ..Default::default()
        });
        let key = CachedKey::from((crate::db_state::SsTableId::Wal(1), 0u64));

        cache.insert(key.clone(), block_entry()).await;

        assert!(cache.get_block(&key).await.unwrap().is_some());
    }

    #[test]
    fn test_foyer_cache_options_default_eviction_policy() {
        assert_eq!(
            FoyerCacheOptions::default().eviction_policy,
            FoyerEvictionPolicy::Lru
        );
    }
}
