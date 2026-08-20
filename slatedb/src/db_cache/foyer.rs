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
use std::sync::Arc;
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
}

impl FoyerCache {
    pub fn new() -> Self {
        Self::new_with_opts(FoyerCacheOptions::default())
    }

    pub fn new_with_opts(options: FoyerCacheOptions) -> Self {
        let builder = foyer::CacheBuilder::new(options.max_capacity as _)
            .with_weighter(|_, v: &CachedEntry| v.size())
            .with_shards(options.shards);
        let builder = match options.eviction_policy {
            FoyerEvictionPolicy::Lru => builder.with_eviction_config(foyer::LruConfig::default()),
            FoyerEvictionPolicy::Fifo => builder.with_eviction_config(foyer::FifoConfig::default()),
        };
        Self {
            inner: builder.build(),
        }
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
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
    }

    async fn get_index(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
    }

    async fn get_filter(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
    }

    async fn get_stats(&self, key: &CachedKey) -> Result<Option<CachedEntry>, crate::Error> {
        Ok(self.inner.get(key).map(|entry| entry.value().clone()))
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
        self.dedup_fetch(key, loader).await
    }

    async fn fetch_filter(
        &self,
        key: CachedKey,
        loader: CacheLoader,
    ) -> Result<CachedEntry, crate::Error> {
        self.dedup_fetch(key, loader).await
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
        let fetch = self
            .inner
            .get_or_fetch(&key, move || async move { loader().await });
        match fetch.await {
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
