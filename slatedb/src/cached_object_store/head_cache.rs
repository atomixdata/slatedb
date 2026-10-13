use object_store::{path::Path, Attributes, ObjectMeta};

/// The default maximum number of heads in a [`HeadCache`]. One entry uses a
/// few hundred bytes, so a full cache uses tens of megabytes.
pub(crate) const DEFAULT_HEAD_CACHE_CAPACITY: usize = 100_000;

/// A cached object head. It holds the metadata that a HEAD request returns and
/// the attributes of the object.
#[derive(Debug, Clone)]
pub(crate) struct CachedHead {
    pub(crate) meta: ObjectMeta,
    pub(crate) attributes: Attributes,
}

/// An in-memory cache of object heads (HEAD metadata) that sits in front of the
/// on-disk head cache in [`super::CachedObjectStore`].
///
/// These paths put heads into the cache: the write paths (`cached_put_opts`
/// and the multipart commit), the startup preload (`warm()`), and reads that
/// find the head on disk. Delete, rename and copy remove the head of each
/// object that they change. A hit skips the on-disk head read (spawn_blocking,
/// file-handle lookup and JSON parse) and the upstream HEAD request.
///
/// The cache is a [`quick_cache::sync::Cache`] with a fixed maximum number of
/// entries. The cache is split into shards, and a hit takes only a read lock on
/// one shard. When the cache is full, it evicts heads that are used the least.
/// An evicted head is not an error, because the next read gets the head from
/// disk or from the object store.
pub(crate) struct HeadCache {
    entries: quick_cache::sync::Cache<Path, CachedHead>,
}

impl std::fmt::Debug for HeadCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadCache")
            .field("len", &self.entries.len())
            .field("cap", &self.entries.capacity())
            .finish()
    }
}

impl HeadCache {
    /// Creates a cache that holds at most `max_entries` heads.
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            entries: quick_cache::sync::Cache::new(max_entries),
        }
    }

    /// Returns a copy of the cached head for `location`, or `None` on a miss.
    pub(crate) fn get(&self, location: &Path) -> Option<CachedHead> {
        self.entries.get(location)
    }

    /// Inserts the head for `location`, or replaces the current head. The
    /// write paths call it after the object is durable in the object store.
    pub(crate) fn insert(&self, location: &Path, meta: ObjectMeta, attributes: Attributes) {
        self.entries
            .insert(location.clone(), CachedHead { meta, attributes });
    }

    /// Removes the head for `location`. Delete, rename and copy call it. If
    /// there is no head for `location`, it does nothing.
    pub(crate) fn remove(&self, location: &Path) {
        self.entries.remove(location);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn meta(location: &Path, size: u64) -> ObjectMeta {
        ObjectMeta {
            location: location.clone(),
            last_modified: Utc::now(),
            size,
            e_tag: None,
            version: None,
        }
    }

    #[test]
    fn test_insert_get_remove() {
        let cache = HeadCache::new(DEFAULT_HEAD_CACHE_CAPACITY);
        let path = Path::from("a/b/c");
        assert!(cache.get(&path).is_none());

        cache.insert(&path, meta(&path, 42), Attributes::new());
        let hit = cache.get(&path).expect("should be cached");
        assert_eq!(hit.meta.size, 42);

        cache.remove(&path);
        assert!(cache.get(&path).is_none());
    }

    #[test]
    fn test_insert_overwrites_in_place() {
        let cache = HeadCache::new(DEFAULT_HEAD_CACHE_CAPACITY);
        let path = Path::from("k");
        cache.insert(&path, meta(&path, 1), Attributes::new());
        cache.insert(&path, meta(&path, 2), Attributes::new());
        assert_eq!(cache.get(&path).unwrap().meta.size, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_concurrent_reads_during_writes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(HeadCache::new(DEFAULT_HEAD_CACHE_CAPACITY));
        let path = Path::from("hot");
        cache.insert(&path, meta(&path, 1), Attributes::new());

        let stop = Arc::new(AtomicBool::new(false));
        // Readers call get in a loop while a writer replaces the entry.
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let cache = cache.clone();
                let path = path.clone();
                let stop = stop.clone();
                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        if let Some(h) = cache.get(&path) {
                            assert!(h.meta.size >= 1);
                        }
                    }
                })
            })
            .collect();

        for i in 1..10_000u64 {
            cache.insert(&path, meta(&path, i), Attributes::new());
        }
        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }

    #[test]
    fn test_len_stays_within_capacity() {
        let capacity = 100;
        let cache = HeadCache::new(capacity);
        for i in 0..(capacity * 10) {
            let path = Path::from(format!("obj-{i}"));
            cache.insert(&path, meta(&path, i as u64), Attributes::new());
        }
        assert!(
            cache.len() <= capacity,
            "len {} > {}",
            cache.len(),
            capacity
        );
    }
}
