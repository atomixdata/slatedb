use crossbeam_skiplist::SkipMap;
use object_store::{path::Path, Attributes, ObjectMeta};
use std::sync::Arc;

/// A cached object head: the metadata returned by a HEAD request together with
/// the object's attributes. Stored behind an `Arc` so a cache hit clones only a
/// pointer rather than the underlying `ObjectMeta`/`Attributes`.
#[derive(Debug)]
pub(crate) struct CachedHead {
    pub(crate) meta: ObjectMeta,
    pub(crate) attributes: Attributes,
}

/// An in-memory cache of object heads (HEAD metadata) that sits in front of the
/// on-disk head cache in [`super::CachedObjectStore`].
///
/// It is populated only on write paths (`cached_put_opts` after the on-disk
/// `save_head` succeeds, and the multipart commit). Reads never fill it: that
/// keeps the map coherent with the authoritative on-disk head files without any
/// invalidation protocol, and avoids amplifying read-side memory pressure when
/// reads stream over many short-lived objects. A hit here skips both the on-disk
/// head read and the upstream HEAD round trip.
///
/// Entries are removed when the underlying object is deleted (or overwritten via
/// rename/copy) so a later reader never sees stale metadata. There is no size
/// bound: each entry is ~hundreds of bytes and the live SST working set is
/// small, so the map stays in the low megabytes.
///
/// Backed by a lock-free [`SkipMap`] (the same structure the memtable uses):
/// reads take no lock and never block concurrent writers, which is what this
/// read-heavy, write-rare cache wants.
pub(crate) struct HeadCache {
    entries: SkipMap<Path, Arc<CachedHead>>,
}

impl std::fmt::Debug for HeadCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadCache")
            .field("len", &self.entries.len())
            .finish()
    }
}

impl HeadCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: SkipMap::new(),
        }
    }

    /// Returns the cached head for `location`, or `None` on a miss. Lock-free:
    /// clones an `Arc` out of the map guard.
    pub(crate) fn get(&self, location: &Path) -> Option<Arc<CachedHead>> {
        self.entries.get(location).map(|entry| entry.value().clone())
    }

    /// Inserts (or overwrites) the head for `location`. Called from write paths
    /// once the object is durable upstream, so the entry always reflects the
    /// just-written content.
    pub(crate) fn insert(&self, location: &Path, meta: ObjectMeta, attributes: Attributes) {
        self.entries
            .insert(location.clone(), Arc::new(CachedHead { meta, attributes }));
    }

    /// Drops the head for `location`, if present. Called when the underlying
    /// object is deleted or overwritten via rename/copy. Best-effort: the
    /// absence of an entry is fine.
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
        let cache = HeadCache::new();
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
        let cache = HeadCache::new();
        let path = Path::from("k");
        cache.insert(&path, meta(&path, 1), Attributes::new());
        cache.insert(&path, meta(&path, 2), Attributes::new());
        assert_eq!(cache.get(&path).unwrap().meta.size, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_concurrent_reads_during_writes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let cache = Arc::new(HeadCache::new());
        let path = Path::from("hot");
        cache.insert(&path, meta(&path, 1), Attributes::new());

        let stop = Arc::new(AtomicBool::new(false));
        // Readers spin on lock-free gets while a writer churns the entry.
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
}
