//! Inserts a write group's entries into the memtable from the batch writer and
//! a pool of helper threads.
//!
//! The batch writer is the single task that orders every write, so the time it
//! spends inserting a group bounds write throughput. Spawned as tokio tasks,
//! the inserts waited to be scheduled behind other tasks for several times as
//! long as they took to run. Here the writer and dedicated helper threads take
//! small chunks of the group from a shared cursor until none are left: the
//! writer does whatever the helpers have not started, so its wait at the end is
//! at most one chunk. The writer unparks each helper itself, so they wake
//! together: a condition variable's `notify_all` hands its mutex from one
//! woken thread to the next, and the last helper would start only after a
//! chain of wakeups.

use crate::mem_table::KVTable;
use crate::types::RowEntry;
use parking_lot::Mutex;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

/// Entries in one chunk, a few tens of microseconds of inserts.
const CHUNK_ENTRIES: usize = 32;

/// Spins the writer does while helpers finish their last chunks before it
/// yields its thread between checks.
const SPINS_BEFORE_YIELD: u32 = 1 << 14;

pub(crate) struct MemtableInsertPool {
    shared: Arc<Shared>,
    helpers: Vec<thread::JoinHandle<()>>,
}

#[derive(Default)]
struct Shared {
    job: Mutex<Option<Arc<Job>>>,
    /// Incremented for each job, so a helper joins each job at most once.
    generation: AtomicU64,
    shutdown: AtomicBool,
}

/// The chunks of one group, taken in turn by the writer and the helpers.
struct Job {
    table: Arc<KVTable>,
    chunks: Vec<Mutex<Option<Vec<RowEntry>>>>,
    next: AtomicUsize,
    finished: AtomicUsize,
    panicked: AtomicBool,
}

impl Job {
    fn work(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(chunk) = self.chunks.get(index) else {
                return;
            };
            let entries = chunk.lock().take().expect("each chunk is taken once");
            let _done = ChunkDone(self);
            self.table.put_all(entries);
        }
    }
}

/// Counts a chunk as finished even if one of its inserts panics, so the
/// writer never waits for it forever.
struct ChunkDone<'a>(&'a Job);

impl Drop for ChunkDone<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.panicked.store(true, Ordering::Relaxed);
        }
        self.0.finished.fetch_add(1, Ordering::Release);
    }
}

impl MemtableInsertPool {
    /// Starts `helpers` threads; with none, the writer inserts on its own.
    ///
    /// Helpers are OS threads rather than tasks so that they are never
    /// scheduled behind other tasks. Deterministic simulation runs leave
    /// [`crate::batch_write`]'s insert setting at one thread, so they start
    /// none.
    #[allow(clippy::disallowed_types, clippy::disallowed_methods)]
    pub(crate) fn new(helpers: usize) -> Self {
        let shared = Arc::new(Shared::default());
        let helpers = (0..helpers)
            .map(|index| {
                let shared = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("slatedb-memtable-insert-{index}"))
                    .spawn(move || help(&shared))
                    .expect("failed to start a memtable insert thread")
            })
            .collect();
        Self { shared, helpers }
    }

    /// Inserts `entries` into `table`. Returns `false` if an insert on a
    /// helper thread panicked.
    pub(crate) fn insert(&self, table: &Arc<KVTable>, entries: Vec<RowEntry>) -> bool {
        if self.helpers.is_empty() || entries.len() <= CHUNK_ENTRIES {
            table.put_all(entries);
            return true;
        }
        let chunks = into_chunks(entries)
            .into_iter()
            .map(|chunk| Mutex::new(Some(chunk)))
            .collect();
        let job = Arc::new(Job {
            table: Arc::clone(table),
            chunks,
            next: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        });
        *self.shared.job.lock() = Some(Arc::clone(&job));
        self.shared.generation.fetch_add(1, Ordering::Release);
        for helper in &self.helpers {
            helper.thread().unpark();
        }
        job.work();
        let mut spins = 0u32;
        while job.finished.load(Ordering::Acquire) < job.chunks.len() {
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }
        *self.shared.job.lock() = None;
        !job.panicked.load(Ordering::Relaxed)
    }
}

impl Drop for MemtableInsertPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        for helper in self.helpers.drain(..) {
            helper.thread().unpark();
            let _ = helper.join();
        }
    }
}

/// Splits `entries` into chunks of at most [`CHUNK_ENTRIES`], moving each entry
/// once. Each chunk owns an allocation sized to its entries.
fn into_chunks(entries: Vec<RowEntry>) -> Vec<Vec<RowEntry>> {
    let mut entries = entries.into_iter();
    std::iter::from_fn(|| {
        let chunk: Vec<RowEntry> = entries.by_ref().take(CHUNK_ENTRIES).collect();
        (!chunk.is_empty()).then_some(chunk)
    })
    .collect()
}

/// Joins each new job until the pool shuts down.
fn help(shared: &Shared) {
    let mut seen = 0;
    loop {
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        let generation = shared.generation.load(Ordering::Acquire);
        if generation == seen {
            // An unpark that came before this park is not lost.
            thread::park();
            continue;
        }
        seen = generation;
        let job = shared.job.lock().clone();
        if let Some(job) = job {
            // A panic is recorded by the chunk's guard, and the thread keeps
            // serving later jobs.
            let _ = panic::catch_unwind(AssertUnwindSafe(|| job.work()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(count: u64) -> Vec<RowEntry> {
        (0..count)
            .map(|i| RowEntry::new_value(format!("key{i:05}").as_bytes(), b"value", i + 1))
            .collect()
    }

    #[test]
    fn should_insert_every_entry_with_helpers() {
        let pool = MemtableInsertPool::new(3);
        let table = Arc::new(KVTable::new());
        // Later groups repeat the same keys and sequence numbers, so the count
        // stays the same.
        for _ in 0..20 {
            assert!(pool.insert(&table, entries(1000)));
            assert_eq!(table.metadata().entry_num, 1000);
        }
        let table = Arc::new(KVTable::new());
        assert!(pool.insert(&table, entries(5000)));
        assert_eq!(table.metadata().entry_num, 5000);
        assert_eq!(table.metadata().last_seq, 5000);
    }

    #[test]
    fn should_split_into_chunks_that_own_their_entries() {
        let chunks = into_chunks(entries(1000));
        assert_eq!(chunks.len(), 1000usize.div_ceil(CHUNK_ENTRIES));
        // No chunk keeps a share of the group's allocation.
        assert!(chunks
            .iter()
            .all(|chunk| chunk.len() <= CHUNK_ENTRIES && chunk.capacity() <= CHUNK_ENTRIES));
        let seqs: Vec<u64> = chunks.iter().flatten().map(|entry| entry.seq).collect();
        assert_eq!(seqs, (1..=1000).collect::<Vec<_>>());
    }

    #[test]
    fn should_insert_small_groups_without_helpers() {
        let pool = MemtableInsertPool::new(0);
        let table = Arc::new(KVTable::new());
        assert!(pool.insert(&table, entries(10)));
        assert!(pool.insert(&table, entries(500)));
        assert_eq!(table.metadata().entry_num, 500);
    }
}
