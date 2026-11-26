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
//!
//! With `memtable_prune_overwrites`, each insert removes the next older version
//! of its key, so a key's entries must land in order. Different keys do not
//! interact, so such a group is split by key: every entry of a key goes to the
//! same chunk, in group order, and the chunks are inserted in parallel.

use crate::mem_table::KVTable;
use crate::types::RowEntry;
use parking_lot::Mutex;
use std::hash::{DefaultHasher, Hash, Hasher};
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
    /// Whether each insert prunes the next older version of its key.
    prune: bool,
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
            if self.prune {
                for entry in entries {
                    self.table.put_and_prune_overwritten(entry);
                }
            } else {
                self.table.put_all(entries);
            }
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
        self.run(table, into_chunks(entries), false)
    }

    /// Inserts `entries` into `table` with
    /// [`KVTable::put_and_prune_overwritten`], in group order for each key.
    /// Returns `false` if an insert on a helper thread panicked.
    pub(crate) fn insert_pruning(&self, table: &Arc<KVTable>, entries: Vec<RowEntry>) -> bool {
        if self.helpers.is_empty() || entries.len() <= CHUNK_ENTRIES {
            for entry in entries {
                table.put_and_prune_overwritten(entry);
            }
            return true;
        }
        self.run(table, into_key_chunks(entries), true)
    }

    /// Inserts `chunks` from the writer and the helper threads.
    fn run(&self, table: &Arc<KVTable>, chunks: Vec<Vec<RowEntry>>, prune: bool) -> bool {
        let chunks = chunks
            .into_iter()
            .map(|chunk| Mutex::new(Some(chunk)))
            .collect();
        let job = Arc::new(Job {
            table: Arc::clone(table),
            chunks,
            prune,
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
        wait_until(|| job.finished.load(Ordering::Acquire) == job.chunks.len());
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

/// Spins until `done`, yielding the thread between checks after a while.
fn wait_until(done: impl Fn() -> bool) {
    let mut spins = 0u32;
    while !done() {
        if spins < SPINS_BEFORE_YIELD {
            spins += 1;
            std::hint::spin_loop();
        } else {
            thread::yield_now();
        }
    }
}

/// The memtable inserts of one write group, shared by the writer and the
/// group's callers (`SLATEDB_CALLER_INSERTS=1`). Each write's entries sit in a
/// slot inserted by whoever takes it first: its caller, once handed an
/// [`InsertTicket`], or the writer, which inserts every slot no caller has
/// taken once it has handed out the tickets. Callers that wake promptly insert
/// in parallel; one that is slow to wake, or gone, costs no more than the
/// writer inserting its entries itself.
///
/// With `prune`, each insert prunes the next older version of its key, so the
/// writer uses a group only if no key is in two of its writes.
///
/// With `callers_publish` (`SLATEDB_CALLER_PUBLISH=1`), the writer hands out
/// the tickets and moves on without inserting or waiting: a caller inserts its
/// own slot, then every slot of the group no one has taken, and the insert that
/// completes the group tells its caller to publish.
pub(crate) struct InsertGroup {
    table: Arc<KVTable>,
    slots: Vec<Mutex<Option<Vec<RowEntry>>>>,
    prune: bool,
    callers_publish: bool,
    /// Slots whose entries are not yet in the memtable.
    pending: AtomicUsize,
    panicked: AtomicBool,
}

impl InsertGroup {
    pub(crate) fn new(
        table: Arc<KVTable>,
        writes: impl IntoIterator<Item = Vec<RowEntry>>,
        prune: bool,
        callers_publish: bool,
    ) -> Arc<Self> {
        let slots: Vec<_> = writes
            .into_iter()
            .map(|entries| Mutex::new(Some(entries)))
            .collect();
        Arc::new(Self {
            table,
            pending: AtomicUsize::new(slots.len()),
            slots,
            prune,
            callers_publish,
            panicked: AtomicBool::new(false),
        })
    }

    /// Inserts the entries in `slot` unless someone already took them.
    /// Returns whether this insert was the group's last.
    pub(crate) fn insert(&self, slot: usize) -> bool {
        let Some(entries) = self.slots[slot].lock().take() else {
            return false;
        };
        let done = SlotDone(self);
        if self.prune {
            for entry in entries {
                self.table.put_and_prune_overwritten(entry);
            }
        } else {
            self.table.put_all(entries);
        }
        std::mem::forget(done);
        self.pending.fetch_sub(1, Ordering::AcqRel) == 1
    }

    /// Inserts every slot no one has taken, starting at `from` and wrapping.
    /// Returns whether one of these inserts was the group's last.
    pub(crate) fn insert_rest(&self, from: usize) -> bool {
        let len = self.slots.len();
        let mut last = false;
        for offset in 0..len {
            last |= self.insert((from + offset) % len);
        }
        last
    }

    /// Whether every slot is in the memtable.
    pub(crate) fn is_inserted(&self) -> bool {
        self.pending.load(Ordering::Acquire) == 0
    }

    /// Whether an insert panicked.
    pub(crate) fn panicked(&self) -> bool {
        self.panicked.load(Ordering::Relaxed)
    }

    /// Inserts every slot no caller has taken, latest first since the earliest
    /// callers were woken first, then waits for the callers' inserts. Returns
    /// `false` if an insert panicked.
    pub(crate) fn finish(&self) -> bool {
        for slot in (0..self.slots.len()).rev() {
            self.insert(slot);
        }
        wait_until(|| self.pending.load(Ordering::Acquire) == 0);
        !self.panicked.load(Ordering::Relaxed)
    }
}

/// Counts a slot as inserted even if its insert panics, so the writer never
/// waits for it forever.
struct SlotDone<'a>(&'a InsertGroup);

impl Drop for SlotDone<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.panicked.store(true, Ordering::Relaxed);
        }
        self.0.pending.fetch_sub(1, Ordering::Release);
    }
}

/// Lets a caller insert its own write's entries into the memtable.
pub(crate) struct InsertTicket {
    group: Arc<InsertGroup>,
    slot: usize,
}

impl InsertTicket {
    pub(crate) fn new(group: Arc<InsertGroup>, slot: usize) -> Self {
        Self { group, slot }
    }

    /// Inserts the write's entries, unless the writer already has. If the
    /// group's callers publish it, also inserts the rest of the group and
    /// returns whether the group is now inserted and this caller must publish.
    pub(crate) fn insert(self) -> bool {
        let last = self.group.insert(self.slot);
        if !self.group.callers_publish {
            return false;
        }
        last | self.group.insert_rest(self.slot + 1)
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

/// Splits `entries` into about one chunk per [`CHUNK_ENTRIES`] entries by the
/// hash of the key, so that every entry of a key is in one chunk, in the order
/// of `entries`.
fn into_key_chunks(entries: Vec<RowEntry>) -> Vec<Vec<RowEntry>> {
    let count = entries.len().div_ceil(CHUNK_ENTRIES);
    let mut chunks: Vec<Vec<RowEntry>> = (0..count)
        .map(|_| Vec::with_capacity(CHUNK_ENTRIES))
        .collect();
    for entry in entries {
        let mut hasher = DefaultHasher::new();
        entry.key.hash(&mut hasher);
        let index =
            usize::try_from(hasher.finish() % count as u64).expect("a chunk index fits in usize");
        chunks[index].push(entry);
    }
    chunks.retain(|chunk| !chunk.is_empty());
    chunks
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

    /// `versions` versions of each of `keys` keys, interleaved, with sequence
    /// numbers rising in group order.
    fn versioned_entries(keys: u64, versions: u64) -> Vec<RowEntry> {
        (0..keys * versions)
            .map(|i| {
                let key = format!("key{:05}", i % keys);
                RowEntry::new_value(key.as_bytes(), format!("value{i}").as_bytes(), i + 1)
            })
            .collect()
    }

    #[test]
    fn should_prune_with_helpers_as_one_thread_does() {
        let pool = MemtableInsertPool::new(3);
        for _ in 0..50 {
            let expected = Arc::new(KVTable::new());
            for entry in versioned_entries(64, 20) {
                expected.put_and_prune_overwritten(entry);
            }
            let table = Arc::new(KVTable::new());
            assert!(pool.insert_pruning(&table, versioned_entries(64, 20)));
            let (got, want) = (table.metadata(), expected.metadata());
            // Each insert pruned the version before it, so one version of
            // each key is left.
            assert_eq!(want.entry_num, 64);
            assert_eq!(got.entry_num, want.entry_num);
            assert_eq!(got.entries_size_in_bytes, want.entries_size_in_bytes);
            assert_eq!(got.last_seq, want.last_seq);
        }
    }

    #[test]
    fn should_split_into_key_chunks_that_keep_each_key_in_order() {
        let chunks = into_key_chunks(versioned_entries(64, 20));
        assert!(chunks.len() > 1);
        assert_eq!(chunks.iter().map(Vec::len).sum::<usize>(), 64 * 20);
        for chunk in &chunks {
            for key in chunk.iter().map(|entry| &entry.key) {
                let seqs: Vec<u64> = chunk
                    .iter()
                    .filter(|entry| entry.key == *key)
                    .map(|entry| entry.seq)
                    .collect();
                assert_eq!(seqs.len(), 20, "every version of a key is in one chunk");
                assert!(seqs.is_sorted());
            }
        }
    }

    /// One write of two entries per slot, with sequence numbers 1..=2 * count.
    fn writes(count: u64) -> Vec<Vec<RowEntry>> {
        (0..count)
            .map(|i| {
                vec![
                    RowEntry::new_value(format!("key{i:05}a").as_bytes(), b"value", 2 * i + 1),
                    RowEntry::new_value(format!("key{i:05}b").as_bytes(), b"value", 2 * i + 2),
                ]
            })
            .collect()
    }

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn should_insert_each_slot_once_with_callers_and_writer() {
        let table = Arc::new(KVTable::new());
        let group = InsertGroup::new(Arc::clone(&table), writes(200), false, false);
        // Callers insert the even slots on their own threads while the writer
        // finishes the group.
        let callers: Vec<_> = (0..200)
            .step_by(2)
            .map(|slot| {
                let ticket = InsertTicket::new(Arc::clone(&group), slot);
                thread::spawn(move || ticket.insert())
            })
            .collect();
        assert!(group.finish());
        assert_eq!(group.pending.load(Ordering::Acquire), 0);
        for caller in callers {
            caller.join().unwrap();
        }
        assert_eq!(table.metadata().entry_num, 400);
        assert_eq!(table.metadata().last_seq, 400);
    }

    #[test]
    fn should_skip_a_ticket_for_a_slot_already_inserted() {
        let table = Arc::new(KVTable::new());
        let group = InsertGroup::new(Arc::clone(&table), writes(3), false, false);
        InsertTicket::new(Arc::clone(&group), 1).insert();
        assert_eq!(table.metadata().entry_num, 2);
        assert!(group.finish());
        assert_eq!(table.metadata().entry_num, 6);
        // A late ticket for a slot the writer inserted does nothing.
        InsertTicket::new(Arc::clone(&group), 2).insert();
        assert_eq!(table.metadata().entry_num, 6);
        assert_eq!(group.pending.load(Ordering::Acquire), 0);
    }

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn should_prune_older_versions_with_callers() {
        let table = Arc::new(KVTable::new());
        // Older versions of every key, which the group's inserts prune.
        for (i, write) in writes(100).into_iter().enumerate() {
            for entry in write {
                let older = RowEntry::new_value(&entry.key, b"older", u64::try_from(i).unwrap());
                table.put(older);
            }
        }
        assert_eq!(table.metadata().entry_num, 200);
        let newer: Vec<Vec<RowEntry>> = writes(100)
            .into_iter()
            .map(|write| {
                write
                    .into_iter()
                    .map(|entry| RowEntry::new_value(&entry.key, b"newer", entry.seq + 1000))
                    .collect()
            })
            .collect();
        let group = InsertGroup::new(Arc::clone(&table), newer, true, false);
        let callers: Vec<_> = (0..100)
            .step_by(3)
            .map(|slot| {
                let ticket = InsertTicket::new(Arc::clone(&group), slot);
                thread::spawn(move || ticket.insert())
            })
            .collect();
        assert!(group.finish());
        for caller in callers {
            caller.join().unwrap();
        }
        assert_eq!(table.metadata().entry_num, 200);
    }

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn should_tell_exactly_one_caller_to_publish() {
        let table = Arc::new(KVTable::new());
        let group = InsertGroup::new(Arc::clone(&table), writes(64), false, true);
        let callers: Vec<_> = (0..64)
            .map(|slot| {
                let ticket = InsertTicket::new(Arc::clone(&group), slot);
                thread::spawn(move || ticket.insert())
            })
            .collect();
        let publishers = callers
            .into_iter()
            .map(|caller| caller.join().unwrap())
            .filter(|publish| *publish)
            .count();
        assert_eq!(publishers, 1);
        assert!(group.is_inserted());
        assert_eq!(table.metadata().entry_num, 128);
    }

    #[test]
    fn should_insert_the_rest_of_the_group_from_one_caller() {
        let table = Arc::new(KVTable::new());
        let group = InsertGroup::new(Arc::clone(&table), writes(5), false, true);
        assert!(InsertTicket::new(Arc::clone(&group), 3).insert());
        assert!(group.is_inserted());
        assert_eq!(table.metadata().entry_num, 10);
        // Later tickets find their slots inserted and do not publish again.
        assert!(!InsertTicket::new(Arc::clone(&group), 0).insert());
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
