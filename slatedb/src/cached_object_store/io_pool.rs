//! A fixed pool of threads for disk-cache reads.
//!
//! Cache reads are synchronous file I/O, so they have to run somewhere other
//! than a runtime worker. `tokio::task::spawn_blocking` is the obvious home,
//! but its submit path takes one process-wide mutex per task. That is fine
//! when reads trickle in, and not fine when thousands of connections wake
//! together and every runtime worker tries to submit in the same instant:
//! they pile onto the mutex, spin, and `sched_yield`, and the read waits on
//! scheduling rather than on the disk.
//!
//! What drives it is arrivals colliding, not arrival rate. A few dozen
//! connections issuing a higher total rate never collide and never contend;
//! thousands of connections at a lower rate do, because their wakeups arrive
//! in batches.
//!
//! This pool removes the shared mutex from the submit path. Threads are
//! created once, up front, so no submission waits on a thread spawn, and work
//! is handed over through an MPMC channel that submitters do not serialize on.

// OS threads are the point of this module: the work is blocking file I/O, so
// it cannot run on the runtime, and a fixed set of threads is what avoids the
// shared pool's submit-path mutex. `tokio::task::Builder` would put the work
// back where it came from.
#![allow(clippy::disallowed_types)]

use crate::error::SlateDBError;
use log::warn;
use std::sync::Arc;

/// Work handed to a pool thread.
///
/// Boxed because each job has its own closure and result type; the result
/// goes back through a channel the caller holds rather than being returned.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// Threads dedicated to disk-cache reads.
///
/// Cloning shares one pool. When the last clone drops, the channel closes and
/// the threads exit.
#[derive(Clone)]
pub(crate) struct IoPool {
    sender: async_channel::Sender<Job>,
    inner: Arc<PoolThreads>,
}

/// Holds the thread handles so they are joined once the last [`IoPool`] clone
/// is dropped.
struct PoolThreads {
    threads: parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl IoPool {
    /// Starts `threads` threads immediately.
    ///
    /// Sizing is about reads in flight rather than cores: these threads spend
    /// nearly all their time parked inside `pread`, so the count needs to
    /// cover concurrent reads, and idle ones cost only a stack.
    pub(crate) fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        // Unbounded, because a full bounded channel would make the submitter
        // wait - reintroducing the queueing this exists to remove. Depth is
        // bounded in practice by how many reads callers have outstanding.
        let (sender, receiver) = async_channel::unbounded::<Job>();

        let handles = (0..threads)
            .filter_map(|index| {
                let receiver = receiver.clone();
                let spawned = std::thread::Builder::new()
                    .name(format!("slatedb-cache-io-{index}"))
                    .spawn(move || {
                        // Returns `Err` once the last sender drops, which is
                        // how the thread learns to exit.
                        while let Ok(job) = receiver.recv_blocking() {
                            job();
                        }
                    });
                match spawned {
                    Ok(handle) => Some(handle),
                    Err(error) => {
                        warn!("Failed to start cache I/O thread {index}: {error}");
                        None
                    }
                }
            })
            .collect();

        Self {
            sender,
            inner: Arc::new(PoolThreads {
                threads: parking_lot::Mutex::new(handles),
            }),
        }
    }

    /// Sizes a pool from the machine, for callers without a better number.
    pub(crate) fn with_default_size() -> Self {
        let parallelism = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(DEFAULT_PARALLELISM);
        // Several reads per core: they block rather than compute, so the
        // useful count is set by how many can be outstanding, not by cores.
        Self::new((parallelism * READS_PER_CORE).clamp(MIN_THREADS, MAX_THREADS))
    }

    /// Runs `job` on a pool thread and waits for its result.
    ///
    /// Returns [`SlateDBError::BackgroundTaskCancelled`] when the pool can no
    /// longer run work, which happens once it has been dropped.
    pub(crate) async fn run<F, T>(&self, job: F) -> Result<T, SlateDBError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let queued = self
            .sender
            .send(Box::new(move || {
                // A dropped receiver means the caller stopped waiting; the
                // result is simply discarded.
                let _ = result_tx.send(job());
            }))
            .await;
        if queued.is_err() {
            return Err(SlateDBError::BackgroundTaskCancelled(POOL_STOPPED.into()));
        }
        result_rx
            .await
            .map_err(|_| SlateDBError::BackgroundTaskCancelled(POOL_STOPPED.into()))
    }
}

/// Reported when the pool is gone and a read can no longer be run.
const POOL_STOPPED: &str = "cache I/O pool stopped";

/// Assumed core count when the machine won't say.
const DEFAULT_PARALLELISM: usize = 8;
/// Threads per core. Reads block, so this is about overlap, not compute.
const READS_PER_CORE: usize = 4;
const MIN_THREADS: usize = 8;
const MAX_THREADS: usize = 128;

impl std::fmt::Debug for IoPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoPool")
            .field("threads", &self.inner.threads.lock().len())
            .field("queued", &self.sender.len())
            .finish()
    }
}

impl Drop for PoolThreads {
    fn drop(&mut self) {
        // By the time this runs the last `IoPool` clone is going away, so the
        // senders are gone and every `recv_blocking` has returned.
        for handle in self.threads.lock().drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn test_run() {
        let pool = IoPool::new(2);

        assert_eq!(pool.run(|| 1 + 1).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_run_concurrently() {
        let pool = IoPool::new(4);
        let ran = Arc::new(AtomicUsize::new(0));

        let jobs: Vec<_> = (0..64)
            .map(|_| {
                let ran = ran.clone();
                pool.run(move || ran.fetch_add(1, Ordering::Relaxed))
            })
            .collect();
        for job in jobs {
            job.await.unwrap();
        }

        assert_eq!(ran.load(Ordering::Relaxed), 64);
    }

    #[tokio::test]
    async fn test_run_after_pool_closed() {
        let pool = IoPool::new(1);
        pool.sender.close();

        let result = pool.run(|| ()).await;

        assert!(matches!(
            result,
            Err(SlateDBError::BackgroundTaskCancelled(_))
        ));
    }
}
