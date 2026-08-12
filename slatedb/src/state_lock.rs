// Real monotonic time by design: these counters diagnose actual lock
// wait/hold durations in production, which a mockable clock must not
// distort, and the measurements happen inside synchronous lock paths
// where the async clock is unavailable.
#![allow(clippy::disallowed_types)]

use crate::db_state::DbState;
use crate::db_stats::DbStats;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use slatedb_common::metrics::CounterFn;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::{Duration, Instant};

const OVER_1MS: Duration = Duration::from_millis(1);
const OVER_10MS: Duration = Duration::from_millis(10);

/// The database state lock, instrumented to attribute request latency
/// spikes.
///
/// parking_lot's `RwLock` is write-preferring: one queued writer blocks
/// every newly arriving reader, so a writer that waits (or holds) for
/// milliseconds stalls the whole read path. These counters split that
/// into read waits, write waits, and write holds past 1ms/10ms, which
/// tells us whether spikes come from writers holding too long or from
/// writers themselves queueing behind reader bursts.
///
/// The uncontended read path is a bare `try_read` (same cost as
/// `read`), so the hot path takes no timing overhead.
pub(crate) struct StateLock {
    lock: RwLock<DbState>,
    read_wait_over_1ms: Arc<dyn CounterFn>,
    read_wait_over_10ms: Arc<dyn CounterFn>,
    write_wait_over_1ms: Arc<dyn CounterFn>,
    write_wait_over_10ms: Arc<dyn CounterFn>,
    write_hold_over_1ms: Arc<dyn CounterFn>,
    write_hold_over_10ms: Arc<dyn CounterFn>,
}

impl std::fmt::Debug for StateLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateLock").finish_non_exhaustive()
    }
}

impl StateLock {
    pub(crate) fn new(state: DbState, stats: &DbStats) -> Self {
        Self {
            lock: RwLock::new(state),
            read_wait_over_1ms: stats.state_lock_read_wait_over_1ms.clone(),
            read_wait_over_10ms: stats.state_lock_read_wait_over_10ms.clone(),
            write_wait_over_1ms: stats.state_lock_write_wait_over_1ms.clone(),
            write_wait_over_10ms: stats.state_lock_write_wait_over_10ms.clone(),
            write_hold_over_1ms: stats.state_lock_write_hold_over_1ms.clone(),
            write_hold_over_10ms: stats.state_lock_write_hold_over_10ms.clone(),
        }
    }

    pub(crate) fn read(&self) -> RwLockReadGuard<'_, DbState> {
        // Fast path: succeeds exactly when no writer holds or waits.
        if let Some(guard) = self.lock.try_read() {
            return guard;
        }
        let start = Instant::now();
        let guard = self.lock.read();
        let waited = start.elapsed();
        if waited >= OVER_1MS {
            self.read_wait_over_1ms.increment(1);
            if waited >= OVER_10MS {
                self.read_wait_over_10ms.increment(1);
            }
        }
        guard
    }

    pub(crate) fn write(&self) -> StateWriteGuard<'_> {
        let start = Instant::now();
        let guard = self.lock.write();
        let waited = start.elapsed();
        if waited >= OVER_1MS {
            self.write_wait_over_1ms.increment(1);
            if waited >= OVER_10MS {
                self.write_wait_over_10ms.increment(1);
            }
        }
        StateWriteGuard {
            guard,
            acquired_at: Instant::now(),
            lock: self,
        }
    }
}

pub(crate) struct StateWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, DbState>,
    acquired_at: Instant,
    lock: &'a StateLock,
}

impl Deref for StateWriteGuard<'_> {
    type Target = DbState;

    fn deref(&self) -> &DbState {
        &self.guard
    }
}

impl DerefMut for StateWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut DbState {
        &mut self.guard
    }
}

impl Drop for StateWriteGuard<'_> {
    fn drop(&mut self) {
        let held = self.acquired_at.elapsed();
        if held >= OVER_1MS {
            self.lock.write_hold_over_1ms.increment(1);
            if held >= OVER_10MS {
                self.lock.write_hold_over_10ms.increment(1);
            }
        }
    }
}
