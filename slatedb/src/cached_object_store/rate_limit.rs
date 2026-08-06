//! Global pacing of SST uploads to the upstream object store.
//!
//! A single shared leaky bucket bounds the aggregate upload rate across
//! all concurrent writers (memtable flushes, every sub-compaction, and
//! each in-flight multipart part). Unpaced, concurrent compaction
//! output uploads can saturate the instance's network egress allowance,
//! causing packet loss and TCP retransmit stalls on foreground reads.

use chrono::{DateTime, Utc};
use slatedb_common::clock::SystemClock;
use std::sync::Arc;
use std::time::Duration;

/// Env var holding the aggregate upload rate limit in MiB/s. Absent,
/// unparsable, or non-positive means uploads are not limited.
pub(crate) const UPLOAD_RATE_LIMIT_ENV: &str = "SLATEDB_UPLOAD_RATE_LIMIT_MIB";

#[derive(Debug)]
pub(crate) struct UploadRateLimiter {
    bytes_per_sec: f64,
    clock: Arc<dyn SystemClock>,
    /// Virtual time when the next payload may start; claiming a payload
    /// advances it by `bytes / bytes_per_sec`.
    next_free: parking_lot::Mutex<DateTime<Utc>>,
}

impl UploadRateLimiter {
    pub(crate) fn new(bytes_per_sec: f64, clock: Arc<dyn SystemClock>) -> Arc<Self> {
        let now = clock.now();
        Arc::new(Self {
            bytes_per_sec,
            clock,
            next_free: parking_lot::Mutex::new(now),
        })
    }

    /// Build from [UPLOAD_RATE_LIMIT_ENV], or None when unlimited.
    pub(crate) fn from_env(clock: Arc<dyn SystemClock>) -> Option<Arc<Self>> {
        let mib: f64 = std::env::var(UPLOAD_RATE_LIMIT_ENV).ok()?.parse().ok()?;
        if mib <= 0.0 || !mib.is_finite() {
            return None;
        }
        Some(Self::new(mib * 1048576.0, clock))
    }

    /// Wait until `bytes` may be sent without exceeding the aggregate
    /// rate. Claims are serialized through the shared schedule, so any
    /// number of concurrent uploads together average the configured
    /// rate. An idle limiter admits immediately (no credit accrues).
    pub(crate) async fn acquire(&self, bytes: u64) {
        let cost =
            chrono::Duration::from_std(Duration::from_secs_f64(bytes as f64 / self.bytes_per_sec))
                .unwrap_or_else(|_| chrono::Duration::zero());
        let wait = {
            let mut next = self.next_free.lock();
            let now = self.clock.now();
            let start = (*next).max(now);
            *next = start + cost;
            (start - now).to_std().unwrap_or(Duration::ZERO)
        };
        if !wait.is_zero() {
            self.clock.sleep(wait).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slatedb_common::clock::DefaultSystemClock;
    use tokio::time::Instant;

    fn limiter(bytes_per_sec: f64) -> Arc<UploadRateLimiter> {
        UploadRateLimiter::new(bytes_per_sec, Arc::new(DefaultSystemClock::new()))
    }

    #[tokio::test(start_paused = true)]
    async fn test_acquire_paces_sequential_claims() {
        let limiter = limiter(1048576.0); // 1 MiB/s
        let t0 = Instant::now();
        for _ in 0..3 {
            limiter.acquire(1048576).await;
        }
        // The third claim starts once the first two have used 2s of
        // budget; the idle first claim starts immediately.
        assert_eq!(t0.elapsed().as_secs(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn test_acquire_bounds_concurrent_uploads() {
        let limiter = limiter(10.0 * 1048576.0); // 10 MiB/s
        let t0 = Instant::now();
        let tasks: Vec<_> = (0..5)
            .map(|_| {
                let l = limiter.clone();
                tokio::spawn(async move { l.acquire(10 * 1048576).await })
            })
            .collect();
        for t in tasks {
            t.await.unwrap();
        }
        // 5 x 10 MiB at 10 MiB/s: the last claim is scheduled 4s in.
        assert_eq!(t0.elapsed().as_secs(), 4);
    }

    #[tokio::test]
    async fn test_acquire_idle_admits_immediately() {
        let limiter = limiter(1048576.0);
        // The limiter starts idle, so a first small claim must not wait
        // (and must not bank credit for later claims beyond one cost).
        let t0 = Instant::now();
        limiter.acquire(1024).await;
        assert!(t0.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn test_from_env() {
        let clock: Arc<dyn SystemClock> = Arc::new(DefaultSystemClock::new());
        // Unset or invalid values disable the limiter.
        std::env::remove_var(UPLOAD_RATE_LIMIT_ENV);
        assert!(UploadRateLimiter::from_env(clock.clone()).is_none());
        std::env::set_var(UPLOAD_RATE_LIMIT_ENV, "0");
        assert!(UploadRateLimiter::from_env(clock.clone()).is_none());
        std::env::set_var(UPLOAD_RATE_LIMIT_ENV, "junk");
        assert!(UploadRateLimiter::from_env(clock.clone()).is_none());
        std::env::set_var(UPLOAD_RATE_LIMIT_ENV, "500");
        let l = UploadRateLimiter::from_env(clock).expect("500 MiB/s should parse");
        assert_eq!(l.bytes_per_sec, 500.0 * 1048576.0);
        std::env::remove_var(UPLOAD_RATE_LIMIT_ENV);
    }
}
