//! Measure durable writes with a fixed delay on the dedicated WAL store.
//!
//! Run with:
//! `cargo run -p slatedb --release --example wal_latency --no-default-features`
//!
//! Set `WAL_MAX_INFLIGHT_UPLOADS` above 1 for concurrent uploads.

// This standalone benchmark uses real time and prints JSON results.
#![allow(clippy::disallowed_methods, clippy::disallowed_macros)]

use bytes::Bytes;
use object_store::{memory::InMemory, throttle::ThrottledStore};
use slatedb::{Db, Settings};
use std::{error::Error, sync::Arc, time::Duration};
use tokio::{task::JoinSet, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn number(name: &str, default: u32) -> Result<u32> {
    match std::env::var(name) {
        Ok(value) => Ok(value
            .parse()
            .map_err(|_| format!("{name} must be an integer"))?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

async fn measure(
    db: &Arc<Db>,
    count: u32,
    key_offset: u32,
    arrival: Duration,
    max_inflight: usize,
    value: Bytes,
) -> Result<(Vec<Duration>, bool)> {
    let started = Instant::now();
    let mut tasks = JoinSet::new();
    let mut samples = Vec::with_capacity(count as usize);
    let mut capacity_limited = false;
    for id in 0..count {
        let scheduled = started + arrival * id;
        tokio::time::sleep_until(scheduled).await;
        while let Some(result) = tasks.try_join_next() {
            samples.push(result??);
        }
        // Limit outstanding writes while keeping the original arrival schedule.
        if tasks.len() >= max_inflight {
            capacity_limited = true;
            samples.push(tasks.join_next().await.expect("pending write")??);
        }
        let db = db.clone();
        let value = value.clone();
        tasks.spawn(async move {
            let key = (u64::from(key_offset) + u64::from(id)).to_be_bytes();
            let submitted = Instant::now();
            let handle = db.put(key, value).await?;
            handle.await_durable().await?;
            Ok::<_, slatedb::Error>(submitted.elapsed())
        });
    }
    while let Some(result) = tasks.join_next().await {
        samples.push(result??);
    }
    Ok((samples, capacity_limited))
}

fn distribution(values: impl Iterator<Item = Duration>) -> serde_json::Value {
    let mut ms: Vec<_> = values.map(|v| v.as_secs_f64() * 1000.0).collect();
    ms.sort_by(f64::total_cmp);
    let percentile = |p: usize| ms[(ms.len() * p).div_ceil(100).saturating_sub(1)];
    serde_json::json!({
        "min": ms[0], "p50": percentile(50), "p95": percentile(95),
        "p99": percentile(99), "max": ms[ms.len() - 1],
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let delay_ms = number("WAL_UPLOAD_DELAY_MS", 20)?;
    let flush_ms = number("WAL_FLUSH_INTERVAL_MS", 1)?;
    let arrival_ms = number("WAL_ARRIVAL_INTERVAL_MS", 1)?;
    let count = number("WAL_WRITES", 5000)?;
    let warmup = number("WAL_WARMUP_WRITES", 500)?;
    let max_inflight = number("WAL_MAX_INFLIGHT", 256)?;
    let value_bytes = number("WAL_VALUE_BYTES", 1024)?;
    let max_inflight_uploads = number("WAL_MAX_INFLIGHT_UPLOADS", 1)?;
    if flush_ms == 0
        || count == 0
        || max_inflight == 0
        || value_bytes == 0
        || max_inflight_uploads == 0
    {
        return Err(
            "flush interval, writes, max inflight, value bytes, and max inflight uploads must be positive"
                .into(),
        );
    }
    let wal = Arc::new(ThrottledStore::new(InMemory::new(), Default::default()));
    let db = Arc::new(
        Db::builder("wal-latency", Arc::new(InMemory::new()))
            .with_wal_object_store(wal.clone())
            .with_settings(Settings {
                flush_interval: Some(Duration::from_millis(flush_ms.into())),
                max_inflight_wal_uploads: max_inflight_uploads as usize,
                compactor_options: None,
                garbage_collector_options: None,
                ..Default::default()
            })
            .build()
            .await?,
    );
    // Startup fencing is outside the measurement. Main-store writes remain
    // undelayed.
    wal.config_mut(|config| config.wait_put_per_call = Duration::from_millis(delay_ms.into()));
    let arrival = Duration::from_millis(arrival_ms.into());
    let value = Bytes::from(vec![0; value_bytes as usize]);
    measure(
        &db,
        warmup,
        0,
        arrival,
        max_inflight as usize,
        value.clone(),
    )
    .await?;
    let (samples, capacity_limited) =
        measure(&db, count, warmup, arrival, max_inflight as usize, value).await?;
    if capacity_limited {
        eprintln!(
            "Warning: WAL_MAX_INFLIGHT delayed scheduled writes. \
             Arrival pacing changed, and durable_latency_ms excludes the wait for task capacity."
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "upload_delay_ms": delay_ms, "flush_interval_ms": flush_ms,
            "max_inflight_wal_uploads": max_inflight_uploads,
            "arrival_interval_ms": arrival_ms, "writes": count, "warmup_writes": warmup,
            "max_inflight": max_inflight, "value_bytes": value_bytes,
            "durable_latency_ms": distribution(samples.iter().copied()),
        }))?
    );
    if let Ok(path) = std::env::var("WAL_SAMPLES_PATH") {
        let mut csv = String::from("durable_latency_ms\n");
        for sample in &samples {
            csv.push_str(&format!("{}\n", sample.as_secs_f64() * 1000.0));
        }
        std::fs::write(path, csv)?;
    }
    db.close().await?;
    Ok(())
}
