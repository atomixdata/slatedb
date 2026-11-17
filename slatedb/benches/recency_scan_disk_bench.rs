// our microbenchmarks use pprof, but it doesn't work on windows
#![cfg(not(windows))]
// The bench prints the LSM shape and I/O counters next to criterion's output.
#![allow(clippy::disallowed_macros)]

//! Measures first-entry `scan_prefix_by_recency` reads through an
//! `ObjectStoreMirror` when most data block reads miss the page cache.
//!
//! The remote store is a `LocalFileSystem`, so the dataset is not held in
//! memory. Run the bench under a memory limit much smaller than the dataset,
//! so that the page cache holds only a small part of it:
//!
//! ```sh
//! cargo bench -p slatedb --features foyer,wal_disable \
//!     --bench recency_scan_disk_bench --no-run
//! systemd-run --scope -p MemoryMax=3G <bench binary> --bench
//! ```
//!
//! The bench writes `DISK_BENCH_KEYS` keys (28 million by default) in random
//! key order. Each value has between `MIN_VALUE_BYTES` and `MAX_VALUE_BYTES`
//! bytes, picked from a hash of the key, so each aligned 4 KiB data block
//! holds about 6 entries, and each copy of the dataset is about 16 GiB. The
//! embedded compactor shapes the SSTs into sorted runs, and a few L0 SSTs
//! stay. Set `TMPDIR` to put the dataset on a fast disk. Set
//! `DISK_BENCH_DIR` to keep the dataset in that directory and reuse it in
//! later runs with the same keys and value sizes.
//!
//! The bench then reopens the database once for each `Config`, with filters
//! and indexes in the meta cache and no block cache. Each scan reads the first
//! entry of the next existing key, in write order, so consecutive scans read
//! unrelated blocks and each scan reads one data block. A case with an `/xN`
//! suffix runs N scans at the same time in separate tasks.
//!
//! On Linux, the bench also prints, for each case, the bytes per scan that
//! the process read from storage and through `read` system calls.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use chrono::TimeDelta;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use pprof::criterion::{Output, PProfProfiler};
use slatedb::config::{
    CompactionWorkerOptions, CompactorOptions, FlushOptions, FlushType, PutOptions, ScanOptions,
    Settings, WriteOptions,
};
use slatedb::db_cache::foyer::{FoyerCache, FoyerCacheOptions};
use slatedb::db_cache::SplitCache;
use slatedb::{
    BloomFilterPolicy, Db, FilterPolicy, PrefixExtractor, PrefixTarget, SlateDbMirrorPolicy,
    SstBlockSize,
};
use slatedb_common::clock::{DefaultSystemClock, SystemClock};
use slatedb_mirror::{ObjectStoreMirror, StdVfs};
use tokio::runtime::Runtime;
use tokio::task::JoinSet;

const PATH: &str = "/bench/recency_disk";
/// A key is a 16-digit hex prefix followed by `:0`.
const PREFIX_LEN: usize = 16;
/// The value sizes of a typical Atomix row.
const MIN_VALUE_BYTES: usize = 100;
const MAX_VALUE_BYTES: usize = 1024;
const DEFAULT_KEYS: u64 = 28_000_000;
const SST_BYTES: usize = 64 * 1024 * 1024;
/// The numbers of scans that the cases run at the same time.
const CONCURRENCY: [usize; 4] = [1, 64, 256, 1024];
/// Scans before the first case of each `Config`, so that the filters and
/// indexes of every SST are in the meta cache.
const WARM_UP_SCANS: u64 = 20_000;
const WARM_UP_CONCURRENCY: usize = 256;

/// How the mirror runs its `pread` calls.
#[derive(Clone, Copy)]
enum Config {
    SpawnBlocking,
    /// Dedicated read threads, with this many threads.
    ReadPool(usize),
}

impl Config {
    const ALL: [Config; 4] = [
        Config::SpawnBlocking,
        Config::ReadPool(16),
        Config::ReadPool(64),
        Config::ReadPool(256),
    ];

    fn name(self) -> String {
        match self {
            Config::SpawnBlocking => "spawn_blocking".to_string(),
            Config::ReadPool(threads) => format!("read_pool_{threads}"),
        }
    }

    fn vfs(self) -> StdVfs {
        match self {
            Config::SpawnBlocking => StdVfs::new(),
            Config::ReadPool(threads) => StdVfs::new()
                .with_read_threads(threads)
                .expect("failed to start read threads"),
        }
    }
}

fn keys() -> u64 {
    std::env::var("DISK_BENCH_KEYS")
        .map(|keys| keys.parse().expect("DISK_BENCH_KEYS must be a number"))
        .unwrap_or(DEFAULT_KEYS)
}

/// Spreads consecutive IDs over the key space.
fn splitmix64(id: u64) -> u64 {
    let mut z = id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The size of the value of the key with `id`.
fn value_len(id: u64) -> usize {
    let span = (MAX_VALUE_BYTES - MIN_VALUE_BYTES + 1) as u64;
    MIN_VALUE_BYTES + (splitmix64(id ^ 0x5bd1_e995) % span) as usize
}

/// Identifies the dataset that `keys` keys build.
fn dataset_name(keys: u64) -> String {
    format!("{keys} keys, values of {MIN_VALUE_BYTES} to {MAX_VALUE_BYTES} bytes")
}

fn prefix(id: u64) -> Vec<u8> {
    format!("{:016x}", splitmix64(id)).into_bytes()
}

fn key(id: u64) -> Vec<u8> {
    let mut key = prefix(id);
    key.extend_from_slice(b":0");
    key
}

struct FixedPrefixExtractor;

impl PrefixExtractor for FixedPrefixExtractor {
    fn name(&self) -> &str {
        "fixed16"
    }

    fn prefix_len(&self, target: &PrefixTarget) -> Option<usize> {
        let input = match target {
            PrefixTarget::Point(k) => k.as_ref(),
            PrefixTarget::Prefix(p) => p.as_ref(),
        };
        (input.len() >= PREFIX_LEN).then_some(PREFIX_LEN)
    }
}

fn filter_policies() -> Vec<Arc<dyn FilterPolicy>> {
    vec![Arc::new(
        BloomFilterPolicy::new(10).with_prefix_extractor(Arc::new(FixedPrefixExtractor)),
    )]
}

/// A meta cache large enough for every filter and index, and no block cache.
fn cache() -> Arc<SplitCache> {
    let meta_cache = Arc::new(FoyerCache::new_with_opts(FoyerCacheOptions {
        max_capacity: 1024 * 1024 * 1024,
        ..Default::default()
    }));
    Arc::new(
        SplitCache::new()
            .with_block_cache(None)
            .with_meta_cache(Some(meta_cache))
            .build(),
    )
}

/// Small SSTs and few concurrent compactions keep memory low under the limit.
fn settings(compactor: bool) -> Settings {
    let compactor_options = compactor.then(|| CompactorOptions {
        max_concurrent_compactions: 2,
        worker: Some(CompactionWorkerOptions {
            max_concurrent_compactions: 2,
            max_subcompactions: 2,
            max_sst_size: SST_BYTES,
            ..CompactionWorkerOptions::default()
        }),
        ..CompactorOptions::default()
    });
    Settings {
        wal_enabled: false,
        l0_sst_size_bytes: SST_BYTES,
        max_unflushed_bytes: 4 * SST_BYTES,
        compactor_options,
        ..Settings::default()
    }
}

async fn open(store: Arc<dyn ObjectStore>, compactor: bool) -> Db {
    Db::builder(PATH, store)
        .with_settings(settings(compactor))
        .with_db_cache(cache(), 0)
        .with_filter_policies(filter_policies())
        .with_sst_block_size(SstBlockSize::Block4Kib)
        .with_sst_block_alignment(true)
        .build()
        .await
        .expect("failed to build db")
}

/// Builds a mirror over `remote` in `dir`. A dropped mirror can hold the lock
/// on `dir` for a moment, so this retries.
async fn mirror(dir: &Path, remote: Arc<dyn ObjectStore>, vfs: StdVfs) -> Arc<dyn ObjectStore> {
    // The direct mirror holds a handle to the previous mirror, which holds
    // the lock on `dir`.
    slatedb::direct_mirror::set_direct_mirror(None);
    let clock = DefaultSystemClock::new();
    let mut attempts = 0;
    loop {
        let result =
            ObjectStoreMirror::builder(dir, remote.clone(), SlateDbMirrorPolicy::builder().build())
                .with_vfs(Arc::new(vfs.clone()))
                .with_remote_scan_interval(None)
                .build()
                .await;
        match result {
            Ok(mirror) => {
                // Prototype: read mirrored blocks through concrete types.
                slatedb::direct_mirror::set_direct_mirror(Some((mirror.handle(), vfs.clone())));
                return mirror;
            }
            Err(err) if attempts < 100 => {
                attempts += 1;
                if attempts == 1 {
                    eprintln!("retrying mirror build: {err}");
                }
                clock.sleep(Duration::from_millis(100)).await;
            }
            Err(err) => panic!("failed to build mirror: {err}"),
        }
    }
}

/// Waits until compaction leaves fewer L0 SSTs than one compaction takes and
/// the manifest stops changing for 10 seconds.
async fn await_compactions(db: &Db) {
    let clock = DefaultSystemClock::new();
    let deadline = clock.now() + TimeDelta::minutes(60);
    let mut last = None;
    let mut stable_polls = 0;
    while stable_polls < 10 {
        assert!(clock.now() < deadline, "compaction did not settle");
        clock.sleep(Duration::from_secs(1)).await;
        db.refresh_manifest()
            .await
            .expect("refresh_manifest failed");
        let manifest = db.manifest();
        let shape = (manifest.l0().len(), manifest.compacted().len());
        if shape.0 < 4 && last == Some(shape) {
            stable_polls += 1;
        } else {
            stable_polls = 0;
        }
        last = Some(shape);
    }
}

fn print_shape(db: &Db) {
    let manifest = db.manifest();
    let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
    let l0: u64 = manifest.l0().iter().map(|sst| sst.estimate_size()).sum();
    eprintln!("L0: {} SSTs, {:.2} GiB", manifest.l0().len(), gib(l0));
    for (i, run) in manifest.compacted().iter().enumerate() {
        eprintln!(
            "sorted run {i}: {} SSTs, {:.2} GiB",
            run.sst_views().len(),
            gib(run.estimate_size())
        );
    }
}

/// Writes every key through a mirror in `dir/mirror`, with the remote store in
/// `dir/remote`, and returns the remote store.
async fn build(dir: &Path, keys: u64) -> Arc<dyn ObjectStore> {
    let remote_dir = dir.join("remote");
    std::fs::create_dir_all(&remote_dir).expect("failed to create remote dir");
    let remote: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(&remote_dir).expect("failed to open remote"));
    let store = mirror(&dir.join("mirror"), remote.clone(), StdVfs::new()).await;
    let db = open(store, true).await;
    let value = vec![b'v'; MAX_VALUE_BYTES];
    for id in 0..keys {
        db.put_with_options(
            &key(id),
            &value[..value_len(id)],
            &PutOptions::default(),
            &WriteOptions::default(),
        )
        .await
        .expect("put failed");
    }
    db.flush_with_options(FlushOptions {
        flush_type: FlushType::MemTable,
    })
    .await
    .expect("flush failed");
    await_compactions(&db).await;
    print_shape(&db);
    db.close().await.expect("close failed");
    remote
}

/// Reads the first entry of the key with `id`.
async fn read_first(db: &Db, id: u64) {
    let options = ScanOptions {
        cache_blocks: true,
        ..ScanOptions::default()
    };
    let mut iter = db
        .scan_prefix_by_recency_with_options(prefix(id), &options)
        .await
        .expect("scan_prefix_by_recency failed");
    let entry = iter.next_entry().await.expect("next_entry failed");
    assert!(entry.is_some(), "missing key {id}");
}

/// Runs `scans` scans of consecutive IDs, `concurrency` at a time.
async fn run_scans(db: &Db, next: &Arc<AtomicU64>, keys: u64, concurrency: usize, scans: u64) {
    let mut done = 0;
    while done < scans {
        let batch = (scans - done).min(concurrency as u64);
        let mut tasks = JoinSet::new();
        for _ in 0..batch {
            let db = db.clone();
            let next = Arc::clone(next);
            tasks.spawn(async move {
                read_first(&db, next.fetch_add(1, Ordering::Relaxed) % keys).await;
            });
        }
        while let Some(result) = tasks.join_next().await {
            result.expect("scan task failed");
        }
        done += batch;
    }
}

/// Process I/O counters from `/proc/self/io`.
#[derive(Clone, Copy, Default)]
struct IoStats {
    /// Bytes returned by `read` system calls.
    rchar: u64,
    /// `read` system calls.
    syscr: u64,
    /// Bytes read from storage.
    read_bytes: u64,
}

impl IoStats {
    fn read() -> Option<IoStats> {
        let text = std::fs::read_to_string("/proc/self/io").ok()?;
        let field = |name: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
        };
        Some(IoStats {
            rchar: field("rchar:")?,
            syscr: field("syscr:")?,
            read_bytes: field("read_bytes:")?,
        })
    }
}

fn bench_recency_scan_disk(c: &mut Criterion) {
    let runtime = Runtime::new().expect("failed to create runtime");
    let keys = keys();
    let (dir, _temp_dir) = match std::env::var("DISK_BENCH_DIR") {
        Ok(dir) => (PathBuf::from(dir), None),
        Err(_) => {
            let temp_dir = tempfile::tempdir().expect("failed to create bench dir");
            (temp_dir.path().to_path_buf(), Some(temp_dir))
        }
    };
    let built = dir.join("BUILT");
    let remote: Arc<dyn ObjectStore> = match std::fs::read_to_string(&built) {
        Ok(built_name) => {
            assert_eq!(
                built_name,
                dataset_name(keys),
                "{} has another dataset",
                dir.display()
            );
            eprintln!("reusing {} in {}", dataset_name(keys), dir.display());
            Arc::new(
                LocalFileSystem::new_with_prefix(dir.join("remote"))
                    .expect("failed to open remote"),
            )
        }
        Err(_) => {
            eprintln!("building {} in {}", dataset_name(keys), dir.display());
            let remote = runtime.block_on(build(&dir, keys));
            std::fs::write(&built, dataset_name(keys)).expect("failed to write marker");
            remote
        }
    };
    let next = Arc::new(AtomicU64::new(0));

    for config in Config::ALL {
        let store = runtime.block_on(mirror(&dir.join("mirror"), remote.clone(), config.vfs()));
        let db = runtime.block_on(open(store, false));
        runtime.block_on(run_scans(
            &db,
            &next,
            keys,
            WARM_UP_CONCURRENCY,
            WARM_UP_SCANS,
        ));

        let mut group = c.benchmark_group(format!("recency_scan_disk/{}", config.name()));
        for concurrency in CONCURRENCY {
            let name = format!("first_entry/x{concurrency}");
            let before = (IoStats::read(), next.load(Ordering::Relaxed));
            group.throughput(Throughput::Elements(concurrency as u64));
            group.bench_function(&name, |b| {
                b.to_async(&runtime)
                    .iter(|| run_scans(&db, &next, keys, concurrency, concurrency as u64));
            });
            if let (Some(start), Some(end)) = (before.0, IoStats::read()) {
                let scans = (next.load(Ordering::Relaxed) - before.1).max(1) as f64;
                eprintln!(
                    "{}/{name}: {:.0} scans, {:.2} KiB from storage per scan, \
                     {:.2} KiB and {:.2} read calls per scan",
                    config.name(),
                    scans,
                    (end.read_bytes - start.read_bytes) as f64 / 1024.0 / scans,
                    (end.rchar - start.rchar) as f64 / 1024.0 / scans,
                    (end.syscr - start.syscr) as f64 / scans,
                );
            }
        }
        group.finish();
        runtime.block_on(async { db.close().await.expect("close failed") });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Protobuf));
    targets = bench_recency_scan_disk
}

criterion_main!(benches);
