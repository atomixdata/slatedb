// our microbenchmarks use pprof, but it doesn't work on windows
#![cfg(not(windows))]

// Run with: cargo bench -p slatedb --features bench-internal --bench sst_index_search
//
// Measures one point search in an SST index: the search for the blocks that can hold a key,
// and the read of the first block offset. Each search waits for the one before it, as the
// SST lookups of a get do. With one index, the index stays in the CPU cache. With 64 indexes
// of 65K blocks (about 140 to 210 MB), most searches read an index that is not in the CPU cache.

use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};
use slatedb::sst_index_benches::{sst_index_search_bench, KeyShape, SstIndexSearchBenchConfig};

#[allow(clippy::redundant_closure)]
fn criterion_benchmark(c: &mut Criterion) {
    let shapes = [
        ("decimal", KeyShape::Decimal),
        ("hash", KeyShape::Hash),
        ("sequential", KeyShape::Sequential),
    ];
    // (blocks, indexes). A compacted SST with the default size has about 65K blocks.
    let sizes = [(16_384, 1), (65_536, 1), (65_536, 64)];

    let mut group = c.benchmark_group("sst_index_search");
    for (shape_name, shape) in shapes {
        for (blocks, indexes) in sizes {
            sst_index_search_bench(
                SstIndexSearchBenchConfig {
                    blocks,
                    indexes,
                    shape,
                },
                |search| {
                    group.bench_function(
                        format!("{shape_name}/{blocks}_blocks/{indexes}_indexes"),
                        |b| b.iter(|| search()),
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        // This only runs when `--profile-time <num_seconds>` is set
        .with_profiler(PProfProfiler::new(100, Output::Protobuf));
    targets = criterion_benchmark
}

criterion_main!(benches);
