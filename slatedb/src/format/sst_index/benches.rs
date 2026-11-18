//! Helpers for the `sst_index_search` benchmark. The benchmark is a separate
//! crate, so it cannot build an index itself. These helpers build the indexes
//! and give the benchmark one point search as a closure.

use std::hint::black_box;
use std::ops::Bound;

use bytes::Bytes;
use flatbuffers::FlatBufferBuilder;

use super::SsTableIndexOwned;
use crate::flatbuffer_types::{BlockMeta, BlockMetaArgs, SsTableIndex, SsTableIndexArgs};
use crate::partitioned_keyspace::partitions_covering_range;

/// The shape of the index keys.
#[derive(Clone, Copy, Debug)]
pub enum KeyShape {
    /// `tenant-0001/users/NNNNNNNNNNNN`: decimal text after a shared prefix.
    Decimal,
    /// A 4-byte prefix and 8 random bytes, as with hashed keys.
    Hash,
    /// A 4-byte prefix and a big-endian sequence number. The prefix values are
    /// evenly spaced.
    Sequential,
}

/// The configuration of one benchmark case.
pub struct SstIndexSearchBenchConfig {
    /// The number of blocks in each index.
    pub blocks: usize,
    /// The number of indexes. Each search picks one at random. With many
    /// indexes, most searches read an index that is not in the CPU cache.
    pub indexes: usize,
    /// The shape of the keys.
    pub shape: KeyShape,
}

/// The number of lookup keys. They are in one buffer, so reading a lookup key
/// adds little to the time of a search.
const LOOKUPS: usize = 1 << 16;

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn key(shape: KeyShape, id: u64) -> Vec<u8> {
    match shape {
        KeyShape::Decimal => format!("tenant-0001/users/{id:012}").into_bytes(),
        KeyShape::Hash | KeyShape::Sequential => {
            let mut key = b"\x02ks\x00".to_vec();
            key.extend_from_slice(&id.to_be_bytes());
            key
        }
    }
}

/// Builds an index of `blocks` blocks. The first index key is empty, as
/// `compute_index_key` makes it.
fn build_index(shape: KeyShape, blocks: usize, rng: &mut XorShift) -> SsTableIndexOwned {
    let mut ids: Vec<u64> = match shape {
        KeyShape::Decimal | KeyShape::Sequential => (0..blocks as u64).map(|i| i * 100).collect(),
        KeyShape::Hash => (0..blocks).map(|_| rng.next()).collect(),
    };
    ids.sort_unstable();
    let mut builder = FlatBufferBuilder::new();
    let metas: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let first_key = if i == 0 { Vec::new() } else { key(shape, id) };
            let first_key = builder.create_vector(&first_key);
            BlockMeta::create(
                &mut builder,
                &BlockMetaArgs {
                    offset: i as u64 * 4096,
                    first_key: Some(first_key),
                    encoded_len: 0,
                },
            )
        })
        .collect();
    let block_meta = builder.create_vector(&metas);
    let index = SsTableIndex::create(
        &mut builder,
        &SsTableIndexArgs {
            block_meta: Some(block_meta),
        },
    );
    builder.finish(index, None);
    SsTableIndexOwned::new(Bytes::copy_from_slice(builder.finished_data()))
        .expect("benchmark index is valid")
}

/// Builds the indexes and the lookup keys, then calls `run_bench` with one
/// point search.
///
/// Each call of the search picks the next index and lookup key in a fixed
/// random order. It finds the blocks that can hold the key, and reads the
/// offset of the first one, as a get does. The next search waits for that
/// offset, so the time is the latency of one search, not the throughput of
/// many.
pub fn sst_index_search_bench<F>(config: SstIndexSearchBenchConfig, mut run_bench: F)
where
    F: FnMut(&mut dyn FnMut()),
{
    let mut rng = XorShift(0x51A7_EDBB_E4C4);
    let indexes: Vec<SsTableIndexOwned> = (0..config.indexes)
        .map(|_| build_index(config.shape, config.blocks, &mut rng))
        .collect();
    let max_id = match config.shape {
        KeyShape::Decimal | KeyShape::Sequential => config.blocks as u64 * 100,
        KeyShape::Hash => u64::MAX,
    };
    let key_len = key(config.shape, 0).len();
    let mut lookups = Vec::with_capacity(LOOKUPS * key_len);
    let mut order = Vec::with_capacity(LOOKUPS);
    for i in 0..LOOKUPS {
        lookups.extend_from_slice(&key(config.shape, rng.next() % max_id));
        order.push(((rng.next() % config.indexes as u64) as usize, i));
    }

    let zero = black_box(0usize);
    let mut next = 0;
    let mut previous = 0usize;
    let mut search = || {
        let (index, lookup) = order[next];
        next = (next + 1) % order.len();
        // `previous & zero` is always 0, but the compiler cannot know it. So
        // the search waits for the result of the search before it.
        let index = &indexes[index + (previous & zero)];
        let start = lookup * key_len + (previous & zero);
        let key = &lookups[start..start + key_len];
        let blocks = partitions_covering_range(index, Bound::Included(key), Bound::Included(key));
        previous = index.offset(blocks.start) as usize;
    };
    run_bench(&mut search);
    black_box(previous);
}
