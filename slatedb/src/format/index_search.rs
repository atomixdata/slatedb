//! Prototype of RFC 0038 (slatedb PR #2254): a faster search of the SST index.
//!
//! When an index is loaded, it is copied into flat arrays, with one numeric
//! prefix value for each block and a static B+ tree over these values. Most
//! search steps then compare two `u64` values, and only blocks with the same
//! prefix value as the key need a full key compare.

use std::ops::Range;

use crate::flatbuffer_types::SsTableIndex;
use crate::partitioned_keyspace::RangePartitionedKeySpace;

/// The values in one tree node. A node fills one 64-byte cache line.
const NODE_VALUES: usize = 8;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Node([u64; NODE_VALUES]);

/// The first keys of an index, in flat arrays, with optional prefix values.
pub(crate) struct IndexSearch {
    /// All first keys, one after the other.
    keys: Vec<u8>,
    /// The end of each key in `keys`.
    key_ends: Vec<u32>,
    prefixes: Option<PrefixValues>,
}

/// One prefix value for each block, in the leaf level of a static B+ tree.
struct PrefixValues {
    /// The prefix that every key at or after `first` starts with.
    shared: Vec<u8>,
    /// 1 if the first key is empty and has no shared prefix, else 0.
    first: usize,
    /// The leaf level first. Each upper level holds the first value of each
    /// node of the level below. `u64::MAX` pads each level to whole nodes.
    levels: Vec<Vec<Node>>,
    len: usize,
}

/// The 8 key bytes after the shared prefix, padded with zeros, as a
/// big-endian number. A smaller value means a smaller key.
fn prefix_value(suffix: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    let len = suffix.len().min(8);
    bytes[..len].copy_from_slice(&suffix[..len]);
    u64::from_be_bytes(bytes)
}

impl PrefixValues {
    fn build(search: &IndexSearch) -> Option<Self> {
        let len = search.len();
        if len < 2 {
            return None;
        }
        let first = usize::from(search.key(0).is_empty());
        if search.key(first).is_empty() {
            return None;
        }
        // The keys are sorted, so the common prefix of the first non-empty key
        // and the last key is a prefix of every key between them.
        let (low, high) = (search.key(first), search.key(len - 1));
        let shared_len = low.iter().zip(high).take_while(|(a, b)| a == b).count();
        let shared = low[..shared_len].to_vec();
        let values: Vec<u64> = (0..len)
            .map(|i| {
                if i < first {
                    0
                } else {
                    prefix_value(&search.key(i)[shared_len..])
                }
            })
            .collect();
        // With many equal values, most searches need full compares anyway.
        let distinct = 1 + values.windows(2).filter(|w| w[0] != w[1]).count();
        if distinct * 2 < len {
            return None;
        }
        let mut levels = vec![Self::nodes(&values)];
        while levels.last().expect("at least one level").len() > 1 {
            let below = levels.last().expect("at least one level");
            let firsts: Vec<u64> = below.iter().map(|node| node.0[0]).collect();
            levels.push(Self::nodes(&firsts));
        }
        Some(Self {
            shared,
            first,
            levels,
            len,
        })
    }

    fn nodes(values: &[u64]) -> Vec<Node> {
        values
            .chunks(NODE_VALUES)
            .map(|chunk| {
                let mut node = Node([u64::MAX; NODE_VALUES]);
                node.0[..chunk.len()].copy_from_slice(chunk);
                node
            })
            .collect()
    }

    fn value(&self, i: usize) -> u64 {
        self.levels[0][i / NODE_VALUES].0[i % NODE_VALUES]
    }

    /// The number of values at or below `value`.
    fn count_le(&self, value: u64) -> usize {
        let mut node = 0;
        for (depth, level) in self.levels.iter().enumerate().rev() {
            let node_index = node.min(level.len() - 1);
            let count = level[node_index]
                .0
                .iter()
                .filter(|&&candidate| candidate <= value)
                .count();
            if depth == 0 {
                return (node_index * NODE_VALUES + count).min(self.len);
            }
            if count == 0 {
                // Only the root can be fully above the value.
                return 0;
            }
            node = node_index * NODE_VALUES + count - 1;
        }
        unreachable!("the leaf level returns")
    }
}

impl IndexSearch {
    pub(crate) fn new(index: &SsTableIndex<'_>) -> Self {
        let metas = index.block_meta();
        let mut keys = Vec::new();
        let mut key_ends = Vec::with_capacity(metas.len());
        for meta in metas.iter() {
            keys.extend_from_slice(meta.first_key().bytes());
            key_ends.push(u32::try_from(keys.len()).expect("index keys under 4 GiB"));
        }
        let mut search = Self {
            keys,
            key_ends,
            prefixes: None,
        };
        search.prefixes = PrefixValues::build(&search);
        search
    }

    pub(crate) fn size(&self) -> usize {
        let prefixes = self.prefixes.as_ref().map_or(0, |p| {
            p.levels
                .iter()
                .map(|level| level.len() * size_of::<Node>())
                .sum::<usize>()
        });
        self.keys.len() + self.key_ends.len() * 4 + prefixes
    }

    fn len(&self) -> usize {
        self.key_ends.len()
    }

    fn key(&self, i: usize) -> &[u8] {
        let start = if i == 0 {
            0
        } else {
            self.key_ends[i - 1] as usize
        };
        &self.keys[start..self.key_ends[i] as usize]
    }

    /// The blocks whose first key can compare equal to `key`. Every block
    /// before the range has a first key below `key`, and every block after it
    /// has a first key above `key`.
    fn candidates(&self, key: &[u8]) -> Range<usize> {
        let len = self.len();
        let Some(prefixes) = &self.prefixes else {
            return 0..len;
        };
        if !key.starts_with(&prefixes.shared) {
            return if key > prefixes.shared.as_slice() {
                len..len
            } else {
                0..prefixes.first
            };
        }
        let value = prefix_value(&key[prefixes.shared.len()..]);
        let end = prefixes.count_le(value);
        if end == 0 || prefixes.value(end - 1) != value {
            return end..end;
        }
        let mut start = end - 1;
        while start > 0 && prefixes.value(start - 1) == value {
            start -= 1;
        }
        start..end
    }

    /// The number of blocks in `range` before the first block whose first key
    /// fails `pred`, plus `range.start`.
    fn partition_in(&self, range: Range<usize>, pred: impl Fn(&[u8]) -> bool) -> usize {
        let (mut low, mut high) = (range.start, range.end);
        while low < high {
            let mid = low + (high - low) / 2;
            if pred(self.key(mid)) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }
}

impl RangePartitionedKeySpace for &IndexSearch {
    fn partitions(&self) -> usize {
        self.len()
    }

    fn partition_first_key(&self, partition: usize) -> &[u8] {
        self.key(partition)
    }

    fn partition_point_le(&self, key: &[u8]) -> usize {
        self.partition_in(self.candidates(key), |first_key| first_key <= key)
    }

    fn partition_point_lt(&self, key: &[u8]) -> usize {
        self.partition_in(self.candidates(key), |first_key| first_key < key)
    }

    fn partitions_covering_key(&self, key: &[u8]) -> Range<usize> {
        let candidates = self.candidates(key);
        let end = self.partition_in(candidates.clone(), |first_key| first_key <= key);
        let first_equal = self.partition_in(candidates.start..end, |first_key| first_key < key);
        first_equal.saturating_sub(1)..end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioned_keyspace::partition_point;
    use proptest::collection::vec;
    use proptest::{prop_assert_eq, proptest};

    fn index_bytes(keys: &[Vec<u8>]) -> bytes::Bytes {
        use crate::flatbuffer_types::{BlockMeta, BlockMetaArgs, SsTableIndexArgs};
        let mut builder = flatbuffers::FlatBufferBuilder::new();
        let metas: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, key)| {
                let first_key = builder.create_vector(key);
                BlockMeta::create(
                    &mut builder,
                    &BlockMetaArgs {
                        offset: i as u64 * 4096,
                        first_key: Some(first_key),
                        ..Default::default()
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
        bytes::Bytes::copy_from_slice(builder.finished_data())
    }

    proptest! {
        #[test]
        fn should_match_binary_search(
            mut keys in vec(vec(proptest::sample::select(vec![0u8, 1, 0x7f, 0xfe, 0xff]), 0..12), 1..300),
            probes in vec(vec(proptest::sample::select(vec![0u8, 1, 0x7f, 0xfe, 0xff]), 0..12), 1..50),
            shared in vec(proptest::sample::select(vec![b'a', b'b']), 0..10),
            empty_first in proptest::bool::ANY,
        ) {
            for key in keys.iter_mut() {
                key.splice(0..0, shared.iter().copied());
            }
            keys.sort();
            if empty_first {
                keys[0].clear();
            }
            let bytes = index_bytes(&keys);
            let owned = crate::flatbuffer_types::SsTableIndexOwned::new(bytes).unwrap();
            let search = IndexSearch::new(&owned.borrow());
            let fast = &search;
            let flat = owned.borrow();
            for mut probe in probes {
                probe.splice(0..0, shared.iter().copied());
                for probe in [probe.clone(), probe[..probe.len() / 2].to_vec(), vec![0xff; 3]] {
                    prop_assert_eq!(
                        fast.partition_point_le(&probe),
                        partition_point(&flat, |k| k <= probe.as_slice())
                    );
                    prop_assert_eq!(
                        fast.partition_point_lt(&probe),
                        partition_point(&flat, |k| k < probe.as_slice())
                    );
                    prop_assert_eq!(
                        fast.partitions_covering_key(&probe),
                        flat.partitions_covering_key(&probe)
                    );
                }
            }
        }
    }
}
