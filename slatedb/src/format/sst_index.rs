//! The in-memory form of an SST index, and the search over it.
//!
//! An index has one entry for each data block: the block offset, the block
//! length, and the index key. An index key is a lower bound for the keys in
//! its block. The first index key is empty. [`SsTableIndexOwned::new`]
//! verifies the `SsTableIndex` FlatBuffer and copies it into three parts:
//!
//! 1. Flat arrays. Each field has its own array. All index keys are in one
//!    buffer, with an array of end positions. A search reads only the keys and
//!    their end positions.
//! 2. Prefix values. The shared prefix is the prefix that all index keys but
//!    the first share. The prefix value of a key is the 8 bytes after the
//!    shared prefix, padded with zeros and read as a big-endian `u64`. If one
//!    key is smaller than another, its prefix value is smaller or equal.
//! 3. A static B+ tree over the prefix values. Each node is one cache line,
//!    and the target architecture sets the node size at compile time. Each
//!    upper level holds the first value of each node in the level below.
//!
//! A search first finds the candidate blocks: the blocks whose prefix value
//! equals the prefix value of the key. To find them, the search reads one tree
//! node for each level. Only the candidates need a full key compare.
//!
//! The index builds the prefix values and the tree only if it has at least 2
//! blocks, an empty first key, and no other empty key. At least half of the
//! prefix values must also be distinct. Otherwise, a search is a binary search
//! over the full keys.

use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::{FlatBufferBuilder, InvalidFlatbuffer};

use crate::flatbuffer_types::{
    verifier_options, BlockMeta, BlockMetaArgs, SsTableIndex, SsTableIndexArgs,
};
use crate::partitioned_keyspace::RangePartitionedKeySpace;

/// The decoded index of an SST: one entry for each data block, in block order.
#[derive(PartialEq, Eq, Clone)]
pub(crate) struct SsTableIndexOwned {
    inner: Arc<IndexArrays>,
}

/// The flat arrays of an index, with one array for each field of a block.
/// Block `i` is at position `i` in every array.
///
/// The arrays never change after [`SsTableIndexOwned::new`] builds them.
#[derive(PartialEq, Eq)]
struct IndexArrays {
    /// The offset of each block in the SST.
    offsets: Box<[u64]>,
    /// The encoded length of each block. `None` when every length is zero.
    encoded_lens: Option<Box<[u32]>>,
    /// The first key of each block, one after the other.
    keys: Box<[u8]>,
    /// The end of each key in `keys`. Key `i` starts at the end of key `i - 1`.
    key_ends: Box<[u32]>,
    /// The prefix array, or `None` when it does not help the search.
    prefixes: Option<PrefixArray>,
}

/// The prefix values of an index.
///
/// All index keys but the first start with the same bytes, the shared prefix.
/// The bytes after the shared prefix tell the keys apart. The prefix value of
/// a key is the first 8 of those bytes, padded with zeros and read as a
/// big-endian `u64`. The prefix value of a block is the prefix value of its
/// first key.
#[derive(PartialEq, Eq)]
struct PrefixArray {
    /// The length of the shared prefix that all keys but the first share. The
    /// shared prefix can be read from the start of the last key in `keys`.
    shared_len: usize,
    /// A copy of the shared prefix, if it is at most `INLINE_PREFIX_LEN` bytes
    /// long.
    shared_inline: [u8; INLINE_PREFIX_LEN],
    /// A B+ tree over the prefix value of each block: the 8 key bytes after
    /// the shared prefix, padded with zeros and read as big-endian.
    btree: StaticBTree,
}

/// The size of a cache line on the target, in bytes.
///
/// Apple M-series CPUs have 128-byte lines. x86 and other ARM CPUs, such as
/// AWS Graviton, have 64-byte lines. Other targets use 64.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
pub(crate) const CACHE_LINE: usize = 128;
#[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
pub(crate) const CACHE_LINE: usize = 64;

/// The number of values in one node of a [`StaticBTree`].
const NODE_LEN: usize = CACHE_LINE / size_of::<u64>();

/// One node of a [`StaticBTree`]: `NODE_LEN` values in one cache line.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    all(target_arch = "aarch64", target_vendor = "apple"),
    repr(C, align(128))
)]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_vendor = "apple")),
    repr(C, align(64))
)]
struct Node([u64; NODE_LEN]);

const _: () = assert!(size_of::<Node>() == CACHE_LINE && align_of::<Node>() == CACHE_LINE);

impl Node {
    /// Returns the number of values in the node that are at or below `target`.
    #[inline]
    fn count_le(&self, target: u64) -> usize {
        self.0.iter().filter(|&&v| v <= target).count()
    }
}

/// A B+ tree over sorted values.
///
/// It is static: it is built one time and never changes. It knows nothing
/// about keys; [`PrefixArray`] maps keys to its values.
#[derive(PartialEq, Eq)]
struct StaticBTree {
    /// The number of values.
    len: usize,
    /// The levels of the tree, from the leaves to the root.
    ///
    /// The leaf level, `levels[0]`, holds the values. Each next level holds
    /// the first value of each node of the level before it. All levels are
    /// padded with `u64::MAX` to whole nodes. The topmost level has one node.
    levels: Box<[Box<[Node]>]>,
}

impl StaticBTree {
    /// Builds the tree over sorted `values`.
    fn new(values: &[u64]) -> Self {
        let mut levels = vec![to_nodes(values.iter().copied())];
        while levels[levels.len() - 1].len() > 1 {
            let below = &levels[levels.len() - 1];
            let next = to_nodes(below.iter().map(|node| node.0[0]));
            levels.push(next);
        }
        Self {
            len: values.len(),
            levels: levels.into_boxed_slice(),
        }
    }

    /// Returns the value at index `i`.
    fn value(&self, i: usize) -> u64 {
        self.levels[0][i / NODE_LEN].0[i % NODE_LEN]
    }

    /// Returns the number of values at or below `target`.
    fn count_le(&self, target: u64) -> usize {
        self.descend(self.levels.len() - 1, 0, target)
    }

    /// Descends the tree from node `node` at level `level` and counts the
    /// values at or below `target`.
    fn descend(&self, level: usize, mut node: usize, target: u64) -> usize {
        for level in (1..=level).rev() {
            // Each value in this node is the first value of one child.
            // Count the number of children that start at or below `target`.
            let count = self.levels[level][node].count_le(target);
            // Go to the last of those children. A point search picks a block
            // in the same way. If no child starts at or below `target`, go to
            // the first child. Then limit the child index to the last node in
            // the level below.
            node = (node * NODE_LEN + count)
                .saturating_sub(1)
                .min(self.levels[level - 1].len() - 1);
        }
        // `node` is now a leaf. The values in the leaves before it are all at
        // or below `target`, so add their number to the count in this leaf.
        // The padding slots of the last leaf can add to the count if `target`
        // is `u64::MAX`, so limit the result to the number of values.
        (node * NODE_LEN + self.levels[0][node].count_le(target)).min(self.len)
    }

    /// Returns the memory of the nodes, in bytes.
    fn size(&self) -> usize {
        self.levels
            .iter()
            .map(|l| l.len() * size_of::<Node>())
            .sum()
    }
}

/// Packs `values` into nodes, and pads the last node with `u64::MAX`.
fn to_nodes(values: impl ExactSizeIterator<Item = u64>) -> Box<[Node]> {
    let mut nodes = vec![Node([u64::MAX; NODE_LEN]); values.len().div_ceil(NODE_LEN).max(1)];
    for (i, v) in values.enumerate() {
        nodes[i / NODE_LEN].0[i % NODE_LEN] = v;
    }
    nodes.into_boxed_slice()
}

/// The longest shared prefix that [`PrefixArray`] keeps a copy of.
const INLINE_PREFIX_LEN: usize = 16;

/// The longest run of equal prefix values that the search scans one by one.
const MAX_LINEAR_STEP_BACK: usize = 8;

/// The index keeps the prefix array only if at least one in
/// `MIN_DISTINCT_DENOMINATOR` prefix values is distinct. With 2, at least half
/// of the values must be distinct.
const MIN_DISTINCT_DENOMINATOR: usize = 2;

impl SsTableIndexOwned {
    /// Verifies an `SsTableIndex` FlatBuffer, and copies it into arrays.
    ///
    /// The verifier does not check that the keys are sorted. If the keys are
    /// not sorted, a search can return a wrong block, but it does not panic
    /// or read out of bounds.
    pub(crate) fn new(data: Bytes) -> Result<Self, InvalidFlatbuffer> {
        let index = flatbuffers::root_with_opts::<SsTableIndex>(&verifier_options(), &data)?;
        let arrays = decode_index(index)?;
        Ok(Self {
            inner: Arc::new(arrays),
        })
    }

    /// Returns the number of data blocks.
    pub(crate) fn len(&self) -> usize {
        self.inner.offsets.len()
    }

    /// Returns whether the index has no data blocks.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the offset of block `block` in the SST.
    pub(crate) fn offset(&self, block: usize) -> u64 {
        self.inner.offsets[block]
    }

    /// Returns the first key of block `block`.
    pub(crate) fn first_key(&self, block: usize) -> &[u8] {
        self.inner.key(block)
    }

    /// Returns the encoded length of `block`, or zero if the SST does not
    /// store it.
    pub(crate) fn encoded_len(&self, block: usize) -> u32 {
        match &self.inner.encoded_lens {
            Some(lens) => lens[block],
            None => 0,
        }
    }

    /// Encodes the index as an `SsTableIndex` FlatBuffer.
    pub(crate) fn data(&self) -> Bytes {
        let mut builder = FlatBufferBuilder::new();
        let block_metas: Vec<_> = (0..self.len())
            .map(|block| {
                let first_key = builder.create_vector(self.first_key(block));
                BlockMeta::create(
                    &mut builder,
                    &BlockMetaArgs {
                        offset: self.offset(block),
                        first_key: Some(first_key),
                        encoded_len: self.encoded_len(block),
                    },
                )
            })
            .collect();
        let block_meta = builder.create_vector(&block_metas);
        let index = SsTableIndex::create(
            &mut builder,
            &SsTableIndexArgs {
                block_meta: Some(block_meta),
            },
        );
        builder.finish(index, None);
        Bytes::copy_from_slice(builder.finished_data())
    }

    /// Returns a clone. The arrays have exact sizes, so they need no copy.
    pub(crate) fn clamp_allocated_size(&self) -> Self {
        self.clone()
    }

    /// Returns whether `self` and `other` share the same arrays.
    #[cfg(test)]
    pub(crate) fn shares_arrays_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Returns the memory that the index uses, in bytes.
    pub(crate) fn size(&self) -> usize {
        let a = &self.inner;
        size_of::<IndexArrays>()
            + a.offsets.len() * size_of::<u64>()
            + a.encoded_lens
                .as_ref()
                .map_or(0, |l| l.len() * size_of::<u32>())
            + a.keys.len()
            + a.key_ends.len() * size_of::<u32>()
            + a.prefixes.as_ref().map_or(0, |p| p.btree.size())
    }

    /// Returns the number of blocks whose first key is at or below `key`. If
    /// `inclusive` is false, returns the number of blocks whose first key is
    /// below `key`.
    ///
    /// The result is the position where `key` would be inserted, as in
    /// `slice::partition_point`. The block that can hold `key` is the block
    /// before it.
    fn partition_point(&self, key: &[u8], inclusive: bool) -> usize {
        let a = &self.inner;
        let candidates = a.candidates(key);
        partition_point_by_predicate(candidates.start, candidates.end, |block| {
            let k = a.key(block);
            if inclusive {
                k <= key
            } else {
                k < key
            }
        })
    }
}

/// Returns the first position in `low..high` for which `pred` is false. `pred`
/// must be true for a contiguous start of the range.
fn partition_point_by_predicate(
    mut low: usize,
    mut high: usize,
    pred: impl Fn(usize) -> bool,
) -> usize {
    while low < high {
        let mid = low + (high - low) / 2;
        if pred(mid) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

impl IndexArrays {
    /// Returns the candidate blocks for `key`: the blocks whose prefix value
    /// equals the prefix value of `key`. The returned candidates are the
    /// blocks that need a full key compare to decide whether they can hold
    /// `key`.
    ///
    /// Every block before the range has a first key below `key`. Every block
    /// after the range has a first key above `key`. The range is empty if no
    /// block has the same prefix value.
    ///
    /// If `key` does not start with the shared prefix, the shared prefix alone
    /// decides the order. The range is then empty, or it holds only the first
    /// block, whose key is empty. If the index has no prefix array, the range
    /// holds all blocks.
    fn candidates(&self, key: &[u8]) -> Range<usize> {
        let n = self.offsets.len();
        let Some(prefixes) = &self.prefixes else {
            return 0..n;
        };
        let shared = if prefixes.shared_len <= INLINE_PREFIX_LEN {
            &prefixes.shared_inline[..prefixes.shared_len]
        } else {
            &self.key(n - 1)[..prefixes.shared_len]
        };
        if !key.starts_with(shared) {
            // Every key but the first starts with `shared`, and the first key
            // is empty. If `key` is above `shared`, it is above every key.
            // Otherwise it is below every key but the first. An empty `key`
            // equals the first key, so it needs the full compare.
            if key > shared {
                return n..n;
            }
            if key.is_empty() {
                return 0..1;
            }
            return 1..1;
        }
        let target = head(key, prefixes.shared_len);
        let btree = &prefixes.btree;
        // The values are in key order. A smaller value means a smaller key, and
        // a larger value means a larger key. Only the keys with an equal value
        // need a full compare.
        let end = btree.count_le(target);
        if end == 0 || btree.value(end - 1) != target {
            return end..end;
        }
        // Step back over the run of equal values. With distinct values, the run
        // is one block long, so the loop reads one more value and stops.
        let mut start = end - 1;
        let mut steps = 0;
        while start > 0 && btree.value(start - 1) == target {
            steps += 1;
            if steps > MAX_LINEAR_STEP_BACK {
                start = partition_point_by_predicate(0, start, |i| btree.value(i) < target);
                break;
            }
            start -= 1;
        }
        start..end
    }

    fn key(&self, block: usize) -> &[u8] {
        let start = if block == 0 {
            0
        } else {
            self.key_ends[block - 1] as usize
        };
        &self.keys[start..self.key_ends[block] as usize]
    }
}

impl RangePartitionedKeySpace for SsTableIndexOwned {
    fn partitions(&self) -> usize {
        self.len()
    }

    fn partition_first_key(&self, partition: usize) -> &[u8] {
        self.first_key(partition)
    }

    fn partition_point_le(&self, key: &[u8]) -> usize {
        self.partition_point(key, true)
    }

    fn partition_point_lt(&self, key: &[u8]) -> usize {
        self.partition_point(key, false)
    }

    fn partitions_covering_key(&self, key: &[u8]) -> Range<usize> {
        let a = &self.inner;
        let candidates = a.candidates(key);
        // Blocks before `first_equal` start below `key`. The blocks from
        // `first_equal` to `end` start with `key`. If the candidates are empty,
        // no block starts with `key`, and no full key compare happens.
        let first_equal = partition_point_by_predicate(candidates.start, candidates.end, |block| {
            a.key(block) < key
        });
        let mut end = first_equal;
        while end < candidates.end && a.key(end) == key {
            end += 1;
        }
        // The block before the first candidate starting at `key` may still end
        // with `key`.
        first_equal.saturating_sub(1)..end
    }
}

/// Returns the 8 bytes of `key` after its first `skip` bytes, padded with
/// zeros, as a big-endian `u64`.
fn head(key: &[u8], skip: usize) -> u64 {
    let rest = &key[skip..];
    let len = rest.len().min(8);
    let mut bytes = [0u8; 8];
    bytes[..len].copy_from_slice(&rest[..len]);
    u64::from_be_bytes(bytes)
}

/// Builds the prefix array, or returns `None` when it does not help the search.
///
/// The index must have at least 2 blocks, an empty first key, and no other
/// empty key, as `compute_index_key` makes them.
fn build_prefixes<'a>(n: usize, key: impl Fn(usize) -> &'a [u8]) -> Option<PrefixArray> {
    if n < 2 || !key(0).is_empty() || (1..n).any(|block| key(block).is_empty()) {
        return None;
    }
    let first = key(1);
    let last = key(n - 1);
    let shared_len = first.iter().zip(last).take_while(|(a, b)| a == b).count();
    let values: Vec<u64> = (0..n)
        .map(|block| {
            let k = key(block);
            // Only the empty first key, or a key in an unsorted index, is
            // shorter.
            if k.len() < shared_len {
                0
            } else {
                head(k, shared_len)
            }
        })
        .collect();
    // If few values are distinct, most searches compare full keys, and the
    // array does not help.
    let rest = &values[1..];
    let distinct = 1 + rest.windows(2).filter(|w| w[0] != w[1]).count();
    if distinct * MIN_DISTINCT_DENOMINATOR < rest.len() {
        return None;
    }
    let mut shared_inline = [0u8; INLINE_PREFIX_LEN];
    if shared_len <= INLINE_PREFIX_LEN {
        shared_inline[..shared_len].copy_from_slice(&last[..shared_len]);
    }
    Some(PrefixArray {
        shared_len,
        shared_inline,
        btree: StaticBTree::new(&values),
    })
}

/// Copies a verified `SsTableIndex` into arrays.
fn decode_index(index: SsTableIndex<'_>) -> Result<IndexArrays, InvalidFlatbuffer> {
    let block_meta = index.block_meta();
    let n = block_meta.len();
    let key_bytes: usize = block_meta.iter().map(|meta| meta.first_key().len()).sum();

    let mut offsets = Vec::with_capacity(n);
    let mut encoded_lens = Vec::with_capacity(n);
    let mut keys = Vec::with_capacity(key_bytes);
    let mut key_ends = Vec::with_capacity(n);
    for meta in block_meta.iter() {
        offsets.push(meta.offset());
        encoded_lens.push(meta.encoded_len());
        keys.extend_from_slice(meta.first_key().bytes());
        // The verifier limits the size of the buffer that it accepts to 2 GiB.
        key_ends
            .push(u32::try_from(keys.len()).map_err(|_| InvalidFlatbuffer::ApparentSizeTooLarge)?);
    }
    let keys = keys.into_boxed_slice();
    let key_ends = key_ends.into_boxed_slice();
    let prefixes = {
        let key = |block: usize| {
            let start = if block == 0 {
                0
            } else {
                key_ends[block - 1] as usize
            };
            &keys[start..key_ends[block] as usize]
        };
        build_prefixes(n, key)
    };
    let encoded_lens = if encoded_lens.iter().all(|&len| len == 0) {
        None
    } else {
        Some(encoded_lens.into_boxed_slice())
    };
    Ok(IndexArrays {
        offsets: offsets.into_boxed_slice(),
        encoded_lens,
        keys,
        key_ends,
        prefixes,
    })
}

#[cfg(feature = "bench-internal")]
pub mod benches;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioned_keyspace::partition_point as generic_partition_point;
    use crate::partitioned_keyspace::RangePartitionedKeySpace;
    use proptest::collection::vec;
    use proptest::prelude::*;

    fn build(blocks: &[(u64, &[u8], u32)]) -> Bytes {
        let mut builder = FlatBufferBuilder::new();
        let metas: Vec<_> = blocks
            .iter()
            .map(|(offset, key, encoded_len)| {
                let first_key = builder.create_vector(key);
                BlockMeta::create(
                    &mut builder,
                    &BlockMetaArgs {
                        offset: *offset,
                        first_key: Some(first_key),
                        encoded_len: *encoded_len,
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
        Bytes::copy_from_slice(builder.finished_data())
    }

    fn build_keys(keys: &[Vec<u8>]) -> SsTableIndexOwned {
        let blocks: Vec<_> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| (i as u64 * 4096, k.as_slice(), 0))
            .collect();
        SsTableIndexOwned::new(build(&blocks)).unwrap()
    }

    /// Sorted keys that share a prefix, with an empty first key, as
    /// `compute_index_key` makes them.
    fn sorted_keys() -> impl Strategy<Value = Vec<Vec<u8>>> {
        sorted_keys_from(0u8..4, 1..200, Just(true))
    }

    /// Like [`sorted_keys`], with suffix bytes from `bytes` and `count` keys.
    /// The first key is empty when `empty_first` gives `true`.
    fn sorted_keys_from(
        bytes: impl Strategy<Value = u8>,
        count: Range<usize>,
        empty_first: impl Strategy<Value = bool>,
    ) -> impl Strategy<Value = Vec<Vec<u8>>> {
        (
            vec(any::<u8>(), 0..24),
            vec(vec(bytes, 0..12), count),
            empty_first,
        )
            .prop_map(|(shared, suffixes, empty_first)| {
                let mut keys: Vec<Vec<u8>> = suffixes
                    .into_iter()
                    .map(|s| [shared.as_slice(), s.as_slice()].concat())
                    .collect();
                keys.sort();
                if empty_first {
                    keys[0] = Vec::new();
                }
                keys
            })
    }

    #[test]
    fn should_decode_all_fields() {
        let buf = build(&[(0, b"", 0), (4096, b"apple", 100), (8192, b"banana", 0)]);
        let index = SsTableIndexOwned::new(buf).unwrap();
        assert_eq!(index.len(), 3);
        assert_eq!(index.offset(1), 4096);
        assert_eq!(index.first_key(0), b"");
        assert_eq!(index.first_key(2), b"banana");
        assert_eq!(index.encoded_len(1), 100);
        assert_eq!(index.encoded_len(2), 0);
    }

    #[test]
    fn should_decode_empty_index() {
        let index = SsTableIndexOwned::new(build(&[])).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.partition_point_le(b"a"), 0);
        assert_eq!(index.partition_point_lt(b"a"), 0);
    }

    #[test]
    fn should_encode_to_same_index() {
        let buf = build(&[(0, b"", 0), (4096, b"apple", 100), (8192, b"banana", 7)]);
        let index = SsTableIndexOwned::new(buf).unwrap();
        let again = SsTableIndexOwned::new(index.data()).unwrap();
        assert!(index == again);
    }

    #[test]
    fn should_skip_prefix_array_when_few_values_are_distinct() {
        // The shared prefix is `p/`, and the next 8 bytes take only two values.
        let keys: Vec<Vec<u8>> = std::iter::once(Vec::new())
            .chain([b"aaaaaaaa", b"bbbbbbbb"].iter().flat_map(|group| {
                (0..32u8).map(move |i| [b"p/".as_slice(), *group, &[i]].concat())
            }))
            .collect();
        let index = build_keys(&keys);
        assert!(index.inner.prefixes.is_none());
        assert_eq!(index.partition_point_le(&keys[10]), 11);
        assert_eq!(index.partition_point_lt(&keys[40]), 40);
    }

    #[test]
    fn should_build_prefix_array_only_with_one_empty_first_key() {
        let has_prefixes = |keys: &[&[u8]]| {
            let keys: Vec<Vec<u8>> = keys.iter().map(|k| k.to_vec()).collect();
            build_keys(&keys).inner.prefixes.is_some()
        };
        assert!(has_prefixes(&[b"", b"p/a", b"p/b"]));
        // One block.
        assert!(!has_prefixes(&[b""]));
        // The first key is not empty.
        assert!(!has_prefixes(&[b"p/a", b"p/b", b"p/c"]));
        // A second empty key.
        assert!(!has_prefixes(&[b"", b"", b"p/a", b"p/b"]));
    }

    /// The verifier does not check that the keys are sorted. With unsorted
    /// keys, a search can return a wrong block, but it must not panic, and its
    /// results must stay in range.
    #[test]
    fn should_search_unsorted_keys_without_panic() {
        let long = |c: u8| [vec![b'x'; 20], vec![c]].concat();
        let cases: Vec<Vec<Vec<u8>>> = vec![
            // A shared prefix of `p/`, with the values out of order.
            vec![
                b"".to_vec(),
                b"p/zz".to_vec(),
                b"p/aa".to_vec(),
                b"p/mm".to_vec(),
                b"p/bb".to_vec(),
            ],
            // A key shorter than the shared prefix, and a key without it.
            vec![
                b"".to_vec(),
                b"p/zz".to_vec(),
                b"a".to_vec(),
                b"q".to_vec(),
                b"p/cc".to_vec(),
            ],
            // A shared prefix longer than the inline copy.
            vec![
                b"".to_vec(),
                long(b'b'),
                long(b'a'),
                b"y".to_vec(),
                long(b'c'),
            ],
        ];
        for keys in cases {
            let index = build_keys(&keys);
            assert!(index.inner.prefixes.is_some());
            check_in_range(&index, &keys);
        }
    }

    fn check_in_range(index: &SsTableIndexOwned, keys: &[Vec<u8>]) {
        let n = index.len();
        let mut targets: Vec<Vec<u8>> = vec![Vec::new(), vec![0xFF; 30]];
        for k in keys {
            targets.push(k.clone());
            targets.push([k.as_slice(), &[0]].concat());
            if let Some((_, shorter)) = k.split_last() {
                targets.push(shorter.to_vec());
            }
        }
        for t in &targets {
            assert!(index.partition_point_le(t) <= n);
            assert!(index.partition_point_lt(t) <= n);
            let r = index.partitions_covering_key(t);
            assert!(r.start <= r.end && r.end <= n);
        }
    }

    /// Makes sure that the index search gives the same results as a binary
    /// search over the full keys, for `targets` and for keys at, after, and
    /// before about 20 of `keys`.
    fn check_search(keys: &[Vec<u8>], targets: Vec<Vec<u8>>) -> Result<(), TestCaseError> {
        let index = build_keys(keys);
        let mut all_targets = targets;
        let stride = keys.len().div_ceil(20);
        for k in keys.iter().step_by(stride) {
            all_targets.push(k.clone());
            let mut longer = k.clone();
            longer.push(0);
            all_targets.push(longer);
            if let Some(shorter) = k.split_last().map(|(_, s)| s.to_vec()) {
                all_targets.push(shorter);
            }
        }
        all_targets.push(Vec::new());
        for t in &all_targets {
            let le = generic_partition_point(&index, |k| k <= t.as_slice());
            let lt = generic_partition_point(&index, |k| k < t.as_slice());
            prop_assert_eq!(index.partition_point_le(t), le);
            prop_assert_eq!(index.partition_point_lt(t), lt);
            // The range that the default trait method returns.
            let mut first_equal = le;
            while first_equal > 0 && index.first_key(first_equal - 1) == t.as_slice() {
                first_equal -= 1;
            }
            prop_assert_eq!(
                index.partitions_covering_key(t),
                first_equal.saturating_sub(1)..le
            );
        }
        Ok(())
    }

    #[test]
    fn should_build_tree_levels() {
        let values: Vec<u64> = (0..300).collect();
        let tree = StaticBTree::new(&values);
        // Each level has one node for each `NODE_LEN` nodes of the level
        // below, until one node is left. With 16 values in each node, 300
        // values need 19, 2, and 1 nodes. With 8, they need 38, 5, and 1.
        let nodes: Vec<usize> = tree.levels.iter().map(|l| l.len()).collect();
        let mut expected = vec![300usize.div_ceil(NODE_LEN)];
        while expected[expected.len() - 1] > 1 {
            expected.push(expected[expected.len() - 1].div_ceil(NODE_LEN));
        }
        assert_eq!(nodes, expected);
        // Each separator is the first value of its child.
        let l = NODE_LEN as u64;
        assert_eq!(tree.levels[1][0].0[1], l);
        assert_eq!(tree.levels[1][1].0[0], l * l);
        assert_eq!(tree.levels[2][0].0[..2], [0, l * l]);
        // 300 is not a multiple of the node size, so the last leaf has padding.
        assert_eq!(tree.levels[0][nodes[0] - 1].0[NODE_LEN - 1], u64::MAX);
        assert_eq!(tree.count_le(u64::MAX), 300);
        assert_eq!(tree.count_le(150), 151);
        assert_eq!(tree.count_le(256), 257);
        assert_eq!(StaticBTree::new(&[1, 2]).count_le(0), 0);
        assert_eq!(StaticBTree::new(&[]).count_le(u64::MAX), 0);
    }

    proptest! {
        #[test]
        fn should_search_shuffled_keys_without_panic(
            keys in sorted_keys(),
            swaps in vec((any::<prop::sample::Index>(), any::<prop::sample::Index>()), 1..8),
        ) {
            let mut keys = keys;
            for (a, b) in swaps {
                let (a, b) = (a.index(keys.len()), b.index(keys.len()));
                keys.swap(a, b);
            }
            check_in_range(&build_keys(&keys), &keys);
        }

        #[test]
        fn should_search_like_binary_search(
            keys in sorted_keys(),
            targets in vec(vec(any::<u8>(), 0..16), 1..20),
        ) {
            check_search(&keys, targets)?;
        }

        /// Without an empty first key, the index has no prefix array, and the
        /// search is a binary search over the full keys.
        #[test]
        fn should_search_without_prefix_array_like_binary_search(
            keys in sorted_keys_from(0u8..4, 1..200, Just(false)),
            targets in vec(vec(any::<u8>(), 0..16), 1..20),
        ) {
            // An empty shared prefix and an empty suffix can still make an
            // empty first key.
            prop_assume!(!keys[0].is_empty());
            prop_assert!(build_keys(&keys).inner.prefixes.is_none());
            check_search(&keys, targets)?;
        }

        #[test]
        fn should_count_like_partition_point(
            len in 0usize..5000,
            seed in any::<u64>(),
            targets in vec(any::<u64>(), 1..20),
        ) {
            // The values come from a seed, because proptest generates large
            // vectors slowly. A quarter of them are small, and some are 0 or
            // u64::MAX, so values repeat.
            let mut x = seed;
            let mut values: Vec<u64> = (0..len)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    match x >> 61 {
                        0 => 0,
                        1 => u64::MAX,
                        2 | 3 => x % 64,
                        _ => x,
                    }
                })
                .collect();
            values.sort_unstable();
            let tree = StaticBTree::new(&values);
            let mut all_targets = targets;
            all_targets.extend([0, 1, u64::MAX - 1, u64::MAX]);
            all_targets.extend(values.iter().step_by(values.len().div_ceil(20).max(1)));
            for t in all_targets {
                prop_assert_eq!(tree.count_le(t), values.partition_point(|&v| v <= t));
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        /// Larger indexes have several tree levels. Bytes `0xFE` and `0xFF`
        /// make values near `u64::MAX`, the value that pads the nodes.
        #[test]
        fn should_search_large_index_like_binary_search(
            keys in sorted_keys_from(
                prop_oneof![Just(0u8), Just(1), Just(0xFE), Just(0xFF)],
                1..5000,
                Just(true),
            ),
            targets in vec(vec(prop_oneof![Just(0u8), Just(1), Just(0xFE), Just(0xFF), any::<u8>()], 0..16), 1..20),
        ) {
            check_search(&keys, targets)?;
        }
    }

    proptest! {

        #[test]
        fn should_reject_or_decode_corrupt_buffers(
            keys in sorted_keys(),
            flips in vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
            truncate in any::<prop::sample::Index>(),
        ) {
            let blocks: Vec<_> = keys.iter().enumerate().map(|(i, k)| (i as u64, k.as_slice(), 0)).collect();
            let buf = build(&blocks).to_vec();
            let mut corrupt = buf.clone();
            for (pos, byte) in flips {
                let pos = pos.index(corrupt.len());
                corrupt[pos] = byte;
            }
            // The decode can return an error or an index, but it must not
            // panic.
            let _ = SsTableIndexOwned::new(Bytes::from(corrupt));
            let _ = SsTableIndexOwned::new(Bytes::copy_from_slice(&buf[..truncate.index(buf.len())]));
        }

        #[test]
        fn should_decode_like_flatbuffer_reader(keys in sorted_keys()) {
            let blocks: Vec<_> = keys.iter().enumerate().map(|(i, k)| (i as u64 * 7, k.as_slice(), i as u32 % 3)).collect();
            let buf = build(&blocks);
            let fb = flatbuffers::root::<SsTableIndex>(&buf).unwrap();
            let index = SsTableIndexOwned::new(buf.clone()).unwrap();
            prop_assert_eq!(index.len(), fb.block_meta().len());
            for (i, meta) in fb.block_meta().iter().enumerate() {
                prop_assert_eq!(index.offset(i), meta.offset());
                prop_assert_eq!(index.first_key(i), meta.first_key().bytes());
                prop_assert_eq!(index.encoded_len(i), meta.encoded_len());
            }
        }
    }
}
