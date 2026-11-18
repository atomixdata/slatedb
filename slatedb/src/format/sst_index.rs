//! The in-memory form of an SST index, and the search over it.
//!
//! An index has one entry for each data block: the block offset, the block
//! length, and the index key. An index key is a lower bound for the keys in
//! its block. [`SsTableIndexOwned::new`] verifies the `SsTableIndex`
//! FlatBuffer and copies it into two parts:
//!
//! 1. Flat arrays. Each field has its own array. All index keys are in one
//!    buffer, with an array of start positions.
//! 2. A static B+ tree over prefix values. Each node has 128 bytes, aligned to
//!    cache lines. Each node has its own shared prefix: the common prefix of
//!    its boundary keys.
//!    The prefix value of a key in a node is the 8 bytes of the key after that
//!    prefix, padded with zeros and read as a big-endian `u64`.
//!
//! A search finds the candidate blocks: the blocks whose prefix value equals
//! the prefix value of the key. The search goes down one node on each level,
//! and the prefix values decide which child to go to. Only ties in the prefix
//! values need an index key compare. In the leaf level, the tied blocks are
//! the candidate blocks.

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
        self.inner.len()
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

    /// Returns the index key of block `block`.
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
        let index_arrays = &self.inner;
        size_of::<IndexArrays>()
            + index_arrays.offsets.len() * size_of::<u64>()
            + index_arrays
                .encoded_lens
                .as_ref()
                .map_or(0, |l| l.len() * size_of::<u32>())
            + index_arrays.index_keys.len()
            + index_arrays.key_ends.len() * size_of::<u32>()
            + index_arrays.tree.as_ref().map_or(0, |t| t.size())
    }

    /// Returns the number of blocks whose index key is at or below `key`. If
    /// `inclusive` is false, returns the number of blocks whose index key is
    /// below `key`.
    ///
    /// The result is the position where `key` would be inserted, as in
    /// `slice::partition_point`. The block that can hold `key` is the block
    /// before it.
    fn partition_point(&self, key: &[u8], inclusive: bool) -> usize {
        let index_arrays = &self.inner;
        let candidates = index_arrays.candidates(key, inclusive);
        partition_point_by_predicate(candidates.start, candidates.end, |block| {
            let k = index_arrays.key(block);
            if inclusive {
                k <= key
            } else {
                k < key
            }
        })
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
        let index_arrays = &self.inner;
        let candidates = index_arrays.candidates(key, false);
        // Blocks before `first_equal` have an index key below `key`. The blocks
        // from `first_equal` to `end` have an index key equal to `key`. These
        // blocks can go past the candidates, into the next leaf.
        let first_equal = partition_point_by_predicate(candidates.start, candidates.end, |block| {
            index_arrays.key(block) < key
        });
        // Read the key at `first_equal` only if it can equal `key`: when the
        // block is a candidate, or is the first block of the next leaf, which
        // this search did not count. Any other block after the candidates has
        // a larger prefix value, so its index key is above `key`.
        let may_equal = first_equal < candidates.end || first_equal % NUM_ENTRIES_PER_NODE == 0;
        let mut end = first_equal;
        while may_equal && end < self.len() && index_arrays.key(end) == key {
            end += 1;
        }
        // The block before `first_equal` can still hold `key`.
        first_equal.saturating_sub(1)..end
    }
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
    /// The index key of each block, one after the other.
    index_keys: Box<[u8]>,
    /// The start of each key in `index_keys`, and then the end of the last
    /// key. Key `i` is `index_keys[key_ends[i]..key_ends[i + 1]]`.
    key_ends: Box<[u32]>,
    /// The tree over the prefix values, or `None` for an empty index.
    tree: Option<StaticBTree>,
}

impl IndexArrays {
    /// Returns the candidate blocks for `key`.
    ///
    /// See [`StaticBTree::candidates`] for details.
    fn candidates(&self, key: &[u8], inclusive: bool) -> Range<usize> {
        match &self.tree {
            Some(tree) => tree.candidates(key, inclusive, self),
            // If the index is empty, the range is empty.
            None => 0..0,
        }
    }

    fn key(&self, block: usize) -> &[u8] {
        &self.index_keys[self.key_ends[block] as usize..self.key_ends[block + 1] as usize]
    }

    fn len(&self) -> usize {
        self.offsets.len()
    }
}

/// The number of entries below a target, and the number at or below it.
///
/// The entries from `lt` to `le` equal the target: they are the ties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Count {
    /// The number of entries below the target.
    lt: usize,
    /// The number of entries at or below the target.
    le: usize,
}

impl Count {
    fn new(lt: usize, le: usize) -> Self {
        Self { lt, le }
    }

    /// Limits both numbers to `real`, the number of entries that are not
    /// padding.
    fn clamp(self, real: usize) -> Self {
        Self {
            lt: self.lt.min(real),
            le: self.le.min(real),
        }
    }
}

/// The size of one node of a [`StaticBTree`], in bytes.
///
/// A node is one cache line on CPUs with 128-byte lines. On CPUs with 64-byte
/// cache lines, such as x86 and AWS Graviton, a node is two adjacent lines.
/// These CPUs usually fetch the two lines together, and a node of 128 bytes
/// holds twice as many entries as a node of one line, so the tree has fewer
/// levels. In benchmarks on Graviton and x86, 128-byte nodes were faster than
/// 64-byte nodes.
pub(crate) const NODE_BYTES: usize = 128;

/// The number of entries in one node. The first 8 bytes of a node hold the
/// location of its shared prefix, and each entry is one `u64`.
const NUM_ENTRIES_PER_NODE: usize = NODE_BYTES / size_of::<u64>() - 1;

/// One node of a [`StaticBTree`]: the location of the shared prefix, then up
/// to `NUM_ENTRIES_PER_NODE` entries, padded with `u64::MAX`.
///
/// The shared prefix is the start of the lower boundary key of the node, so
/// its bytes are already in `index_keys`. The node stores only where they
/// start there, and the prefix length.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(128))]
struct Node {
    /// The start of the shared prefix in `index_keys`.
    prefix_start: u32,
    /// The length of the shared prefix.
    prefix_len: u32,
    /// The entries. Each entry holds the prefix value of the index key of one
    /// block. The padding is `u64::MAX`.
    entries: [u64; NUM_ENTRIES_PER_NODE],
}

impl Node {
    /// Builds node `node_idx` of `level`.
    fn build(arrays: &IndexArrays, level: &Level, node_idx: usize) -> Self {
        let (prefix_start, prefix_len) = level.node_prefix(arrays, node_idx);
        let mut node = Node {
            prefix_start,
            prefix_len,
            entries: [u64::MAX; NUM_ENTRIES_PER_NODE],
        };
        let skip = node.prefix_len();
        for (entry, entry_idx) in node.entries.iter_mut().zip(level.node_entries(node_idx)) {
            *entry = head(arrays.key(level.block(entry_idx)), skip);
        }
        node
    }

    fn prefix_len(&self) -> usize {
        self.prefix_len as usize
    }

    /// Returns the bytes of the shared prefix, from `index_keys`.
    fn prefix<'a>(&self, index_keys: &'a [u8]) -> &'a [u8] {
        let start = self.prefix_start as usize;
        &index_keys[start..start + self.prefix_len()]
    }

    /// Counts the entries of the node against `target`.
    #[inline]
    fn count(&self, target: u64) -> Count {
        let (mut lt, mut le) = (0, 0);
        for &entry in &self.entries {
            lt += usize::from(entry < target);
            le += usize::from(entry <= target);
        }
        Count::new(lt, le)
    }
}

const _: () = assert!(size_of::<Node>() == NODE_BYTES && align_of::<Node>() == NODE_BYTES);

/// One level of a [`StaticBTree`].
///
/// An entry is one slot in one of the nodes on the level, and holds one
/// prefix value. On level 0 it stands for a block, and on a higher level it
/// stands for a node of the level below.
#[derive(PartialEq, Eq)]
struct Level {
    nodes: Box<[Node]>,
    /// The number of entries on the level.
    num_entries: usize,
    /// The number of blocks for each entry: `NUM_ENTRIES_PER_NODE^l` on level
    /// `l`.
    stride: usize,
}

impl Level {
    /// Builds a level of `num_entries` entries, with `stride` blocks for each
    /// entry.
    fn build(arrays: &IndexArrays, stride: usize, num_entries: usize) -> Self {
        let mut level = Self {
            nodes: Box::default(),
            num_entries,
            stride,
        };
        let num_nodes = num_entries.div_ceil(NUM_ENTRIES_PER_NODE);
        let nodes = (0..num_nodes)
            .map(|node_idx| Node::build(arrays, &level, node_idx))
            .collect();
        level.nodes = nodes;
        level
    }

    /// Returns the entries of node `node_idx`.
    #[inline]
    fn node_entries(&self, node_idx: usize) -> Range<usize> {
        let first_entry = node_idx * NUM_ENTRIES_PER_NODE;
        first_entry..(first_entry + NUM_ENTRIES_PER_NODE).min(self.num_entries)
    }

    /// Returns the block of entry `entry_idx`.
    fn block(&self, entry_idx: usize) -> usize {
        entry_idx * self.stride
    }

    /// Returns the shared prefix of node `node_idx`, as its start in
    /// `index_keys` and its length: the common prefix of the boundary keys,
    /// at the start of the lower boundary key.
    fn node_prefix(&self, arrays: &IndexArrays, node_idx: usize) -> (u32, u32) {
        let (lower_block, upper_block) = self.boundary_blocks(arrays, node_idx);
        let len = common_prefix_len(arrays.key(lower_block), arrays.key(upper_block));
        // A prefix is part of a key, and `decode_index` makes sure that all
        // keys fit in `u32` positions.
        (arrays.key_ends[lower_block], len as u32)
    }

    /// Returns the blocks of the boundary keys of node `node_idx`.
    ///
    /// Usually, the lower boundary key is the first key of the node, and the
    /// upper boundary key is the first key of the next node on the level.
    /// There are two special cases:
    ///
    /// 1. If the node starts with an empty key and has a second key, the lower
    ///    boundary key is the second key.
    /// 2. The last node of a level has no next node, so its upper boundary key
    ///    is its own last key.
    fn boundary_blocks(&self, arrays: &IndexArrays, node_idx: usize) -> (usize, usize) {
        let entries = self.node_entries(node_idx);
        let mut lower_entry_idx = entries.start;
        if arrays.key(self.block(lower_entry_idx)).is_empty() && entries.len() >= 2 {
            lower_entry_idx += 1;
        }
        let upper_entry_idx = if entries.end < self.num_entries {
            entries.end
        } else {
            entries.end - 1
        };
        (self.block(lower_entry_idx), self.block(upper_entry_idx))
    }
}

/// A static B+ tree over prefix values, with a shared prefix for each node.
///
/// It is built one time and never changes. Entry `i` of level `l` stands for
/// block `i * NUM_ENTRIES_PER_NODE^l`, the first block below it. It holds the
/// prefix value of the index key of that block, at the shared prefix of its
/// node.
///
/// The shared prefix of a node is the common prefix of its two boundary keys
/// (see [`Level::boundary_blocks`]). The keys between the boundary keys all
/// start with it. So the prefix values of the node use only the bytes after it.
///
/// In most nodes, the search key is between the boundary keys, because the
/// parent node sent the search there. So the search key also starts with the
/// shared prefix, and the search skips it without a compare.
///
/// The first and the last node of a level are different. A search key below
/// every index key reaches the first node, and a search key above every
/// index key reaches the last node. Such a key can be outside the boundary
/// keys. So in these two nodes, [`StaticBTree::count_first_or_last_node`]
/// compares the search key with the prefix bytes first.
#[derive(PartialEq, Eq)]
struct StaticBTree {
    /// The levels of the tree, from the leaves to the root.
    levels: Box<[Level]>,
    /// Whether the first index key is empty.
    first_key_empty: bool,
}

impl StaticBTree {
    /// Builds the tree over the index keys of `arrays`, which must have at
    /// least one block.
    fn new(arrays: &IndexArrays) -> Self {
        let n = arrays.len();
        debug_assert!(n > 0);
        let mut levels = Vec::new();
        let (mut len, mut stride) = (n, 1usize);
        loop {
            let level = Level::build(arrays, stride, len);
            let num_nodes = level.nodes.len();
            levels.push(level);
            if num_nodes == 1 {
                break;
            }
            len = num_nodes;
            stride = stride.saturating_mul(NUM_ENTRIES_PER_NODE);
        }
        Self {
            levels: levels.into_boxed_slice(),
            first_key_empty: arrays.key(0).is_empty(),
        }
    }

    /// Returns the candidate blocks for `key`: the blocks whose prefix value
    /// equals the prefix value of `key`. Only these blocks need an index key
    /// compare to decide whether they can hold `key`.
    ///
    /// If `inclusive`, every block before the range has an index key at or
    /// below `key`, and every block after it has an index key above `key`.
    /// Otherwise, every block before the range has an index key below `key`,
    /// and every block after it has an index key at or above `key`.
    fn candidates(&self, key: &[u8], inclusive: bool, arrays: &IndexArrays) -> Range<usize> {
        let mut node_idx = 0;
        for (depth, level) in self.levels.iter().enumerate().rev() {
            let first_entry_idx = node_idx * NUM_ENTRIES_PER_NODE;
            let count = self.count_node(level, node_idx, key, arrays);
            if depth == 0 {
                return (first_entry_idx + count.lt)..(first_entry_idx + count.le);
            }
            // Above the leaves, each tied entry is a child. The search compares
            // the first index key of each tied child, and descends to the last
            // child whose index key is below the key, or at or below it if
            // `inclusive`.
            let child_idx = partition_point_by_predicate(count.lt, count.le, |pos| {
                let k = arrays.key(level.block(first_entry_idx + pos));
                if inclusive {
                    k <= key
                } else {
                    k < key
                }
            });
            node_idx = (first_entry_idx + child_idx)
                .saturating_sub(1)
                .min(self.levels[depth - 1].nodes.len() - 1);
        }
        0..0
    }

    /// Counts the entries of node `node_idx` of `level` against `key`. The
    /// count leaves out the padding.
    #[inline]
    fn count_node(
        &self,
        level: &Level,
        node_idx: usize,
        key: &[u8],
        arrays: &IndexArrays,
    ) -> Count {
        let node = &level.nodes[node_idx];
        let is_first = node_idx == 0;
        let is_first_or_last = is_first || node_idx + 1 == level.nodes.len();
        let count = if is_first_or_last && node.prefix_len() > 0 {
            self.count_first_or_last_node(node, key, node.prefix(&arrays.index_keys), is_first)
        } else {
            node.count(head(key, node.prefix_len()))
        };
        count.clamp(level.node_entries(node_idx).len())
    }

    /// Counts like [`Node::count`] in the first or last node of a level, where
    /// the search key can be outside the boundary keys. `prefix` is the shared
    /// prefix of the node.
    fn count_first_or_last_node(
        &self,
        node: &Node,
        key: &[u8],
        prefix: &[u8],
        is_first: bool,
    ) -> Count {
        if key.starts_with(prefix) {
            return node.count(head(key, prefix.len()));
        }
        if key > prefix {
            return Count::new(NUM_ENTRIES_PER_NODE, NUM_ENTRIES_PER_NODE);
        }
        // `key` is below every key of the node, except an empty first index
        // key.
        match (is_first && self.first_key_empty, key.is_empty()) {
            (true, true) => Count::new(0, 1),
            (true, false) => Count::new(1, 1),
            (false, _) => Count::new(0, 0),
        }
    }

    /// Returns the memory of the nodes, in bytes.
    fn size(&self) -> usize {
        self.levels
            .iter()
            .map(|l| l.nodes.len() * size_of::<Node>() + size_of::<Level>())
            .sum::<usize>()
    }
}

/// Returns the length of the common prefix of `a` and `b`.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Returns the first position in `low..high` for which `pred` is false.
fn partition_point_by_predicate(
    mut low: usize,
    mut high: usize,
    predicate: impl Fn(usize) -> bool,
) -> usize {
    while low < high {
        let mid = low + (high - low) / 2;
        if predicate(mid) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

/// Returns the 8 bytes of `key` after its first `skip` bytes, padded with
/// zeros, as a big-endian `u64`. A key with at most `skip` bytes gives 0.
fn head(key: &[u8], skip: usize) -> u64 {
    let rest = key.get(skip..).unwrap_or(&[]);
    let len = rest.len().min(8);
    let mut bytes = [0u8; 8];
    bytes[..len].copy_from_slice(&rest[..len]);
    u64::from_be_bytes(bytes)
}

/// Copies a verified `SsTableIndex` into arrays.
fn decode_index(index: SsTableIndex<'_>) -> Result<IndexArrays, InvalidFlatbuffer> {
    let block_meta = index.block_meta();
    let n = block_meta.len();
    let key_bytes: usize = block_meta.iter().map(|meta| meta.first_key().len()).sum();

    let mut offsets = Vec::with_capacity(n);
    let mut encoded_lens = Vec::with_capacity(n);
    let mut index_keys = Vec::with_capacity(key_bytes);
    let mut key_ends = Vec::with_capacity(n + 1);
    key_ends.push(0);
    for meta in block_meta.iter() {
        offsets.push(meta.offset());
        encoded_lens.push(meta.encoded_len());
        index_keys.extend_from_slice(meta.first_key().bytes());
        // The verifier limits the size of the buffer that it accepts to 2 GiB.
        key_ends.push(
            u32::try_from(index_keys.len()).map_err(|_| InvalidFlatbuffer::ApparentSizeTooLarge)?,
        );
    }
    let encoded_lens = if encoded_lens.iter().all(|&len| len == 0) {
        None
    } else {
        Some(encoded_lens.into_boxed_slice())
    };
    let mut arrays = IndexArrays {
        offsets: offsets.into_boxed_slice(),
        encoded_lens,
        index_keys: index_keys.into_boxed_slice(),
        key_ends: key_ends.into_boxed_slice(),
        tree: None,
    };
    if n > 0 {
        arrays.tree = Some(StaticBTree::new(&arrays));
    }
    Ok(arrays)
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

    /// Sorted keys that share a prefix, with an empty first index key, as
    /// `compute_index_key` makes them.
    fn sorted_keys() -> impl Strategy<Value = Vec<Vec<u8>>> {
        sorted_keys_from(0u8..4, 1..200, Just(true))
    }

    /// Like [`sorted_keys`], with suffix bytes from `bytes` and `count` keys.
    /// The first index key is empty when `empty_first` gives `true`.
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
    fn should_count_node_entries() {
        let mut node = Node {
            prefix_start: 0,
            prefix_len: 3,
            entries: [u64::MAX; NUM_ENTRIES_PER_NODE],
        };
        node.entries[..4].copy_from_slice(&[0, 5, 5, 9]);
        assert_eq!(node.prefix_len(), 3);
        assert_eq!(node.count(0), Count::new(0, 1));
        assert_eq!(node.count(5), Count::new(1, 3));
        assert_eq!(node.count(7), Count::new(3, 3));
        assert_eq!(node.count(10), Count::new(4, 4));
        // The padding equals `u64::MAX`, so it counts as at or below it. The
        // search clamps the counts to the real entries.
        assert_eq!(node.count(u64::MAX), Count::new(4, NUM_ENTRIES_PER_NODE));
        assert_eq!(node.count(u64::MAX).clamp(4), Count::new(4, 4));
    }

    #[test]
    fn should_build_tree_unless_empty() {
        let has_tree = |keys: &[&[u8]]| {
            let keys: Vec<Vec<u8>> = keys.iter().map(|k| k.to_vec()).collect();
            build_keys(&keys).inner.tree.is_some()
        };
        assert!(!has_tree(&[]));
        assert!(has_tree(&[b""]));
        assert!(has_tree(&[b"p/a"]));
        assert!(has_tree(&[b"", b"p/a"]));
        // The first index key need not be empty, as in a WAL SST.
        assert!(has_tree(&[b"p/a", b"p/b", b"p/c"]));
    }

    /// The verifier does not check that the keys are sorted. With unsorted
    /// keys, a search can return a wrong block, but it must not panic, and its
    /// results must stay in range.
    #[test]
    fn should_search_unsorted_keys_without_panic() {
        let long = |c: u8| [vec![b'x'; 20], vec![c]].concat();
        let cases: Vec<Vec<Vec<u8>>> = vec![
            // A shared prefix of `p/`, with the keys out of order.
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
            assert!(index.inner.tree.is_some());
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
    /// search over the index keys, for `targets` and for keys at, after, and
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
        // An empty first index key, then `k` and a 3-byte number.
        let keys: Vec<Vec<u8>> = std::iter::once(Vec::new())
            .chain((1..302u32).map(|i| [&b"k"[..], &i.to_be_bytes()[1..]].concat()))
            .collect();
        let index = build_keys(&keys);
        let tree = index.inner.tree.as_ref().unwrap();
        // Each level has one node for each `NUM_ENTRIES_PER_NODE` entries of
        // the level below, until one node is left.
        let nodes: Vec<usize> = tree.levels.iter().map(|l| l.nodes.len()).collect();
        let mut expected = vec![302usize.div_ceil(NUM_ENTRIES_PER_NODE)];
        while expected[expected.len() - 1] > 1 {
            expected.push(expected[expected.len() - 1].div_ceil(NUM_ENTRIES_PER_NODE));
        }
        assert_eq!(nodes, expected);
        // The second leaf holds keys `NUM_ENTRIES_PER_NODE` to
        // `2 * NUM_ENTRIES_PER_NODE - 1`. Its boundary keys share `k`, the
        // first number byte, and the start of the second one.
        let leaf = &tree.levels[0].nodes[1];
        let lower = &keys[NUM_ENTRIES_PER_NODE];
        let upper = &keys[2 * NUM_ENTRIES_PER_NODE];
        let prefix_len = common_prefix_len(lower, upper);
        let index_keys = &index.inner.index_keys;
        assert_eq!(leaf.prefix_len(), prefix_len);
        assert_eq!(leaf.prefix(index_keys), &lower[..prefix_len]);
        assert_eq!(leaf.entries[0], head(lower, prefix_len));
        // The first leaf starts with the empty key, so its lower boundary key
        // is key 1.
        assert_eq!(tree.levels[0].nodes[0].entries[0], 0);
        let first_len = common_prefix_len(&keys[1], &keys[NUM_ENTRIES_PER_NODE]);
        assert_eq!(tree.levels[0].nodes[0].prefix_len(), first_len);
        assert_eq!(
            tree.levels[0].nodes[0].prefix(index_keys),
            &keys[1][..first_len]
        );
        // Each upper entry holds the prefix value of the index key of a child,
        // at the prefix of its node.
        let upper_node = &tree.levels[1].nodes[0];
        assert_eq!(
            upper_node.entries[1],
            head(&keys[NUM_ENTRIES_PER_NODE], upper_node.prefix_len())
        );
        // 302 is not a multiple of the node size, so the last leaf has padding.
        assert_eq!(
            tree.levels[0].nodes[nodes[0] - 1].entries[NUM_ENTRIES_PER_NODE - 1],
            u64::MAX
        );
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

        /// Without an empty first index key, as in a WAL SST.
        #[test]
        fn should_search_without_empty_first_key_like_binary_search(
            keys in sorted_keys_from(0u8..4, 1..200, Just(false)),
            targets in vec(vec(any::<u8>(), 0..16), 1..20),
        ) {
            check_search(&keys, targets)?;
        }

    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        /// Larger indexes have several tree levels. Bytes `0xFE` and `0xFF`
        /// make prefix values near `u64::MAX`, the value that pads the nodes.
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
