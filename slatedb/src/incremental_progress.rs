//! Builds the manifest views for an incremental commit of a running compaction job.
//!
//! An incremental commit publishes output that a worker fully stored before the job
//! finishes. For each subcompaction, the coordinator finds the durable interval, which
//! is the key range whose output is complete. It removes those keys from the input
//! views of the job and adds views of the output SSTs to the destination run.
//!
//! With a key group extractor, a commit publishes only whole key groups. A key group
//! is the set of keys that share the prefix that the extractor returns for a stored
//! key, for example all versions of one record that an application stores as
//! separate keys. A read that stops at the first version that it finds then never
//! sees the newer versions of a record in the destination run while the older
//! versions stay in a newer input run.

use std::collections::HashSet;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::ops::RangeBounds;

use bytes::Bytes;
use slatedb_common::DbRand;

use crate::bytes_range::BytesRange;
use crate::compactor_state::{Compaction, CompactionSpec, SourceId};
use crate::db_state::{SortedRun, SsTableHandle, SsTableId, SsTableView};
use crate::error::SlateDBError;
use crate::manifest::ManifestCore;
use crate::prefix_extractor::{PrefixExtractor, PrefixTarget};
use crate::subcompaction::Subcompaction;

/// Durable progress of one job, built from its stored subcompactions.
pub(crate) struct IncrementalProgress {
    /// Durable intervals in key order, at most one per subcompaction. A subcompaction
    /// that has no stored output and is not `Completed` has no interval.
    intervals: Vec<DurableInterval>,
    /// Views of the output SSTs inside the durable intervals, in key order. When two
    /// SSTs share a key, the SST with the newer versions comes first. Only an SST that
    /// ends at the boundary key of an unfinished range is cut, so that it ends before
    /// that key. With key groups, each SST that crosses an end of its interval is cut
    /// to the interval.
    output: Vec<SsTableView>,
    /// IDs of all recorded output SSTs, including SSTs left out of `output`.
    /// `apply_to_manifest_core` uses them to remove output views of earlier commits.
    output_ids: HashSet<SsTableId>,
}

impl IncrementalProgress {
    /// Builds the durable progress of `job` from its stored subcompactions.
    ///
    /// Without `key_group_extractor`, each subcompaction adds one durable interval or
    /// none:
    /// - `Completed`: the whole planned range, even when the range has no output.
    /// - Unfinished with output: from the range start up to the last key of the last
    ///   output SST, without that key, because later output SSTs can still hold
    ///   versions of that key.
    /// - Unfinished without output: no interval.
    ///
    /// With `key_group_extractor`, the progress holds only whole key groups. See
    /// [`Self::from_compaction_by_key_group`].
    ///
    /// Each cut output view gets a new view ID from `rand`.
    ///
    /// Panics if an output SST that the rules need has no `first_entry` or
    /// `last_entry`.
    pub(crate) fn from_compaction(
        job: &Compaction,
        key_group_extractor: Option<&dyn PrefixExtractor>,
        rand: &DbRand,
    ) -> Self {
        match key_group_extractor {
            Some(extractor) => Self::from_compaction_by_key_group(job, extractor, rand),
            None => Self::from_compaction_by_key(job, rand),
        }
    }

    /// Builds the durable progress of `job` without key groups. See
    /// [`Self::from_compaction`].
    fn from_compaction_by_key(job: &Compaction, rand: &DbRand) -> Self {
        let mut intervals = Vec::new();
        let mut output = Vec::new();
        let mut output_ids = HashSet::new();
        for sub_compaction in job.subcompactions() {
            let output_ssts = sub_compaction.output_ssts();
            // Record every output SST, even one that this commit leaves out.
            output_ids.extend(output_ssts.iter().map(|sst| sst.id));
            if sub_compaction.completed() {
                output.extend(output_ssts.iter().cloned().map(SsTableView::identity));
                intervals.push(DurableInterval::from_planned_range(sub_compaction.range()));
                continue;
            }
            let Some(last) = output_ssts.last() else {
                continue;
            };
            let boundary = last
                .info
                .last_entry
                .clone()
                .expect("SST writers record the last entry");
            debug_assert!(
                sub_compaction.range().contains(&boundary),
                "a subcompaction writes only keys inside its range"
            );
            // The boundary key can still have versions in later output SSTs.
            let interval = DurableInterval {
                end: Some(boundary.clone()),
                ..DurableInterval::from_planned_range(sub_compaction.range())
            };
            // If the boundary key is the range start, no key of the range is durable yet.
            if interval.start.as_ref() == Some(&boundary) {
                continue;
            }
            for sst in output_ssts {
                let view = SsTableView::identity(sst.clone());
                // The versions of the boundary key can span several SSTs, and each of
                // them ends at that key. Cut such an SST to end before the key. An SST
                // that holds only the boundary key drops out.
                if sst.info.last_entry.as_ref() == Some(&boundary) {
                    if let Some(cut_view) = view.try_with_visible_range(interval.to_range(), rand) {
                        output.push(cut_view);
                    }
                } else {
                    output.push(view);
                }
            }
            intervals.push(interval);
        }
        Self {
            intervals,
            output,
            output_ids,
        }
    }

    /// Builds the durable progress of `job` so that it holds only whole key groups.
    ///
    /// A commit must not split a key group between the destination run and the input
    /// runs. This function applies `extractor` only to stored keys, which are the first
    /// and last keys of output SSTs, and not to planned range bounds. A planned bound is
    /// often a shortened SST index key, not a key that the application wrote, so the
    /// extractor can return a wrong prefix for it. Such a bound can also sort between
    /// two keys of one group. A bound is inside a group when it starts with the group
    /// prefix and is longer than that prefix.
    ///
    /// Each subcompaction adds one durable interval or none. The interval holds only
    /// whole key groups:
    /// 1. It starts at the start of the group of the first output key. If the planned
    ///    start bound is inside that group, the group can have keys in the previous
    ///    subcompaction. Then the interval starts after the group.
    /// 2. For a `Completed` subcompaction, it ends after the group of the last output
    ///    key. If the planned end bound is inside that group, the group can have keys
    ///    in the next subcompaction. Then the interval ends at the start of the group.
    /// 3. For an unfinished subcompaction, it ends at the start of the group of its
    ///    last written key. That group stays out, because later output SSTs can still
    ///    add keys to it.
    ///
    /// A subcompaction with no output publishes nothing, because it has no stored key
    /// that shows where its groups are. The trim also leaves out keys that a filter
    /// dropped next to a bound, because the group of such a key can continue in the
    /// next subcompaction.
    ///
    /// This can be optimized. When the subcompactions on both sides of a planned bound
    /// are durable at that bound, the group that crosses the bound and the dropped keys
    /// next to it are durable too. Then one interval across the bound can publish them,
    /// and a `Completed` subcompaction without output can also be published. Now these
    /// keys wait for the final commit.
    fn from_compaction_by_key_group(
        job: &Compaction,
        extractor: &dyn PrefixExtractor,
        rand: &DbRand,
    ) -> Self {
        let mut intervals = Vec::new();
        let mut output = Vec::new();
        let mut output_ids = HashSet::new();
        for subcompaction in job.subcompactions() {
            let output_ssts = subcompaction.output_ssts();
            // Record every output SST, even one that this commit leaves out.
            output_ids.extend(output_ssts.iter().map(|sst| sst.id));
            let Some(interval) = grouped_interval(subcompaction, extractor) else {
                continue;
            };
            for sst in output_ssts {
                push_output_view(&mut output, sst, &interval, rand);
            }
            intervals.push(interval);
        }
        Self {
            intervals,
            output,
            output_ids,
        }
    }

    /// Rebuilds the input runs of the job with `spec` in `core` from this progress.
    ///
    /// The result depends only on the current input views and this progress, so a
    /// commit with more progress replaces the views of an earlier commit, and the
    /// same progress applied twice gives the same result:
    /// - Every view of a recorded output SST is removed first.
    /// - Each durable interval is cut out of each remaining input view. Each piece
    ///   whose range changes gets a new view ID from `rand`. A view with no keys left
    ///   is removed.
    /// - The destination run gets the views in `output`, sorted by start key.
    ///
    /// Input runs stay in the manifest, even when empty, so recovery can still find
    /// them by ID. Updates only `core`. The caller writes the manifest.
    ///
    /// Panics if `spec` has L0 inputs or if its destination is not one of its input
    /// runs. The coordinator passes only jobs that can commit incremental progress.
    ///
    /// Returns `InvalidCompaction` if the segment or an input run of `spec` is missing
    /// from `core`. This can happen when another job removed an input run that both
    /// jobs share.
    pub(crate) fn apply_to_manifest_core(
        &self,
        core: &mut ManifestCore,
        spec: &CompactionSpec,
        rand: &DbRand,
    ) -> Result<(), SlateDBError> {
        assert!(
            !spec.has_l0_sources() && spec.reuses_input_run(),
            "an incremental commit needs a job with only sorted-run inputs and an input \
             run as destination"
        );
        let destination = spec
            .destination()
            .expect("reuses_input_run requires a destination");
        let sources: HashSet<u32> = spec
            .sources()
            .iter()
            .map(SourceId::unwrap_sorted_run)
            .collect();
        let tree = core
            .tree_for_segment_mut(spec.segment())
            .ok_or(SlateDBError::InvalidCompaction)?;
        if !sources
            .iter()
            .all(|id| tree.compacted.iter().any(|sr| sr.id == *id))
        {
            return Err(SlateDBError::InvalidCompaction);
        }
        // Rebuild each input run of the job. Other runs stay unchanged.
        for run in tree
            .compacted
            .iter_mut()
            .filter(|run| sources.contains(&run.id))
        {
            let mut views = Vec::new();
            for view in run.sst_views() {
                // Remove output views of earlier commits from the destination run, the only
                // run that holds them. `output` adds them again below.
                if run.id == destination && self.output_ids.contains(&view.sst.id) {
                    continue;
                }
                // Keep only the keys outside every durable interval.
                let mut retained = vec![view.compacted_effective_range().clone()];
                for interval in &self.intervals {
                    retained = retained
                        .into_iter()
                        .flat_map(|range| interval.subtract_from(&range))
                        .collect();
                }
                // A view can split into several pieces. A view with no keys left is gone.
                for range in retained {
                    views.push(if &range == view.compacted_effective_range() {
                        view.clone()
                    } else {
                        view.try_with_visible_range(range, rand)
                            .expect("retained range intersects its SST")
                    });
                }
            }
            // The destination run also gets the output views, sorted by start key.
            if run.id == destination {
                views.extend(self.output.iter().cloned());
                // Stable sorting preserves version order when output SSTs start at the same key.
                views.sort_by(|left, right| {
                    left.compacted_effective_range()
                        .comparable_start_bound()
                        .cmp(&right.compacted_effective_range().comparable_start_bound())
                });
            }
            *run = SortedRun::new(run.id, views);
        }
        Ok(())
    }
}

/// Adds the part of `sst` that `interval` covers to `output`. An SST inside the
/// interval keeps its whole view. Another SST is cut to the interval with a new view ID
/// from `rand`, and an SST with no key in the interval adds nothing.
fn push_output_view(
    output: &mut Vec<SsTableView>,
    sst: &SsTableHandle,
    interval: &DurableInterval,
    rand: &DbRand,
) {
    let view = SsTableView::identity(sst.clone());
    let range = interval.to_range();
    let inside = [&sst.info.first_entry, &sst.info.last_entry]
        .iter()
        .all(|key| key.as_ref().is_some_and(|key| range.contains(key)));
    if inside {
        output.push(view);
    } else if let Some(cut_view) = view.try_with_visible_range(range, rand) {
        output.push(cut_view);
    }
}

/// Returns the durable interval of `subcompaction` with whole key groups, or `None` if
/// `subcompaction` has no output or no key is left after the trim. See
/// [`IncrementalProgress::from_compaction_by_key_group`] for the steps.
fn grouped_interval(
    subcompaction: &Subcompaction,
    extractor: &dyn PrefixExtractor,
) -> Option<DurableInterval> {
    let output_ssts = subcompaction.output_ssts();
    let first_key = output_ssts
        .first()?
        .info
        .first_entry
        .as_ref()
        .expect("SST writers record the first entry");
    // Step 1: trim the start.
    let start = match subcompaction.range().start_bound() {
        Unbounded => None,
        Included(bound) => {
            let group = KeyGroup::of(first_key, extractor);
            if group.cuts(bound) {
                // The earlier part of the group can be in the previous subcompaction.
                // Start after the group.
                Some(group.end()?)
            } else {
                Some(group.start)
            }
        }
        Excluded(_) => unreachable!("planned ranges start inclusively"),
    };
    let end = if subcompaction.completed() {
        // Step 2: trim the end of a `Completed` subcompaction.
        match subcompaction.range().end_bound() {
            Unbounded => None,
            Excluded(bound) => {
                let last_key = output_ssts
                    .last()?
                    .info
                    .last_entry
                    .as_ref()
                    .expect("SST writers record the last entry");
                let group = KeyGroup::of(last_key, extractor);
                match group.end() {
                    Some(group_end) if !group.cuts(bound) => Some(group_end),
                    // The later part of the group can be in the next subcompaction.
                    // End before the group.
                    _ => Some(group.start),
                }
            }
            Included(_) => unreachable!("planned ranges end exclusively"),
        }
    } else {
        // Step 3: end an unfinished subcompaction before the group of its last written
        // key. The largest group start over all output SSTs only grows when the worker
        // adds output, even if the extractor breaks its contract. So the end never
        // moves back between commits.
        Some(
            output_ssts
                .iter()
                .filter_map(|sst| sst.info.last_entry.as_ref())
                .map(|key| KeyGroup::of(key, extractor).start)
                .max()?,
        )
    };
    let interval = DurableInterval { start, end };
    (!interval.is_empty()).then_some(interval)
}

/// The key group of a stored key: every key that starts with `start`.
///
/// If the extractor returns no prefix for a key, the whole key is the prefix. Then the
/// group also holds the longer keys that start with that key. A trim can hold back
/// those keys too, but it never splits a group that the extractor defines.
struct KeyGroup {
    /// The prefix of the group, which is also its first key.
    start: Bytes,
}

impl KeyGroup {
    /// Returns the group of the stored key `key`.
    fn of(key: &Bytes, extractor: &dyn PrefixExtractor) -> Self {
        let len = extractor
            .prefix_len(&PrefixTarget::Point(key.clone()))
            .unwrap_or(key.len());
        Self {
            start: key.slice(..len),
        }
    }

    /// Returns the first key after the group, or `None` if no key sorts after it.
    fn end(&self) -> Option<Bytes> {
        BytesRange::increment_prefix(&self.start)
    }

    /// Returns `true` if `bound` sorts strictly inside the group, so that the group can
    /// have keys on both sides of `bound`. All keys that start with one prefix form one
    /// contiguous interval, so a prefix test is enough, also for a shortened bound.
    fn cuts(&self, bound: &[u8]) -> bool {
        bound.len() > self.start.len() && bound.starts_with(&self.start)
    }
}

/// A durable interval `[start, end)`: the key range of one subcompaction whose output
/// is fully stored, so an incremental commit can publish it.
///
/// A `Completed` subcompaction has its whole planned range as its durable interval.
/// An unfinished subcompaction has the prefix of its planned range that ends before
/// the last key of its last output SST, so the interval grows as the worker writes
/// more output. With key groups, the interval holds only whole key groups (see
/// [`IncrementalProgress::from_compaction_by_key_group`]).
///
/// The start is always inclusive and the end always exclusive. So `subtract_from`
/// needs no cases for other bound kinds, and its pieces are always valid view ranges.
#[derive(Clone, Debug, PartialEq)]
struct DurableInterval {
    /// First key of the interval. `None` means that the interval starts at the lowest key.
    start: Option<Bytes>,
    /// First key after the interval. `None` means that the interval has no end.
    end: Option<Bytes>,
}

impl DurableInterval {
    /// Converts a planned subcompaction range.
    ///
    /// The planner builds every range as `[boundary, next boundary)`. The first range
    /// has no start bound, and the last range has no end bound. A `Completed`
    /// subcompaction uses the result as it is. An unfinished subcompaction keeps the
    /// start and replaces the end with its boundary key.
    fn from_planned_range(range: &BytesRange) -> Self {
        let start = match range.start_bound() {
            Included(key) => Some(key.clone()),
            Unbounded => None,
            Excluded(_) => unreachable!("planned ranges start inclusively"),
        };
        let end = match range.end_bound() {
            Excluded(key) => Some(key.clone()),
            Unbounded => None,
            Included(_) => unreachable!("planned ranges end exclusively"),
        };
        Self { start, end }
    }

    /// Returns `true` if no key is in the interval.
    fn is_empty(&self) -> bool {
        match (&self.start, &self.end) {
            (_, None) => false,
            // The empty key is the lowest key, so no key is before it.
            (None, Some(end)) => end.is_empty(),
            (Some(start), Some(end)) => start >= end,
        }
    }

    /// Returns the same keys as a `BytesRange`, for example to cut an output SST view.
    fn to_range(&self) -> BytesRange {
        BytesRange::new(
            self.start.clone().map_or(Unbounded, Included),
            self.end.clone().map_or(Unbounded, Excluded),
        )
    }

    /// Returns the parts of `range` outside this interval, in key order.
    ///
    /// The parts are the keys of `range` before `start` and the keys of `range` from
    /// `end` on. A part is missing when this interval has no bound on that side or
    /// when `range` has no keys there. So the result has zero, one, or two ranges, and
    /// each one starts inclusively.
    fn subtract_from(&self, range: &BytesRange) -> Vec<BytesRange> {
        // Keys of `range` in `(.., start)`.
        let before = self
            .start
            .clone()
            .and_then(|start| range.intersect(&BytesRange::new(Unbounded, Excluded(start))));
        // Keys of `range` in `[end, ..)`.
        let after = self
            .end
            .clone()
            .and_then(|end| range.intersect(&BytesRange::new(Included(end), Unbounded)));
        before.into_iter().chain(after).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compactor_state::{CompactionContext, CompactionSpec};
    use crate::db_state::{SsTableHandle, SsTableInfo};
    use crate::format::sst::SST_FORMAT_VERSION_LATEST;
    use crate::subcompaction::Subcompaction;
    use crate::test_utils::VersionSuffixExtractor;
    use bytes::Bytes;
    use std::sync::Arc;
    use ulid::Ulid;

    fn view(first: &'static [u8], last: &'static [u8]) -> SsTableView {
        SsTableView::identity(SsTableHandle::new(
            Ulid::new().into(),
            SST_FORMAT_VERSION_LATEST,
            SsTableInfo {
                first_entry: Some(Bytes::from_static(first)),
                last_entry: Some(Bytes::from_static(last)),
                ..Default::default()
            },
        ))
    }

    #[test]
    fn subtract_preserves_edges_and_gaps() {
        let range = BytesRange::from_slice(b"a".as_slice()..b"z".as_slice());
        let removed = BytesRange::from_slice(b"g".as_slice()..b"m".as_slice());
        assert_eq!(
            DurableInterval::from_planned_range(&removed).subtract_from(&range),
            vec![
                BytesRange::from_slice(b"a".as_slice()..b"g".as_slice()),
                BytesRange::from_slice(b"m".as_slice()..b"z".as_slice()),
            ]
        );
        assert_eq!(
            DurableInterval::from_planned_range(&BytesRange::unbounded()).subtract_from(&range),
            vec![]
        );
        assert_eq!(
            DurableInterval::from_planned_range(&BytesRange::from_slice(b"z".as_slice()..))
                .subtract_from(&range),
            vec![range]
        );
    }

    #[test]
    fn incremental_views_get_new_ids_keep_projections_and_empty_runs() {
        let original = view(b"a", b"z")
            .with_visible_range(BytesRange::from_slice(b"c".as_slice()..b"x".as_slice()));
        let released = view(b"g", b"j");
        let output = view(b"g", b"l");
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).compacted = vec![
            SortedRun::new(2, [released]),
            SortedRun::new(1, [original.clone()]),
        ];
        let job = Compaction::new(
            Ulid::new(),
            CompactionSpec::new(vec![SourceId::SortedRun(2), SourceId::SortedRun(1)], 1),
        );
        let progress = IncrementalProgress {
            intervals: vec![DurableInterval::from_planned_range(
                &BytesRange::from_slice(b"g".as_slice()..b"m".as_slice()),
            )],
            output_ids: HashSet::from([output.sst.id]),
            output: vec![output],
        };
        progress
            .apply_to_manifest_core(&mut core, job.spec(), &DbRand::default())
            .unwrap();
        assert_eq!(
            core.tree
                .compacted
                .iter()
                .map(|run| run.id)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(core.tree.compacted[0].sst_views().is_empty());
        let views = core.tree.compacted[1].sst_views();
        assert_eq!(views.len(), 3);
        assert_ne!(views[0].id, views[2].id);
        for piece in [&views[0], &views[2]] {
            assert_ne!(piece.id, original.id);
            assert_eq!(piece.id.timestamp_ms(), original.id.timestamp_ms());
        }
        assert_eq!(
            views[0].compacted_effective_range(),
            &BytesRange::from_slice(b"c".as_slice()..b"g".as_slice())
        );
        assert_eq!(
            views[2].compacted_effective_range(),
            &BytesRange::from_slice(b"m".as_slice()..b"x".as_slice())
        );
        let before = core.clone();
        progress
            .apply_to_manifest_core(&mut core, job.spec(), &DbRand::default())
            .unwrap();
        assert_eq!(core, before);
    }

    #[test]
    fn clone_projection_preserves_incremental_input_gaps_and_boundary_versions() {
        use crate::manifest::{Manifest, ProjectionConfig};
        let original = view(b"a", b"z");
        let left = original.with_visible_range(BytesRange::from_slice(..b"g".as_slice()));
        let mut right = original.with_visible_range(BytesRange::from_slice(b"m".as_slice()..));
        right.id = Ulid::new();
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).compacted = vec![
            SortedRun::new(3, [view(b"g", b"l")]),
            SortedRun::new(2, [left, right]),
            SortedRun::new(1, [view(b"m", b"m"), view(b"m", b"m"), view(b"z", b"z")]),
        ];
        let manifest = Manifest::initial(core);
        let projected = Manifest::projected(
            &manifest,
            &ProjectionConfig::from_global_range(BytesRange::from_slice(
                b"b".as_slice()..b"y".as_slice(),
            )),
            &DbRand::default(),
        )
        .unwrap();
        let retained = &projected.core.tree.compacted[1];
        assert!(retained.sst_views().iter().all(|view| !view
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"h"))));
        let point = Manifest::projected(
            &manifest,
            &ProjectionConfig::from_global_range(BytesRange::from_slice(
                b"m".as_slice()..=b"m".as_slice(),
            )),
            &DbRand::default(),
        )
        .unwrap();
        assert_eq!(
            point
                .core
                .tree
                .compacted
                .iter()
                .find(|run| run.id == 1)
                .unwrap()
                .sst_views()
                .len(),
            2
        );
    }

    #[test]
    fn durable_progress_includes_empty_completion() {
        let output = view(b"a", b"m").sst;
        let make_job = |output| {
            Compaction::new(
                Ulid::new(),
                CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
            )
            .with_ctx(Some(CompactionContext::new(
                vec![
                    Subcompaction::new(BytesRange::from_slice(..b"z".as_slice()))
                        .with_output_ssts(vec![output]),
                    Subcompaction::new(BytesRange::from_slice(b"z".as_slice()..))
                        .with_status(crate::subcompaction::SubcompactionStatus::Completed),
                ],
                Some(0),
            )))
        };
        let progress =
            IncrementalProgress::from_compaction(&make_job(output), None, &DbRand::default());
        assert_eq!(
            progress.intervals,
            vec![
                DurableInterval {
                    start: None,
                    end: Some(Bytes::from_static(b"m")),
                },
                DurableInterval {
                    start: Some(Bytes::from_static(b"z")),
                    end: None,
                },
            ]
        );
        assert_eq!(progress.output.len(), 1);
    }

    #[test]
    fn durable_prefix_excludes_all_versions_of_the_boundary_key() {
        let untouched = view(b"a", b"e");
        let first = view(b"f", b"m");
        let last = view(b"m", b"m");
        let job = Compaction::new(
            Ulid::new(),
            CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
        )
        .with_ctx(Some(CompactionContext::new(
            vec![
                Subcompaction::new(BytesRange::unbounded()).with_output_ssts(vec![
                    untouched.sst,
                    first.sst,
                    last.sst,
                ]),
            ],
            Some(0),
        )));
        let progress = IncrementalProgress::from_compaction(&job, None, &DbRand::default());
        assert_eq!(progress.output.len(), 2);
        assert!(progress.output[0].visible_range().is_none());
        assert!(!progress.output[1]
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"m")));
    }

    #[test]
    fn destination_rebuild_preserves_output_order_and_expands_previous_views() {
        let input = view(b"a", b"z");
        let outputs = [view(b"g", b"g"), view(b"g", b"g"), view(b"g", b"m")];
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).compacted = vec![SortedRun::new(1, [input.clone()])];
        let mut job = Compaction::new(
            Ulid::new(),
            CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
        )
        .with_ctx(Some(CompactionContext::new(
            vec![
                Subcompaction::new(BytesRange::from_slice(b"g".as_slice()..b"n".as_slice()))
                    .with_output_ssts(outputs.iter().map(|view| view.sst.clone()).collect()),
            ],
            Some(0),
        )));
        let progress = IncrementalProgress::from_compaction(&job, None, &DbRand::default());
        progress
            .apply_to_manifest_core(&mut core, job.spec(), &DbRand::default())
            .unwrap();
        let views = core.tree.compacted[0].sst_views();
        assert_eq!(views.len(), 5);
        assert_eq!(views[0].sst.id, input.sst.id);
        assert_eq!(
            views[1..4]
                .iter()
                .map(|view| view.sst.id)
                .collect::<Vec<_>>(),
            outputs.iter().map(|view| view.sst.id).collect::<Vec<_>>()
        );
        assert!(!views[3]
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"m")));

        let mut ctx = job.ctx().unwrap().clone();
        ctx.mark_completed(0);
        job.set_ctx(Some(ctx));
        let completed = IncrementalProgress::from_compaction(&job, None, &DbRand::default());
        completed
            .apply_to_manifest_core(&mut core, job.spec(), &DbRand::default())
            .unwrap();
        let views = core.tree.compacted[0].sst_views();
        assert_eq!(views.len(), 5);
        assert_eq!(views[4].sst.id, input.sst.id);
        assert!(views[3]
            .compacted_effective_range()
            .contains(&Bytes::from_static(b"m")));
        assert_eq!(
            views[4].compacted_effective_range().start_bound(),
            Included(&Bytes::from_static(b"n"))
        );
        let before = core.clone();
        completed
            .apply_to_manifest_core(&mut core, job.spec(), &DbRand::default())
            .unwrap();
        assert_eq!(core, before);
    }

    /// A job over sorted run 1, which is also its destination.
    fn job_with(subcompactions: Vec<Subcompaction>) -> Compaction {
        Compaction::new(
            Ulid::new(),
            CompactionSpec::new(vec![SourceId::SortedRun(1)], 1),
        )
        .with_ctx(Some(CompactionContext::new(subcompactions, Some(0))))
    }

    fn grouped(subcompactions: Vec<Subcompaction>) -> IncrementalProgress {
        IncrementalProgress::from_compaction(
            &job_with(subcompactions),
            Some(&VersionSuffixExtractor),
            &DbRand::default(),
        )
    }

    fn subcompaction(range: BytesRange, outputs: &[SsTableView]) -> Subcompaction {
        Subcompaction::new(range)
            .with_output_ssts(outputs.iter().map(|view| view.sst.clone()).collect())
    }

    fn completed(subcompaction: Subcompaction) -> Subcompaction {
        subcompaction.with_status(crate::subcompaction::SubcompactionStatus::Completed)
    }

    fn interval(start: Option<&'static [u8]>, end: Option<&'static [u8]>) -> DurableInterval {
        DurableInterval {
            start: start.map(Bytes::from_static),
            end: end.map(Bytes::from_static),
        }
    }

    /// The range below the planned bound `c@5`, which cuts the group `c@`.
    fn below_cut() -> BytesRange {
        BytesRange::from_slice(..b"c@5".as_slice())
    }

    /// The range from the planned bound `c@5` on.
    fn above_cut() -> BytesRange {
        BytesRange::from_slice(b"c@5".as_slice()..)
    }

    fn contains(view: &SsTableView, key: &[u8]) -> bool {
        view.compacted_effective_range()
            .contains(&Bytes::copy_from_slice(key))
    }

    #[test]
    fn grouped_progress_ends_unfinished_range_before_group_of_last_key() {
        // The keys of the group `c@` span the last two SSTs.
        let outputs = [
            view(b"a@1", b"a@9"),
            view(b"b@1", b"c@2"),
            view(b"c@3", b"c@7"),
        ];

        let progress = grouped(vec![subcompaction(BytesRange::unbounded(), &outputs)]);

        // No key of `c@` is durable yet. The first SST stays whole, the second SST
        // is cut before `c@`, and the third SST drops out.
        assert_eq!(progress.intervals, vec![interval(None, Some(b"c@"))]);
        assert_eq!(progress.output.len(), 2);
        assert_eq!(progress.output[0].id, outputs[0].id);
        assert!(progress.output[0].visible_range().is_none());
        assert!(contains(&progress.output[1], b"b@1"));
        assert!(!contains(&progress.output[1], b"c@1"));
    }

    #[test]
    fn grouped_progress_holds_back_cut_group_at_end_of_completed_range() {
        let outputs = [view(b"a@1", b"c@3")];

        let progress = grouped(vec![
            completed(subcompaction(below_cut(), &outputs)),
            subcompaction(above_cut(), &[]),
        ]);

        // The next range holds the rest of `c@` and is not durable, so the
        // interval ends at the start of `c@`.
        assert_eq!(progress.intervals, vec![interval(None, Some(b"c@"))]);
        assert_eq!(progress.output.len(), 1);
        assert!(contains(&progress.output[0], b"b@1"));
        assert!(!contains(&progress.output[0], b"c@1"));
    }

    #[test]
    fn grouped_progress_holds_back_dropped_keys_before_a_bound() {
        // The filter dropped every key after `a@9`, so no output key shows whether a
        // group crosses the bound.
        let outputs = [view(b"a@1", b"a@9")];

        let progress = grouped(vec![
            completed(subcompaction(below_cut(), &outputs)),
            subcompaction(above_cut(), &[]),
        ]);

        // The interval ends right after the group `a@`.
        assert_eq!(progress.intervals, vec![interval(None, Some(b"aA"))]);
        assert!(progress.output[0].visible_range().is_none());
    }

    #[test]
    fn grouped_progress_starts_after_cut_group() {
        let outputs = [view(b"c@6", b"e@1")];

        let progress = grouped(vec![
            subcompaction(below_cut(), &[]),
            completed(subcompaction(above_cut(), &outputs)),
        ]);

        // The range before holds the start of `c@` and is not durable, so the
        // interval starts after `c@`.
        assert_eq!(progress.intervals, vec![interval(Some(b"cA"), None)]);
        assert!(!contains(&progress.output[0], b"c@6"));
        assert!(contains(&progress.output[0], b"d@1"));
    }

    #[test]
    fn grouped_progress_holds_back_cut_group_between_completed_ranges() {
        let below = [view(b"a@1", b"c@3")];
        let above = [view(b"c@6", b"e@1")];

        let progress = grouped(vec![
            completed(subcompaction(below_cut(), &below)),
            completed(subcompaction(above_cut(), &above)),
        ]);

        // Each range is trimmed by itself, so the group `c@` waits for the final
        // commit, even though both ranges are durable.
        assert_eq!(
            progress.intervals,
            vec![interval(None, Some(b"c@")), interval(Some(b"cA"), None)]
        );
        assert_eq!(progress.output.len(), 2);
        assert!(!contains(&progress.output[0], b"c@1"));
        assert!(!contains(&progress.output[1], b"c@6"));
        assert!(contains(&progress.output[1], b"d@1"));
    }

    #[test]
    fn grouped_progress_starts_unfinished_range_after_cut_group() {
        let below = [view(b"a@1", b"c@3")];
        let above = [view(b"c@6", b"d@4"), view(b"d@5", b"e@2")];

        let progress = grouped(vec![
            completed(subcompaction(below_cut(), &below)),
            subcompaction(above_cut(), &above),
        ]);

        // The unfinished range starts after the cut group `c@` and ends at the group
        // of its last written key.
        assert_eq!(
            progress.intervals,
            vec![
                interval(None, Some(b"c@")),
                interval(Some(b"cA"), Some(b"e@"))
            ]
        );
        assert_eq!(progress.output.len(), 3);
        assert!(!contains(&progress.output[1], b"c@6"));
        assert!(contains(&progress.output[1], b"d@1"));
        assert!(contains(&progress.output[2], b"d@5"));
        assert!(!contains(&progress.output[2], b"e@2"));
    }

    #[test]
    fn grouped_progress_uses_whole_key_without_prefix_as_prefix() {
        // `b` has no `@`, so its group holds `b` and every key that starts with `b`.
        let outputs = [view(b"a@1", b"b")];
        let below = |end: &'static [u8]| BytesRange::from_slice(..end);
        let above = |start: &'static [u8]| BytesRange::from_slice(start..);

        // An unfinished range ends before `b`, because later output SSTs can still
        // hold versions of `b`.
        let unfinished = grouped(vec![subcompaction(BytesRange::unbounded(), &outputs)]);
        assert_eq!(unfinished.intervals, vec![interval(None, Some(b"b"))]);

        // A completed range ends after every key that starts with `b`.
        let after_group = grouped(vec![
            completed(subcompaction(below(b"c"), &outputs)),
            subcompaction(above(b"c"), &[]),
        ]);
        assert_eq!(after_group.intervals, vec![interval(None, Some(b"c"))]);

        // The bound `b@5` is inside the group of `b`, so the range ends before `b`.
        let cut = grouped(vec![
            completed(subcompaction(below(b"b@5"), &outputs)),
            subcompaction(above(b"b@5"), &[]),
        ]);
        assert_eq!(cut.intervals, vec![interval(None, Some(b"b"))]);
    }

    #[test]
    fn grouped_progress_publishes_nothing_without_output_next_to_a_bound() {
        let progress = grouped(vec![
            subcompaction(below_cut(), &[]),
            completed(subcompaction(above_cut(), &[])),
        ]);

        assert!(progress.intervals.is_empty());
        assert!(progress.output.is_empty());
    }

    /// Applies three commits in a row, as the coordinator does, and makes sure that
    /// the group `c@` is all in the input or all in the output after each commit.
    #[test]
    fn grouped_progress_publishes_whole_groups_in_every_state() {
        // Run 1 is the only input and the destination. Its one view covers every key.
        let input = view(b"a@1", b"e@9");
        let mut core = ManifestCore::new();
        Arc::make_mut(&mut core.tree).compacted = vec![SortedRun::new(1, [input.clone()])];
        let below = [view(b"a@1", b"c@3")];
        let above = [
            view(b"c@6", b"d@4"),
            view(b"d@5", b"e@2"),
            view(b"e@3", b"e@9"),
        ];
        let states = [
            // Range 1 completes before range 2 starts.
            vec![
                completed(subcompaction(below_cut(), &below)),
                subcompaction(above_cut(), &[]),
            ],
            // Range 2 writes past the group `c@`.
            vec![
                completed(subcompaction(below_cut(), &below)),
                subcompaction(above_cut(), &above[..2]),
            ],
            // Range 2 completes.
            vec![
                completed(subcompaction(below_cut(), &below)),
                completed(subcompaction(above_cut(), &above)),
            ],
        ];
        // For each view that holds `key`, whether it is an output view.
        let holders = |core: &ManifestCore, key: &[u8]| -> Vec<bool> {
            core.tree
                .compacted
                .iter()
                .flat_map(|run| run.sst_views())
                .filter(|view| contains(view, key))
                .map(|view| view.sst.id != input.sst.id)
                .collect()
        };

        for (state, subcompactions) in states.into_iter().enumerate() {
            let job = job_with(subcompactions);
            IncrementalProgress::from_compaction(
                &job,
                Some(&VersionSuffixExtractor),
                &DbRand::default(),
            )
            .apply_to_manifest_core(&mut core, job.spec(), &DbRand::default())
            .unwrap();

            for key in [b"a@5".as_slice(), b"c@3", b"c@6", b"e@5"] {
                assert_eq!(
                    holders(&core, key).len(),
                    1,
                    "state {state}: one view holds {key:?}"
                );
            }
            assert_eq!(
                holders(&core, b"c@3"),
                holders(&core, b"c@6"),
                "state {state}: the group c@ is split between input and output"
            );
        }
    }
}
