//! Wire round ids and the `RoundContext` payloads the round loop publishes.
//!
//! Wire round ids are nonzero, because a fresh client polls with the proto
//! default `last_seen_round_id = 0` and must still receive the current context,
//! and strictly increasing across the session: stats is 1, gradient rounds pack
//! `(tree + 1, depth)`, and the completed context takes an id above them all.
//! Clients read `tree_idx` and `depth` from the context fields; the id itself
//! is opaque to them.

use super::gather::SourceSlot;
use pbr_proto::convert::{bin_config_to_edges, model_to_proto, split_to_proto};
use pbr_proto::v1::RoundContext;
use pbr_core::{Aggregator, RoundContext as CoreRoundContext, SplitDecision};
use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

/// Carried by OpenRound/CloseRound and in the published context, where it
/// tells clients "submit stats".
pub(super) const STATS_DEPTH_SENTINEL: u32 = u32::MAX;
pub(super) const STATS_ROUND_ID: u64 = 1;

pub(super) fn gradient_round_id(tree: u64, depth: usize) -> u64 {
    ((tree + 1) << 32) | depth as u64
}

fn completed_round_id(n_trees: usize) -> u64 {
    gradient_round_id(n_trees as u64, u32::MAX as usize)
}

fn deadline_after(window: Duration) -> prost_types::Timestamp {
    prost_types::Timestamp::from(SystemTime::now() + window)
}

/// Children of the split set that are not themselves split (depth 0: just the
/// root). Must equal the node set `pbr-core`'s client derives from the same
/// splits under path hiding, or the gather asks shareholders for node ids no
/// client submitted to.
fn active_node_ids(splits: &BTreeMap<usize, SplitDecision>, depth: usize) -> Vec<u32> {
    if depth == 0 {
        return vec![0];
    }
    let mut active: Vec<u32> = Vec::new();
    for split in splits.values() {
        if !splits.contains_key(&split.left_child_id) {
            active.push(split.left_child_id as u32);
        }
        if !splits.contains_key(&split.right_child_id) {
            active.push(split.right_child_id as u32);
        }
    }
    if active.is_empty() {
        vec![0]
    } else {
        active.sort_unstable();
        active
    }
}

pub(super) fn stats_ctx(window: Duration, session_id: &str) -> RoundContext {
    RoundContext {
        tree_idx: 0,
        depth: STATS_DEPTH_SENTINEL,
        round_id: STATS_ROUND_ID,
        active_node_ids: Vec::new(),
        splits_so_far: Default::default(),
        bin_edges: Vec::new(),
        model: None,
        submission_deadline: Some(deadline_after(window)),
        session_id: session_id.to_string(),
    }
}

pub(super) fn gradient_ctx(
    pctx: &CoreRoundContext,
    round_id: u64,
    window: Duration,
    session_id: &str,
) -> RoundContext {
    RoundContext {
        tree_idx: pctx.round_id as u32,
        depth: pctx.depth as u32,
        round_id,
        active_node_ids: active_node_ids(&pctx.splits, pctx.depth),
        splits_so_far: pctx
            .splits
            .iter()
            .map(|(id, s)| (*id as u32, split_to_proto(s)))
            .collect(),
        bin_edges: pctx.bins.iter().map(bin_config_to_edges).collect(),
        model: Some(model_to_proto(&pctx.model)),
        submission_deadline: Some(deadline_after(window)),
        session_id: session_id.to_string(),
    }
}

pub(super) fn completed_ctx(
    agg: &Aggregator<SourceSlot>,
    n_trees: usize,
    session_id: &str,
) -> RoundContext {
    let pctx = agg.round_context();
    RoundContext {
        tree_idx: n_trees as u32,
        depth: STATS_DEPTH_SENTINEL,
        round_id: completed_round_id(n_trees),
        active_node_ids: Vec::new(),
        splits_so_far: Default::default(),
        bin_edges: pctx.bins.iter().map(bin_config_to_edges).collect(),
        model: Some(model_to_proto(agg.model())),
        submission_deadline: None,
        session_id: session_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_ids_are_nonzero_and_strictly_increasing() {
        let mut ids = vec![STATS_ROUND_ID];
        for tree in 0..3u64 {
            for depth in 0..4usize {
                ids.push(gradient_round_id(tree, depth));
            }
        }
        ids.push(completed_round_id(3));
        assert!(ids.iter().all(|&id| id > 0 && id <= i64::MAX as u64));
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "ids not increasing: {ids:?}"
        );
    }

    #[test]
    fn active_nodes_match_client_leaf_rule() {
        let split = |node_id, left, right| SplitDecision {
            node_id,
            feature_idx: 0,
            threshold: 0.0,
            gain: 0.0,
            left_child_id: left,
            right_child_id: right,
            g_left: 0.0,
            h_left: 0.0,
            g_right: 0.0,
            h_right: 0.0,
        };
        let mut splits = BTreeMap::new();
        // Depth 0 is always the root, even with stale splits present.
        splits.insert(0, split(0, 1, 2));
        assert_eq!(active_node_ids(&splits, 0), vec![0]);
        assert_eq!(active_node_ids(&splits, 1), vec![1, 2]);
        splits.insert(1, split(1, 3, 4));
        assert_eq!(active_node_ids(&splits, 2), vec![2, 3, 4]);
        assert_eq!(active_node_ids(&BTreeMap::new(), 1), vec![0]);
    }
}
