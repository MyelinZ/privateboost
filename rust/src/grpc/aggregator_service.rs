use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use sha2::{Digest, Sha256};

use crate::domain::aggregator::Aggregator;
use crate::domain::model::*;
use crate::grpc::remote_shareholder::RemoteShareHolder;
use crate::grpc::vec_to_ndarray;
use crate::proto;

/// Run the aggregator polling loop for a single run.
/// The aggregator polls shareholders for phase changes and computes results when frozen.
pub async fn run_aggregator_loop(
    aggregator_id: i32,
    aggregator: Aggregator,
    remote_shareholders: Vec<Arc<RemoteShareHolder>>,
    run_id: String,
    n_trees: usize,
    max_depth: usize,
) -> Result<()> {
    tracing::info!(aggregator_id, %run_id, "Starting aggregator loop");

    // Wait for stats to freeze
    loop {
        let state = remote_shareholders[0].get_run_state(&run_id).await?;
        if state.phase == proto::Phase::FrozenStats as i32 {
            break;
        }
        if state.phase == proto::Phase::TrainingComplete as i32 {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // Compute bins
    let (bin_configs, initial_prediction, _n_clients, n_features) =
        aggregator.compute_bins().await?;

    // Build BinsResult proto and submit to all shareholders
    let bins_result = proto::BinsResult {
        bins: bin_configs.iter().map(bin_config_to_proto).collect(),
        initial_prediction,
    };
    let bins_bytes = prost::Message::encode_to_vec(&bins_result);
    let bins_hash = sha256(&bins_bytes);

    let step = proto::StepId {
        step_type: proto::StepType::Stats as i32,
        round_id: 0,
        depth: 0,
    };

    for rsh in &remote_shareholders {
        rsh.submit_result(
            &run_id,
            &step,
            aggregator_id,
            &bins_hash,
            proto::submit_result_request::Result::BinsResult(bins_result.clone()),
        )
        .await?;
    }

    // Training rounds
    for round_id in 0..n_trees as i32 {
        let mut node_totals: HashMap<NodeId, NodeTotals> = HashMap::new();
        let mut splits: HashMap<NodeId, SplitDecision> = HashMap::new();
        let mut next_node_id: NodeId = 1;

        for depth in 0..max_depth as i32 {
            // Wait for gradients to freeze
            loop {
                let state = remote_shareholders[0].get_run_state(&run_id).await?;
                if state.phase == proto::Phase::FrozenGradients as i32
                    && state.round_id == round_id
                    && state.depth == depth
                {
                    break;
                }
                if state.phase == proto::Phase::TrainingComplete as i32 {
                    return Ok(());
                }
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }

            if depth == 0 {
                node_totals.clear();
                splits.clear();
                next_node_id = 1;
            }

            let found_splits = aggregator
                .compute_splits(
                    depth,
                    1,
                    &bin_configs,
                    n_features,
                    &mut node_totals,
                    &mut splits,
                    &mut next_node_id,
                )
                .await?;

            // Submit splits result
            let splits_result = proto::SplitsResult {
                splits: splits
                    .iter()
                    .map(|(&nid, sd)| (nid, split_decision_to_proto(sd)))
                    .collect(),
            };
            let splits_bytes = prost::Message::encode_to_vec(&splits_result);
            let splits_hash = sha256(&splits_bytes);

            let step = proto::StepId {
                step_type: proto::StepType::Gradients as i32,
                round_id,
                depth,
            };

            for rsh in &remote_shareholders {
                rsh.submit_result(
                    &run_id,
                    &step,
                    aggregator_id,
                    &splits_hash,
                    proto::submit_result_request::Result::SplitsResult(splits_result.clone()),
                )
                .await?;
            }

            if !found_splits {
                break;
            }
        }

        // Build tree and submit as TreeResult
        let tree = aggregator.build_tree(&splits, &node_totals);
        let tree_result = proto::TreeResult {
            tree: Some(tree_to_proto(&tree)),
        };
        let tree_bytes = prost::Message::encode_to_vec(&tree_result);
        let tree_hash = sha256(&tree_bytes);

        let step = proto::StepId {
            step_type: proto::StepType::Gradients as i32,
            round_id,
            depth: max_depth as i32,
        };

        for rsh in &remote_shareholders {
            rsh.submit_result(
                &run_id,
                &step,
                aggregator_id,
                &tree_hash,
                proto::submit_result_request::Result::TreeResult(tree_result.clone()),
            )
            .await?;
        }
    }

    tracing::info!(aggregator_id, %run_id, "Aggregator loop complete");
    Ok(())
}

fn sha256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

fn bin_config_to_proto(bc: &BinConfiguration) -> proto::BinConfiguration {
    proto::BinConfiguration {
        feature_idx: bc.feature_idx as i32,
        edges: Some(vec_to_ndarray(&bc.edges)),
        inner_edges: Some(vec_to_ndarray(&bc.inner_edges)),
        n_bins: bc.n_bins as i32,
    }
}

fn split_decision_to_proto(sd: &SplitDecision) -> proto::SplitDecision {
    proto::SplitDecision {
        node_id: sd.node_id,
        feature_idx: sd.feature_idx as i32,
        threshold: sd.threshold,
        gain: sd.gain,
        left_child_id: sd.left_child_id,
        right_child_id: sd.right_child_id,
    }
}

fn tree_to_proto(tree: &Tree) -> proto::Tree {
    proto::Tree {
        root: Some(tree_node_to_proto(&tree.root)),
    }
}

fn tree_node_to_proto(node: &TreeNode) -> proto::TreeNode {
    match node {
        TreeNode::Split(s) => proto::TreeNode {
            node: Some(proto::tree_node::Node::Split(Box::new(proto::SplitNode {
                feature_idx: s.feature_idx as i32,
                threshold: s.threshold,
                left: Some(Box::new(tree_node_to_proto(&s.left))),
                right: Some(Box::new(tree_node_to_proto(&s.right))),
            }))),
        },
        TreeNode::Leaf(l) => proto::TreeNode {
            node: Some(proto::tree_node::Node::Leaf(proto::LeafNode {
                value: l.value,
            })),
        },
    }
}
