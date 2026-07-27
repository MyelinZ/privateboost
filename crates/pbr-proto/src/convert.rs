//! Wire messages to `pbr-core` types and back.
//!
//! The `*_from_proto` direction is the decode trust boundary: it rejects what
//! `pbr-core`'s own types would otherwise accept as well-formed: a zero
//! evaluation point, a field element that is not a canonical residue, a
//! commitment that is not 32 bytes, a flattened tree whose child indices do
//! not describe a tree.
//!
//! Encoding is lossy in one place: `SplitNode`, `SplitProto` and `LeafProto`
//! carry only the fields prediction and routing read, so `gain` and a split's
//! per-side g/h sums come back 0.0. A decoded model predicts identically to
//! the one that was encoded; it is not field-equal to it.

use crate::v1::{
    LeafProto, ModelProto, NumericEdges, ShareVector, SplitNode, SplitProto, TreeNodeProto,
    TreeProto, tree_node_proto,
};
use pbr_core::crypto::field::PRIME;
use pbr_core::{BinConfiguration, Commitment, F, Model, Share, SplitDecision, Tree, TreeNode};

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("field element {0} >= PRIME")]
    FieldElementOutOfRange(u64),
    #[error("evaluation point x must be nonzero")]
    InvalidEvaluationPoint,
    #[error("commitment must be 32 bytes, got {0}")]
    BadCommitmentLength(usize),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("malformed flattened tree: {0}")]
    MalformedTree(&'static str),
}

pub fn share_to_proto(share: &Share) -> ShareVector {
    ShareVector {
        // x is a shareholder index in 1..=n_parties, so the narrow cannot
        // truncate; core `Share.x` stays u64 to live in the F_p domain.
        x: share.x as u32,
        values: share.values.iter().map(|v| v.inner()).collect(),
    }
}

pub fn share_from_proto(proto: ShareVector) -> Result<Share, ConvertError> {
    if proto.x == 0 {
        return Err(ConvertError::InvalidEvaluationPoint);
    }
    let mut values = Vec::with_capacity(proto.values.len());
    for raw in proto.values {
        if raw >= PRIME {
            return Err(ConvertError::FieldElementOutOfRange(raw));
        }
        values.push(F::from_u64(raw));
    }
    Ok(Share {
        x: proto.x.into(),
        values,
    })
}

pub fn commitment_to_bytes(c: &Commitment) -> Vec<u8> {
    c.0.to_vec()
}

pub fn commitment_from_bytes(bytes: &[u8]) -> Result<Commitment, ConvertError> {
    let arr: [u8; 32] = bytes
        .try_into()
        .map_err(|_| ConvertError::BadCommitmentLength(bytes.len()))?;
    Ok(Commitment(arr))
}

pub fn bin_config_to_edges(cfg: &BinConfiguration) -> NumericEdges {
    NumericEdges {
        feature_idx: cfg.feature_idx as u32,
        edges: cfg.edges.clone(),
    }
}

pub fn edges_to_bin_config(proto: NumericEdges) -> BinConfiguration {
    BinConfiguration {
        feature_idx: proto.feature_idx as usize,
        edges: proto.edges,
    }
}

pub fn split_to_proto(split: &SplitDecision) -> SplitNode {
    SplitNode {
        feature_idx: split.feature_idx as u32,
        threshold: split.threshold,
        left_child_id: split.left_child_id as u32,
        right_child_id: split.right_child_id as u32,
    }
}

/// The `SplitDecision` a wire `SplitNode` describes. `node_id` is the
/// `RoundContext.splits_so_far` map key the message was stored under; the
/// message carries no id of its own. `gain` and the per-side g/h sums are not
/// on the wire and are filled in with 0.0, so the result routes records
/// correctly but says nothing about split quality.
pub fn split_from_proto(proto: SplitNode, node_id: u32) -> SplitDecision {
    SplitDecision {
        node_id: node_id as usize,
        feature_idx: proto.feature_idx as usize,
        threshold: proto.threshold,
        gain: 0.0,
        left_child_id: proto.left_child_id as usize,
        right_child_id: proto.right_child_id as usize,
        g_left: 0.0,
        h_left: 0.0,
        g_right: 0.0,
        h_right: 0.0,
    }
}

/// Flatten `node` into `out` in pre-order and return its index. A split is
/// pushed before its subtrees with placeholder child indices, then
/// backpatched once the subtrees know theirs, so every child index in the
/// output is strictly greater than its parent's, the invariant
/// `tree_from_proto` enforces.
fn flatten_tree_node(node: &TreeNode, out: &mut Vec<TreeNodeProto>) -> u32 {
    let idx = out.len() as u32;
    match node {
        TreeNode::Leaf { value } => out.push(TreeNodeProto {
            node: Some(tree_node_proto::Node::Leaf(LeafProto { value: *value })),
        }),
        TreeNode::Split {
            feature_idx,
            threshold,
            left,
            right,
            ..
        } => {
            out.push(TreeNodeProto {
                node: Some(tree_node_proto::Node::Split(SplitProto {
                    feature_idx: *feature_idx as u32,
                    threshold: *threshold,
                    left: 0,
                    right: 0,
                })),
            });
            let l = flatten_tree_node(left, out);
            let r = flatten_tree_node(right, out);
            if let Some(tree_node_proto::Node::Split(s)) = &mut out[idx as usize].node {
                s.left = l;
                s.right = r;
            }
        }
    }
    idx
}

/// Claim the child at index `child` for the split at `parent`. Pre-order
/// bounds (`parent < child < built.len()`) reject out-of-range, self, and
/// backward references, which is what makes reference cycles impossible, and
/// `take` leaves an empty slot behind, so a node claimed by two splits
/// surfaces as an error instead of silently duplicating a subtree.
fn take_child(
    built: &mut [Option<TreeNode>],
    parent: usize,
    child: u32,
) -> Result<Box<TreeNode>, ConvertError> {
    let child = child as usize;
    if child <= parent || child >= built.len() {
        return Err(ConvertError::MalformedTree(
            "child index outside pre-order range",
        ));
    }
    built[child]
        .take()
        .map(Box::new)
        .ok_or(ConvertError::MalformedTree("child referenced twice"))
}

fn tree_to_proto(tree: &Tree) -> TreeProto {
    let mut nodes = Vec::new();
    flatten_tree_node(&tree.root, &mut nodes);
    TreeProto { nodes }
}

/// Decoding never recurses, but the rebuilt `pbr_core::TreeNode` is a Box
/// chain whose `Drop` does, so decoded tree DEPTH is still a stack hazard;
/// the node count bounds it. Real trees are tiny (a complete depth-11 tree
/// is 4095 nodes), so anything past this cap is hostile or corrupt, and 4096
/// recursive drop frames sit comfortably inside any thread stack.
const MAX_TREE_NODES: usize = 4096;

/// Rebuild the recursive `TreeNode` from the flat wire array without
/// recursing on untrusted input: nodes are built in reverse index order, so
/// a split's children (strictly higher indices) always exist before it.
fn tree_from_proto(proto: TreeProto) -> Result<Tree, ConvertError> {
    let n = proto.nodes.len();
    if n > MAX_TREE_NODES {
        return Err(ConvertError::MalformedTree("more than MAX_TREE_NODES nodes"));
    }
    let mut built: Vec<Option<TreeNode>> = (0..n).map(|_| None).collect();
    for (idx, node) in proto.nodes.into_iter().enumerate().rev() {
        let rebuilt = match node.node.ok_or(ConvertError::MissingField("node"))? {
            tree_node_proto::Node::Leaf(leaf) => TreeNode::Leaf { value: leaf.value },
            tree_node_proto::Node::Split(split) => TreeNode::Split {
                feature_idx: split.feature_idx as usize,
                threshold: split.threshold,
                gain: 0.0,
                left: take_child(&mut built, idx, split.left)?,
                right: take_child(&mut built, idx, split.right)?,
            },
        };
        built[idx] = Some(rebuilt);
    }
    let root = built
        .first_mut()
        .and_then(Option::take)
        .ok_or(ConvertError::MalformedTree("no nodes"))?;
    if built.iter().any(Option::is_some) {
        return Err(ConvertError::MalformedTree("unreferenced node"));
    }
    Ok(Tree { root })
}

pub fn model_to_proto(model: &Model) -> ModelProto {
    ModelProto {
        initial_prediction: model.initial_prediction,
        learning_rate: model.learning_rate,
        trees: model.trees.iter().map(tree_to_proto).collect(),
    }
}

pub fn model_from_proto(proto: ModelProto) -> Result<Model, ConvertError> {
    let mut model = Model::new(proto.initial_prediction, proto.learning_rate);
    for tree in proto.trees {
        model.add_tree(tree_from_proto(tree)?);
    }
    Ok(model)
}
