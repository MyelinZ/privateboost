use std::collections::HashMap;

pub type NodeId = i32;
pub type Depth = i32;

#[derive(Debug, Clone)]
pub enum TreeNode {
    Split(SplitNode),
    Leaf(LeafNode),
}

#[derive(Debug, Clone)]
pub struct SplitNode {
    pub feature_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left: Box<TreeNode>,
    pub right: Box<TreeNode>,
}

#[derive(Debug, Clone)]
pub struct LeafNode {
    pub value: f64,
    pub n_samples: usize,
}

#[derive(Debug, Clone)]
pub struct Tree {
    pub root: TreeNode,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub initial_prediction: f64,
    pub learning_rate: f64,
    pub trees: Vec<Tree>,
}

impl Model {
    pub fn new(initial_prediction: f64, learning_rate: f64) -> Self {
        Self {
            initial_prediction,
            learning_rate,
            trees: Vec::new(),
        }
    }

    pub fn add_tree(&mut self, tree: Tree) {
        self.trees.push(tree);
    }

    /// Predict for a single sample.
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        let mut pred = self.initial_prediction;
        for tree in &self.trees {
            pred += self.learning_rate * tree_predict(&tree.root, features);
        }
        pred
    }
}

fn tree_predict(node: &TreeNode, features: &[f64]) -> f64 {
    match node {
        TreeNode::Leaf(leaf) => leaf.value,
        TreeNode::Split(split) => {
            if features[split.feature_idx] <= split.threshold {
                tree_predict(&split.left, features)
            } else {
                tree_predict(&split.right, features)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinConfiguration {
    pub feature_idx: usize,
    pub edges: Vec<f64>,
    pub inner_edges: Vec<f64>,
    pub n_bins: usize,
}

#[derive(Debug, Clone)]
pub struct SplitDecision {
    pub node_id: NodeId,
    pub feature_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left_child_id: NodeId,
    pub right_child_id: NodeId,
    pub g_left: f64,
    pub h_left: f64,
    pub g_right: f64,
    pub h_right: f64,
}

/// Aggregated gradient/hessian totals for a node.
#[derive(Debug, Clone)]
pub struct NodeTotals {
    pub gradient_sum: f64,
    pub hessian_sum: f64,
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub run_id: String,
    pub n_bins: usize,
    pub threshold: usize,
    pub min_clients: usize,
    pub learning_rate: f64,
    pub lambda_reg: f64,
    pub n_trees: usize,
    pub max_depth: usize,
    pub loss: String,
    pub target_count: usize,
    pub features: Vec<String>,
    pub target_column: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepType {
    Stats,
    Gradients,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepId {
    pub step_type: StepType,
    pub round_id: i32,
    pub depth: i32,
}

impl StepId {
    pub fn stats() -> Self {
        Self {
            step_type: StepType::Stats,
            round_id: 0,
            depth: 0,
        }
    }

    pub fn gradients(round_id: i32, depth: i32) -> Self {
        Self {
            step_type: StepType::Gradients,
            round_id,
            depth,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConsensusResult {
    Bins {
        bins: Vec<BinConfiguration>,
        initial_prediction: f64,
    },
    Splits {
        splits: HashMap<NodeId, SplitDecision>,
    },
    Tree {
        tree: Tree,
    },
}
