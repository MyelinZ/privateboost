use crate::crypto::{Commitment, Share};

#[derive(Clone, Debug)]
pub enum Loss {
    Squared,
    Logistic,
}

pub struct CommittedStatsShare {
    pub commitment: Commitment,
    pub share: Share,
}

pub struct CommittedGradientShare {
    pub round_id: u64,
    pub depth: usize,
    pub commitment: Commitment,
    pub share: Share,
    pub node_id: usize,
}

#[derive(Clone, Debug)]
pub struct NodeTotals {
    pub gradient_sum: f64,
    pub hessian_sum: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct GradientHistogram {
    pub gradient: Vec<f64>,
    pub hessian: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct BinConfiguration {
    pub feature_idx: usize,
    pub edges: Vec<f64>,
    pub inner_edges: Vec<f64>,
    pub n_bins: usize,
}

#[derive(Clone, Debug)]
pub struct SplitDecision {
    pub node_id: usize,
    pub feature_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left_child_id: usize,
    pub right_child_id: usize,
    pub g_left: f64,
    pub h_left: f64,
    pub g_right: f64,
    pub h_right: f64,
}
