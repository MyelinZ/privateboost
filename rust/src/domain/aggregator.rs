use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use async_trait::async_trait;

use crate::crypto::commitment::Commitment;
use crate::crypto::shamir::{reconstruct, Share};
use crate::domain::model::*;

const BIN_RANGE_STDS: f64 = 3.0;
const MIN_HESSIAN_THRESHOLD: f64 = 0.1;

/// Which type of commitments to fetch from a shareholder.
pub enum CommitmentKind {
    Stats,
    Gradient(Depth),
}

/// Abstraction over shareholder communication.
///
/// In-process implementations wrap a `ShareHolder` directly; remote
/// implementations will call out over gRPC.
#[async_trait]
pub trait ShareHolderClient: Send + Sync {
    fn x_coord(&self) -> i32;
    async fn get_stats_commitments(&self) -> Result<HashSet<Commitment>>;
    async fn get_gradient_commitments(&self, depth: Depth) -> Result<HashSet<Commitment>>;
    async fn get_gradient_node_ids(&self, depth: Depth) -> Result<HashSet<NodeId>>;
    async fn get_stats_sum(&self, commitments: &[Commitment]) -> Result<(i32, Vec<f64>)>;
    async fn get_gradients_sum(
        &self,
        depth: Depth,
        commitments: &[Commitment],
        node_id: NodeId,
    ) -> Result<(i32, Vec<f64>)>;
}

pub struct Aggregator {
    pub shareholders: Vec<Box<dyn ShareHolderClient>>,
    pub n_bins: usize,
    pub threshold: usize,
    pub min_clients: usize,
    pub learning_rate: f64,
    pub lambda_reg: f64,
}

impl Aggregator {
    pub fn new(
        shareholders: Vec<Box<dyn ShareHolderClient>>,
        n_bins: usize,
        threshold: usize,
        min_clients: usize,
        learning_rate: f64,
        lambda_reg: f64,
    ) -> Result<Self> {
        if shareholders.len() < threshold {
            bail!(
                "Need at least {} shareholders, got {}",
                threshold,
                shareholders.len()
            );
        }
        Ok(Self {
            shareholders,
            n_bins,
            threshold,
            min_clients,
            learning_rate,
            lambda_reg,
        })
    }

    /// Select `threshold` shareholders whose commitment sets have the
    /// largest overlap. Returns (indices into `self.shareholders`,
    /// sorted commitment list).
    async fn select_shareholders(
        &self,
        kind: CommitmentKind,
    ) -> Result<(Vec<usize>, Vec<Commitment>)> {
        let mut sh_commitments: Vec<(usize, HashSet<Commitment>)> = Vec::new();
        for (i, sh) in self.shareholders.iter().enumerate() {
            let commits = match &kind {
                CommitmentKind::Stats => sh.get_stats_commitments().await?,
                CommitmentKind::Gradient(depth) => sh.get_gradient_commitments(*depth).await?,
            };
            sh_commitments.push((i, commits));
        }

        let mut best_overlap: HashSet<Commitment> = HashSet::new();
        let mut best_group: Vec<usize> = Vec::new();

        for combo in combinations(sh_commitments.len(), self.threshold) {
            let commit_sets: Vec<&HashSet<Commitment>> =
                combo.iter().map(|&idx| &sh_commitments[idx].1).collect();

            let overlap = intersect_all(&commit_sets);

            if overlap.len() > best_overlap.len() {
                best_overlap = overlap;
                best_group = combo.iter().map(|&idx| sh_commitments[idx].0).collect();
            }
        }

        if best_overlap.len() < self.min_clients {
            bail!(
                "Best overlap has {} clients, need {}",
                best_overlap.len(),
                self.min_clients
            );
        }

        let commitments: Vec<Commitment> = best_overlap.into_iter().collect();
        Ok((best_group, commitments))
    }

    async fn collect_stats_shares(
        &self,
        selected_indices: &[usize],
        commitments: &[Commitment],
    ) -> Vec<Share> {
        let mut shares = Vec::new();
        for &idx in selected_indices {
            let sh = &self.shareholders[idx];
            match sh.get_stats_sum(commitments).await {
                Ok((x, y)) => shares.push(Share { x, y }),
                Err(e) => {
                    tracing::warn!(shareholder = sh.x_coord(), "stats_sum failed: {e}");
                    continue;
                }
            }
        }
        shares
    }

    async fn collect_gradient_shares(
        &self,
        selected_indices: &[usize],
        depth: Depth,
        commitments: &[Commitment],
        node_id: NodeId,
    ) -> Vec<Share> {
        let mut shares = Vec::new();
        for &idx in selected_indices {
            let sh = &self.shareholders[idx];
            match sh.get_gradients_sum(depth, commitments, node_id).await {
                Ok((x, y)) => shares.push(Share { x, y }),
                Err(e) => {
                    tracing::warn!(shareholder = sh.x_coord(), "gradients_sum failed: {e}");
                    continue;
                }
            }
        }
        shares
    }

    /// Collect stats shares, reconstruct mean/variance per feature,
    /// and produce `BinConfiguration`s. Returns (bin_configs, initial_prediction, n_clients, n_features).
    pub async fn compute_bins(&self) -> Result<(Vec<BinConfiguration>, f64, usize, usize)> {
        let (selected, commitments) =
            self.select_shareholders(CommitmentKind::Stats).await?;
        let n_clients = commitments.len();

        let shares = self.collect_stats_shares(&selected, &commitments).await;
        let totals = reconstruct(&shares, self.threshold)?;

        let n_values = totals.len();
        let n_total = n_values / 2;
        let n_features = n_total - 1;

        let mut means = vec![0.0; n_total];
        let mut variances = vec![0.0; n_total];
        for idx in 0..n_total {
            let total_x = totals[idx * 2];
            let total_x2 = totals[idx * 2 + 1];
            means[idx] = total_x / n_clients as f64;
            variances[idx] = (total_x2 / n_clients as f64) - (means[idx] * means[idx]);
        }

        let stds: Vec<f64> = variances.iter().map(|&v| v.max(0.0).sqrt()).collect();

        let mut bin_configs = Vec::new();
        for idx in 0..n_features {
            let mean = means[idx];
            let std = stds[idx];
            let range_min = mean - BIN_RANGE_STDS * std;
            let range_max = mean + BIN_RANGE_STDS * std;

            let inner_edges = linspace(range_min, range_max, self.n_bins + 1);

            let mut edges = Vec::with_capacity(inner_edges.len() + 2);
            edges.push(f64::NEG_INFINITY);
            edges.extend_from_slice(&inner_edges);
            edges.push(f64::INFINITY);

            bin_configs.push(BinConfiguration {
                feature_idx: idx,
                edges,
                inner_edges,
                n_bins: self.n_bins,
            });
        }

        let initial_prediction = means[n_total - 1];
        Ok((bin_configs, initial_prediction, n_clients, n_features))
    }

    /// For each active node at `depth`, reconstruct gradient histograms
    /// and find the best split. Returns `true` if any new splits were found.
    pub async fn compute_splits(
        &self,
        depth: Depth,
        min_samples: usize,
        bin_configs: &[BinConfiguration],
        n_features: usize,
        node_totals: &mut HashMap<NodeId, NodeTotals>,
        splits: &mut HashMap<NodeId, SplitDecision>,
        next_node_id: &mut NodeId,
    ) -> Result<bool> {
        let n_splits_before = splits.len();

        let (selected, commitments) = self
            .select_shareholders(CommitmentKind::Gradient(depth))
            .await?;

        let mut active_nodes: HashSet<NodeId> = HashSet::new();
        for &idx in &selected {
            let nodes = self.shareholders[idx].get_gradient_node_ids(depth).await?;
            active_nodes.extend(nodes);
        }

        for node_id in active_nodes {
            let shares = self
                .collect_gradient_shares(&selected, depth, &commitments, node_id)
                .await;
            if shares.len() < self.threshold {
                continue;
            }

            let totals = reconstruct(&shares, self.threshold)?;

            let n_bins_total = self.n_bins + 2;
            let grad_size = n_features * n_bins_total;

            let gradient_flat = &totals[..grad_size];
            let hessian_flat = &totals[grad_size..];

            let mut histograms: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(n_features);
            for f in 0..n_features {
                let start = f * n_bins_total;
                let end = start + n_bins_total;
                histograms.push((
                    gradient_flat[start..end].to_vec(),
                    hessian_flat[start..end].to_vec(),
                ));
            }

            let total_g: f64 = histograms[0].0.iter().sum();
            let total_h: f64 = histograms[0].1.iter().sum();

            node_totals.insert(
                node_id,
                NodeTotals {
                    gradient_sum: total_g,
                    hessian_sum: total_h,
                },
            );

            let n_samples = total_h.round() as usize;
            if n_samples < min_samples {
                continue;
            }

            let base_score = (total_g * total_g) / (total_h + self.lambda_reg);

            let mut best_gain = 0.0_f64;
            let mut best_split: Option<SplitDecision> = None;

            for (feature_idx, (grad_hist, hess_hist)) in histograms.iter().enumerate() {
                if feature_idx >= bin_configs.len() {
                    continue;
                }
                let config = &bin_configs[feature_idx];

                let mut g_cumsum = vec![0.0; grad_hist.len()];
                let mut h_cumsum = vec![0.0; hess_hist.len()];
                g_cumsum[0] = grad_hist[0];
                h_cumsum[0] = hess_hist[0];
                for i in 1..grad_hist.len() {
                    g_cumsum[i] = g_cumsum[i - 1] + grad_hist[i];
                    h_cumsum[i] = h_cumsum[i - 1] + hess_hist[i];
                }

                for i in 0..(grad_hist.len() - 1) {
                    let g_left = g_cumsum[i];
                    let h_left = h_cumsum[i];
                    let g_right = total_g - g_left;
                    let h_right = total_h - h_left;

                    if h_left < MIN_HESSIAN_THRESHOLD || h_right < MIN_HESSIAN_THRESHOLD {
                        continue;
                    }

                    let left_score = (g_left * g_left) / (h_left + self.lambda_reg);
                    let right_score = (g_right * g_right) / (h_right + self.lambda_reg);
                    let gain = left_score + right_score - base_score;

                    if gain > best_gain {
                        best_gain = gain;
                        let threshold = config.edges[i + 1];
                        best_split = Some(SplitDecision {
                            node_id,
                            feature_idx,
                            threshold,
                            gain,
                            left_child_id: *next_node_id,
                            right_child_id: *next_node_id + 1,
                            g_left,
                            h_left,
                            g_right,
                            h_right,
                        });
                    }
                }
            }

            if let Some(split) = best_split {
                node_totals.insert(
                    split.left_child_id,
                    NodeTotals {
                        gradient_sum: split.g_left,
                        hessian_sum: split.h_left,
                    },
                );
                node_totals.insert(
                    split.right_child_id,
                    NodeTotals {
                        gradient_sum: split.g_right,
                        hessian_sum: split.h_right,
                    },
                );
                splits.insert(node_id, split);
                *next_node_id += 2;
            }
        }

        Ok(splits.len() > n_splits_before)
    }

    /// Build a tree from splits and node_totals (pure function, no mutation).
    pub fn build_tree(
        &self,
        splits: &HashMap<NodeId, SplitDecision>,
        node_totals: &HashMap<NodeId, NodeTotals>,
    ) -> Tree {
        Tree {
            root: self.build_node(0, splits, node_totals),
        }
    }

    fn build_node(
        &self,
        node_id: NodeId,
        splits: &HashMap<NodeId, SplitDecision>,
        node_totals: &HashMap<NodeId, NodeTotals>,
    ) -> TreeNode {
        if let Some(split) = splits.get(&node_id) {
            TreeNode::Split(SplitNode {
                feature_idx: split.feature_idx,
                threshold: split.threshold,
                gain: split.gain,
                left: Box::new(self.build_node(split.left_child_id, splits, node_totals)),
                right: Box::new(self.build_node(split.right_child_id, splits, node_totals)),
            })
        } else {
            self.compute_leaf(node_id, node_totals)
        }
    }

    fn compute_leaf(
        &self,
        node_id: NodeId,
        node_totals: &HashMap<NodeId, NodeTotals>,
    ) -> TreeNode {
        let (total_g, total_h) = match node_totals.get(&node_id) {
            Some(t) => (t.gradient_sum, t.hessian_sum),
            None => (0.0, 0.0),
        };

        let value = if total_h + self.lambda_reg > 0.0 {
            -total_g / (total_h + self.lambda_reg)
        } else {
            0.0
        };

        TreeNode::Leaf(LeafNode {
            value,
            n_samples: total_h.round() as usize,
        })
    }
}

/// All k-element subsets of 0..n.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(k);
    combinations_inner(n, k, 0, &mut current, &mut result);
    result
}

fn combinations_inner(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    for i in start..n {
        current.push(i);
        combinations_inner(n, k, i + 1, current, result);
        current.pop();
    }
}

/// Intersection of all sets. Returns empty set if input is empty.
fn intersect_all(sets: &[&HashSet<Commitment>]) -> HashSet<Commitment> {
    match sets.first() {
        None => HashSet::new(),
        Some(&first) => {
            let mut result = first.clone();
            for &s in &sets[1..] {
                result.retain(|item| s.contains(item));
            }
            result
        }
    }
}

/// Evenly spaced values from start to end (inclusive), n points.
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}
