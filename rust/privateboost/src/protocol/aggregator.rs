use super::messages::*;
use super::shareholder::ShareHolder;
use crate::crypto::{Commitment, Share, decode_all, reconstruct};
use crate::math::FeatureStats;
use crate::model::{Model, Tree, TreeNode};
use crate::traffic::{Direction, TrafficLog};
use crate::{Error, Result};
use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum BinMethod {
    Uniform,
    #[default]
    Gaussian,
}

pub struct RoundContext {
    pub bins: Vec<BinConfiguration>,
    pub model: Model,
    pub splits: BTreeMap<usize, SplitDecision>,
    pub round_id: u64,
    pub depth: usize,
}

pub struct Aggregator {
    shareholders: Vec<ShareHolder>,
    n_bins: usize,
    threshold: usize,
    min_clients: usize,
    learning_rate: f64,
    lambda_reg: f64,
    n_clients: usize,
    n_features: usize,
    feature_stats: Option<Vec<FeatureStats>>,
    bin_configs: Vec<BinConfiguration>,
    model: Model,
    next_node_id: usize,
    node_totals: BTreeMap<usize, NodeTotals>,
    splits: BTreeMap<usize, SplitDecision>,
    bin_method: BinMethod,
    current_round: u64,
    current_depth: usize,
    traffic_log: Option<TrafficLog>,
}

pub struct AggregatorBuilder {
    shareholders: Vec<ShareHolder>,
    n_bins: usize,
    threshold: usize,
    min_clients: usize,
    learning_rate: f64,
    lambda_reg: f64,
    bin_method: BinMethod,
}

impl AggregatorBuilder {
    pub fn n_bins(mut self, v: usize) -> Self {
        self.n_bins = v;
        self
    }
    pub fn threshold(mut self, v: usize) -> Self {
        self.threshold = v;
        self
    }
    pub fn min_clients(mut self, v: usize) -> Self {
        self.min_clients = v;
        self
    }
    pub fn learning_rate(mut self, v: f64) -> Self {
        self.learning_rate = v;
        self
    }
    pub fn lambda_reg(mut self, v: f64) -> Self {
        self.lambda_reg = v;
        self
    }
    pub fn bin_method(mut self, v: BinMethod) -> Self {
        self.bin_method = v;
        self
    }
    pub fn build(self) -> Result<Aggregator> {
        Aggregator::new(
            self.shareholders,
            self.n_bins,
            self.threshold,
            self.min_clients,
            self.learning_rate,
            self.lambda_reg,
            self.bin_method,
        )
    }
}

impl Aggregator {
    pub fn builder(shareholders: Vec<ShareHolder>) -> AggregatorBuilder {
        AggregatorBuilder {
            shareholders,
            n_bins: 10,
            threshold: 2,
            min_clients: 10,
            learning_rate: 0.1,
            lambda_reg: 1.0,
            bin_method: BinMethod::default(),
        }
    }

    pub fn new(
        shareholders: Vec<ShareHolder>,
        n_bins: usize,
        threshold: usize,
        min_clients: usize,
        learning_rate: f64,
        lambda_reg: f64,
        bin_method: BinMethod,
    ) -> Result<Self> {
        if shareholders.len() < threshold {
            return Err(Error::ThresholdExceedsParties {
                threshold,
                n_parties: shareholders.len(),
            });
        }
        Ok(Self {
            shareholders,
            n_bins,
            threshold,
            min_clients,
            learning_rate,
            lambda_reg,
            n_clients: 0,
            n_features: 0,
            feature_stats: None,
            bin_configs: Vec::new(),
            model: Model::new(0.0, learning_rate),
            next_node_id: 1,
            node_totals: BTreeMap::new(),
            splits: BTreeMap::new(),
            bin_method,
            current_round: 0,
            current_depth: 0,
            traffic_log: None,
        })
    }

    pub fn enable_traffic_log(&mut self) {
        self.traffic_log = Some(TrafficLog::new());
    }

    pub fn traffic_log(&self) -> Option<&TrafficLog> {
        self.traffic_log.as_ref()
    }

    pub fn traffic_log_mut(&mut self) -> Option<&mut TrafficLog> {
        self.traffic_log.as_mut()
    }

    pub fn shareholders_mut(&mut self) -> &mut [ShareHolder] {
        &mut self.shareholders
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn feature_stats(&self) -> Option<&[FeatureStats]> {
        self.feature_stats.as_deref()
    }
    pub fn n_clients(&self) -> usize {
        self.n_clients
    }

    pub fn round_context(&self) -> RoundContext {
        RoundContext {
            bins: self.bin_configs.clone(),
            model: self.model.clone(),
            splits: self.splits.clone(),
            round_id: self.current_round,
            depth: self.current_depth,
        }
    }

    pub fn reset(&mut self) {
        self.feature_stats = None;
        self.bin_configs.clear();
        self.model = Model::new(0.0, self.learning_rate);
        self.n_clients = 0;
        self.n_features = 0;
        self.next_node_id = 1;
        self.node_totals.clear();
        self.splits.clear();
        self.current_round = 0;
        self.current_depth = 0;
        if self.traffic_log.is_some() {
            self.traffic_log = Some(TrafficLog::new());
        }
    }

    fn select_shareholders<F>(&self, get_commitments: F) -> Result<(Vec<usize>, Vec<Commitment>)>
    where
        F: Fn(&ShareHolder) -> BTreeSet<Commitment>,
    {
        let sh_commitments: Vec<BTreeSet<Commitment>> =
            self.shareholders.iter().map(get_commitments).collect();

        let mut best_overlap: BTreeSet<Commitment> = BTreeSet::new();
        let mut best_indices: Vec<usize> = Vec::new();

        for combo in (0..self.shareholders.len()).combinations(self.threshold) {
            let overlap: BTreeSet<Commitment> = combo
                .iter()
                .map(|&i| &sh_commitments[i])
                .fold(sh_commitments[combo[0]].clone(), |acc, s| {
                    acc.intersection(s).cloned().collect()
                });

            if overlap.len() > best_overlap.len() {
                best_overlap = overlap;
                best_indices = combo;
            }
        }

        if best_overlap.len() < self.min_clients {
            return Err(Error::InsufficientClients {
                needed: self.min_clients,
                got: best_overlap.len(),
            });
        }

        let commitments: Vec<Commitment> = best_overlap.into_iter().collect();
        Ok((best_indices, commitments))
    }

    fn collect_stats_shares(&self, indices: &[usize], commitments: &[Commitment]) -> Vec<Share> {
        let mut shares = Vec::new();
        for &i in indices {
            if let Ok(s) = self.shareholders[i].get_stats_sum(commitments) {
                shares.push(s);
            }
        }
        shares
    }

    fn collect_gradient_shares(
        &self,
        indices: &[usize],
        depth: usize,
        node_id: usize,
    ) -> Vec<Share> {
        let mut shares = Vec::new();
        for &i in indices {
            if let Ok(s) = self.shareholders[i].get_gradients_sum(depth, node_id) {
                shares.push(s);
            }
        }
        shares
    }

    pub fn define_bins(&mut self) -> Result<Vec<BinConfiguration>> {
        let (indices, commitments) = self.select_shareholders(|sh| sh.get_stats_commitments())?;
        self.n_clients = commitments.len();

        let shares = self.collect_stats_shares(&indices, &commitments);
        if let Some(log) = &mut self.traffic_log && let Some(share) = shares.first() {
            log.record(share, Direction::ShareholderToAggregator, 0, 0,
                       "Share", shares.len() as u64);
        }
        let totals = decode_all(&reconstruct(&shares, self.threshold)?);

        let n_total = totals.len() / 2;
        self.n_features = n_total - 1;

        let stats: Vec<FeatureStats> = (0..n_total)
            .map(|idx| FeatureStats::from_totals(totals[idx * 2], totals[idx * 2 + 1], self.n_clients))
            .collect();

        self.bin_configs = stats[..self.n_features]
            .iter()
            .enumerate()
            .map(|(idx, s)| s.to_bins(idx, self.n_bins, &self.bin_method))
            .collect();

        let target_mean = stats[n_total - 1].mean.clamp(1e-7, 1.0 - 1e-7);
        self.model.initial_prediction = (target_mean / (1.0 - target_mean)).ln();

        self.feature_stats = Some(stats);

        if let Some(log) = &mut self.traffic_log {
            for config in &self.bin_configs {
                log.record(config, Direction::AggregatorBroadcast, 0, 0,
                           "BinConfiguration", self.n_clients as u64);
            }
        }

        Ok(self.bin_configs.clone())
    }

    pub fn compute_splits(
        &mut self,
        min_gain: f64,
        min_child_weight: f64,
    ) -> Result<bool> {
        let depth = self.current_depth;
        if depth == 0 {
            self.next_node_id = 1;
            self.node_totals.clear();
            self.splits.clear();
        }

        let n_splits_before = self.splits.len();
        let (indices, _commitments) =
            self.select_shareholders(|sh| sh.get_gradient_commitments(depth))?;

        let mut active_nodes: BTreeSet<usize> = BTreeSet::new();
        for &i in &indices {
            active_nodes.extend(self.shareholders[i].get_gradient_node_ids(depth));
        }
        let active_nodes: Vec<usize> = active_nodes.into_iter().collect();

        for node_id in active_nodes {
            let shares = self.collect_gradient_shares(&indices, depth, node_id);
            if let Some(log) = &mut self.traffic_log && let Some(share) = shares.first() {
                log.record(share, Direction::ShareholderToAggregator,
                           self.current_round as usize, depth,
                           "Share", shares.len() as u64);
            }
            if shares.len() < self.threshold {
                continue;
            }

            let totals = decode_all(&reconstruct(&shares, self.threshold)?);
            let n_bins_total = self.bin_configs[0].edges.len();
            let n_features = self.n_features;
            let grad_size = n_features * n_bins_total;

            let gradient_flat = &totals[..grad_size];
            let hessian_flat = &totals[grad_size..];

            let mut histograms: Vec<GradientHistogram> = Vec::with_capacity(n_features);
            for f in 0..n_features {
                let start = f * n_bins_total;
                let end = start + n_bins_total;
                histograms.push(GradientHistogram {
                    gradient: gradient_flat[start..end].to_vec(),
                    hessian: hessian_flat[start..end].to_vec(),
                });
            }

            let total_g: f64 = histograms[0].gradient.iter().sum();
            let total_h: f64 = histograms[0].hessian.iter().sum();

            self.node_totals.insert(
                node_id,
                NodeTotals {
                    gradient_sum: total_g,
                    hessian_sum: total_h,
                },
            );

            if total_h < min_child_weight {
                continue;
            }

            let base_score = (total_g * total_g) / (total_h + self.lambda_reg);
            let mut best_gain = min_gain;
            let mut best_split: Option<SplitDecision> = None;

            for (feature_idx, hist) in histograms.iter().enumerate() {
                if feature_idx >= self.bin_configs.len() {
                    continue;
                }
                let config = &self.bin_configs[feature_idx];

                let mut g_cumsum = Vec::with_capacity(hist.gradient.len());
                let mut h_cumsum = Vec::with_capacity(hist.hessian.len());
                let mut g_sum = 0.0;
                let mut h_sum = 0.0;
                for i in 0..hist.gradient.len() {
                    g_sum += hist.gradient[i];
                    h_sum += hist.hessian[i];
                    g_cumsum.push(g_sum);
                    h_cumsum.push(h_sum);
                }

                for i in 0..hist.gradient.len() - 1 {
                    let g_left = g_cumsum[i];
                    let h_left = h_cumsum[i];
                    let g_right = total_g - g_left;
                    let h_right = total_h - h_left;

                    if h_left < min_child_weight || h_right < min_child_weight {
                        continue;
                    }

                    let left_score = (g_left * g_left) / (h_left + self.lambda_reg);
                    let right_score = (g_right * g_right) / (h_right + self.lambda_reg);
                    let gain = left_score + right_score - base_score;

                    if gain > best_gain {
                        best_gain = gain;
                        best_split = Some(SplitDecision {
                            node_id,
                            feature_idx,
                            threshold: config.edges[i + 1],
                            gain,
                            left_child_id: self.next_node_id,
                            right_child_id: self.next_node_id + 1,
                            g_left,
                            h_left,
                            g_right,
                            h_right,
                        });
                    }
                }
            }

            if let Some(split) = best_split {
                self.node_totals.insert(
                    split.left_child_id,
                    NodeTotals {
                        gradient_sum: split.g_left,
                        hessian_sum: split.h_left,
                    },
                );
                self.node_totals.insert(
                    split.right_child_id,
                    NodeTotals {
                        gradient_sum: split.g_right,
                        hessian_sum: split.h_right,
                    },
                );
                self.next_node_id += 2;
                self.splits.insert(node_id, split);
                if let Some(log) = &mut self.traffic_log {
                    let split_ref = &self.splits[&node_id];
                    log.record(split_ref, Direction::AggregatorBroadcast,
                               self.current_round as usize, depth,
                               "SplitDecision", self.n_clients as u64);
                }
            }
        }

        let made_progress = self.splits.len() > n_splits_before;
        self.current_depth += 1;
        Ok(made_progress)
    }

    pub fn finish_round(&mut self) {
        fn build_node(
            node_id: usize,
            splits: &BTreeMap<usize, SplitDecision>,
            node_totals: &BTreeMap<usize, NodeTotals>,
            lambda_reg: f64,
        ) -> TreeNode {
            if let Some(split) = splits.get(&node_id) {
                TreeNode::Split {
                    feature_idx: split.feature_idx,
                    threshold: split.threshold,
                    gain: split.gain,
                    left: Box::new(build_node(
                        split.left_child_id,
                        splits,
                        node_totals,
                        lambda_reg,
                    )),
                    right: Box::new(build_node(
                        split.right_child_id,
                        splits,
                        node_totals,
                        lambda_reg,
                    )),
                }
            } else {
                let (total_g, total_h) = node_totals
                    .get(&node_id)
                    .map(|t| (t.gradient_sum, t.hessian_sum))
                    .unwrap_or((0.0, 0.0));
                let value = if total_h + lambda_reg > 0.0 {
                    -total_g / (total_h + lambda_reg)
                } else {
                    0.0
                };
                TreeNode::Leaf { value }
            }
        }

        let tree = Tree {
            root: build_node(0, &self.splits, &self.node_totals, self.lambda_reg),
        };
        self.model.add_tree(tree);
        self.current_round += 1;
        self.current_depth = 0;
    }
}
