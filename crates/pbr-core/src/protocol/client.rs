//! The protocol client: one site's contribution to a round.
//!
//! A client holds one or more `(features, target)` records and contributes
//! sums over all of them under a single commitment per round, so anonymity
//! floors count clients rather than records and per-round message size is
//! independent of batch size.
//!
//! Two rounds exist. The stats round shares per-feature sums and squared sums
//! with the record count riding inside the shared vector, so only the
//! fleet-wide total ever reconstructs. Gradient rounds share a dense
//! (gradient, hessian) histogram per active node; under path hiding every
//! active node gets an entry, zeros included, so no party learns which node a
//! client's records fell in.

use super::aggregator::RoundContext;
use super::messages::*;
use crate::Result;
use crate::crypto::{commit, encode_all, generate_nonce, share};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::BTreeMap;

fn find_bin_index(value: f64, edges: &[f64], n_total_bins: usize) -> usize {
    let idx = edges.partition_point(|&e| e <= value);
    let idx = if idx == 0 { 0 } else { idx - 1 };
    idx.min(n_total_bins - 1)
}

fn get_node_id(features: &[f64], splits: &BTreeMap<usize, SplitDecision>) -> usize {
    let mut node_id = 0;
    while let Some(split) = splits.get(&node_id) {
        if features[split.feature_idx] <= split.threshold {
            node_id = split.left_child_id;
        } else {
            node_id = split.right_child_id;
        }
    }
    node_id
}

fn get_active_node_ids(splits: &BTreeMap<usize, SplitDecision>, depth: usize) -> Vec<usize> {
    if depth == 0 {
        return vec![0];
    }
    let mut active = Vec::new();
    for split in splits.values() {
        if !splits.contains_key(&split.left_child_id) {
            active.push(split.left_child_id);
        }
        if !splits.contains_key(&split.right_child_id) {
            active.push(split.right_child_id);
        }
    }
    if active.is_empty() {
        vec![0]
    } else {
        active.sort();
        active
    }
}

pub struct Client {
    client_id: String,
    records: Vec<(Vec<f64>, f64)>,
    n_parties: usize,
    threshold: usize,
    rng: StdRng,
}

pub struct ClientBuilder {
    client_id: String,
    features: Vec<f64>,
    target: f64,
    n_parties: usize,
    threshold: usize,
    seed: Option<u64>,
}

impl ClientBuilder {
    pub fn threshold(mut self, v: usize) -> Self {
        self.threshold = v;
        self
    }
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = Some(v);
        self
    }
    pub fn build(self) -> Client {
        Client::new(
            self.client_id,
            self.features,
            self.target,
            self.n_parties,
            self.threshold,
            self.seed,
        )
    }
}

impl Client {
    pub fn builder(
        client_id: impl Into<String>,
        features: Vec<f64>,
        target: f64,
        n_parties: usize,
    ) -> ClientBuilder {
        ClientBuilder {
            client_id: client_id.into(),
            features,
            target,
            n_parties,
            threshold: 2,
            seed: None,
        }
    }

    pub fn new(
        client_id: String,
        features: Vec<f64>,
        target: f64,
        n_parties: usize,
        threshold: usize,
        seed: Option<u64>,
    ) -> Self {
        Self::new_batch(client_id, vec![(features, target)], n_parties, threshold, seed)
    }

    /// Panics if `records` is empty: an empty batch has nothing to
    /// contribute.
    pub fn new_batch(
        client_id: String,
        records: Vec<(Vec<f64>, f64)>,
        n_parties: usize,
        threshold: usize,
        seed: Option<u64>,
    ) -> Self {
        assert!(!records.is_empty(), "batch client requires at least one record");
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self { client_id, records, n_parties, threshold, rng }
    }

    pub fn compute_stat_shares(&mut self) -> Result<Vec<CommittedStatsShare>> {
        let nonce = generate_nonce();
        let commitment = commit(0, &self.client_id, &nonce);

        // [Σf, Σf², …, Σt, Σt², n] over this client's records. `define_bins`
        // needs the count because commitments count submissions, not records.
        let n_features = self.records[0].0.len();
        let mut values = vec![0.0; n_features * 2 + 3];
        for (features, target) in &self.records {
            for (j, &f) in features.iter().enumerate() {
                values[2 * j] += f;
                values[2 * j + 1] += f * f;
            }
            values[2 * n_features] += target;
            values[2 * n_features + 1] += target * target;
        }
        values[2 * n_features + 2] = self.records.len() as f64;

        let encoded = encode_all(&values);
        let shares = share(&encoded, self.n_parties, self.threshold, &mut self.rng)?;

        Ok(shares
            .into_iter()
            .map(|s| CommittedStatsShare {
                commitment: commitment.clone(),
                share: s,
            })
            .collect())
    }

    pub fn compute_gradient_shares(
        &mut self,
        ctx: &RoundContext,
        loss: &Loss,
        hide_path: bool,
    ) -> Result<Vec<CommittedGradientShare>> {
        let nonce = generate_nonce();
        let commitment = commit(ctx.round_id, &self.client_id, &nonce);
        let n_slots: usize = ctx.bins.iter().map(|c| c.edges.len()).sum();

        // Each record lands in its own path's node and its own bin per
        // feature. Shareholders sum across clients anyway, so batching changes
        // no downstream math.
        let mut per_node: BTreeMap<usize, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for (features, target) in &self.records {
            let node_id = if ctx.depth == 0 {
                0
            } else {
                get_node_id(features, &ctx.splits)
            };
            let prediction = ctx.model.predict_one(features);
            let (gradient, hessian) = match loss {
                Loss::Squared => (prediction - target, 1.0),
                Loss::Logistic => {
                    let p = 1.0 / (1.0 + (-prediction).exp());
                    (p - target, p * (1.0 - p))
                }
            };
            let (g_hist, h_hist) = per_node
                .entry(node_id)
                .or_insert_with(|| (vec![0.0; n_slots], vec![0.0; n_slots]));
            let mut offset = 0;
            for config in &ctx.bins {
                let n_total_bins = config.edges.len();
                let bin_idx =
                    find_bin_index(features[config.feature_idx], &config.edges, n_total_bins);
                g_hist[offset + bin_idx] += gradient;
                h_hist[offset + bin_idx] += hessian;
                offset += n_total_bins;
            }
        }

        // Path hiding submits every active node, zero histograms included;
        // without it, only the occupied ones.
        let submit_nodes: Vec<usize> = if hide_path {
            get_active_node_ids(&ctx.splits, ctx.depth)
        } else {
            per_node.keys().copied().collect()
        };

        let zero = (vec![0.0; n_slots], vec![0.0; n_slots]);
        let mut all_shares = Vec::new();
        for &node_id in &submit_nodes {
            let (g_hist, h_hist) = per_node.get(&node_id).unwrap_or(&zero);
            let mut values = g_hist.clone();
            values.extend_from_slice(h_hist);
            let encoded = encode_all(&values);
            let shares = share(&encoded, self.n_parties, self.threshold, &mut self.rng)?;
            for s in shares {
                all_shares.push(CommittedGradientShare {
                    round_id: ctx.round_id,
                    depth: ctx.depth,
                    commitment: commitment.clone(),
                    share: s,
                    node_id,
                });
            }
        }
        Ok(all_shares)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_bin_index_middle() {
        let edges = vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY];
        assert_eq!(find_bin_index(1.5, &edges, 4), 1);
    }

    #[test]
    fn test_find_bin_index_underflow() {
        let edges = vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY];
        assert_eq!(find_bin_index(-100.0, &edges, 4), 0);
    }

    #[test]
    fn test_find_bin_index_overflow() {
        let edges = vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY];
        assert_eq!(find_bin_index(100.0, &edges, 4), 3);
    }

    #[test]
    fn test_client_compute_stat_shares_produces_n_shares() {
        let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(42));
        let shares = client.compute_stat_shares().unwrap();
        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].share.values.len(), 7);
    }

    #[test]
    fn test_get_node_id_no_splits() {
        let splits = BTreeMap::new();
        assert_eq!(get_node_id(&[1.0], &splits), 0);
    }

    #[test]
    fn test_get_node_id_with_split() {
        let mut splits = BTreeMap::new();
        splits.insert(
            0,
            SplitDecision {
                node_id: 0,
                feature_idx: 0,
                threshold: 5.0,
                gain: 1.0,
                left_child_id: 1,
                right_child_id: 2,
                g_left: 0.0,
                h_left: 0.0,
                g_right: 0.0,
                h_right: 0.0,
            },
        );
        assert_eq!(get_node_id(&[3.0], &splits), 1);
        assert_eq!(get_node_id(&[7.0], &splits), 2);
    }

    #[test]
    fn test_get_active_node_ids_depth_zero() {
        let splits = BTreeMap::new();
        assert_eq!(get_active_node_ids(&splits, 0), vec![0]);
    }

    #[test]
    fn test_get_active_node_ids_one_split() {
        let mut splits = BTreeMap::new();
        splits.insert(0, SplitDecision {
            node_id: 0,
            feature_idx: 0,
            threshold: 5.0,
            gain: 1.0,
            left_child_id: 1,
            right_child_id: 2,
            g_left: 0.0, h_left: 0.0,
            g_right: 0.0, h_right: 0.0,
        });
        let mut active = get_active_node_ids(&splits, 1);
        active.sort();
        assert_eq!(active, vec![1, 2]);
    }

    #[test]
    fn test_get_active_node_ids_partial_tree() {
        let mut splits = BTreeMap::new();
        splits.insert(0, SplitDecision {
            node_id: 0, feature_idx: 0, threshold: 5.0, gain: 1.0,
            left_child_id: 1, right_child_id: 2,
            g_left: 0.0, h_left: 0.0, g_right: 0.0, h_right: 0.0,
        });
        splits.insert(1, SplitDecision {
            node_id: 1, feature_idx: 0, threshold: 3.0, gain: 1.0,
            left_child_id: 3, right_child_id: 4,
            g_left: 0.0, h_left: 0.0, g_right: 0.0, h_right: 0.0,
        });
        let mut active = get_active_node_ids(&splits, 2);
        active.sort();
        assert_eq!(active, vec![2, 3, 4]);
    }

    #[test]
    fn test_compute_gradient_shares_hide_path_produces_more_shares() {
        use crate::model::Model;
        use crate::protocol::RoundContext;

        let mut splits = BTreeMap::new();
        splits.insert(0, SplitDecision {
            node_id: 0, feature_idx: 0, threshold: 2.0, gain: 1.0,
            left_child_id: 1, right_child_id: 2,
            g_left: 0.0, h_left: 0.0, g_right: 0.0, h_right: 0.0,
        });

        let ctx = RoundContext {
            bins: vec![BinConfiguration {
                feature_idx: 0,
                edges: vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY],
            }],
            model: Model::new(0.0, 0.15),
            splits,
            round_id: 1,
            depth: 1,
        };
        let n_shareholders = 3;

        let mut client_no_hide = Client::new("c0".into(), vec![1.5], 1.0, n_shareholders, 2, Some(42));
        let shares_no_hide = client_no_hide.compute_gradient_shares(
            &ctx, &Loss::Logistic, false,
        ).unwrap();
        assert_eq!(shares_no_hide.len(), n_shareholders); // 1 node * 3 shareholders

        let mut client_hide = Client::new("c1".into(), vec![1.5], 1.0, n_shareholders, 2, Some(42));
        let shares_hide = client_hide.compute_gradient_shares(
            &ctx, &Loss::Logistic, true,
        ).unwrap();
        assert_eq!(shares_hide.len(), 2 * n_shareholders); // 2 active nodes * 3 shareholders
    }

    /// Reconstruct one node's decoded gradient values from a share set.
    fn node_values(
        shares: Vec<CommittedGradientShare>,
        node_id: usize,
        threshold: usize,
    ) -> Vec<f64> {
        use crate::crypto::{decode_all, reconstruct, Share};
        let node_shares: Vec<Share> = shares
            .into_iter()
            .filter(|s| s.node_id == node_id)
            .map(|s| s.share)
            .collect();
        assert!(!node_shares.is_empty(), "no shares for node {node_id}");
        decode_all(&reconstruct(&node_shares, threshold).unwrap())
    }

    fn stats_values(shares: Vec<CommittedStatsShare>, threshold: usize) -> Vec<f64> {
        use crate::crypto::{decode_all, reconstruct, Share};
        let s: Vec<Share> = shares.into_iter().map(|s| s.share).collect();
        decode_all(&reconstruct(&s, threshold).unwrap())
    }

    fn depth1_ctx() -> RoundContext {
        use crate::model::Model;
        let mut splits = BTreeMap::new();
        splits.insert(0, SplitDecision {
            node_id: 0, feature_idx: 0, threshold: 2.0, gain: 1.0,
            left_child_id: 1, right_child_id: 2,
            g_left: 0.0, h_left: 0.0, g_right: 0.0, h_right: 0.0,
        });
        RoundContext {
            bins: vec![BinConfiguration {
                feature_idx: 0,
                edges: vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, f64::INFINITY],
            }],
            model: Model::new(0.0, 0.15),
            splits,
            round_id: 7,
            depth: 1,
        }
    }

    #[test]
    fn batch_gradient_histogram_equals_sum_of_singles() {
        let ctx = depth1_ctx();
        // All three records fall left of the split (feature <= 2.0) -> node 1.
        let records: Vec<(Vec<f64>, f64)> =
            vec![(vec![0.5], 1.0), (vec![1.5], 0.0), (vec![1.7], 1.0)];

        let mut batch = Client::new_batch("b".into(), records.clone(), 3, 2, Some(1));
        let batch_vals = node_values(
            batch.compute_gradient_shares(&ctx, &Loss::Logistic, false).unwrap(), 1, 2);

        let mut summed = vec![0.0; batch_vals.len()];
        for (i, (f, t)) in records.iter().enumerate() {
            let mut single =
                Client::new(format!("s{i}"), f.clone(), *t, 3, 2, Some(100 + i as u64));
            let vals = node_values(
                single.compute_gradient_shares(&ctx, &Loss::Logistic, false).unwrap(), 1, 2);
            for (acc, v) in summed.iter_mut().zip(vals) {
                *acc += v;
            }
        }
        for (a, b) in batch_vals.iter().zip(&summed) {
            assert!((a - b).abs() < 1e-4, "batch {a} vs summed singles {b}");
        }
    }

    #[test]
    fn batch_without_hiding_submits_only_occupied_nodes() {
        let ctx = depth1_ctx();
        // Records straddle the split: nodes 1 (0.5) and 2 (2.5).
        let mut straddle = Client::new_batch(
            "b".into(), vec![(vec![0.5], 1.0), (vec![2.5], 0.0)], 3, 2, Some(2));
        let shares = straddle.compute_gradient_shares(&ctx, &Loss::Logistic, false).unwrap();
        let mut nodes: Vec<usize> = shares.iter().map(|s| s.node_id).collect();
        nodes.sort();
        nodes.dedup();
        assert_eq!(nodes, vec![1, 2]);

        // All records in node 1: node 2 must not be submitted without hiding.
        let mut left_only = Client::new_batch(
            "b2".into(), vec![(vec![0.5], 1.0), (vec![1.5], 0.0)], 3, 2, Some(3));
        let shares = left_only.compute_gradient_shares(&ctx, &Loss::Logistic, false).unwrap();
        let mut nodes: Vec<usize> = shares.iter().map(|s| s.node_id).collect();
        nodes.sort();
        nodes.dedup();
        assert_eq!(nodes, vec![1]);
    }

    #[test]
    fn batch_with_hiding_submits_zero_histograms_for_empty_active_nodes() {
        let ctx = depth1_ctx();
        // Both records in node 1; hiding must still submit node 2, as zeros.
        let mut c = Client::new_batch(
            "b".into(), vec![(vec![0.5], 1.0), (vec![1.5], 0.0)], 3, 2, Some(4));
        let shares = c.compute_gradient_shares(&ctx, &Loss::Logistic, true).unwrap();
        let mut nodes: Vec<usize> = shares.iter().map(|s| s.node_id).collect();
        nodes.sort();
        nodes.dedup();
        assert_eq!(nodes, vec![1, 2], "hiding submits every active node");

        let empty = node_values(
            c.compute_gradient_shares(&ctx, &Loss::Logistic, true).unwrap(), 2, 2);
        for v in &empty {
            assert!(v.abs() < 1e-4, "node 2 histogram must be all zeros, got {v}");
        }
    }

    #[test]
    fn stats_vector_is_sums_with_trailing_count() {
        // Batch of 2 records, 2 features:
        // [Σf1, Σf1², Σf2, Σf2², Σt, Σt², n] = [4, 10, 6, 20, 1, 1, 2].
        let mut c = Client::new_batch(
            "b".into(),
            vec![(vec![1.0, 2.0], 1.0), (vec![3.0, 4.0], 0.0)],
            3, 2, Some(5));
        let vals = stats_values(c.compute_stat_shares().unwrap(), 2);
        let expected = [4.0, 10.0, 6.0, 20.0, 1.0, 1.0, 2.0];
        assert_eq!(vals.len(), expected.len());
        for (v, e) in vals.iter().zip(&expected) {
            assert!((v - e).abs() < 1e-4, "got {v}, want {e}");
        }
    }

    #[test]
    fn single_record_stats_are_batch_of_one_with_count() {
        // [f, f², t, t², 1] for one record: the batch layout with a count of 1.
        let mut c = Client::new("c0".into(), vec![2.0], 1.0, 3, 2, Some(6));
        let vals = stats_values(c.compute_stat_shares().unwrap(), 2);
        let expected = [2.0, 4.0, 1.0, 1.0, 1.0];
        assert_eq!(vals.len(), expected.len());
        for (v, e) in vals.iter().zip(&expected) {
            assert!((v - e).abs() < 1e-4, "got {v}, want {e}");
        }
    }
}
