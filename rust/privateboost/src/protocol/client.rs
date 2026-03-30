use super::messages::*;
use crate::Result;
use crate::crypto::{commit, encode_all, generate_nonce, share};
use crate::model::Model;
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

pub struct Client {
    client_id: String,
    features: Vec<f64>,
    target: f64,
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
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        Self {
            client_id,
            features,
            target,
            n_parties,
            threshold,
            rng,
        }
    }

    pub fn submit_stats(&mut self) -> Result<Vec<CommittedStatsShare>> {
        let nonce = generate_nonce();
        let commitment = commit(0, &self.client_id, &nonce);

        let mut values = Vec::with_capacity(self.features.len() * 2 + 2);
        for &f in &self.features {
            values.push(f);
            values.push(f * f);
        }
        values.push(self.target);
        values.push(self.target * self.target);

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

    pub fn submit_gradients(
        &mut self,
        bins: &[BinConfiguration],
        model: &Model,
        splits: &BTreeMap<usize, SplitDecision>,
        round_id: u64,
        depth: usize,
        loss: &Loss,
    ) -> Result<Vec<CommittedGradientShare>> {
        let node_id = get_node_id(&self.features, splits);
        let nonce = generate_nonce();
        let commitment = commit(round_id, &self.client_id, &nonce);
        let prediction = model.predict_one(&self.features);

        let (gradient, hessian) = match loss {
            Loss::Squared => (prediction - self.target, 1.0),
            Loss::Logistic => {
                let p = 1.0 / (1.0 + (-prediction).exp());
                (p - self.target, p * (1.0 - p))
            }
        };

        let mut all_gradients = Vec::new();
        let mut all_hessians = Vec::new();

        for config in bins {
            let value = self.features[config.feature_idx];
            let n_total_bins = config.n_bins + 2;
            let bin_idx = find_bin_index(value, &config.edges, n_total_bins);

            let mut g_vec = vec![0.0; n_total_bins];
            let mut h_vec = vec![0.0; n_total_bins];
            g_vec[bin_idx] = gradient;
            h_vec[bin_idx] = hessian;

            all_gradients.extend(g_vec);
            all_hessians.extend(h_vec);
        }

        let mut values = all_gradients;
        values.extend(all_hessians);

        let encoded = encode_all(&values);
        let shares = share(&encoded, self.n_parties, self.threshold, &mut self.rng)?;

        Ok(shares
            .into_iter()
            .map(|s| CommittedGradientShare {
                round_id,
                depth,
                commitment: commitment.clone(),
                share: s,
                node_id,
            })
            .collect())
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
    fn test_client_submit_stats_produces_n_shares() {
        let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(42));
        let shares = client.submit_stats().unwrap();
        assert_eq!(shares.len(), 3);
        assert_eq!(shares[0].share.values.len(), 6);
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
}
