use std::collections::{HashMap, HashSet};

use anyhow::bail;

use crate::crypto::commitment::Commitment;
use crate::crypto::shamir::Share;
use crate::domain::model::{Depth, NodeId};

pub struct ShareHolder {
    pub min_clients: usize,
    target: usize,
    stats: HashMap<Commitment, Share>,
    stats_frozen: bool,
    gradients: HashMap<Depth, HashMap<Commitment, HashMap<NodeId, Share>>>,
    gradients_frozen: HashMap<(i32, i32), bool>,
    current_round_id: i32,
}

impl ShareHolder {
    pub fn new(min_clients: usize) -> Self {
        Self {
            min_clients,
            target: min_clients,
            stats: HashMap::new(),
            stats_frozen: false,
            gradients: HashMap::new(),
            gradients_frozen: HashMap::new(),
            current_round_id: -1,
        }
    }

    pub fn set_target(&mut self, target: usize) {
        self.target = target;
    }

    pub fn is_stats_frozen(&self) -> bool {
        self.stats_frozen
    }

    pub fn is_gradients_frozen(&self, round_id: i32, depth: i32) -> bool {
        self.gradients_frozen
            .get(&(round_id, depth))
            .copied()
            .unwrap_or(false)
    }

    pub fn current_round_id(&self) -> i32 {
        self.current_round_id
    }

    pub fn receive_stats(&mut self, commitment: Commitment, share: Share) -> bool {
        if self.stats_frozen {
            return false;
        }
        self.stats.insert(commitment, share);
        if self.stats.len() >= self.target {
            self.stats_frozen = true;
        }
        true
    }

    pub fn receive_gradients(
        &mut self,
        round_id: i32,
        depth: Depth,
        commitment: Commitment,
        share: Share,
        node_id: NodeId,
    ) -> bool {
        if self.is_gradients_frozen(round_id, depth) {
            return false;
        }
        if round_id > self.current_round_id {
            self.gradients.clear();
            self.current_round_id = round_id;
        }
        self.gradients
            .entry(depth)
            .or_default()
            .entry(commitment)
            .or_default()
            .insert(node_id, share);

        let commitment_count = self.gradients.get(&depth).map_or(0, |d| d.len());
        if commitment_count >= self.target {
            self.gradients_frozen.insert((round_id, depth), true);
        }
        true
    }

    pub fn get_stats_commitments(&self) -> HashSet<Commitment> {
        self.stats.keys().copied().collect()
    }

    pub fn get_gradient_commitments(&self, depth: Depth) -> HashSet<Commitment> {
        match self.gradients.get(&depth) {
            Some(depth_data) => depth_data.keys().copied().collect(),
            None => HashSet::new(),
        }
    }

    pub fn get_gradient_node_ids(&self, depth: Depth) -> HashSet<NodeId> {
        match self.gradients.get(&depth) {
            Some(depth_data) => {
                let mut node_ids = HashSet::new();
                for client_nodes in depth_data.values() {
                    node_ids.extend(client_nodes.keys());
                }
                node_ids
            }
            None => HashSet::new(),
        }
    }

    pub fn get_stats_sum(&self, commitments: &[Commitment]) -> anyhow::Result<Vec<f64>> {
        if commitments.len() < self.min_clients {
            bail!(
                "Requested {} clients, minimum is {}",
                commitments.len(),
                self.min_clients
            );
        }
        let mut total: Option<Vec<f64>> = None;
        for commitment in commitments {
            let s = self
                .stats
                .get(commitment)
                .ok_or_else(|| anyhow::anyhow!("Unknown commitment: {:02x}{:02x}...", commitment[0], commitment[1]))?;
            match &mut total {
                None => total = Some(s.y.clone()),
                Some(t) => {
                    for (a, b) in t.iter_mut().zip(s.y.iter()) {
                        *a += b;
                    }
                }
            }
        }
        total.ok_or_else(|| anyhow::anyhow!("No shares to sum"))
    }

    pub fn get_gradients_sum(
        &self,
        depth: Depth,
        commitments: &[Commitment],
        node_id: NodeId,
    ) -> anyhow::Result<Vec<f64>> {
        if commitments.len() < self.min_clients {
            bail!(
                "Requested {} clients, minimum is {}",
                commitments.len(),
                self.min_clients
            );
        }
        let empty_depth: HashMap<Commitment, HashMap<NodeId, Share>> = HashMap::new();
        let depth_data = self.gradients.get(&depth).unwrap_or(&empty_depth);
        let mut total: Option<Vec<f64>> = None;
        for commitment in commitments {
            if let Some(client_nodes) = depth_data.get(commitment)
                && let Some(s) = client_nodes.get(&node_id)
            {
                match &mut total {
                    None => total = Some(s.y.clone()),
                    Some(t) => {
                        for (a, b) in t.iter_mut().zip(s.y.iter()) {
                            *a += b;
                        }
                    }
                }
            }
        }
        total.ok_or_else(|| anyhow::anyhow!("No shares for node {}", node_id))
    }

    pub fn reset(&mut self) {
        self.stats.clear();
        self.gradients.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_share(x: i32, values: &[f64]) -> Share {
        Share {
            x,
            y: values.to_vec(),
        }
    }

    #[test]
    fn test_receive_and_get_stats() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_stats(c1, make_share(1, &[10.0, 20.0]));
        sh.receive_stats(c2, make_share(1, &[30.0, 40.0]));
        let commitments = sh.get_stats_commitments();
        assert_eq!(commitments.len(), 2);
    }

    #[test]
    fn test_stats_sum() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_stats(c1, make_share(1, &[10.0, 20.0]));
        sh.receive_stats(c2, make_share(1, &[30.0, 40.0]));
        let sum = sh.get_stats_sum(&[c1, c2]).unwrap();
        assert_eq!(sum, vec![40.0, 60.0]);
    }

    #[test]
    fn test_min_clients_enforcement() {
        let mut sh = ShareHolder::new(5);
        let c1 = [1u8; 32];
        sh.receive_stats(c1, make_share(1, &[1.0]));
        assert!(sh.get_stats_sum(&[c1]).is_err());
    }

    #[test]
    fn test_gradient_round_reset() {
        let mut sh = ShareHolder::new(1);
        let c1 = [1u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 1);
        // New round clears
        let c2 = [2u8; 32];
        sh.receive_gradients(1, 0, c2, make_share(1, &[2.0]), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 1);
        assert!(sh.get_gradient_commitments(0).contains(&c2));
    }

    #[test]
    fn test_gradients_sum() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[5.0, 10.0]), 0);
        sh.receive_gradients(0, 0, c2, make_share(1, &[3.0, 7.0]), 0);
        let sum = sh.get_gradients_sum(0, &[c1, c2], 0).unwrap();
        assert_eq!(sum, vec![8.0, 17.0]);
    }

    #[test]
    fn test_gradients_sum_missing_node() {
        let mut sh = ShareHolder::new(1);
        let c1 = [1u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0);
        assert!(sh.get_gradients_sum(0, &[c1], 99).is_err());
    }

    #[test]
    fn test_gradient_node_ids() {
        let mut sh = ShareHolder::new(3);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0);
        sh.receive_gradients(0, 0, c1, make_share(1, &[2.0]), 1);
        sh.receive_gradients(0, 0, c2, make_share(1, &[3.0]), 0);
        sh.receive_gradients(0, 0, c2, make_share(1, &[4.0]), 2);
        let node_ids = sh.get_gradient_node_ids(0);
        assert_eq!(node_ids.len(), 3);
        assert!(node_ids.contains(&0));
        assert!(node_ids.contains(&1));
        assert!(node_ids.contains(&2));
    }

    #[test]
    fn test_reset() {
        let mut sh = ShareHolder::new(1);
        let c1 = [1u8; 32];
        sh.receive_stats(c1, make_share(1, &[1.0]));
        sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0);
        sh.reset();
        assert_eq!(sh.get_stats_commitments().len(), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 0);
    }

    #[test]
    fn test_current_round_id() {
        let mut sh = ShareHolder::new(1);
        assert_eq!(sh.current_round_id(), -1);
        let c1 = [1u8; 32];
        sh.receive_gradients(3, 0, c1, make_share(1, &[1.0]), 0);
        assert_eq!(sh.current_round_id(), 3);
    }

    #[test]
    fn test_min_clients_enforcement_gradients() {
        let mut sh = ShareHolder::new(5);
        let c1 = [1u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0);
        assert!(sh.get_gradients_sum(0, &[c1], 0).is_err());
    }

    // --- Task 4: Freeze tests ---

    #[test]
    fn test_stats_freeze() {
        let mut sh = ShareHolder::new(2);
        assert!(!sh.is_stats_frozen());
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        assert!(sh.receive_stats(c1, make_share(1, &[1.0])));
        assert!(!sh.is_stats_frozen());
        assert!(sh.receive_stats(c2, make_share(1, &[2.0])));
        // 2 commitments >= min_clients=2, should auto-freeze
        assert!(sh.is_stats_frozen());
        // Subsequent submissions rejected
        let c3 = [3u8; 32];
        assert!(!sh.receive_stats(c3, make_share(1, &[3.0])));
        assert_eq!(sh.get_stats_commitments().len(), 2);
    }

    #[test]
    fn test_gradients_freeze() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        let c3 = [3u8; 32];
        assert!(sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0));
        assert!(!sh.is_gradients_frozen(0, 0));
        assert!(sh.receive_gradients(0, 0, c2, make_share(1, &[2.0]), 0));
        assert!(sh.is_gradients_frozen(0, 0));
        // Rejected after freeze
        assert!(!sh.receive_gradients(0, 0, c3, make_share(1, &[3.0]), 0));
    }

    #[test]
    fn test_freeze_with_custom_target() {
        let mut sh = ShareHolder::new(2);
        sh.set_target(5); // freeze at 5, not min_clients
        for i in 0u8..4 {
            assert!(sh.receive_stats([i; 32], make_share(1, &[1.0])));
        }
        assert!(!sh.is_stats_frozen());
        assert!(sh.receive_stats([4u8; 32], make_share(1, &[5.0])));
        assert!(sh.is_stats_frozen());
    }
}
