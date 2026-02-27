use std::collections::{HashMap, HashSet};

use anyhow::bail;
use curve25519_dalek::scalar::Scalar;

use crate::crypto::commitment::Commitment;
use crate::crypto::shamir::Share;
use crate::domain::model::{Depth, NodeId, StepId, StepType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteStatus {
    Pending,
    Consensus,
    Disputed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    CollectingStats,
    FrozenStats,
    CollectingGradients,
    FrozenGradients,
    TrainingComplete,
}

pub struct ShareHolder {
    pub min_clients: usize,
    target: usize,
    stats: HashMap<Commitment, Share>,
    stats_frozen: bool,
    gradients: HashMap<Depth, HashMap<Commitment, HashMap<NodeId, Share>>>,
    gradients_frozen: HashMap<(i32, i32), bool>,
    current_round_id: i32,
    expected_aggregators: usize,
    votes: HashMap<StepId, HashMap<i32, (Vec<u8>, Vec<u8>)>>,
    consensus_results: HashMap<StepId, Vec<u8>>,
    phase: Phase,
    current_depth: i32,
    n_trees: usize,
    max_depth: usize,
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
            expected_aggregators: 1,
            votes: HashMap::new(),
            consensus_results: HashMap::new(),
            phase: Phase::CollectingStats,
            current_depth: 0,
            n_trees: 0,
            max_depth: 0,
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

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn current_depth(&self) -> i32 {
        self.current_depth
    }

    pub fn set_expected_aggregators(&mut self, n: usize) {
        self.expected_aggregators = n;
    }

    pub fn set_training_params(&mut self, n_trees: usize, max_depth: usize) {
        self.n_trees = n_trees;
        self.max_depth = max_depth;
    }

    pub fn receive_stats(&mut self, commitment: Commitment, share: Share) -> bool {
        if self.stats_frozen {
            return false;
        }
        self.stats.insert(commitment, share);
        if self.stats.len() >= self.target {
            self.stats_frozen = true;
            self.phase = Phase::FrozenStats;
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
            if round_id == self.current_round_id && depth == self.current_depth {
                self.phase = Phase::FrozenGradients;
            }
        }
        true
    }

    pub fn submit_vote(
        &mut self,
        step: StepId,
        aggregator_id: i32,
        result_hash: Vec<u8>,
        result_data: Vec<u8>,
    ) -> VoteStatus {
        let step_votes = self.votes.entry(step).or_default();

        if step_votes.contains_key(&aggregator_id) {
            return VoteStatus::Pending;
        }

        step_votes.insert(aggregator_id, (result_hash, result_data));

        if step_votes.len() >= self.expected_aggregators {
            let mut hash_groups: HashMap<&Vec<u8>, (usize, &Vec<u8>)> = HashMap::new();
            for (hash, data) in step_votes.values() {
                let entry = hash_groups.entry(hash).or_insert((0, data));
                entry.0 += 1;
            }

            let (max_count, winning_data) = hash_groups
                .values()
                .max_by_key(|(count, _)| *count)
                .unwrap();

            if *max_count > self.expected_aggregators / 2 {
                self.consensus_results
                    .insert(step, winning_data.to_vec());
                self.advance_phase_on_consensus(step);
                VoteStatus::Consensus
            } else {
                VoteStatus::Disputed
            }
        } else {
            VoteStatus::Pending
        }
    }

    fn advance_phase_on_consensus(&mut self, step: StepId) {
        match step.step_type {
            StepType::Stats => {
                if self.phase == Phase::FrozenStats {
                    self.phase = Phase::CollectingGradients;
                    self.current_round_id = 0;
                    self.current_depth = 0;
                    self.gradients_frozen.clear();
                }
            }
            StepType::Gradients => {
                if step.round_id != self.current_round_id || step.depth != self.current_depth {
                    return;
                }
                if (self.current_depth as usize) < self.max_depth - 1 {
                    self.phase = Phase::CollectingGradients;
                    self.current_depth += 1;
                } else if (self.current_round_id as usize) < self.n_trees - 1 {
                    self.phase = Phase::CollectingGradients;
                    self.current_round_id += 1;
                    self.current_depth = 0;
                } else {
                    self.phase = Phase::TrainingComplete;
                }
            }
        }
    }

    pub fn get_consensus(&self, step: StepId) -> Option<&Vec<u8>> {
        self.consensus_results.get(&step)
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

    pub fn get_stats_sum(&self, commitments: &[Commitment]) -> anyhow::Result<Vec<Scalar>> {
        if commitments.len() < self.min_clients {
            bail!(
                "Requested {} clients, minimum is {}",
                commitments.len(),
                self.min_clients
            );
        }
        let mut total: Option<Vec<Scalar>> = None;
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
    ) -> anyhow::Result<Vec<Scalar>> {
        if commitments.len() < self.min_clients {
            bail!(
                "Requested {} clients, minimum is {}",
                commitments.len(),
                self.min_clients
            );
        }
        let empty_depth: HashMap<Commitment, HashMap<NodeId, Share>> = HashMap::new();
        let depth_data = self.gradients.get(&depth).unwrap_or(&empty_depth);
        let mut total: Option<Vec<Scalar>> = None;
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

    fn scalar(v: u64) -> Scalar {
        Scalar::from(v)
    }

    fn make_share(x: i32, values: &[Scalar]) -> Share {
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
        sh.receive_stats(c1, make_share(1, &[scalar(10), scalar(20)]));
        sh.receive_stats(c2, make_share(1, &[scalar(30), scalar(40)]));
        let commitments = sh.get_stats_commitments();
        assert_eq!(commitments.len(), 2);
    }

    #[test]
    fn test_stats_sum() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_stats(c1, make_share(1, &[scalar(10), scalar(20)]));
        sh.receive_stats(c2, make_share(1, &[scalar(30), scalar(40)]));
        let sum = sh.get_stats_sum(&[c1, c2]).unwrap();
        assert_eq!(sum, vec![scalar(40), scalar(60)]);
    }

    #[test]
    fn test_min_clients_enforcement() {
        let mut sh = ShareHolder::new(5);
        let c1 = [1u8; 32];
        sh.receive_stats(c1, make_share(1, &[scalar(1)]));
        assert!(sh.get_stats_sum(&[c1]).is_err());
    }

    #[test]
    fn test_gradient_round_reset() {
        let mut sh = ShareHolder::new(1);
        let c1 = [1u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(1)]), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 1);
        // New round clears
        let c2 = [2u8; 32];
        sh.receive_gradients(1, 0, c2, make_share(1, &[scalar(2)]), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 1);
        assert!(sh.get_gradient_commitments(0).contains(&c2));
    }

    #[test]
    fn test_gradients_sum() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(5), scalar(10)]), 0);
        sh.receive_gradients(0, 0, c2, make_share(1, &[scalar(3), scalar(7)]), 0);
        let sum = sh.get_gradients_sum(0, &[c1, c2], 0).unwrap();
        assert_eq!(sum, vec![scalar(8), scalar(17)]);
    }

    #[test]
    fn test_gradients_sum_missing_node() {
        let mut sh = ShareHolder::new(1);
        let c1 = [1u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(1)]), 0);
        assert!(sh.get_gradients_sum(0, &[c1], 99).is_err());
    }

    #[test]
    fn test_gradient_node_ids() {
        let mut sh = ShareHolder::new(3);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(1)]), 0);
        sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(2)]), 1);
        sh.receive_gradients(0, 0, c2, make_share(1, &[scalar(3)]), 0);
        sh.receive_gradients(0, 0, c2, make_share(1, &[scalar(4)]), 2);
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
        sh.receive_stats(c1, make_share(1, &[scalar(1)]));
        sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(1)]), 0);
        sh.reset();
        assert_eq!(sh.get_stats_commitments().len(), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 0);
    }

    #[test]
    fn test_current_round_id() {
        let mut sh = ShareHolder::new(1);
        assert_eq!(sh.current_round_id(), -1);
        let c1 = [1u8; 32];
        sh.receive_gradients(3, 0, c1, make_share(1, &[scalar(1)]), 0);
        assert_eq!(sh.current_round_id(), 3);
    }

    #[test]
    fn test_min_clients_enforcement_gradients() {
        let mut sh = ShareHolder::new(5);
        let c1 = [1u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(1)]), 0);
        assert!(sh.get_gradients_sum(0, &[c1], 0).is_err());
    }

    // --- Task 4: Freeze tests ---

    #[test]
    fn test_stats_freeze() {
        let mut sh = ShareHolder::new(2);
        assert!(!sh.is_stats_frozen());
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        assert!(sh.receive_stats(c1, make_share(1, &[scalar(1)])));
        assert!(!sh.is_stats_frozen());
        assert!(sh.receive_stats(c2, make_share(1, &[scalar(2)])));
        // 2 commitments >= min_clients=2, should auto-freeze
        assert!(sh.is_stats_frozen());
        // Subsequent submissions rejected
        let c3 = [3u8; 32];
        assert!(!sh.receive_stats(c3, make_share(1, &[scalar(3)])));
        assert_eq!(sh.get_stats_commitments().len(), 2);
    }

    #[test]
    fn test_gradients_freeze() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        let c3 = [3u8; 32];
        assert!(sh.receive_gradients(0, 0, c1, make_share(1, &[scalar(1)]), 0));
        assert!(!sh.is_gradients_frozen(0, 0));
        assert!(sh.receive_gradients(0, 0, c2, make_share(1, &[scalar(2)]), 0));
        assert!(sh.is_gradients_frozen(0, 0));
        // Rejected after freeze
        assert!(!sh.receive_gradients(0, 0, c3, make_share(1, &[scalar(3)]), 0));
    }

    #[test]
    fn test_freeze_with_custom_target() {
        let mut sh = ShareHolder::new(2);
        sh.set_target(5); // freeze at 5, not min_clients
        for i in 0u8..4 {
            assert!(sh.receive_stats([i; 32], make_share(1, &[scalar(1)])));
        }
        assert!(!sh.is_stats_frozen());
        assert!(sh.receive_stats([4u8; 32], make_share(1, &[scalar(5)])));
        assert!(sh.is_stats_frozen());
    }

    // --- Task 5: Consensus voting tests ---

    #[test]
    fn test_consensus_majority() {
        let mut sh = ShareHolder::new(1);
        let step = StepId::stats();
        let result_a = vec![1u8; 32]; // hash A
        let result_b = vec![2u8; 32]; // hash B

        sh.set_expected_aggregators(3);
        assert_eq!(sh.submit_vote(step, 0, result_a.clone(), b"data_a".to_vec()), VoteStatus::Pending);
        assert_eq!(sh.submit_vote(step, 1, result_b.clone(), b"data_b".to_vec()), VoteStatus::Pending);
        assert_eq!(sh.submit_vote(step, 2, result_a.clone(), b"data_a".to_vec()), VoteStatus::Consensus);

        let consensus = sh.get_consensus(step).unwrap();
        assert_eq!(consensus, b"data_a");
    }

    #[test]
    fn test_consensus_no_majority() {
        let mut sh = ShareHolder::new(1);
        let step = StepId::stats();
        sh.set_expected_aggregators(3);
        sh.submit_vote(step, 0, vec![1u8; 32], b"a".to_vec());
        sh.submit_vote(step, 1, vec![2u8; 32], b"b".to_vec());
        let status = sh.submit_vote(step, 2, vec![3u8; 32], b"c".to_vec());
        assert_eq!(status, VoteStatus::Disputed);
    }

    #[test]
    fn test_consensus_duplicate_aggregator() {
        let mut sh = ShareHolder::new(1);
        let step = StepId::stats();
        sh.set_expected_aggregators(3);
        assert_eq!(sh.submit_vote(step, 0, vec![1u8; 32], b"a".to_vec()), VoteStatus::Pending);
        // Same aggregator_id again -- should be rejected (still Pending)
        assert_eq!(sh.submit_vote(step, 0, vec![1u8; 32], b"a".to_vec()), VoteStatus::Pending);
        // Still only 1 vote counted, no consensus
        assert!(sh.get_consensus(step).is_none());
    }

    // --- Task 6: State machine tests ---

    #[test]
    fn test_state_machine_stats_flow() {
        let mut sh = ShareHolder::new(2);
        sh.set_target(2);
        sh.set_expected_aggregators(1);
        sh.set_training_params(3, 3);

        assert_eq!(sh.phase(), Phase::CollectingStats);

        // Submit stats until frozen
        sh.receive_stats([1u8; 32], make_share(1, &[scalar(1)]));
        sh.receive_stats([2u8; 32], make_share(1, &[scalar(2)]));
        assert_eq!(sh.phase(), Phase::FrozenStats);

        // Aggregator submits bins consensus
        let step = StepId::stats();
        sh.submit_vote(step, 0, vec![1u8; 32], b"bins".to_vec());
        assert_eq!(sh.phase(), Phase::CollectingGradients);
        assert_eq!(sh.current_round_id(), 0);
        assert_eq!(sh.current_depth(), 0);
    }

    #[test]
    fn test_state_machine_gradient_advance_depth() {
        let mut sh = ShareHolder::new(2);
        sh.set_target(2);
        sh.set_expected_aggregators(1);
        sh.set_training_params(3, 3); // 3 trees, max_depth 3

        // Get to gradient phase
        sh.receive_stats([1u8; 32], make_share(1, &[scalar(1)]));
        sh.receive_stats([2u8; 32], make_share(1, &[scalar(2)]));
        sh.submit_vote(StepId::stats(), 0, vec![1u8; 32], b"bins".to_vec());
        assert_eq!(sh.phase(), Phase::CollectingGradients);

        // Submit gradients until frozen
        sh.receive_gradients(0, 0, [10u8; 32], make_share(1, &[scalar(1)]), 0);
        sh.receive_gradients(0, 0, [11u8; 32], make_share(1, &[scalar(2)]), 0);
        assert_eq!(sh.phase(), Phase::FrozenGradients);

        // Aggregator submits splits consensus
        sh.submit_vote(StepId::gradients(0, 0), 0, vec![1u8; 32], b"splits".to_vec());
        // Advances to next depth
        assert_eq!(sh.phase(), Phase::CollectingGradients);
        assert_eq!(sh.current_depth(), 1);
    }

    #[test]
    fn test_state_machine_round_advance() {
        let mut sh = ShareHolder::new(2);
        sh.set_target(2);
        sh.set_expected_aggregators(1);
        sh.set_training_params(2, 2); // 2 trees, max_depth 2

        // Stats phase
        sh.receive_stats([1u8; 32], make_share(1, &[scalar(1)]));
        sh.receive_stats([2u8; 32], make_share(1, &[scalar(2)]));
        sh.submit_vote(StepId::stats(), 0, vec![1u8; 32], b"bins".to_vec());

        // Round 0, depth 0
        sh.receive_gradients(0, 0, [10u8; 32], make_share(1, &[scalar(1)]), 0);
        sh.receive_gradients(0, 0, [11u8; 32], make_share(1, &[scalar(2)]), 0);
        sh.submit_vote(StepId::gradients(0, 0), 0, vec![1u8; 32], b"splits0".to_vec());

        // Round 0, depth 1 (last depth)
        sh.receive_gradients(0, 1, [20u8; 32], make_share(1, &[scalar(1)]), 0);
        sh.receive_gradients(0, 1, [21u8; 32], make_share(1, &[scalar(2)]), 0);
        sh.submit_vote(StepId::gradients(0, 1), 0, vec![1u8; 32], b"tree0".to_vec());

        // Should advance to round 1
        assert_eq!(sh.phase(), Phase::CollectingGradients);
        assert_eq!(sh.current_round_id(), 1);
        assert_eq!(sh.current_depth(), 0);

        // Round 1, depth 0
        sh.receive_gradients(1, 0, [30u8; 32], make_share(1, &[scalar(1)]), 0);
        sh.receive_gradients(1, 0, [31u8; 32], make_share(1, &[scalar(2)]), 0);
        sh.submit_vote(StepId::gradients(1, 0), 0, vec![1u8; 32], b"splits1".to_vec());

        // Round 1, depth 1 (last depth, last round)
        sh.receive_gradients(1, 1, [40u8; 32], make_share(1, &[scalar(1)]), 0);
        sh.receive_gradients(1, 1, [41u8; 32], make_share(1, &[scalar(2)]), 0);
        sh.submit_vote(StepId::gradients(1, 1), 0, vec![1u8; 32], b"tree1".to_vec());

        // Should be training complete
        assert_eq!(sh.phase(), Phase::TrainingComplete);
    }
}
