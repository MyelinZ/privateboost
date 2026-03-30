use super::messages::{CommittedGradientShare, CommittedStatsShare};
use crate::crypto::{Commitment, F, Share};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GradientKey {
    depth: usize,
    commitment: Commitment,
    node_id: usize,
}

pub struct ShareHolder {
    pub party_id: usize,
    pub x_coord: u64,
    pub min_clients: usize,
    stats: BTreeMap<Commitment, Vec<F>>,
    gradients: BTreeMap<GradientKey, Vec<F>>,
    current_round_id: i64,
}

impl ShareHolder {
    pub fn new(party_id: usize, x_coord: u64, min_clients: usize) -> Self {
        Self {
            party_id,
            x_coord,
            min_clients,
            stats: BTreeMap::new(),
            gradients: BTreeMap::new(),
            current_round_id: -1,
        }
    }

    pub fn receive_stats(&mut self, msg: CommittedStatsShare) {
        self.stats.insert(msg.commitment, msg.share.values);
    }

    pub fn receive_gradients(&mut self, msg: CommittedGradientShare) {
        if msg.round_id as i64 > self.current_round_id {
            self.gradients.clear();
            self.current_round_id = msg.round_id as i64;
        }
        let key = GradientKey {
            depth: msg.depth,
            commitment: msg.commitment,
            node_id: msg.node_id,
        };
        self.gradients.insert(key, msg.share.values);
    }

    pub fn get_stats_commitments(&self) -> BTreeSet<Commitment> {
        self.stats.keys().cloned().collect()
    }

    pub fn get_gradient_commitments(&self, depth: usize) -> BTreeSet<Commitment> {
        self.gradients
            .keys()
            .filter(|k| k.depth == depth)
            .map(|k| k.commitment.clone())
            .collect()
    }

    pub fn get_gradient_node_ids(&self, depth: usize) -> BTreeSet<usize> {
        self.gradients
            .keys()
            .filter(|k| k.depth == depth)
            .map(|k| k.node_id)
            .collect()
    }

    pub fn get_stats_sum(&self, commitments: &[Commitment]) -> Result<Share> {
        if commitments.len() < self.min_clients {
            return Err(Error::InsufficientClients {
                needed: self.min_clients,
                got: commitments.len(),
            });
        }
        let mut total: Option<Vec<F>> = None;
        for commitment in commitments {
            let values = self.stats.get(commitment).ok_or(Error::UnknownCommitment)?;
            total = Some(match total {
                None => values.clone(),
                Some(t) => t.iter().zip(values.iter()).map(|(a, b)| *a + *b).collect(),
            });
        }
        let values = total.ok_or(Error::InsufficientClients { needed: 1, got: 0 })?;
        Ok(Share {
            x: self.x_coord,
            values,
        })
    }

    pub fn get_gradients_sum(
        &self,
        depth: usize,
        commitments: &[Commitment],
        node_id: usize,
    ) -> Result<Share> {
        if commitments.len() < self.min_clients {
            return Err(Error::InsufficientClients {
                needed: self.min_clients,
                got: commitments.len(),
            });
        }
        let mut total: Option<Vec<F>> = None;
        for commitment in commitments {
            let key = GradientKey {
                depth,
                commitment: commitment.clone(),
                node_id,
            };
            if let Some(values) = self.gradients.get(&key) {
                total = Some(match total {
                    None => values.clone(),
                    Some(t) => t.iter().zip(values.iter()).map(|(a, b)| *a + *b).collect(),
                });
            }
        }
        let values = total.ok_or(Error::NoSharesForNode(node_id))?;
        Ok(Share {
            x: self.x_coord,
            values,
        })
    }

    pub fn reset(&mut self) {
        self.stats.clear();
        self.gradients.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::field::ZERO;

    fn make_share(x: u64, n: usize) -> Share {
        Share {
            x,
            values: vec![ZERO; n],
        }
    }

    #[test]
    fn test_min_clients_enforcement_stats() {
        let sh = ShareHolder::new(0, 1, 10);
        let result = sh.get_stats_sum(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_receive_and_retrieve_stats() {
        let mut sh = ShareHolder::new(0, 1, 1);
        let commitment = Commitment([0u8; 32]);
        sh.receive_stats(CommittedStatsShare {
            commitment: commitment.clone(),
            share: make_share(1, 4),
        });
        assert!(sh.get_stats_commitments().contains(&commitment));
        let result = sh.get_stats_sum(&[commitment]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().x, 1);
    }

    #[test]
    fn test_gradient_round_reset() {
        let mut sh = ShareHolder::new(0, 1, 1);
        let c = Commitment([0u8; 32]);
        sh.receive_gradients(CommittedGradientShare {
            round_id: 0,
            depth: 0,
            commitment: c.clone(),
            share: make_share(1, 4),
            node_id: 0,
        });
        assert!(!sh.get_gradient_commitments(0).is_empty());
        sh.receive_gradients(CommittedGradientShare {
            round_id: 1,
            depth: 0,
            commitment: c.clone(),
            share: make_share(1, 4),
            node_id: 0,
        });
        assert_eq!(sh.get_gradient_commitments(0).len(), 1);
    }
}
