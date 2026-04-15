use super::messages::{CommittedGradientShare, CommittedStatsShare};
use crate::crypto::{Commitment, F, Share};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NodeKey {
    depth: usize,
    node_id: usize,
}

struct GradientAccum {
    sum: Vec<F>,
    commitments: BTreeSet<Commitment>,
}

pub struct ShareHolder {
    pub party_id: usize,
    pub x_coord: u64,
    pub min_clients: usize,
    stats: BTreeMap<Commitment, Vec<F>>,
    gradient_accums: BTreeMap<NodeKey, GradientAccum>,
    current_round_id: i64,
}

impl ShareHolder {
    pub fn new(party_id: usize, x_coord: u64, min_clients: usize) -> Self {
        Self {
            party_id,
            x_coord,
            min_clients,
            stats: BTreeMap::new(),
            gradient_accums: BTreeMap::new(),
            current_round_id: -1,
        }
    }

    pub fn receive_stats(&mut self, msg: CommittedStatsShare) {
        self.stats.insert(msg.commitment, msg.share.values);
    }

    pub fn receive_gradients(&mut self, msg: CommittedGradientShare) {
        if msg.round_id as i64 > self.current_round_id {
            self.gradient_accums.clear();
            self.current_round_id = msg.round_id as i64;
        }
        let key = NodeKey {
            depth: msg.depth,
            node_id: msg.node_id,
        };
        let accum = self.gradient_accums.entry(key).or_insert_with(|| GradientAccum {
            sum: vec![],
            commitments: BTreeSet::new(),
        });
        if accum.sum.is_empty() {
            accum.sum = msg.share.values;
        } else {
            for (s, v) in accum.sum.iter_mut().zip(msg.share.values.iter()) {
                *s += *v;
            }
        }
        accum.commitments.insert(msg.commitment);
    }

    pub fn get_stats_commitments(&self) -> BTreeSet<Commitment> {
        self.stats.keys().cloned().collect()
    }

    /// Return the set of commitments this shareholder has received for any node at
    /// the given depth. Since every client sends shares for all active nodes (or
    /// just its true node), the commitment set is the same across nodes.
    pub fn get_gradient_commitments(&self, depth: usize) -> BTreeSet<Commitment> {
        self.gradient_accums
            .iter()
            .filter(|(k, _)| k.depth == depth)
            .flat_map(|(_, a)| a.commitments.iter().cloned())
            .collect()
    }

    pub fn get_gradient_node_ids(&self, depth: usize) -> BTreeSet<usize> {
        self.gradient_accums
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

    /// Return the pre-computed sum for a given node. The caller is responsible
    /// for ensuring a valid quorum via `select_shareholders` at the depth level.
    /// The per-node commitment count is used only for min_clients enforcement.
    pub fn get_gradients_sum(
        &self,
        depth: usize,
        node_id: usize,
    ) -> Result<Share> {
        let key = NodeKey { depth, node_id };
        let accum = self.gradient_accums.get(&key).ok_or(Error::NoSharesForNode(node_id))?;
        if accum.commitments.len() < self.min_clients {
            return Err(Error::InsufficientClients {
                needed: self.min_clients,
                got: accum.commitments.len(),
            });
        }
        Ok(Share {
            x: self.x_coord,
            values: accum.sum.clone(),
        })
    }

    pub fn reset(&mut self) {
        self.stats.clear();
        self.gradient_accums.clear();
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
        assert!(!sh.get_gradient_node_ids(0).is_empty());
        sh.receive_gradients(CommittedGradientShare {
            round_id: 1,
            depth: 0,
            commitment: c.clone(),
            share: make_share(1, 4),
            node_id: 0,
        });
        assert_eq!(sh.get_gradient_node_ids(0).len(), 1);
    }

    #[test]
    fn test_streaming_gradient_sum() {
        use crate::crypto::field::MersenneField;

        let mut sh = ShareHolder::new(0, 1, 2);

        let c1 = Commitment([1u8; 32]);
        let c2 = Commitment([2u8; 32]);

        sh.receive_gradients(CommittedGradientShare {
            round_id: 0,
            depth: 0,
            commitment: c1.clone(),
            share: Share { x: 1, values: vec![MersenneField::from_u64(10), MersenneField::from_u64(20)] },
            node_id: 0,
        });
        sh.receive_gradients(CommittedGradientShare {
            round_id: 0,
            depth: 0,
            commitment: c2.clone(),
            share: Share { x: 1, values: vec![MersenneField::from_u64(3), MersenneField::from_u64(7)] },
            node_id: 0,
        });

        let result = sh.get_gradients_sum(0, 0).unwrap();
        assert_eq!(result.values[0], MersenneField::from_u64(13));
        assert_eq!(result.values[1], MersenneField::from_u64(27));
        assert_eq!(result.x, 1);

        // Verify commitment tracking
        let commitments = sh.get_gradient_commitments(0);
        assert!(commitments.contains(&c1));
        assert!(commitments.contains(&c2));
        assert_eq!(commitments.len(), 2);
    }

    #[test]
    fn test_gradient_sum_min_clients_enforcement() {
        use crate::crypto::field::MersenneField;

        let mut sh = ShareHolder::new(0, 1, 5); // min_clients = 5

        sh.receive_gradients(CommittedGradientShare {
            round_id: 0,
            depth: 0,
            commitment: Commitment([1u8; 32]),
            share: Share { x: 1, values: vec![MersenneField::from_u64(10)] },
            node_id: 0,
        });

        // Should fail because only 1 client but min_clients=5
        let result = sh.get_gradients_sum(0, 0);
        assert!(result.is_err());
    }
}
