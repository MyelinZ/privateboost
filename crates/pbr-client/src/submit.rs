use crate::rpc::Shareholders;
use futures::future::join_all;
use pbr_proto::convert::{commitment_to_bytes, share_to_proto};
use pbr_proto::v1::{GradientBatchSubmission, GradientEntry, StatsShareSubmission};
use pbr_core::{CommittedGradientShare, CommittedStatsShare};
use std::collections::BTreeMap;

/// How many shareholders were addressed and how many acknowledged. A round's
/// contribution counts as delivered once `accepted >= threshold`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeliveryReport {
    pub attempted: usize,
    pub accepted: usize,
}

impl Shareholders {
    /// Every shareholder is attempted; a failed RPC is recorded in the report
    /// rather than aborting the fan-out.
    pub async fn submit_stats(
        &mut self,
        shares: Vec<CommittedStatsShare>,
        session_id: &str,
    ) -> DeliveryReport {
        let mut by_idx: BTreeMap<usize, CommittedStatsShare> = shares
            .into_iter()
            .map(|share| (share.share.x as usize - 1, share))
            .collect();
        // Disjoint `&mut` borrows per channel drive every RPC concurrently
        // without cloning the client or spawning a task.
        let calls = self
            .channels
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, chan)| {
                let share = by_idx.remove(&idx)?;
                Some(chan.submit_stats_shares(StatsShareSubmission {
                    commitment: commitment_to_bytes(&share.commitment),
                    share: Some(share_to_proto(&share.share)),
                    session_id: session_id.to_string(),
                }))
            });
        let mut report = DeliveryReport {
            attempted: 0,
            accepted: 0,
        };
        for result in join_all(calls).await {
            report.attempted += 1;
            match result {
                Ok(_) => report.accepted += 1,
                Err(e) => tracing::warn!(error = %e, "submit_stats_shares failed"),
            }
        }
        report
    }

    /// Sends each shareholder its whole per-round gradient contribution as one
    /// atomic `SubmitGradientBatch`. `pbr_core::Client` emits one share per
    /// (shareholder, active node) under a single per-round commitment, and
    /// grouping by evaluation point turns that into one batch of
    /// `(node_id, share)` entries per shareholder, so a shareholder holds
    /// either the client's full contribution or none of it.
    ///
    /// Every batch is attempted and one failure does not abort the others; the
    /// caller decides from the report whether enough accepted.
    pub async fn submit_gradients(
        &mut self,
        shares: Vec<CommittedGradientShare>,
        session_id: &str,
    ) -> DeliveryReport {
        let mut groups: BTreeMap<u64, Vec<CommittedGradientShare>> = BTreeMap::new();
        for share in shares {
            groups.entry(share.share.x).or_default().push(share);
        }
        // Disjoint `&mut` borrows per channel drive every batch concurrently
        // without cloning the client or spawning a task.
        let calls = self
            .channels
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, chan)| {
                let group = groups.remove(&(idx as u64 + 1))?;
                // One commit per round, so every share here carries the same
                // round_id, depth and commitment.
                let first = &group[0];
                let batch = GradientBatchSubmission {
                    round_id: first.round_id,
                    depth: first.depth as u32,
                    commitment: commitment_to_bytes(&first.commitment),
                    entries: group
                        .iter()
                        .map(|s| GradientEntry {
                            node_id: s.node_id as u32,
                            share: Some(share_to_proto(&s.share)),
                        })
                        .collect(),
                    session_id: session_id.to_string(),
                };
                Some(chan.submit_gradient_batch(batch))
            });
        let mut report = DeliveryReport {
            attempted: 0,
            accepted: 0,
        };
        for result in join_all(calls).await {
            report.attempted += 1;
            match result {
                Ok(_) => report.accepted += 1,
                Err(e) => tracing::warn!(error = %e, "submit_gradient_batch failed"),
            }
        }
        report
    }
}
