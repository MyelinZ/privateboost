use crate::grpc_share_source::GrpcShareSource;
use pbr_proto::v1::SharePhase;
use pbr_core::{Commitment, Share, ShareSource};
use std::collections::BTreeSet;

/// One shareholder's gather-side view for the current round. `Dead` stands in
/// when the snapshot could not be built, contributing empty listings so
/// `pbr_core::Aggregator`'s threshold-combination selection routes around it:
/// the 2-of-3 tolerance the sharing scheme exists to provide.
pub(super) enum SourceSlot {
    Live(Box<GrpcShareSource>),
    Dead,
}

impl ShareSource for SourceSlot {
    fn stats_commitments(&self) -> BTreeSet<Commitment> {
        match self {
            SourceSlot::Live(s) => s.stats_commitments(),
            SourceSlot::Dead => BTreeSet::new(),
        }
    }

    fn gradient_commitments(&self, depth: usize) -> BTreeSet<Commitment> {
        match self {
            SourceSlot::Live(s) => s.gradient_commitments(depth),
            SourceSlot::Dead => BTreeSet::new(),
        }
    }

    fn gradient_node_ids(&self, depth: usize) -> BTreeSet<usize> {
        match self {
            SourceSlot::Live(s) => s.gradient_node_ids(depth),
            SourceSlot::Dead => BTreeSet::new(),
        }
    }

    fn stats_sum(&self, commitments: &[Commitment]) -> pbr_core::Result<Share> {
        match self {
            SourceSlot::Live(s) => s.stats_sum(commitments),
            SourceSlot::Dead => Err(pbr_core::Error::UnknownCommitment),
        }
    }

    fn gradients_sum(
        &self,
        depth: usize,
        commitments: &[Commitment],
        node_id: usize,
    ) -> pbr_core::Result<Share> {
        match self {
            SourceSlot::Live(s) => s.gradients_sum(depth, commitments, node_id),
            SourceSlot::Dead => Err(pbr_core::Error::NoSharesForNode(node_id)),
        }
    }
}

/// Build one shareholder's snapshot slot. Blocking (uses `block_on` inside
/// `GrpcShareSource::snapshot`); must only be called from `spawn_blocking`.
pub(super) fn snapshot_slot(
    endpoint: &str,
    phase: SharePhase,
    depth: usize,
    session_id: &str,
    handle: &tokio::runtime::Handle,
) -> SourceSlot {
    match GrpcShareSource::snapshot(endpoint, phase, depth, session_id, handle.clone()) {
        Ok(s) => SourceSlot::Live(Box::new(s)),
        Err(e) => {
            tracing::warn!(
                endpoint,
                error = %e,
                "snapshot failed; shareholder excluded from this round's gather"
            );
            SourceSlot::Dead
        }
    }
}
