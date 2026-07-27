//! Prefetched gRPC snapshot adapter implementing `pbr-core`'s
//! [`ShareSource`] against one remote shareholder's internal service.
//!
//! `pbr_core::Aggregator<S: ShareSource>` holds one `S` per shareholder, so
//! each `GrpcShareSource` wraps exactly one shareholder's
//! `ShareholderInternal` client and the orchestrator builds the `Aggregator`
//! from a `Vec` of them.
//!
//! Async/sync bridge: the `ShareSource` trait methods are synchronous, but the
//! sum methods must issue gRPC calls. Construction prefetches the round's
//! commitments and node ids, so the listing methods (`stats_commitments`,
//! `gradient_commitments`, `gradient_node_ids`) serve purely from the snapshot:
//! sync, infallible, no network. Only the sum methods touch the network, and
//! they do so via `handle.block_on(...)`. They are only ever driven from inside
//! `tokio::task::spawn_blocking` (the aggregator runs `define_bins`/
//! `compute_splits` there), so `block_on` runs on a non-runtime thread and does
//! not panic.
//!
//! `&self` vs `&mut client`: tonic clients need `&mut self` to issue an RPC, but
//! the trait methods take `&self`. Each call clones the client instead, which
//! is cheap for a `Channel`-backed tonic client (they share the underlying
//! HTTP/2 connection pool) and avoids a `Mutex`, with its poisoning and
//! cross-`block_on` contention.

use pbr_proto::convert::{commitment_from_bytes, commitment_to_bytes, share_from_proto};
use pbr_proto::v1::shareholder_internal_client::ShareholderInternalClient;
use pbr_proto::v1::{GetSumsRequest, ListCommitmentsRequest, SharePhase};
use pbr_core::{Commitment, Error, Result, Share, ShareSource};
use std::collections::BTreeSet;
use tonic::transport::Channel;
use tonic::{Code, Status};

/// A read-only, prefetched view of one shareholder's shares for a single
/// `(phase, depth)` round.
pub struct GrpcShareSource {
    client: ShareholderInternalClient<Channel>,
    handle: tokio::runtime::Handle,
    phase: SharePhase,
    depth: usize,
    commitments: BTreeSet<Commitment>,
    node_ids: BTreeSet<usize>,
    /// The session whose pool this source reads. Every ListCommitments and
    /// GetSums call must carry it, or the sum is taken over another session's
    /// pool on the same daemon.
    session_id: String,
}

impl GrpcShareSource {
    /// Connect to one shareholder's internal endpoint and prefetch the
    /// commitments and node-ids for `(phase, depth)`. Blocking (uses
    /// `handle.block_on`); intended to be called from inside `spawn_blocking`.
    pub fn snapshot(
        internal_endpoint: &str,
        phase: SharePhase,
        depth: usize,
        session_id: &str,
        handle: tokio::runtime::Handle,
    ) -> anyhow::Result<Self> {
        let endpoint = internal_endpoint.to_string();
        let (client, commitments, node_ids) = handle.block_on(async move {
            // Same connect/per-RPC budgets as the round loop's internal channels
            // (crate::aggregator::internal_endpoint): a shareholder that wedges
            // after acking CloseRound is in the closed mask, so without a
            // deadline here list_commitments (or a later get_sums) would hang
            // run_session forever inside spawn_blocking, past GIVE_UP_WINDOWS.
            // The Endpoint timeout rides the Channel, so it also bounds the
            // get_sums calls issued over per-call clones of `client`.
            let mut client = ShareholderInternalClient::new(
                crate::aggregator::internal_endpoint(&endpoint)?
                    .connect()
                    .await?,
            );
            let list = client
                .list_commitments(ListCommitmentsRequest {
                    phase: phase as i32,
                    depth: depth as u32,
                    session_id: session_id.to_string(),
                })
                .await?
                .into_inner();
            let mut commitments = BTreeSet::new();
            for c in &list.commitments {
                commitments.insert(commitment_from_bytes(c)?);
            }
            let node_ids: BTreeSet<usize> = list.node_ids.iter().map(|&n| n as usize).collect();
            anyhow::Ok((client, commitments, node_ids))
        })?;
        Ok(Self {
            client,
            handle,
            phase,
            depth,
            commitments,
            node_ids,
            session_id: session_id.to_string(),
        })
    }

    /// Issue a `GetSums` RPC over a cheap per-call clone of the client. Runs on
    /// the calling thread via `block_on`; safe only off the runtime worker
    /// threads (see module docs).
    #[allow(clippy::result_large_err)] // tonic::Status is large; callers map it immediately.
    fn get_sums(&self, req: GetSumsRequest) -> std::result::Result<Share, Status> {
        let mut client = self.client.clone();
        let resp = self
            .handle
            .block_on(async move { client.get_sums(req).await })?;
        let share = resp
            .into_inner()
            .share
            .ok_or_else(|| Status::internal("SumShare missing share"))?;
        share_from_proto(share).map_err(|e| Status::internal(e.to_string()))
    }
}

/// The daemon signals "min_clients not met" with `FAILED_PRECONDITION` and
/// "no shares" with `NOT_FOUND`; map those back to the corresponding
/// `pbr_core::Error` variants so the aggregator's own overlap/threshold
/// logic sees them unchanged.
/// `stats_sum` has no node, so `NOT_FOUND` there means an unknown commitment.
fn stats_status_to_error(status: Status) -> Error {
    match status.code() {
        Code::FailedPrecondition => Error::InsufficientClients { needed: 0, got: 0 },
        Code::NotFound => Error::UnknownCommitment,
        _ => Error::Io(std::io::Error::other(status.to_string())),
    }
}

fn gradient_status_to_error(status: Status, node_id: usize) -> Error {
    match status.code() {
        Code::FailedPrecondition => Error::InsufficientClients { needed: 0, got: 0 },
        Code::NotFound => Error::NoSharesForNode(node_id),
        _ => Error::Io(std::io::Error::other(status.to_string())),
    }
}

impl ShareSource for GrpcShareSource {
    fn stats_commitments(&self) -> BTreeSet<Commitment> {
        if self.phase == SharePhase::Stats {
            self.commitments.clone()
        } else {
            BTreeSet::new()
        }
    }

    fn gradient_commitments(&self, depth: usize) -> BTreeSet<Commitment> {
        if self.phase == SharePhase::Gradient && depth == self.depth {
            self.commitments.clone()
        } else {
            BTreeSet::new()
        }
    }

    fn gradient_node_ids(&self, depth: usize) -> BTreeSet<usize> {
        if self.phase == SharePhase::Gradient && depth == self.depth {
            self.node_ids.clone()
        } else {
            BTreeSet::new()
        }
    }

    fn stats_sum(&self, commitments: &[Commitment]) -> Result<Share> {
        let req = GetSumsRequest {
            phase: SharePhase::Stats as i32,
            depth: 0,
            commitments: commitments.iter().map(commitment_to_bytes).collect(),
            node_id: 0,
            session_id: self.session_id.clone(),
        };
        self.get_sums(req).map_err(stats_status_to_error)
    }

    fn gradients_sum(
        &self,
        depth: usize,
        commitments: &[Commitment],
        node_id: usize,
    ) -> Result<Share> {
        let req = GetSumsRequest {
            phase: SharePhase::Gradient as i32,
            depth: depth as u32,
            commitments: commitments.iter().map(commitment_to_bytes).collect(),
            node_id: node_id as u32,
            session_id: self.session_id.clone(),
        };
        self.get_sums(req)
            .map_err(|status| gradient_status_to_error(status, node_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::WedgedServer;
    use std::time::{Duration, Instant};

    /// A shareholder that wedges AFTER acking CloseRound is still in the closed
    /// mask, so the gather builds a `GrpcShareSource` snapshot against it. Its
    /// `list_commitments` must fail within the internal per-RPC deadline instead
    /// of hanging `run_session` forever inside `spawn_blocking`: once the round
    /// is closed, `await_submissions`'s GIVE_UP_WINDOWS budget is spent and can
    /// no longer rescue a hung gather. Without the deadline the outer timeout
    /// below trips.
    #[tokio::test]
    async fn snapshot_times_out_against_a_wedged_shareholder() {
        let server = WedgedServer::spawn().await;

        // The connection establishes fine (the server speaks HTTP/2); the
        // deadline only bites on the wedged list_commitments. `snapshot` uses
        // `handle.block_on`, so drive it off the runtime worker threads exactly
        // as the gather does (spawn_blocking).
        let handle = tokio::runtime::Handle::current();
        let url = format!("http://{}", server.addr);
        let started = Instant::now();
        let res = tokio::time::timeout(
            Duration::from_secs(20),
            tokio::task::spawn_blocking(move || {
                GrpcShareSource::snapshot(&url, SharePhase::Gradient, 0, "", handle)
            }),
        )
        .await
        .expect("snapshot must return once the per-RPC deadline fires, not hang")
        .unwrap();

        assert!(
            res.is_err(),
            "a wedged shareholder's snapshot must error, not hang"
        );
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "the internal-plane RPC deadline must fire well within the give-up budget"
        );

        server.shutdown().await;
    }
}
