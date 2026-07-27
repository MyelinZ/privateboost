use crate::auth;
use crate::config::ShareholderConfig;
use pbr_proto::convert::{commitment_from_bytes, share_from_proto, share_to_proto};
use pbr_proto::v1::shareholder_internal_server::{ShareholderInternal, ShareholderInternalServer};
use pbr_proto::v1::shareholder_service_server::{ShareholderService, ShareholderServiceServer};
use pbr_proto::v1::{
    Ack, CloseRoundRequest, CommitmentList, EndSessionRequest, GetSumsRequest,
    GradientBatchSubmission, ListCommitmentsRequest, OpenRoundRequest, SharePhase,
    StatsShareSubmission, SumShare,
};
use pbr_core::{CommittedGradientShare, CommittedStatsShare, ShareHolder};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

/// The round the daemon accepts submissions for. Set by `OpenRound`, cleared
/// by `CloseRound` or replaced by the next `OpenRound`. `expected_len` is fixed
/// by the first accepted submission and later mismatches are rejected, so
/// `pbr-core`'s `zip`-based summation never silently truncates a vector.
/// `session_id` always equals the containing pool's, since there is one pool
/// per session.
struct AcceptedRound {
    round_id: u64,
    depth: u32,
    phase: SharePhase,
    session_id: String,
    expected_len: Option<usize>,
}

impl AcceptedRound {
    #[allow(clippy::result_large_err)]
    fn check_len(&mut self, len: usize) -> Result<(), Status> {
        match self.expected_len {
            None => {
                self.expected_len = Some(len);
                Ok(())
            }
            Some(n) if n != len => Err(Status::invalid_argument(format!(
                "share vector length {len} does not match round's expected length {n}"
            ))),
            Some(_) => Ok(()),
        }
    }
}

/// One session's share pool and the round it is accepting. Each session needs
/// its own `ShareHolder`: the holder clears its whole gradient map when an
/// incoming round id exceeds the last one it saw, and round ids repeat across
/// sessions, so two sharing a holder would erase each other every round.
struct SessionPool {
    holder: ShareHolder,
    accepted: Option<AcceptedRound>,
    /// Last time this session advanced: a round opened or closed, or a
    /// submission passed its gate and was stored. Drives the idle sweep, so a
    /// bare lookup must never bump it, including one ending in a rejected
    /// submission; otherwise a client spamming rejected submissions could pin
    /// an orphaned pool indefinitely.
    last_touched: Instant,
}

impl SessionPool {
    /// Stats submissions carry no round_id or depth, so this gates on phase.
    #[allow(clippy::result_large_err)]
    fn gate_stats(&mut self, len: usize) -> Result<(), Status> {
        let round = self
            .accepted
            .as_mut()
            .filter(|r| r.phase == SharePhase::Stats)
            .ok_or_else(|| Status::invalid_argument("no open stats round"))?;
        round.check_len(len)
    }

    #[allow(clippy::result_large_err)]
    fn gate_gradient(&mut self, round_id: u64, depth: u32, len: usize) -> Result<(), Status> {
        let round = self
            .accepted
            .as_mut()
            .filter(|r| {
                r.phase == SharePhase::Gradient && r.round_id == round_id && r.depth == depth
            })
            .ok_or_else(|| Status::invalid_argument("no matching open gradient round"))?;
        round.check_len(len)
    }
}

/// A pool untouched this long belongs to a session whose aggregator died
/// without ending it. Generous on purpose: a live session touches its pool on
/// every OpenRound, CloseRound, gather call and stored submission, so only
/// abandoned pools age out. The backstop, not the normal path, which is
/// `EndSession`.
const SESSION_IDLE_TTL: Duration = Duration::from_secs(2 * 60 * 60);

/// A small fraction of `SESSION_IDLE_TTL`, so an abandoned pool is freed within
/// minutes of aging out rather than waiting for an OpenRound that never comes.
const SESSION_SWEEP_INTERVAL: Duration = Duration::from_secs(5 * 60);

struct DaemonState {
    sessions: HashMap<String, SessionPool>,
    /// Template values for lazily constructing a pool's `ShareHolder`.
    party_id: usize,
    x_coord: u64,
    min_clients: usize,
}

impl DaemonState {
    /// Creates the pool on the session's first OpenRound. Only `open_round`
    /// may create one: a submission or gather naming an unknown session must be
    /// rejected, or any client could grow the map without bound.
    fn pool_for_open(&mut self, session_id: &str) -> &mut SessionPool {
        let (party_id, x_coord, min_clients) = (self.party_id, self.x_coord, self.min_clients);
        let pool = self
            .sessions
            .entry(session_id.to_string())
            .or_insert_with(|| SessionPool {
                holder: ShareHolder::new(party_id, x_coord, min_clients),
                accepted: None,
                last_touched: Instant::now(),
            });
        pool.last_touched = Instant::now();
        pool
    }

    /// A pure lookup, creating nothing and not refreshing `last_touched`: the
    /// round gate runs after this returns, so a caller bumps it itself once it
    /// knows the call is genuine progress.
    fn pool_if_present(&mut self, session_id: &str) -> Option<&mut SessionPool> {
        self.sessions.get_mut(session_id)
    }

    /// The existing pool, or a `FailedPrecondition`: this daemon has none for
    /// that session, whether never opened, already ended, or swept. Like
    /// `pool_if_present` it leaves `last_touched` alone, since a lookup the
    /// round gate goes on to reject is not liveness.
    #[allow(clippy::result_large_err)]
    fn pool(&mut self, session_id: &str) -> Result<&mut SessionPool, Status> {
        self.pool_if_present(session_id).ok_or_else(|| {
            Status::failed_precondition(format!("no open round for session {session_id}"))
        })
    }

    /// Drop pools untouched since `now - SESSION_IDLE_TTL`.
    fn sweep_idle(&mut self, now: Instant) {
        self.sessions
            .retain(|_, p| now.duration_since(p.last_touched) < SESSION_IDLE_TTL);
    }
}

type Shared = Arc<Mutex<DaemonState>>;

/// Calls `sweep_idle` every `interval` until `shutdown` resolves, so reclaiming
/// an abandoned pool does not depend on an `OpenRound` that may never arrive to
/// trigger the inline sweep.
async fn run_idle_sweep(state: Shared, interval: Duration, mut shutdown: oneshot::Receiver<()>) {
    let mut ticker = tokio::time::interval(interval);
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                state.lock().unwrap().sweep_idle(Instant::now());
            }
            _ = &mut shutdown => return,
        }
    }
}

fn to_status(e: pbr_core::Error) -> Status {
    use pbr_core::Error::*;
    match e {
        InsufficientClients { .. } => Status::failed_precondition(e.to_string()),
        NoSharesForNode(_) => Status::not_found(e.to_string()),
        UnknownCommitment => Status::not_found(e.to_string()),
        other => Status::internal(other.to_string()),
    }
}

pub struct ClientFacing {
    state: Shared,
    x_coord: u64,
}

#[tonic::async_trait]
impl ShareholderService for ClientFacing {
    async fn submit_stats_shares(
        &self,
        req: Request<StatsShareSubmission>,
    ) -> Result<Response<Ack>, Status> {
        let msg = req.into_inner();
        let commitment = commitment_from_bytes(&msg.commitment)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let share = share_from_proto(
            msg.share
                .ok_or_else(|| Status::invalid_argument("missing share"))?,
        )
        .map_err(|e| Status::invalid_argument(e.to_string()))?;
        if share.x != self.x_coord {
            return Err(Status::invalid_argument(
                "share x does not match this shareholder",
            ));
        }
        let mut state = self.state.lock().unwrap();
        let pool = state.pool(&msg.session_id)?;
        pool.gate_stats(share.values.len())?;
        pool.holder
            .receive_stats(CommittedStatsShare { commitment, share });
        // Passed the gate and stored: this session just advanced.
        pool.last_touched = Instant::now();
        Ok(Response::new(Ack {}))
    }

    /// A client's whole per-round gradient contribution, stored atomically:
    /// every entry is validated (x coordinate, one vector length, the round
    /// gate) before anything is stored, and all stores share one lock
    /// acquisition. A commitment can therefore never be present with only some
    /// of its entries, which is what makes the aggregator's commitment
    /// intersection a node-level consistency guarantee.
    async fn submit_gradient_batch(
        &self,
        req: Request<GradientBatchSubmission>,
    ) -> Result<Response<Ack>, Status> {
        let msg = req.into_inner();
        if msg.round_id > i64::MAX as u64 {
            return Err(Status::invalid_argument("round_id out of range"));
        }
        let commitment = commitment_from_bytes(&msg.commitment)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        if msg.entries.is_empty() {
            return Err(Status::invalid_argument(
                "batch must contain at least one entry",
            ));
        }
        let mut entries = Vec::with_capacity(msg.entries.len());
        for entry in msg.entries {
            let share = share_from_proto(
                entry
                    .share
                    .ok_or_else(|| Status::invalid_argument("missing share"))?,
            )
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
            if share.x != self.x_coord {
                return Err(Status::invalid_argument(
                    "share x does not match this shareholder",
                ));
            }
            entries.push((entry.node_id as usize, share));
        }
        // All entries share one vector length, checked against the round's
        // expected_len gate once below.
        let len = entries[0].1.values.len();
        if entries.iter().any(|(_, s)| s.values.len() != len) {
            return Err(Status::invalid_argument(
                "batch entries have inconsistent share vector lengths",
            ));
        }
        let mut state = self.state.lock().unwrap();
        let pool = state.pool(&msg.session_id)?;
        pool.gate_gradient(msg.round_id, msg.depth, len)?;
        for (node_id, share) in entries {
            pool.holder.receive_gradients(CommittedGradientShare {
                round_id: msg.round_id,
                depth: msg.depth as usize,
                commitment: commitment.clone(),
                share,
                node_id,
            });
        }
        // Passed the gate and stored: this session just advanced.
        pool.last_touched = Instant::now();
        Ok(Response::new(Ack {}))
    }
}

pub struct Internal {
    state: Shared,
}

#[tonic::async_trait]
impl ShareholderInternal for Internal {
    async fn list_commitments(
        &self,
        req: Request<ListCommitmentsRequest>,
    ) -> Result<Response<CommitmentList>, Status> {
        let msg = req.into_inner();
        let mut state = self.state.lock().unwrap();
        // This and `get_sums` are the aggregator's gather traffic, and both
        // arrive on the loopback-only internal listener, which has no auth
        // interceptor: a device cannot forge them, so receiving one proves this
        // session's aggregator is alive. Refreshing here is what keeps the pool
        // alive through the gather, which runs between a CloseRound and the next
        // OpenRound, a window neither of those brackets.
        let pool = state.pool(&msg.session_id)?;
        pool.last_touched = Instant::now();
        let (commitments, node_ids) = match msg.phase() {
            SharePhase::Stats => (pool.holder.get_stats_commitments(), Default::default()),
            SharePhase::Gradient => (
                pool.holder.get_gradient_commitments(msg.depth as usize),
                pool.holder.get_gradient_node_ids(msg.depth as usize),
            ),
            SharePhase::Unspecified => return Err(Status::invalid_argument("phase required")),
        };
        Ok(Response::new(CommitmentList {
            commitments: commitments.iter().map(|c| c.0.to_vec()).collect(),
            node_ids: node_ids.iter().map(|&n| n as u32).collect(),
        }))
    }

    async fn get_sums(&self, req: Request<GetSumsRequest>) -> Result<Response<SumShare>, Status> {
        let msg = req.into_inner();
        let mut commitments = std::collections::BTreeSet::new();
        for c in &msg.commitments {
            commitments.insert(
                commitment_from_bytes(c).map_err(|e| Status::invalid_argument(e.to_string()))?,
            );
        }
        let commitments: Vec<_> = commitments.into_iter().collect();
        let mut state = self.state.lock().unwrap();
        // Internal-plane traffic; see `list_commitments` for why this counts
        // as liveness.
        let pool = state.pool(&msg.session_id)?;
        pool.last_touched = Instant::now();
        let share = match msg.phase() {
            SharePhase::Stats => pool.holder.get_stats_sum(&commitments),
            SharePhase::Gradient => pool.holder.get_gradients_sum(
                msg.depth as usize,
                &commitments,
                msg.node_id as usize,
            ),
            SharePhase::Unspecified => return Err(Status::invalid_argument("phase required")),
        }
        .map_err(to_status)?;
        Ok(Response::new(SumShare {
            share: Some(share_to_proto(&share)),
        }))
    }

    async fn open_round(&self, req: Request<OpenRoundRequest>) -> Result<Response<Ack>, Status> {
        let msg = req.into_inner();
        let phase = msg.phase();
        if phase == SharePhase::Unspecified {
            return Err(Status::invalid_argument("phase required"));
        }
        let mut state = self.state.lock().unwrap();
        state.sweep_idle(Instant::now());
        let pool = state.pool_for_open(&msg.session_id);
        let changed = !matches!(
            &pool.accepted,
            Some(r) if r.round_id == msg.round_id
                && r.depth == msg.depth
                && r.phase == phase
                && r.session_id == msg.session_id
        );
        if changed {
            // A new round drops stale contents from a prior round or a prior
            // aggregator session. A restart reuses round_id 1 for stats, so only
            // the session_id mismatch distinguishes it from a same-session
            // retry.
            pool.holder.reset();
        }
        // A duplicate OpenRound for the same round and session must not reopen
        // the length gate, so `expected_len` carries forward unless the round
        // actually changed.
        let expected_len = if changed {
            None
        } else {
            pool.accepted.as_ref().and_then(|r| r.expected_len)
        };
        pool.accepted = Some(AcceptedRound {
            round_id: msg.round_id,
            depth: msg.depth,
            phase,
            session_id: msg.session_id,
            expected_len,
        });
        Ok(Response::new(Ack {}))
    }

    async fn close_round(&self, req: Request<CloseRoundRequest>) -> Result<Response<Ack>, Status> {
        let msg = req.into_inner();
        let phase = msg.phase();
        if phase == SharePhase::Unspecified {
            return Err(Status::invalid_argument("phase required"));
        }
        let mut state = self.state.lock().unwrap();
        // Unlike `pool()`, a missing pool is not an error: it may never have
        // been opened here, may have been freed by `EndSession`, or may have
        // been swept, and all three Ack as idempotent no-ops. A pool merely
        // between rounds is handled below.
        let Some(pool) = state.pool_if_present(&msg.session_id) else {
            return Ok(Response::new(Ack {}));
        };
        match &pool.accepted {
            Some(r)
                if r.round_id == msg.round_id
                    && r.depth == msg.depth
                    && r.phase == phase
                    && r.session_id == msg.session_id =>
            {
                pool.accepted = None;
                // The round it was accepting just closed: real progress.
                pool.last_touched = Instant::now();
            }
            Some(r) => {
                // A stale or racing close for another round: keep the current
                // round accepted and reject, so the aggregator's confirmed mask
                // cannot treat this shareholder as frozen for a round it never
                // held.
                let err = Status::failed_precondition(format!(
                    "close_round: requested round (round_id={}, depth={}, phase={:?}) does not \
                     match currently open round (round_id={}, depth={}, phase={:?})",
                    msg.round_id, msg.depth, phase, r.round_id, r.depth, r.phase
                ));
                tracing::warn!(
                    requested_round_id = msg.round_id,
                    requested_depth = msg.depth,
                    requested_phase = ?phase,
                    current_round_id = r.round_id,
                    current_depth = r.depth,
                    current_phase = ?r.phase,
                    "close_round: requested round does not match currently open round; rejecting"
                );
                return Err(err);
            }
            None => {
                // Already closed by an earlier CloseRound. A repeat close Acks
                // idempotently, and nothing advanced, so `last_touched` stays.
            }
        }
        Ok(Response::new(Ack {}))
    }

    async fn end_session(
        &self,
        req: Request<EndSessionRequest>,
    ) -> Result<Response<Ack>, Status> {
        let session_id = req.into_inner().session_id;
        let mut state = self.state.lock().unwrap();
        state.sessions.remove(&session_id);
        Ok(Response::new(Ack {}))
    }
}

pub struct ShutdownHandle {
    tx: Vec<oneshot::Sender<()>>,
    /// Kept for its Drop, which aborts the Firebase key-refresh task so it
    /// cannot outlive the shareholder.
    #[allow(dead_code)]
    refresh: crate::config::RefreshHandle,
}

impl ShutdownHandle {
    pub fn shutdown(self) {
        for tx in self.tx {
            let _ = tx.send(());
        }
    }
}

pub struct RunningShareholder {
    /// Client plane: JWT-gated `SubmitStatsShares` and `SubmitGradientBatch`.
    pub client_addr: SocketAddr,
    /// Internal plane: loopback-only `OpenRound`, `CloseRound` and reads.
    pub internal_addr: SocketAddr,
    pub handle: ShutdownHandle,
}

/// Bind both listeners (port 0 supported) and serve.
pub async fn serve(cfg: ShareholderConfig) -> anyhow::Result<RunningShareholder> {
    if cfg.x_coord == 0 {
        anyhow::bail!("x_coord must be >= 1 (Shamir evaluation points start at 1)");
    }
    if !cfg.internal_listen.ip().is_loopback() {
        anyhow::bail!("internal_listen must be a loopback address");
    }
    let state: Shared = Arc::new(Mutex::new(DaemonState {
        sessions: HashMap::new(),
        party_id: cfg.x_coord as usize - 1,
        x_coord: cfg.x_coord,
        min_clients: cfg.min_clients,
    }));
    let crate::config::RefreshingVerifier { verifier, refresh } =
        cfg.auth.build_and_refresh_verifier().await?;

    let client_listener = tokio::net::TcpListener::bind(cfg.listen).await?;
    let internal_listener = tokio::net::TcpListener::bind(cfg.internal_listen).await?;
    let client_addr = client_listener.local_addr()?;
    let internal_addr = internal_listener.local_addr()?;

    let (tx1, rx1) = oneshot::channel();
    let (tx2, rx2) = oneshot::channel();
    let (tx3, rx3) = oneshot::channel();
    let x_coord = cfg.x_coord;
    let sweep_state = state.clone();

    let svc = ShareholderServiceServer::with_interceptor(
        ClientFacing {
            state: state.clone(),
            x_coord: cfg.x_coord,
        },
        auth::interceptor(verifier),
    );
    // TLS wraps the client-facing listener only; the internal loopback plane
    // below stays plaintext, as does this one without `[tls]`.
    let mut client_builder = tonic::transport::Server::builder();
    if let Some(tls) = &cfg.tls {
        client_builder = client_builder.tls_config(tls.server_tls_config()?)?;
    }
    let client_router = client_builder.add_service(svc);
    tokio::spawn(async move {
        let result = client_router
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(client_listener),
                async {
                    let _ = rx1.await;
                },
            )
            .await;
        if let Err(e) = result {
            tracing::error!(error = %e, x_coord, "shareholder client-facing gRPC server exited with error");
        }
    });
    tokio::spawn(async move {
        let result = tonic::transport::Server::builder()
            .add_service(ShareholderInternalServer::new(Internal { state }))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(internal_listener),
                async {
                    let _ = rx2.await;
                },
            )
            .await;
        if let Err(e) = result {
            tracing::error!(error = %e, x_coord, "shareholder internal-plane gRPC server exited with error");
        }
    });
    tokio::spawn(run_idle_sweep(sweep_state, SESSION_SWEEP_INTERVAL, rx3));

    Ok(RunningShareholder {
        client_addr,
        internal_addr,
        handle: ShutdownHandle {
            tx: vec![tx1, tx2, tx3],
            refresh,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_evicts_only_pools_idle_past_the_ttl() {
        let mut state = DaemonState {
            sessions: HashMap::new(),
            party_id: 0,
            x_coord: 1,
            min_clients: 1,
        };
        let now = Instant::now();
        state.pool_for_open("fresh").last_touched = now;
        state.pool_for_open("stale").last_touched = now - SESSION_IDLE_TTL - Duration::from_secs(1);

        state.sweep_idle(now);

        assert!(state.sessions.contains_key("fresh"));
        assert!(
            !state.sessions.contains_key("stale"),
            "a pool untouched for longer than the TTL is abandoned and must be freed"
        );
    }

    #[test]
    fn rejected_submissions_do_not_keep_an_orphaned_pool_alive() {
        let mut state = DaemonState {
            sessions: HashMap::new(),
            party_id: 0,
            x_coord: 1,
            min_clients: 1,
        };
        let start = Instant::now();
        state.pool_for_open("s").last_touched = start;

        // A lookup leading to a rejected submission must not count as
        // liveness, or a client could pin a dead session's pool.
        let _ = state.pool("s");
        let touched = state.sessions.get("s").unwrap().last_touched;
        assert_eq!(touched, start, "a bare lookup is not progress");

        state.sweep_idle(start + SESSION_IDLE_TTL + Duration::from_secs(1));
        assert!(
            !state.sessions.contains_key("s"),
            "an orphaned pool must age out despite lookup traffic"
        );
    }

    /// Gather traffic alone, with no OpenRound, CloseRound or submission in
    /// between, must keep a pool from aging out past the moment its last
    /// OpenRound touch would have exceeded the TTL. The gather runs between a
    /// CloseRound and the next OpenRound, a window neither brackets.
    #[tokio::test]
    async fn gather_traffic_keeps_a_live_pool_from_being_swept() {
        let state: Shared = Arc::new(Mutex::new(DaemonState {
            sessions: HashMap::new(),
            party_id: 0,
            x_coord: 1,
            min_clients: 1,
        }));
        let stale_touch = Instant::now() - SESSION_IDLE_TTL - Duration::from_secs(1);
        state.lock().unwrap().pool_for_open("s").last_touched = stale_touch;

        let internal = Internal {
            state: state.clone(),
        };
        internal
            .list_commitments(Request::new(ListCommitmentsRequest {
                phase: SharePhase::Stats as i32,
                depth: 0,
                session_id: "s".into(),
            }))
            .await
            .expect("list_commitments on an open pool must succeed");
        // get_sums on a pool with no shares fails with InsufficientClients,
        // but its arrival must refresh last_touched regardless.
        let _ = internal
            .get_sums(Request::new(GetSumsRequest {
                phase: SharePhase::Stats as i32,
                depth: 0,
                commitments: vec![],
                node_id: 0,
                session_id: "s".into(),
            }))
            .await;

        assert!(
            state.lock().unwrap().sessions.get("s").unwrap().last_touched > stale_touch,
            "list_commitments/get_sums must refresh last_touched"
        );

        // Sweeping exactly when the stale OpenRound touch would have aged out:
        // the pool survives because gather traffic kept it fresh.
        state
            .lock()
            .unwrap()
            .sweep_idle(stale_touch + SESSION_IDLE_TTL + Duration::from_secs(1));
        assert!(
            state.lock().unwrap().sessions.contains_key("s"),
            "a pool whose only recent activity is internal-plane gather traffic must not be swept"
        );
    }

    /// `run_idle_sweep` reclaims an idle pool on its timer, with no RPC ever
    /// touching it. The interval is shortened so this runs in milliseconds; it
    /// does not exercise `serve()`'s wiring beyond what is rebuilt here.
    #[tokio::test]
    async fn periodic_sweep_evicts_idle_pool_without_rpc_traffic() {
        let state: Shared = Arc::new(Mutex::new(DaemonState {
            sessions: HashMap::new(),
            party_id: 0,
            x_coord: 1,
            min_clients: 1,
        }));
        let now = Instant::now();
        state.lock().unwrap().pool_for_open("stale").last_touched =
            now - SESSION_IDLE_TTL - Duration::from_secs(1);

        let (_tx, rx) = oneshot::channel();
        let sweep_task = tokio::spawn(run_idle_sweep(state.clone(), Duration::from_millis(10), rx));

        // Polling rather than a fixed sleep: fast when healthy, bounded well
        // under a second.
        let deadline = Instant::now() + Duration::from_secs(2);
        while state.lock().unwrap().sessions.contains_key("stale") && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        assert!(
            !state.lock().unwrap().sessions.contains_key("stale"),
            "periodic sweep task did not evict an idle pool without any RPC traffic"
        );
        sweep_task.abort();
    }
}
