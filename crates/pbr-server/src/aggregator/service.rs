use super::manager::{SessionManager, Sole};
use super::notify::{self, device_uid};
use crate::auth;
use pbr_proto::v1::aggregator_service_server::AggregatorService;
use pbr_proto::v1::poll_session_response::Body;
use pbr_proto::v1::{
    Ack, ClientPlatform, EnrollRequest, ListSessionsRequest, NumericEdges, PollSessionRequest,
    PollSessionResponse, RegisterDeviceRequest, RoundContext, SessionConfig, SessionList,
    SessionPhase, SessionSummary,
};
use super::SessionSpec;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tonic::{Request, Response, Status};

/// Shared session state: the round loop writes, the gRPC service reads.
pub(super) struct SessionState {
    pub(super) phase: SessionPhase,
    /// The currently published round context, served by PollSession.
    pub(super) ctx: Option<RoundContext>,
    /// Filled after define_bins; served by EnrollSession.
    pub(super) bin_edges: Vec<NumericEdges>,
    pub(super) n_features: u32,
    /// Minted per session in `spawn_session`, not per process. `resolve` routes
    /// requests by it, and `SessionManager` keys its rows on it, so a session
    /// reloaded after a restart stays reachable under the same id.
    pub(super) session_id: String,
    /// Carried here rather than only as `run_session`'s parameter so
    /// `SessionManager::persist` reads it under the lock that guards `phase`.
    pub(super) spec: SessionSpec,
    pub(super) created_at: SystemTime,
}

pub(super) type SharedSession = Arc<Mutex<SessionState>>;

pub(super) fn publish(session: &SharedSession, phase: SessionPhase, ctx: RoundContext) {
    let mut s = session.lock().unwrap();
    s.phase = phase;
    s.ctx = Some(ctx);
}

pub(super) fn fail_session(session: &SharedSession) {
    session.lock().unwrap().phase = SessionPhase::Failed;
}

/// Every hosted session, sorted by id for a stable order. Shared by the
/// client-facing and admin-plane `ListSessions` handlers so the two cannot
/// drift in what they report.
pub(super) fn session_list(manager: &SessionManager) -> SessionList {
    let mut sessions = manager.summaries();
    sessions.sort_by(|a, b| a.session_id.cmp(&b.session_id));
    let sessions = sessions
        .into_iter()
        .map(|s| SessionSummary {
            session_id: s.session_id,
            phase: s.phase as i32,
            n_features: s.n_features,
            dataset_id: s.dataset_id,
            created_at: Some(prost_types::Timestamp::from(s.created_at)),
        })
        .collect();
    SessionList { sessions }
}

pub(super) struct AggregatorSvc {
    pub(super) manager: SessionManager,
    /// Static for the process lifetime, so it lives here, not `SessionState`.
    pub(super) threshold: u32,
    /// Number of shareholders (= `internal_shareholder_endpoints.len()`).
    pub(super) n_parties: u32,
    /// Handed to clients verbatim via `EnrollSession`, so a client needs no
    /// out-of-band shareholder config. Not the internal endpoints.
    pub(super) client_shareholder_endpoints: Vec<String>,
}

impl AggregatorSvc {
    /// An empty selector means "the only session", which keeps a client that
    /// never listed sessions working. It is an error when this process hosts
    /// several: guessing would misattribute the caller's contributions.
    #[allow(clippy::result_large_err)] // tonic::Status is large; callers map it immediately.
    fn resolve(&self, session_id: &str) -> Result<SharedSession, Status> {
        if !session_id.is_empty() {
            return self
                .manager
                .get(session_id)
                .ok_or_else(|| Status::not_found(format!("no session {session_id}")));
        }
        match self.manager.sole() {
            Sole::One(state) => Ok(state),
            Sole::Zero => Err(Status::failed_precondition(
                "session_id required: no live session is hosted; create one on the admin plane first",
            )),
            Sole::Many(n) => Err(Status::failed_precondition(format!(
                "session_id required: this process hosts {n} live sessions"
            ))),
        }
    }
}

fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[tonic::async_trait]
impl AggregatorService for AggregatorSvc {
    async fn enroll_session(
        &self,
        req: Request<EnrollRequest>,
    ) -> Result<Response<SessionConfig>, Status> {
        let identity = req
            .extensions()
            .get::<auth::Identity>()
            .cloned()
            .ok_or_else(|| Status::unauthenticated("missing identity"))?;
        let session = self.resolve(&req.into_inner().session_id)?;
        let config = {
            let s = session.lock().unwrap();
            SessionConfig {
                bin_edges: s.bin_edges.clone(),
                phase: s.phase as i32,
                n_features: s.n_features,
                session_id: s.session_id.clone(),
                threshold: self.threshold,
                n_parties: self.n_parties,
                shareholder_endpoints: self.client_shareholder_endpoints.clone(),
            }
        };
        // Recorded under the resolved id, which is what the notify tick looks
        // up, so an empty selector enrolls the caller in the session it landed
        // in. Best-effort: the wake loop re-enrolls on every resume, so a lost
        // row self-repairs, and failing the RPC over push bookkeeping would
        // block contribution on a nicety. `register_device` stays fatal, since
        // the store write is that RPC's entire purpose.
        if let Err(e) = notify::record_enrollment(
            self.manager.store(),
            &config.session_id,
            &device_uid(&identity),
            now_epoch_secs(),
        )
        .await
        {
            tracing::warn!(
                error = %e,
                "enrollment write failed; device will re-enroll on its next wake"
            );
        }
        Ok(Response::new(config))
    }

    async fn poll_session(
        &self,
        req: Request<PollSessionRequest>,
    ) -> Result<Response<PollSessionResponse>, Status> {
        let msg = req.into_inner();
        let last_seen = msg.last_seen_round_id;
        let session = self.resolve(&msg.session_id)?;
        let s = session.lock().unwrap();
        let body = match &s.ctx {
            Some(ctx) if ctx.round_id != last_seen => Body::Ctx(ctx.clone()),
            _ => Body::NothingNew(()),
        };
        // The cadence comes from this session's own submission window: sessions
        // are created independently, so there is no process-wide window.
        Ok(Response::new(PollSessionResponse {
            body: Some(body),
            next_poll_after: Some(poll_hint(s.spec.submission_window())),
            phase: s.phase as i32,
            session_id: s.session_id.clone(),
        }))
    }

    /// Binds the caller's FCM token to the verified (iss, sub) from the JWT
    /// interceptor's `Identity`, so a client cannot register a token under
    /// another uid. A re-register updates the existing entry.
    async fn register_device(
        &self,
        req: Request<RegisterDeviceRequest>,
    ) -> Result<Response<Ack>, Status> {
        let identity = req
            .extensions()
            .get::<auth::Identity>()
            .cloned()
            .ok_or_else(|| Status::unauthenticated("missing identity"))?;
        let msg = req.into_inner();
        let platform = ClientPlatform::try_from(msg.platform)
            .ok()
            .filter(|p| *p != ClientPlatform::Unspecified)
            .ok_or_else(|| Status::invalid_argument("platform must be a known ClientPlatform"))?;
        notify::upsert_device(
            self.manager.store(),
            &device_uid(&identity),
            &msg.fcm_token,
            platform as i32,
            now_epoch_secs(),
        )
        .await
        .map_err(|e| Status::internal(format!("store device: {e}")))?;
        Ok(Response::new(Ack {}))
    }

    async fn list_sessions(
        &self,
        _req: Request<ListSessionsRequest>,
    ) -> Result<Response<SessionList>, Status> {
        Ok(Response::new(session_list(&self.manager)))
    }
}

/// Server-suggested poll cadence: window/20, clamped to something sane.
/// Purely advisory; clients may clamp further.
pub(super) fn poll_hint(window: Duration) -> prost_types::Duration {
    let d = (window / 20).clamp(Duration::from_millis(50), Duration::from_secs(5));
    prost_types::Duration {
        seconds: d.as_secs() as i64,
        nanos: d.subsec_nanos() as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_shared_session(id: &str) -> SharedSession {
        Arc::new(Mutex::new(SessionState {
            phase: SessionPhase::StatsPending,
            ctx: None,
            bin_edges: Vec::new(),
            n_features: 0,
            session_id: id.into(),
            spec: SessionSpec {
                dataset_id: "d".into(),
                title: "t".into(),
                n_trees: 1,
                max_depth: 1,
                n_bins: 4,
                learning_rate: 0.1,
                lambda: 1.0,
                min_clients: 1,
                target_clients: 1,
                submission_window_ms: 1000,
            },
            created_at: SystemTime::now(),
        }))
    }

    async fn svc_hosting(ids: &[&str]) -> AggregatorSvc {
        let manager = SessionManager::load(std::path::Path::new(":memory:")).await.unwrap();
        for id in ids {
            manager.insert((*id).into(), new_shared_session(id), None);
        }
        AggregatorSvc {
            manager,
            threshold: 2,
            n_parties: 3,
            client_shareholder_endpoints: Vec::new(),
        }
    }

    /// `resolve`'s refusal branch (empty selector, not exactly one session
    /// hosted) has no coverage elsewhere: nothing before `CreateSession`
    /// exists can put a second session into a `SessionManager`. This test
    /// builds one directly so a regression here is caught instead of only
    /// ever being verified by reading the code.
    #[tokio::test]
    async fn resolve_disambiguates_by_session_id() {
        let svc = svc_hosting(&["sess-a", "sess-b"]).await;

        match svc.resolve("") {
            Ok(_) => panic!("empty selector over two sessions must not guess"),
            Err(e) => assert_eq!(e.code(), tonic::Code::FailedPrecondition),
        }

        let a = svc.resolve("sess-a").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.lock().unwrap().session_id, "sess-a");
        let b = svc.resolve("sess-b").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(b.lock().unwrap().session_id, "sess-b");

        match svc.resolve("sess-missing") {
            Ok(_) => panic!("unknown session id must not resolve"),
            Err(e) => assert_eq!(e.code(), tonic::Code::NotFound),
        }
    }
}
