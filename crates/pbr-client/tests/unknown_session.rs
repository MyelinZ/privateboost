//! An aggregator's session store is always-on SQLite: a session_id it has
//! never hosted is genuinely unknown, not evidence of a restart (a restarted
//! aggregator re-serves a previously-live session under its original id,
//! demoted to `Failed`, which the driver's ordinary `Failed`-phase handling
//! already covers). `PollSession` against such an id fails the RPC with
//! `NOT_FOUND`, and retrying cannot help: the id will never start existing.
//! This must still not hang the caller, it has to fall through the
//! driver's generic transient-retry path and abort once that budget is
//! exhausted, the same way any other persistent poll fault does.

use pbr_client::driver::{SessionParams, run_to_completion};
use pbr_client::jwt::mint;
use pbr_proto::v1::aggregator_service_server::{AggregatorService, AggregatorServiceServer};
use pbr_proto::v1::{
    Ack, EnrollRequest, ListSessionsRequest, PollSessionRequest, PollSessionResponse,
    RegisterDeviceRequest, SessionConfig, SessionList, SessionPhase,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tonic::{Request, Response, Status};

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("../../pbr-server/tests/fixtures/test_key.pem");
const ENROLLED_SESSION: &str = "session-unknown-to-server";

/// Enrolls the client under `ENROLLED_SESSION` but answers every poll with
/// `NOT_FOUND`, a real `resolve()`-backed server's response for a
/// session_id its store has never heard of. `polls` counts every
/// `poll_session` call reaching the server, so a test can assert exactly how
/// many the driver spent before giving up.
struct UnknownSessionAggregator {
    polls: Arc<AtomicU32>,
}

#[tonic::async_trait]
impl AggregatorService for UnknownSessionAggregator {
    async fn enroll_session(
        &self,
        _req: Request<EnrollRequest>,
    ) -> Result<Response<SessionConfig>, Status> {
        Ok(Response::new(SessionConfig {
            bin_edges: Vec::new(),
            phase: SessionPhase::StatsPending as i32,
            n_features: 0,
            session_id: ENROLLED_SESSION.to_string(),
            threshold: 2,
            n_parties: 1,
            // A lazily-connected fan-out target; never actually contacted
            // because every poll fails before a round context ever arrives.
            shareholder_endpoints: vec!["http://127.0.0.1:1".to_string()],
        }))
    }

    async fn poll_session(
        &self,
        _req: Request<PollSessionRequest>,
    ) -> Result<Response<PollSessionResponse>, Status> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Err(Status::not_found(format!("no session {ENROLLED_SESSION}")))
    }

    async fn register_device(
        &self,
        _req: Request<RegisterDeviceRequest>,
    ) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {}))
    }

    async fn list_sessions(
        &self,
        _req: Request<ListSessionsRequest>,
    ) -> Result<Response<SessionList>, Status> {
        Ok(Response::new(SessionList { sessions: Vec::new() }))
    }
}

#[tokio::test]
async fn unknown_session_poll_aborts_within_the_retry_bound() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let polls = Arc::new(AtomicU32::new(0));
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(AggregatorServiceServer::new(UnknownSessionAggregator {
                polls: polls.clone(),
            }))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = rx.await;
                },
            ),
    );

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();

    // Drive the whole session through `run_to_completion`, the same entry
    // point the CLI and app bridge use, so this exercises the retry-budget
    // decision itself (not just one isolated `step` call).
    let err = run_to_completion(
        SessionParams {
            agg_endpoint: format!("http://{addr}"),
            shareholder_endpoints: None,
            token,
            records: vec![(vec![1.0, 2.0], 0.0)],
            threshold: Some(2),
            hide_path: true,
            ca_pem: None,
            session_id: None,
        },
        |_| {},
    )
    .await
    .expect_err("a NOT_FOUND poll against an unknown session must not hang the client");

    assert!(
        err.to_string().contains("consecutive step failures"),
        "abort must name the exhausted retry budget, got: {err}"
    );

    // The bound is three consecutive failures, so the server sees exactly
    // three polls before the driver gives up: an unbounded retry loop would
    // instead poll forever and this test would never reach this assertion.
    assert_eq!(
        polls.load(Ordering::SeqCst),
        3,
        "an unknown session_id must be retried like any other transient poll fault, bounded the same way"
    );

    let _ = tx.send(());
    let _ = server.await;
}
