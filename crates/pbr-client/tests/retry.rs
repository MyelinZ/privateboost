//! A transient poll failure must not abandon a whole participation: the
//! driver retries a bounded number of consecutive `step` failures before
//! giving up, so one network blip on a flaky radio is consumed rather than
//! ending the session (and, conversely, a persistent fault still aborts
//! within that bound instead of polling forever).

use pbr_client::driver::{SessionParams, run_to_completion};
use pbr_client::jwt::mint;
use pbr_proto::v1::aggregator_service_server::{AggregatorService, AggregatorServiceServer};
use pbr_proto::v1::poll_session_response::Body;
use pbr_proto::v1::{
    Ack, EnrollRequest, ListSessionsRequest, PollSessionRequest, PollSessionResponse,
    RegisterDeviceRequest, RoundContext, SessionConfig, SessionList, SessionPhase,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tonic::{Request, Response, Status};

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("../../pbr-server/tests/fixtures/test_key.pem");
const SESSION: &str = "session-flaky";

/// Enrolls under `SESSION`, then fails the first `fail_polls` polls with a
/// transient `UNAVAILABLE`; the next poll reports the session `COMPLETED`.
/// Every poll (failing or not) bumps `polls` so a test can assert exactly how
/// many the driver issued.
struct FlakyAggregator {
    polls: Arc<AtomicU32>,
    fail_polls: u32,
}

#[tonic::async_trait]
impl AggregatorService for FlakyAggregator {
    async fn enroll_session(
        &self,
        _req: Request<EnrollRequest>,
    ) -> Result<Response<SessionConfig>, Status> {
        Ok(Response::new(SessionConfig {
            bin_edges: Vec::new(),
            phase: SessionPhase::StatsPending as i32,
            n_features: 0,
            session_id: SESSION.to_string(),
            threshold: 2,
            n_parties: 1,
            // Lazily-connected fan-out target; never contacted because the
            // session completes before any submit.
            shareholder_endpoints: vec!["http://127.0.0.1:1".to_string()],
        }))
    }

    async fn poll_session(
        &self,
        _req: Request<PollSessionRequest>,
    ) -> Result<Response<PollSessionResponse>, Status> {
        let n = self.polls.fetch_add(1, Ordering::SeqCst);
        if n < self.fail_polls {
            return Err(Status::unavailable("transient network blip"));
        }
        Ok(Response::new(PollSessionResponse {
            body: Some(Body::Ctx(RoundContext {
                tree_idx: 0,
                depth: u32::MAX,
                round_id: 999,
                active_node_ids: Vec::new(),
                splits_so_far: Default::default(),
                bin_edges: Vec::new(),
                model: None,
                submission_deadline: None,
                session_id: SESSION.to_string(),
            })),
            next_poll_after: None,
            phase: SessionPhase::Completed as i32,
            session_id: SESSION.to_string(),
        }))
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

/// A running `FlakyAggregator`: its endpoint URL, the shared poll counter, the
/// shutdown sender, and the serving task's join handle.
struct FlakyServer {
    endpoint: String,
    polls: Arc<AtomicU32>,
    shutdown: tokio::sync::oneshot::Sender<()>,
    server: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

async fn spawn_flaky(fail_polls: u32) -> FlakyServer {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let polls = Arc::new(AtomicU32::new(0));
    let svc = FlakyAggregator {
        polls: polls.clone(),
        fail_polls,
    };
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(AggregatorServiceServer::new(svc))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = rx.await;
                },
            ),
    );
    FlakyServer {
        endpoint: format!("http://{addr}"),
        polls,
        shutdown: tx,
        server,
    }
}

#[tokio::test]
async fn transient_poll_failures_are_retried_then_session_completes() {
    // Two transient poll failures, then a completing poll.
    let FlakyServer {
        endpoint,
        polls,
        shutdown,
        server,
    } = spawn_flaky(2).await;
    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();

    run_to_completion(
        SessionParams {
            agg_endpoint: endpoint,
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
    .expect("two transient poll failures must be retried, not abort the session");

    // Exactly three polls reached the server: two consumed retries, then the
    // completing poll.
    assert_eq!(polls.load(Ordering::SeqCst), 3);

    let _ = shutdown.send(());
    let _ = server.await;
}

#[tokio::test]
async fn persistent_poll_failures_abort_within_the_retry_bound() {
    // Every poll fails: the driver must give up rather than poll forever.
    let FlakyServer {
        endpoint,
        polls,
        shutdown,
        server,
    } = spawn_flaky(u32::MAX).await;
    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();

    let err = run_to_completion(
        SessionParams {
            agg_endpoint: endpoint,
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
    .expect_err("a shareholder-independent poll fault that never clears must abort the session");
    assert!(
        err.to_string().contains("consecutive step failures"),
        "abort must name the exhausted retry budget, got: {err}"
    );

    // The bound is three consecutive failures, so the server sees exactly
    // three polls before the driver gives up.
    assert_eq!(polls.load(Ordering::SeqCst), 3);

    let _ = shutdown.send(());
    let _ = server.await;
}
