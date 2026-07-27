//! `AdminService` (`CreateSession`, `ListSessions`, `DeleteSession`):
//! admin-token auth (and its deliberate rejection of device identities),
//! dataset validation, session enumeration, and deletion. Bring-up follows
//! the same in-process 3-shareholder + aggregator cluster pattern as
//! `aggregator_loop.rs`.

use pbr_client::jwt::mint;
use pbr_proto::v1::admin_service_client::AdminServiceClient;
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_proto::v1::{
    CreateSessionRequest, DeleteSessionRequest, EnrollRequest, ListSessionsRequest,
    PollSessionRequest,
};
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{
    AggregatorHandle, DatasetTable, MAX_LIVE_SESSIONS, RunningAggregator, SessionSpec,
    serve as serve_aggregator,
};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
use pbr_server::shareholder::{RunningShareholder, ShutdownHandle, serve as serve_shareholder};
use tonic::Request;

mod common;
use common::bearer;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("fixtures/test_key.pem");
const ADMIN_TOKEN: &str = "test-admin-token";

fn auth_cfg() -> AuthConfig {
    AuthConfig {
        issuer: ISS.into(),
        audience: AUD.into(),
        static_keys: vec![StaticKey {
            kid: KID.into(),
            public_key_pem_path: concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/test_key.pub.pem"
            )
            .into(),
        }],
        google_jwks_url: None,
    }
}

fn shareholder_cfg(x: u64) -> ShareholderConfig {
    ShareholderConfig {
        x_coord: x,
        min_clients: 5,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: None,
    }
}

fn mint_device_token() -> String {
    mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap()
}

/// A valid `CreateSessionRequest` against the cluster's one accepted
/// dataset. Individual tests override whichever field they're exercising.
fn create_session_request() -> CreateSessionRequest {
    CreateSessionRequest {
        dataset_id: "heart_disease".into(),
        title: "admin test session".into(),
        n_trees: 1,
        max_depth: 1,
        n_bins: 4,
        learning_rate: 0.1,
        lambda: 1.0,
        min_clients: 1,
        target_clients: 1,
        submission_window_ms: 20_000,
    }
}

struct Cluster {
    agg_url: String,
    agg_handle: AggregatorHandle,
    sh_handles: Vec<ShutdownHandle>,
}

impl Cluster {
    fn shutdown(self) {
        self.agg_handle.shutdown();
        for h in self.sh_handles {
            h.shutdown();
        }
    }
}

/// A running 3-shareholder + aggregator cluster with the admin plane
/// enabled and `heart_disease` the only accepted dataset.
async fn start_cluster() -> Cluster {
    start_cluster_with_admin_token(Some(ADMIN_TOKEN.into())).await
}

async fn start_cluster_with_admin_token(admin_token: Option<String>) -> Cluster {
    start_cluster_with(admin_token, ":memory:".into()).await
}

/// Same cluster bring-up as `start_cluster`, with the admin token and the
/// aggregator's `state_path` under test control: `admin_token: None` stands
/// up a cluster with the admin plane disabled; `state_path` picks the store
/// (`":memory:"` for an ephemeral one, a file path to persist across a
/// restart).
async fn start_cluster_with(
    admin_token: Option<String>,
    state_path: std::path::PathBuf,
) -> Cluster {
    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: addr,
            internal_addr: internal,
            handle: h,
        } = serve_shareholder(shareholder_cfg(x)).await.unwrap();
        client_eps.push(format!("http://{addr}"));
        internal_eps.push(format!("http://{internal}"));
        sh_handles.push(h);
    }

    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: internal_eps,
        client_shareholder_endpoints: client_eps,
        threshold: 2,
        auth: auth_cfg(),
        fcm: None,
        tls: None,
        datasets: DatasetTable::from_pairs(vec![("heart_disease".into(), 13)]),
        admin_token,
        state_path,
        eval: None,
    })
    .await
    .unwrap();

    Cluster {
        agg_url: format!("http://{agg_addr}"),
        agg_handle,
        sh_handles,
    }
}

/// Creating a session with the admin token spawns it: it becomes listable and
/// a device can enroll in it by id.
#[tokio::test(flavor = "multi_thread")]
async fn create_session_spawns_a_listable_session() {
    let cluster = start_cluster().await;

    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();
    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let summary = admin
        .create_session(req)
        .await
        .expect("a well-formed request with the admin token must succeed")
        .into_inner();
    assert!(!summary.session_id.is_empty());
    assert_eq!(
        summary.dataset_id, "heart_disease",
        "the summary must echo the dataset the session was created for"
    );
    assert_eq!(
        summary.n_features, 13,
        "a StatsPending session must advertise the dataset table's declared \
         width, not 0 — devices width-check before joining"
    );

    let device_token = mint_device_token();
    let mut agg = AggregatorServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let mut list_req = Request::new(ListSessionsRequest {});
    list_req
        .metadata_mut()
        .insert("authorization", bearer(&device_token));
    let list = agg.list_sessions(list_req).await.unwrap().into_inner();
    assert_eq!(
        list.sessions.len(),
        1,
        "the admin-created session must be the one hosted session on a fresh cluster"
    );
    let created = &list.sessions[0];
    assert_eq!(created.session_id, summary.session_id);
    assert_eq!(created.dataset_id, "heart_disease");

    let mut enroll_req = Request::new(EnrollRequest {
        session_id: summary.session_id.clone(),
    });
    enroll_req
        .metadata_mut()
        .insert("authorization", bearer(&device_token));
    let cfg = agg
        .enroll_session(enroll_req)
        .await
        .expect("a device must be able to enroll in the admin-created session by id")
        .into_inner();
    assert_eq!(cfg.session_id, summary.session_id);

    cluster.shutdown();
}

/// `created_at` must be each session's own stored creation time, not
/// `SystemTime::now()` sampled while `ListSessions` builds its response. It is
/// the wire's only recency signal, since session ids are random UUIDs, so a
/// list-time `now()` bug would report every session as equally new. Two
/// sessions are created with a real gap between them: a
/// server that stamps `created_at` once at construction reports that same
/// gap back; a server that stamps it at list time reports both sessions from
/// the single instant `ListSessions` ran, so their timestamps would come back
/// equal (or differ only by the sub-millisecond cost of building the second
/// summary) regardless of the gap below.
#[tokio::test(flavor = "multi_thread")]
async fn created_at_reflects_stored_creation_time_not_list_time() {
    let cluster = start_cluster().await;

    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let before_ms = now_ms();

    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let first = admin
        .create_session(req)
        .await
        .expect("the first session must be created")
        .into_inner();

    // A generous, deterministic gap: the work between two creations (a
    // mutex-guarded insert, a UUID, spawning the round loop) is
    // sub-millisecond, so a real construction-time stamp reports a gap close
    // to this sleep; a list-time stamp cannot, no matter how long this is.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let second = admin
        .create_session(req)
        .await
        .expect("the second session must be created")
        .into_inner();

    let device_token = mint_device_token();
    let mut agg = AggregatorServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();
    let mut list_req = Request::new(ListSessionsRequest {});
    list_req
        .metadata_mut()
        .insert("authorization", bearer(&device_token));
    let list = agg.list_sessions(list_req).await.unwrap().into_inner();

    let after_ms = now_ms();

    let first_summary = list
        .sessions
        .iter()
        .find(|s| s.session_id == first.session_id)
        .expect("the first created session must be listed");
    let second_summary = list
        .sessions
        .iter()
        .find(|s| s.session_id == second.session_id)
        .expect("the second created session must be listed");

    let first_at = first_summary
        .created_at
        .expect("created_at must be populated, not left at the proto default");
    let second_at = second_summary
        .created_at
        .expect("created_at must be populated, not left at the proto default");
    // Ordering compares the timestamps themselves, (seconds, nanos); the
    // window and gap checks convert to millis to compare with now_ms().
    let ts_ms = |t: prost_types::Timestamp| t.seconds as u64 * 1000 + t.nanos as u64 / 1_000_000;
    let (first_ms, second_ms) = (ts_ms(first_at), ts_ms(second_at));

    assert!(
        before_ms <= first_ms && first_ms <= after_ms,
        "the first session's created_at ({first_ms}ms) must fall within [{before_ms}, {after_ms}]",
    );
    assert!(
        before_ms <= second_ms && second_ms <= after_ms,
        "the second session's created_at ({second_ms}ms) must fall within [{before_ms}, {after_ms}]",
    );

    assert!(
        (first_at.seconds, first_at.nanos) < (second_at.seconds, second_at.nanos),
        "the session created first ({first_at:?}) must report an earlier created_at than the \
         one created 50ms later ({second_at:?}); equal timestamps would mean the server stamped \
         both at list time instead of at each session's own construction",
    );
    let gap_ms = second_ms - first_ms;
    assert!(
        gap_ms >= 20,
        "the two sessions were created ~50ms apart; a gap this small ({gap_ms}ms) means \
         created_at is not tracking each session's own creation time"
    );

    cluster.shutdown();
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// The admin plane must not be reachable with a device's Firebase token, and
/// must not be reachable with no token or a wrong one. An unauthenticated
/// CreateSession would let anyone run training rounds against the fleet.
#[tokio::test(flavor = "multi_thread")]
async fn create_session_rejects_missing_wrong_and_device_tokens() {
    let cluster = start_cluster().await;
    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let req = Request::new(create_session_request());
    let err = admin
        .create_session(req)
        .await
        .expect_err("a request with no authorization header must be rejected");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);

    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer("wrong-token"));
    let err = admin
        .create_session(req)
        .await
        .expect_err("a wrong admin token must be rejected");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);

    // The critical case: a perfectly valid DEVICE identity must not reach
    // the admin plane at all, it is authenticated by a wholly separate,
    // static token, never by the device identity provider.
    let device_token = mint_device_token();
    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(&device_token));
    let err = admin
        .create_session(req)
        .await
        .expect_err("a signed-in device's Firebase JWT must never authorize CreateSession");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);

    cluster.shutdown();
}

/// With no admin token configured, `AdminService` must never be registered on
/// the router at all: `CreateSession` must fail as `Unimplemented` (tonic's
/// response for a method with no registered service), never as
/// `Unauthenticated`. `Unauthenticated` here would mean the service got
/// registered and is merely refusing the request, that would be a silent
/// regression of "no admin token means the admin plane is unreachable", not
/// just unauthorized.
#[tokio::test(flavor = "multi_thread")]
async fn create_session_is_unreachable_with_no_admin_token_configured() {
    let cluster = start_cluster_with_admin_token(None).await;
    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let err = admin
        .create_session(req)
        .await
        .expect_err("with no admin_token configured, AdminService must not be on the router");
    assert_eq!(
        err.code(),
        tonic::Code::Unimplemented,
        "Unimplemented means the service was never registered; Unauthenticated would mean it \
         was registered and merely refused the request, which is the regression this test \
         exists to catch"
    );

    cluster.shutdown();
}

/// Once the cluster already hosts `MAX_LIVE_SESSIONS` live sessions,
/// `CreateSession` must refuse to spawn another rather than letting an
/// unbounded run of admin requests hold arbitrarily many share pools and
/// internal-plane connections open at once.
#[tokio::test(flavor = "multi_thread")]
async fn create_session_is_rejected_once_the_live_session_cap_is_reached() {
    let cluster = start_cluster().await;
    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    // The cluster boots hosting no session, so filling the cap takes
    // MAX_LIVE_SESSIONS admin-created ones.
    for i in 0..MAX_LIVE_SESSIONS {
        let mut req = Request::new(create_session_request());
        req.metadata_mut()
            .insert("authorization", bearer(ADMIN_TOKEN));
        admin
            .create_session(req)
            .await
            .unwrap_or_else(|e| panic!("session {i} must succeed while under the live-session cap: {e}"));
    }

    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let err = admin.create_session(req).await.expect_err(
        "the live-session cap must reject a request once MAX_LIVE_SESSIONS live sessions are \
         already hosted",
    );
    assert_eq!(err.code(), tonic::Code::ResourceExhausted);

    cluster.shutdown();
}

/// A spec naming a dataset the cluster does not accept is refused before any
/// round loop is spawned.
#[tokio::test(flavor = "multi_thread")]
async fn create_session_rejects_an_unknown_dataset() {
    let cluster = start_cluster().await;
    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let mut req = Request::new(CreateSessionRequest {
        dataset_id: "no_such_dataset".into(),
        ..create_session_request()
    });
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let err = admin
        .create_session(req)
        .await
        .expect_err("an unknown dataset must be refused");
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    cluster.shutdown();
}

/// The admin plane must be able to enumerate sessions itself: the
/// client-facing ListSessions requires a device JWT the operator does not
/// hold.
#[tokio::test(flavor = "multi_thread")]
async fn admin_list_sessions_enumerates_hosted_sessions() {
    let cluster = start_cluster().await;
    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let created = admin.create_session(req).await.unwrap().into_inner();

    let mut list_req = Request::new(ListSessionsRequest {});
    list_req
        .metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let list = admin.list_sessions(list_req).await.unwrap().into_inner();
    assert_eq!(
        list.sessions.len(),
        1,
        "the admin-created session is the only one hosted (the process boots hosting nothing)"
    );
    assert!(
        list.sessions.iter().any(|s| s.session_id == created.session_id),
        "the created session must be discoverable by id on the admin plane"
    );

    // Same interceptor as CreateSession: a valid DEVICE identity must not
    // reach the admin plane.
    let mut list_req = Request::new(ListSessionsRequest {});
    list_req
        .metadata_mut()
        .insert("authorization", bearer(&mint_device_token()));
    let err = admin
        .list_sessions(list_req)
        .await
        .expect_err("a device JWT must never authorize admin ListSessions");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);

    cluster.shutdown();
}

/// Deleting an id the aggregator does not host must be NOT_FOUND, never a
/// silent success the operator reads as "the dead card is gone". And the
/// same interceptor as CreateSession guards it: a valid DEVICE identity
/// must not reach it.
#[tokio::test(flavor = "multi_thread")]
async fn delete_session_of_an_unknown_id_is_not_found() {
    let cluster = start_cluster().await;
    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let mut req = Request::new(DeleteSessionRequest {
        session_id: "no-such-session".into(),
    });
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let err = admin
        .delete_session(req)
        .await
        .expect_err("an unknown id must not report success");
    assert_eq!(err.code(), tonic::Code::NotFound);

    let mut req = Request::new(DeleteSessionRequest {
        session_id: "x".into(),
    });
    req.metadata_mut()
        .insert("authorization", bearer(&mint_device_token()));
    let err = admin
        .delete_session(req)
        .await
        .expect_err("a device JWT must never authorize DeleteSession");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);

    cluster.shutdown();
}

/// Deleting a LIVE session is the kill switch: it must vanish from the
/// admin list and answer NOT_FOUND on the client plane. pbr-client's
/// restart tests (`not_found_poll_against_restarted_aggregator_is_terminal_
/// not_retried`) pin that clients treat that NOT_FOUND as terminal.
#[tokio::test(flavor = "multi_thread")]
async fn deleting_a_live_session_removes_it_from_both_planes() {
    let cluster = start_cluster().await;
    let mut admin = AdminServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let mut req = Request::new(create_session_request());
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let created = admin.create_session(req).await.unwrap().into_inner();

    let mut req = Request::new(DeleteSessionRequest {
        session_id: created.session_id.clone(),
    });
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    admin
        .delete_session(req)
        .await
        .expect("deleting a live session must succeed, not FAILED_PRECONDITION");

    let mut list_req = Request::new(ListSessionsRequest {});
    list_req
        .metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let list = admin.list_sessions(list_req).await.unwrap().into_inner();
    assert!(
        !list.sessions.iter().any(|s| s.session_id == created.session_id),
        "a deleted session must not be listed"
    );

    let device_token = mint_device_token();
    let mut agg = AggregatorServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();
    let mut poll = Request::new(PollSessionRequest {
        last_seen_round_id: 0,
        session_id: created.session_id.clone(),
    });
    poll.metadata_mut()
        .insert("authorization", bearer(&device_token));
    let err = agg
        .poll_session(poll)
        .await
        .expect_err("polling a deleted session must fail");
    assert_eq!(err.code(), tonic::Code::NotFound);

    let mut enroll = Request::new(EnrollRequest {
        session_id: created.session_id.clone(),
    });
    enroll
        .metadata_mut()
        .insert("authorization", bearer(&device_token));
    let err = agg
        .enroll_session(enroll)
        .await
        .expect_err("enrolling in a deleted session must fail");
    assert_eq!(err.code(), tonic::Code::NotFound);

    cluster.shutdown();
}

/// Deletion is checkpointed to the store before the RPC returns, so a restart
/// right after a delete must not resurrect the session by reloading its row.
/// The store is binary SQLite, so this is a black-box restart rather than a
/// state-file text scan: a second aggregator on the same `state_path` must not
/// list the deleted session. A second, undeleted session proves the reload
/// actually restores state, the deleted one is absent because it was deleted,
/// not because nothing reloaded.
#[tokio::test(flavor = "multi_thread")]
async fn a_deleted_session_stays_gone_across_an_aggregator_restart() {
    let dir = tempfile::tempdir().unwrap();
    let state_path = dir.path().join("sessions.sqlite");

    // The shareholders outlive the aggregator restart; the second aggregator
    // never contacts them (admin ListSessions reads the in-memory map).
    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr,
            internal_addr,
            handle,
        } = serve_shareholder(shareholder_cfg(x)).await.unwrap();
        client_eps.push(format!("http://{client_addr}"));
        internal_eps.push(format!("http://{internal_addr}"));
        sh_handles.push(handle);
    }

    let agg_config = || AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: internal_eps.clone(),
        client_shareholder_endpoints: client_eps.clone(),
        threshold: 2,
        auth: auth_cfg(),
        fcm: None,
        tls: None,
        datasets: DatasetTable::from_pairs(vec![("heart_disease".into(), 13)]),
        admin_token: Some(ADMIN_TOKEN.into()),
        state_path: state_path.clone(),
        eval: None,
    };

    // First aggregator: create two sessions, delete one.
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(agg_config()).await.unwrap();
    let mut admin = AdminServiceClient::connect(format!("http://{agg_addr}"))
        .await
        .unwrap();
    let mut keep_req = Request::new(create_session_request());
    keep_req
        .metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let keep = admin.create_session(keep_req).await.unwrap().into_inner();
    let mut gone_req = Request::new(create_session_request());
    gone_req
        .metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let gone = admin.create_session(gone_req).await.unwrap().into_inner();

    // The creation checkpoint runs as the round loop's first await, since the
    // sync create path cannot await it, so both upserts must land before the
    // delete: it has to remove a row that was really persisted rather than lean
    // on the create/delete-race compensation.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let mut req = Request::new(DeleteSessionRequest {
        session_id: gone.session_id.clone(),
    });
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    admin.delete_session(req).await.unwrap();

    agg_handle.shutdown();
    // Let the first aggregator's single store connection close before the
    // second opens the same file.
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Second aggregator on the same state_path: it reloads the persisted list.
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(agg_config()).await.unwrap();
    let mut admin = AdminServiceClient::connect(format!("http://{agg_addr}"))
        .await
        .unwrap();
    let mut list_req = Request::new(ListSessionsRequest {});
    list_req
        .metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let list = admin.list_sessions(list_req).await.unwrap().into_inner();
    let ids: Vec<&str> = list.sessions.iter().map(|s| s.session_id.as_str()).collect();
    assert!(
        ids.contains(&keep.session_id.as_str()),
        "the undeleted session must reload after a restart"
    );
    assert!(
        !ids.contains(&gone.session_id.as_str()),
        "the deleted session must not resurrect after a restart"
    );

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// The in-process handle skips the dataset-table check only for an EMPTY
/// `dataset_id`; a non-empty, unknown one gets the same table check
/// `CreateSession` runs, so it is refused here too rather than silently
/// spawning a session for a dataset this cluster never advertised.
#[tokio::test(flavor = "multi_thread")]
async fn handle_create_session_rejects_a_non_empty_unknown_dataset() {
    let cluster = start_cluster().await;

    let err = cluster
        .agg_handle
        .create_session(SessionSpec {
            dataset_id: "no_such_dataset".into(),
            title: "handle test session".into(),
            n_trees: 1,
            max_depth: 1,
            n_bins: 4,
            learning_rate: 0.1,
            lambda: 1.0,
            min_clients: 1,
            target_clients: 1,
            submission_window_ms: 20_000,
        })
        .expect_err("a non-empty unknown dataset must be refused even via the handle");
    assert!(
        err.to_string().contains("unknown dataset"),
        "expected an unknown-dataset error, got: {err}"
    );

    cluster.shutdown();
}
