//! Plumbing-correctness test for the aggregator round loop: 3 real
//! shareholder daemons + the aggregator orchestrator, driven by a
//! synchronous in-test client loop over the REAL PollSession surface.
//!
//! This is not the AUC gate (that is the `pbr-e2e` crate); it checks that a
//! 2-tree depth-2 session runs open -> publish -> window -> close ->
//! gather -> advance to completion and that the resulting model actually
//! learned something (beats the base rate on separable synthetic data).

use pbr_client::jwt::mint;
use pbr_client::rpc::Shareholders;
use pbr_client::wire_metrics::WireCounters;
use pbr_proto::convert::{edges_to_bin_config, model_from_proto, split_from_proto};
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_proto::v1::poll_session_response::Body;
use pbr_proto::v1::{EnrollRequest, ListSessionsRequest, PollSessionRequest, SessionPhase};
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{
    AggregatorHandle, DatasetTable, RunningAggregator, SessionSpec, serve as serve_aggregator,
};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
use pbr_server::shareholder::{RunningShareholder, ShutdownHandle, serve as serve_shareholder};
use pbr_core::{Client, Loss, Model, RoundContext as CoreRoundContext, accuracy};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use tonic::Request;

mod common;
use common::bearer;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("fixtures/test_key.pem");

const N_CLIENTS: usize = 40;
const N_TREES: usize = 2;
const MAX_DEPTH: usize = 2;

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

fn mint_token() -> String {
    mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap()
}

/// The 2-tree depth-2 session every cluster test in this file trains. The
/// aggregator boots hosting no session, so each test creates this one
/// explicitly through `AggregatorHandle::create_session`; `dataset_id` is
/// empty so an empty-selector enroll resolves to it.
fn training_spec() -> SessionSpec {
    SessionSpec {
        dataset_id: String::new(),
        title: "loop test".into(),
        n_trees: N_TREES,
        max_depth: MAX_DEPTH,
        n_bins: 8,
        learning_rate: 0.3,
        lambda: 1.0,
        min_clients: 5,
        target_clients: N_CLIENTS,
        submission_window_ms: 20_000,
    }
}

/// A running 3-shareholder + aggregator cluster, as every test in this file
/// needs. The aggregator boots hosting no session; `start_cluster` creates
/// the one session under test through the handle, so even a test that never
/// submits anything needs the real daemons behind it.
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

/// The cluster above with its one training session already created, as every
/// test that drives or lists a session needs.
async fn start_cluster() -> Cluster {
    let cluster = start_cluster_no_session().await;
    cluster
        .agg_handle
        .create_session(training_spec())
        .expect("the session under test must be created");
    cluster
}

/// A cluster that boots hosting no session, for the one test that asserts
/// exactly that before creating a session itself.
async fn start_cluster_no_session() -> Cluster {
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
        datasets: DatasetTable::default(),
        admin_token: None,
        state_path: ":memory:".into(),
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

/// The aggregator boots hosting nothing: a fresh cluster lists zero sessions,
/// and only a session created in-process through `AggregatorHandle` becomes
/// listable and enrollable. This is the structural guarantee of "no implicit
/// sessions", there is no boot session to enroll into.
#[tokio::test(flavor = "multi_thread")]
async fn fresh_cluster_hosts_no_sessions_until_one_is_created() {
    let cluster = start_cluster_no_session().await;
    let token = mint_token();
    let mut agg = AggregatorServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();

    let mut req = Request::new(ListSessionsRequest {});
    req.metadata_mut().insert("authorization", bearer(&token));
    let list = agg.list_sessions(req).await.unwrap().into_inner();
    assert_eq!(
        list.sessions.len(),
        0,
        "a freshly booted aggregator hosts no session until one is created"
    );

    let summary = cluster
        .agg_handle
        .create_session(training_spec())
        .expect("an in-process session must be creatable through the handle");
    assert!(!summary.session_id.is_empty(), "a created session has an id");

    let mut req = Request::new(ListSessionsRequest {});
    req.metadata_mut().insert("authorization", bearer(&token));
    let list = agg.list_sessions(req).await.unwrap().into_inner();
    assert_eq!(
        list.sessions.len(),
        1,
        "the handle-created session must now be the one hosted session"
    );
    assert_eq!(list.sessions[0].session_id, summary.session_id);

    let mut enroll_req = Request::new(EnrollRequest {
        session_id: summary.session_id.clone(),
    });
    enroll_req
        .metadata_mut()
        .insert("authorization", bearer(&token));
    let cfg = agg
        .enroll_session(enroll_req)
        .await
        .expect("a device must be able to enroll in the handle-created session by id")
        .into_inner();
    assert_eq!(cfg.session_id, summary.session_id);

    cluster.shutdown();
}

/// Enrolling in a session this aggregator does not host must be refused, not
/// silently served from whatever session happens to exist, otherwise a
/// client's contributions would be misattributed to another session.
#[tokio::test(flavor = "multi_thread")]
async fn enroll_for_an_unknown_session_is_rejected() {
    let cluster = start_cluster().await;
    let token = mint_token();

    let mut agg = AggregatorServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();
    let mut req = Request::new(EnrollRequest {
        session_id: "no-such-session".to_string(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));

    let err = agg
        .enroll_session(req)
        .await
        .expect_err("an unknown session id must be refused");
    assert_eq!(err.code(), tonic::Code::NotFound);

    cluster.shutdown();
}

/// A signed-in device can enumerate the sessions this aggregator hosts, with
/// enough detail to decide whether to contribute.
#[tokio::test(flavor = "multi_thread")]
async fn list_sessions_reports_the_hosted_session() {
    let cluster = start_cluster().await;
    let token = mint_token();

    let mut agg = AggregatorServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();
    let mut req = Request::new(ListSessionsRequest {});
    req.metadata_mut().insert("authorization", bearer(&token));

    let list = agg.list_sessions(req).await.unwrap().into_inner();
    assert_eq!(list.sessions.len(), 1, "one session is hosted");
    let s = &list.sessions[0];
    assert!(!s.session_id.is_empty(), "a session always has an id");
    assert_ne!(
        s.phase,
        SessionPhase::Unspecified as i32,
        "a hosted session always has a real phase"
    );
    assert_eq!(
        s.dataset_id, "",
        "this dataset-less session names no dataset; a device must never guess one for it"
    );

    cluster.shutdown();
}

/// Separable synthetic set: label depends only on feature 0; feature 1 is
/// noise. Roughly balanced (21 negatives / 19 positives), so the base rate
/// is ~0.525 and the 0.6 accuracy gate means the model must have learned.
fn synthetic_rows() -> Vec<(Vec<f64>, f64)> {
    (0..N_CLIENTS)
        .map(|i| {
            let f0 = i as f64 * 0.25; // 0.0 .. 9.75
            let f1 = (i % 7) as f64; // uninformative
            let label = if f0 > 5.0 { 1.0 } else { 0.0 };
            (vec![f0, f1], label)
        })
        .collect()
}

#[tokio::test]
async fn two_tree_session_completes_and_model_beats_base_rate() {
    // Three real shareholder daemons.
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

    // The aggregator. The window is generous because the test's client loop
    // is synchronous:
    // the loop early-closes as soon as all target_clients submissions have
    // landed on every shareholder, so the window only acts as a timeout.
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: internal_eps,
        client_shareholder_endpoints: client_eps.clone(),
        threshold: 2,
        auth: auth_cfg(),
        fcm: None,
        tls: None,
        datasets: DatasetTable::default(),
        admin_token: None,
        state_path: ":memory:".into(),
        eval: None,
    })
    .await
    .unwrap();

    // No boot session exists; create the one this test drives, dataset-less so
    // the empty-selector enroll/poll below resolve to it.
    agg_handle
        .create_session(training_spec())
        .expect("the session under test must be created");

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let mut agg = AggregatorServiceClient::connect(format!("http://{agg_addr}"))
        .await
        .unwrap();

    // Enrollment sanity: the session exists and has not failed.
    let mut req = Request::new(EnrollRequest {
        session_id: String::new(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    let sess = agg.enroll_session(req).await.unwrap().into_inner();
    assert_ne!(sess.phase(), SessionPhase::Failed);
    let session_id = sess.session_id.clone();
    assert!(
        !session_id.is_empty(),
        "EnrollSession must stamp a session_id"
    );

    // Synchronous client loop over the real poll surface.
    let rows = synthetic_rows();
    let mut clients: Vec<Client> = rows
        .iter()
        .enumerate()
        .map(|(i, (feats, label))| {
            Client::new(format!("c{i}"), feats.clone(), *label, 3, 2, Some(i as u64))
        })
        .collect();
    let mut fanout = Shareholders::connect_best_effort(
        &client_eps,
        token.clone(),
        None,
        std::sync::Arc::new(WireCounters::default()),
    )
    .unwrap();

    let deadline = Instant::now() + Duration::from_secs(120);
    let mut last_seen = 0u64;
    let final_model: Model = loop {
        assert!(
            Instant::now() < deadline,
            "session did not complete within 120s"
        );
        // Poll by the concrete id, as the real client does once enrolled: the
        // empty selector resolves to the sole *live* session, so it stops
        // resolving the instant this session reaches Completed.
        let mut req = Request::new(PollSessionRequest {
            last_seen_round_id: last_seen,
            session_id: session_id.clone(),
        });
        req.metadata_mut().insert("authorization", bearer(&token));
        let resp = agg.poll_session(req).await.unwrap().into_inner();
        let phase = resp.phase();
        assert_ne!(phase, SessionPhase::Failed, "session failed");

        let Some(Body::Ctx(ctx)) = resp.body else {
            tokio::time::sleep(Duration::from_millis(25)).await;
            continue;
        };
        assert_eq!(
            ctx.session_id, session_id,
            "session_id must stay stable across every published context in one session"
        );
        last_seen = ctx.round_id;

        if phase == SessionPhase::Completed {
            let model = ctx
                .model
                .expect("completed context carries the final model");
            break model_from_proto(model).unwrap();
        }

        if ctx.depth == u32::MAX {
            // Stats sentinel: submit statistics shares.
            for client in clients.iter_mut() {
                let shares = client.compute_stat_shares().unwrap();
                fanout.submit_stats(shares, &session_id).await;
            }
        } else {
            // Gradient round: rebuild the `pbr-core` round context and
            // submit.
            let core_ctx = CoreRoundContext {
                bins: ctx
                    .bin_edges
                    .iter()
                    .cloned()
                    .map(edges_to_bin_config)
                    .collect(),
                model: model_from_proto(ctx.model.clone().expect("gradient ctx carries model"))
                    .unwrap(),
                splits: ctx
                    .splits_so_far
                    .iter()
                    .map(|(id, s)| (*id as usize, split_from_proto(*s, *id)))
                    .collect::<BTreeMap<_, _>>(),
                round_id: ctx.round_id,
                depth: ctx.depth as usize,
            };
            for client in clients.iter_mut() {
                let shares = client
                    .compute_gradient_shares(&core_ctx, &Loss::Logistic, true)
                    .unwrap();
                fanout.submit_gradients(shares, &session_id).await;
            }
        }
    };

    // The session trained the configured number of trees...
    assert_eq!(final_model.trees.len(), N_TREES);

    // ...and the model actually learned: accuracy on the (separable)
    // training rows must beat the ~0.525 base rate with margin.
    let features: Vec<Vec<f64>> = rows.iter().map(|(f, _)| f.clone()).collect();
    let targets: Vec<f64> = rows.iter().map(|(_, t)| *t).collect();
    let preds = final_model.predict(&features);
    let acc = accuracy(&preds, &targets, 0.5);
    assert!(
        acc > 0.6,
        "trained model should beat the base rate on the training rows, got accuracy {acc}"
    );

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}
