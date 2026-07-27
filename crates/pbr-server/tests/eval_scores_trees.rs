//! The aggregator's own per-tree evaluation, driven against an in-process
//! plaintext cluster with an injected fake `MetricSink` (no live Firestore).
//! Real driver clients train a short session; the aggregator must emit one
//! metrics row per finished tree with tree indices increasing and every AUC
//! in [0, 1], and emit nothing for a session whose dataset has no
//! `[eval.datasets]` entry.

use pbr_client::driver::{SessionParams, run_to_completion};
use pbr_client::jwt::mint;
use pbr_proto::v1::EnrollRequest;
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{
    AggregatorHandle, DatasetTable, RunningAggregator, SessionSpec, serve_with_eval,
};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
use pbr_server::eval::metrics::THRESHOLD;
use pbr_server::eval::{Evaluator, HeldOutSplit, MetricSink, TreeMetricDoc};
use pbr_server::shareholder::{RunningShareholder, ShutdownHandle, serve as serve_shareholder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tonic::Request;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("fixtures/test_key.pem");

const THRESHOLD_SHARES: usize = 2;
const N_TREES: usize = 3;
const N_CLIENTS: usize = 4;

/// A `MetricSink` that collects documents in memory instead of POSTing to
/// Firestore.
#[derive(Clone, Default)]
struct FakeSink(Arc<Mutex<Vec<TreeMetricDoc>>>);

#[tonic::async_trait]
impl MetricSink for FakeSink {
    async fn write(&self, doc: &TreeMetricDoc) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(doc.clone());
        Ok(())
    }
}

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
        min_clients: 1,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: None,
    }
}

/// A labelled held-out split, injected directly (never read from disk here).
fn held_out_split() -> HeldOutSplit {
    HeldOutSplit {
        features: vec![
            vec![0.3, 0.7],
            vec![0.85, 0.15],
            vec![0.6, 0.4],
            vec![0.15, 0.55],
        ],
        targets: vec![1.0, 0.0, 1.0, 0.0],
    }
}

/// An `Evaluator` whose only split is registered under `dataset_id`, sinking
/// into `sink`.
fn evaluator_for(dataset_id: &str, sink: FakeSink) -> Arc<Evaluator> {
    Arc::new(Evaluator {
        splits: HashMap::from([(dataset_id.to_string(), held_out_split())]),
        sink: Arc::new(sink),
    })
}

/// 3 shareholders + an aggregator over plaintext loopback, with `eval`
/// injected via `serve_with_eval`, and 4 real driver clients training a
/// dataset-less session (`dataset_id` empty) created through the handle.
async fn spawn_cluster(
    eval: Arc<Evaluator>,
) -> (
    String,
    Vec<tokio::task::JoinHandle<anyhow::Result<()>>>,
    AggregatorHandle,
    Vec<ShutdownHandle>,
) {
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

    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_with_eval(
        AggregatorConfig {
            listen: "127.0.0.1:0".parse().unwrap(),
            internal_shareholder_endpoints: internal_eps,
            client_shareholder_endpoints: client_eps.clone(),
            threshold: THRESHOLD_SHARES,
            auth: auth_cfg(),
            fcm: None,
            tls: None,
            datasets: DatasetTable::default(),
            admin_token: None,
            state_path: ":memory:".into(),
            eval: None,
        },
        Some(eval),
    )
    .await
    .unwrap();

    // The dataset-less session these clients train; its empty `dataset_id` is
    // what an evaluator registered under "" scores.
    agg_handle
        .create_session(SessionSpec {
            dataset_id: String::new(),
            title: "eval test".into(),
            n_trees: N_TREES,
            max_depth: 2,
            n_bins: 8,
            learning_rate: 0.3,
            lambda: 1.0,
            min_clients: N_CLIENTS,
            target_clients: N_CLIENTS,
            submission_window_ms: 2_000,
        })
        .expect("the session under test must be created");

    let agg_url = format!("http://{agg_addr}");

    let rows: [([f64; 2], f64); N_CLIENTS] = [
        ([0.2, 0.8], 1.0),
        ([0.9, 0.1], 0.0),
        ([0.7, 0.3], 1.0),
        ([0.1, 0.6], 0.0),
    ];
    let mut client_tasks = Vec::new();
    for (features, label) in rows {
        let token = mint(ISS, AUD, KID, "eval-test-client", 300, PRIV).unwrap();
        let eps = client_eps.clone();
        let url = agg_url.clone();
        client_tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: url,
                    shareholder_endpoints: Some(eps),
                    token,
                    records: vec![(features.to_vec(), label)],
                    threshold: Some(THRESHOLD_SHARES),
                    hide_path: true,
                    ca_pem: None,
                    session_id: None,
                },
                |_| {},
            )
            .await
        }));
    }

    (agg_url, client_tasks, agg_handle, sh_handles)
}

/// `EnrollSession` is read-only on the aggregator, so calling it once with an
/// empty selector is a side-effect-free way to learn the sole session's
/// minted id.
async fn enrolled_session_id(agg_url: &str) -> String {
    let token = mint(ISS, AUD, KID, "eval-test-enroll", 300, PRIV).unwrap();
    let mut agg = AggregatorServiceClient::connect(agg_url.to_string())
        .await
        .unwrap();
    let mut req = Request::new(EnrollRequest {
        session_id: String::new(),
    });
    req.metadata_mut()
        .insert("authorization", format!("Bearer {token}").parse().unwrap());
    agg.enroll_session(req).await.unwrap().into_inner().session_id
}

/// The Firestore writes are spawned off the round loop, so the last document
/// can land after the session publishes `Completed`; poll (bounded) instead
/// of asserting immediately.
async fn wait_for_docs(sink: &FakeSink, n: usize) -> Vec<TreeMetricDoc> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let docs = sink.0.lock().unwrap().clone();
        if docs.len() >= n {
            return docs;
        }
        assert!(
            Instant::now() < deadline,
            "expected {n} metric docs, still at {} after 10s",
            docs.len()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn aggregator_writes_one_metric_row_per_tree() {
    let sink = FakeSink::default();
    // The session's dataset_id is the empty string; registering the split
    // under "" makes that session the scored one.
    let (agg_url, client_tasks, agg_handle, sh_handles) =
        spawn_cluster(evaluator_for("", sink.clone())).await;
    let session_id = enrolled_session_id(&agg_url).await;

    for task in client_tasks {
        task.await.unwrap().expect("driver client completes");
    }
    let mut docs = wait_for_docs(&sink, N_TREES).await;

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }

    assert_eq!(docs.len(), N_TREES, "one metrics row per tree, no extras");
    // Writes are spawned, so guard against reorderings instead of assuming
    // push order.
    docs.sort_by_key(|d| d.tree_idx);
    for (i, d) in docs.iter().enumerate() {
        assert_eq!(d.tree_idx as usize, i, "treeIdx covers 0..N_TREES exactly");
        assert!(
            (0.0..=1.0).contains(&d.metrics.auc),
            "auc {} must be in [0, 1]",
            d.metrics.auc
        );
        assert_eq!(d.n_test, held_out_split().targets.len());
        assert_eq!(d.threshold_used, THRESHOLD);
        assert_eq!(d.session_id, session_id, "sessionId matches the scored session");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_session_without_an_eval_entry_is_not_scored() {
    let sink = FakeSink::default();
    // The evaluator only knows a dataset the session ("") does not train;
    // the gate must skip the session entirely.
    let (_agg_url, client_tasks, agg_handle, sh_handles) =
        spawn_cluster(evaluator_for("some_other_dataset", sink.clone())).await;

    for task in client_tasks {
        task.await.unwrap().expect("driver client completes");
    }
    // Misses never spawn a write task, but give any bug a moment to surface
    // before asserting emptiness.
    tokio::time::sleep(Duration::from_millis(200)).await;

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }

    assert!(
        sink.0.lock().unwrap().is_empty(),
        "a dataset with no [eval.datasets] entry must produce zero metric rows"
    );
}
