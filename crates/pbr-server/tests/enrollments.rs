//! EnrollSession persistence: the verified identity is recorded under the
//! RESOLVED session id (empty selector included), refreshed rather than
//! duplicated on re-enroll.

use pbr_client::jwt::mint;
use pbr_proto::v1::EnrollRequest;
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{
    DatasetTable, RunningAggregator, SessionSpec, serve as serve_aggregator,
};
use pbr_server::config::{AuthConfig, StaticKey};
use tonic::Request;

mod common;
use common::bearer;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("fixtures/test_key.pem");

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

/// A minimal aggregator config: EnrollSession only exercises the gRPC
/// listener, so the round loop's shareholder endpoints never need to be
/// real (the loop retries/fails in the background, off the RPC path).
fn test_agg_config() -> AggregatorConfig {
    AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: vec!["http://127.0.0.1:1".into()],
        client_shareholder_endpoints: vec!["http://127.0.0.1:2".into()],
        threshold: 1,
        auth: auth_cfg(),
        fcm: None,
        tls: None,
        datasets: DatasetTable::default(),
        admin_token: None,
        state_path: ":memory:".into(),
        eval: None,
    }
}

/// Dataset-less spec (the in-process `create_session` path accepts it), so
/// no shareholder cluster is needed: the round loop fails in the background,
/// off the RPC path this test exercises.
fn dataset_less_spec() -> SessionSpec {
    SessionSpec {
        dataset_id: String::new(),
        title: "enroll test".into(),
        n_trees: 1,
        max_depth: 1,
        n_bins: 8,
        learning_rate: 0.3,
        lambda: 1.0,
        min_clients: 1,
        target_clients: 1,
        submission_window_ms: 5_000,
    }
}

#[tokio::test]
async fn enroll_records_verified_identity_under_resolved_session_id() {
    let RunningAggregator { addr, handle } = serve_aggregator(test_agg_config()).await.unwrap();
    let summary = handle.create_session(dataset_less_spec()).unwrap();

    let mut agg = AggregatorServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    let token = mint(ISS, AUD, KID, "device-e", 300, PRIV).unwrap();

    // Empty selector: must resolve to the only session and record THAT id.
    let mut req = Request::new(EnrollRequest {
        session_id: String::new(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    let cfg = agg.enroll_session(req).await.unwrap().into_inner();
    assert_eq!(cfg.session_id, summary.session_id);

    let rows = handle.enrollments().await.unwrap();
    assert_eq!(
        rows,
        vec![(summary.session_id.clone(), format!("{ISS}|device-e"))]
    );

    // Re-enroll: refreshed, not duplicated.
    let mut req = Request::new(EnrollRequest {
        session_id: summary.session_id.clone(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    agg.enroll_session(req).await.unwrap();
    assert_eq!(handle.enrollments().await.unwrap().len(), 1);
}

#[tokio::test]
async fn enroll_session_only_records_without_a_full_client_session() {
    let RunningAggregator { addr, handle } =
        serve_aggregator(test_agg_config()).await.unwrap();
    let summary = handle.create_session(dataset_less_spec()).unwrap();
    let token = mint(ISS, AUD, KID, "device-j", 300, PRIV).unwrap();

    pbr_client::driver::enroll_session_only(
        &format!("http://{addr}"),
        token,
        summary.session_id.clone(),
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        handle.enrollments().await.unwrap(),
        vec![(summary.session_id, format!("{ISS}|device-j"))]
    );
}
