//! RegisterDevice RPC + persisted device store: binding to the verified
//! identity from the JWT interceptor, and re-register-updates-not-duplicates.

use pbr_client::jwt::mint;
use pbr_proto::v1::{ClientPlatform, RegisterDeviceRequest};
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{DatasetTable, DeviceRow, RunningAggregator, serve as serve_aggregator};
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

/// A minimal aggregator config: RegisterDevice only exercises the gRPC
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

#[tokio::test]
async fn register_device_binds_to_verified_identity() {
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(test_agg_config()).await.unwrap();
    let mut agg = AggregatorServiceClient::connect(format!("http://{agg_addr}"))
        .await
        .unwrap();

    let token = mint(ISS, AUD, KID, "device-x", 300, PRIV).unwrap();
    let expected_uid = format!("{ISS}|device-x");

    // First registration.
    let mut req = Request::new(RegisterDeviceRequest {
        fcm_token: "tok-A".into(),
        platform: ClientPlatform::Android.into(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    agg.register_device(req).await.unwrap();

    let devices: Vec<DeviceRow> = agg_handle.registered_devices().await.unwrap();
    assert_eq!(devices.len(), 1, "expected exactly one registered device");
    assert_eq!(
        devices[0].uid, expected_uid,
        "device must be keyed by verified (iss, sub)"
    );
    assert_eq!(devices[0].fcm_token, "tok-A");
    assert_eq!(
        devices[0].platform,
        ClientPlatform::Android as i32,
        "the wire enum must round-trip into the devices table"
    );

    // Re-register with a new token: must UPDATE the entry, not duplicate it.
    let mut req = Request::new(RegisterDeviceRequest {
        fcm_token: "tok-B".into(),
        platform: ClientPlatform::Android.into(),
    });
    req.metadata_mut().insert("authorization", bearer(&token));
    agg.register_device(req).await.unwrap();

    let devices: Vec<DeviceRow> = agg_handle.registered_devices().await.unwrap();
    assert_eq!(
        devices.len(),
        1,
        "re-register must update the existing entry, not add a duplicate"
    );
    assert_eq!(devices[0].uid, expected_uid);
    assert_eq!(devices[0].fcm_token, "tok-B", "re-register must update the fcm_token");

    agg_handle.shutdown();
}

#[tokio::test]
async fn register_device_rejects_missing_bearer_token() {
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(test_agg_config()).await.unwrap();
    let mut agg = AggregatorServiceClient::connect(format!("http://{agg_addr}"))
        .await
        .unwrap();

    let req = Request::new(RegisterDeviceRequest {
        fcm_token: "tok-A".into(),
        platform: ClientPlatform::Android.into(),
    });
    let result = agg.register_device(req).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::Unauthenticated);

    agg_handle.shutdown();
}

#[tokio::test]
async fn register_device_rejects_unspecified_or_unknown_platform() {
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(test_agg_config()).await.unwrap();
    let mut agg = AggregatorServiceClient::connect(format!("http://{agg_addr}"))
        .await
        .unwrap();

    let token = mint(ISS, AUD, KID, "device-x", 300, PRIV).unwrap();

    // CLIENT_PLATFORM_UNSPECIFIED and a discriminant outside the enum must
    // both be refused at the RPC, not stored.
    for bad in [ClientPlatform::Unspecified.into(), 99] {
        let mut req = Request::new(RegisterDeviceRequest {
            fcm_token: "tok-A".into(),
            platform: bad,
        });
        req.metadata_mut().insert("authorization", bearer(&token));
        let err = agg.register_device(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }
    assert!(
        agg_handle.registered_devices().await.unwrap().is_empty(),
        "a rejected registration must not create a devices row"
    );

    agg_handle.shutdown();
}
