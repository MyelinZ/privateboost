use pbr_client::jwt::mint;
use pbr_proto::convert::{commitment_to_bytes, share_to_proto};
use pbr_proto::v1::shareholder_internal_client::ShareholderInternalClient;
use pbr_proto::v1::shareholder_service_client::ShareholderServiceClient;
use pbr_proto::v1::{CloseRoundRequest, OpenRoundRequest, SharePhase, StatsShareSubmission};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
use pbr_server::grpc_share_source::GrpcShareSource;
use pbr_server::shareholder::{RunningShareholder, serve};
use pbr_core::{Aggregator, BinMethod, Client};
use tonic::Request;
use tonic::transport::Channel;

mod common;
use common::bearer;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("fixtures/test_key.pem");

// Stats OpenRound/CloseRound carry a sentinel depth (stats gating matches on
// phase alone). The snapshot itself fetches stats listings with depth=0.
const STATS_DEPTH_SENTINEL: u32 = u32::MAX;

fn cfg(x: u64) -> ShareholderConfig {
    ShareholderConfig {
        x_coord: x,
        min_clients: 5,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: AuthConfig {
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
        },
        tls: None,
    }
}

async fn open_stats(int: &mut ShareholderInternalClient<Channel>) {
    int.open_round(OpenRoundRequest {
        round_id: 0,
        depth: STATS_DEPTH_SENTINEL,
        phase: SharePhase::Stats as i32,
        session_id: String::new(),
    })
    .await
    .unwrap();
}

async fn close_stats(int: &mut ShareholderInternalClient<Channel>) {
    int.close_round(CloseRoundRequest {
        round_id: 0,
        depth: STATS_DEPTH_SENTINEL,
        phase: SharePhase::Stats as i32,
        session_id: String::new(),
    })
    .await
    .unwrap();
}

/// Distributed analogue of the `pbr-core` crate's own
/// `aggregator_works_with_custom_share_source` test: two real shareholder daemons,
/// a `GrpcShareSource` snapshot per daemon (one adapter == one shareholder),
/// and `define_bins()` reconstructing the feature stats 2-of-N through the
/// two adapters, all sum calls crossing the process boundary over gRPC.
#[tokio::test]
async fn define_bins_reconstructs_through_two_grpc_adapters() {
    let RunningShareholder {
        client_addr: addr1,
        internal_addr: internal1,
        handle: h1,
    } = serve(cfg(1)).await.unwrap();
    let RunningShareholder {
        client_addr: addr2,
        internal_addr: internal2,
        handle: h2,
    } = serve(cfg(2)).await.unwrap();

    let token = mint(ISS, AUD, KID, "test-device", 300, PRIV).unwrap();

    let mut svc1 = ShareholderServiceClient::connect(format!("http://{addr1}"))
        .await
        .unwrap();
    let mut svc2 = ShareholderServiceClient::connect(format!("http://{addr2}"))
        .await
        .unwrap();
    let mut int1 = ShareholderInternalClient::connect(format!("http://{internal1}"))
        .await
        .unwrap();
    let mut int2 = ShareholderInternalClient::connect(format!("http://{internal2}"))
        .await
        .unwrap();

    open_stats(&mut int1).await;
    open_stats(&mut int2).await;

    // 6 clients (>= min_clients = 5), 2 features each.
    let feature_rows: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64, 10.0 + i as f64]).collect();
    for (i, feats) in feature_rows.iter().enumerate() {
        let mut client = Client::new(format!("c{i}"), feats.clone(), 1.0, 3, 2, Some(i as u64));
        let shares = client.compute_stat_shares().unwrap();
        for share in shares {
            let svc = match share.share.x {
                1 => &mut svc1,
                2 => &mut svc2,
                3 => continue,
                other => panic!("unexpected evaluation point {other}"),
            };
            let mut req = Request::new(StatsShareSubmission {
                commitment: commitment_to_bytes(&share.commitment),
                share: Some(share_to_proto(&share.share)),
                session_id: String::new(),
            });
            req.metadata_mut().insert("authorization", bearer(&token));
            svc.submit_stats_shares(req).await.unwrap();
        }
    }

    close_stats(&mut int1).await;
    close_stats(&mut int2).await;

    let ep1 = format!("http://{internal1}");
    let ep2 = format!("http://{internal2}");

    // Snapshots + define_bins run inside spawn_blocking so the sum methods'
    // `block_on` executes off the runtime worker threads.
    let (n_bins, n_clients, n_features) = tokio::task::spawn_blocking(move || {
        let handle = tokio::runtime::Handle::current();
        let a1 = GrpcShareSource::snapshot(&ep1, SharePhase::Stats, 0, "", handle.clone())?;
        let a2 = GrpcShareSource::snapshot(&ep2, SharePhase::Stats, 0, "", handle)?;
        let mut agg = Aggregator::new(vec![a1, a2], 10, 2, 5, 0.1, 1.0, BinMethod::Gaussian)?;
        let bins = agg.define_bins()?;
        let n_features = agg.feature_stats().map(|s| s.len() - 1).unwrap_or(0);
        anyhow::Ok((bins.len(), agg.n_clients(), n_features))
    })
    .await
    .unwrap()
    .unwrap();

    // One bin config per feature; all 6 clients reconstructed 2-of-N.
    assert_eq!(n_features, 2);
    assert_eq!(n_bins, n_features);
    assert_eq!(n_clients, 6);

    h1.shutdown();
    h2.shutdown();
}
