use pbr_client::jwt::mint;
use pbr_client::rpc::Shareholders;
use pbr_client::wire_metrics::WireCounters;
use pbr_proto::v1::shareholder_internal_client::ShareholderInternalClient;
use pbr_proto::v1::{ListCommitmentsRequest, OpenRoundRequest, SharePhase};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey};
use pbr_server::shareholder::{RunningShareholder, serve};
use pbr_core::protocol::aggregator::RoundContext;
use pbr_core::{BinConfiguration, Model, SplitDecision};
use pbr_core::{Client, Loss};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("../../pbr-server/tests/fixtures/test_key.pem");

fn cfg(x: u64) -> ShareholderConfig {
    ShareholderConfig {
        x_coord: x,
        min_clients: 1,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: AuthConfig {
            issuer: ISS.into(),
            audience: AUD.into(),
            static_keys: vec![StaticKey {
                kid: KID.into(),
                public_key_pem_path: concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../pbr-server/tests/fixtures/test_key.pub.pem"
                )
                .into(),
            }],
            google_jwks_url: None,
        },
        tls: None,
    }
}

#[tokio::test]
async fn fan_out_reaches_all_three_shareholders() {
    let mut addrs = Vec::new();
    let mut internals = Vec::new();
    let mut handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: a,
            internal_addr: i,
            handle: h,
        } = serve(cfg(x)).await.unwrap();
        addrs.push(format!("http://{a}"));
        internals.push(i);
        handles.push(h);
    }
    let token = mint(ISS, AUD, KID, "device-1", 300, PRIV).unwrap();
    let mut sh =
        Shareholders::connect_best_effort(&addrs, token, None, Arc::new(WireCounters::default()))
            .unwrap();

    let mut ints = Vec::new();
    for internal in &internals {
        ints.push(
            ShareholderInternalClient::connect(format!("http://{internal}"))
                .await
                .unwrap(),
        );
    }

    // Stats fan-out
    for int in &mut ints {
        int.open_round(OpenRoundRequest {
            round_id: 0,
            depth: u32::MAX,
            phase: SharePhase::Stats as i32,
            session_id: String::new(),
        })
        .await
        .unwrap();
    }
    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(7));
    sh.submit_stats(client.compute_stat_shares().unwrap(), "")
        .await;

    // Every shareholder holds exactly one stats commitment. Checked here,
    // before the round transitions to gradients: OpenRound for a new round
    // resets stale pools, so the stats pool will not survive past this point.
    for int in &mut ints {
        let stats = int
            .list_commitments(ListCommitmentsRequest {
                phase: SharePhase::Stats as i32,
                depth: 0,
                session_id: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(stats.commitments.len(), 1);
    }

    // Gradient fan-out (depth 0, no splits yet)
    for int in &mut ints {
        int.open_round(OpenRoundRequest {
            round_id: 1,
            depth: 0,
            phase: SharePhase::Gradient as i32,
            session_id: String::new(),
        })
        .await
        .unwrap();
    }
    let ctx = RoundContext {
        bins: vec![
            BinConfiguration {
                feature_idx: 0,
                edges: vec![f64::NEG_INFINITY, 0.0, 2.0],
            },
            BinConfiguration {
                feature_idx: 1,
                edges: vec![f64::NEG_INFINITY, 0.0, 2.0],
            },
        ],
        model: Model::new(0.0, 0.15),
        splits: BTreeMap::new(),
        round_id: 1,
        depth: 0,
    };
    sh.submit_gradients(
        client
            .compute_gradient_shares(&ctx, &Loss::Logistic, false)
            .unwrap(),
        "",
    )
    .await;

    // Every shareholder holds exactly one gradient commitment.
    for int in &mut ints {
        let grads = int
            .list_commitments(ListCommitmentsRequest {
                phase: SharePhase::Gradient as i32,
                depth: 0,
                session_id: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(grads.commitments.len(), 1);
    }
    for h in handles {
        h.shutdown();
    }
}

/// A blackholed shareholder (TCP accepts, HTTP/2 never answers) must not hang
/// the fan-out: the per-RPC / connect deadline on the client-plane channel
/// bounds the submit so it returns with that shareholder simply unacknowledged,
/// instead of parking on the OS TCP timeout (10+ minutes).
#[tokio::test]
async fn submit_to_unresponsive_shareholder_times_out_not_hangs() {
    // Bind a listener and never accept/read: the kernel completes the TCP
    // handshake from its backlog, so the client connects but the HTTP/2
    // handshake never finishes.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let endpoints = vec![format!("http://{addr}")];

    let token = mint(ISS, AUD, KID, "device-1", 300, PRIV).unwrap();
    let mut sh = Shareholders::connect_best_effort(
        &endpoints,
        token,
        None,
        Arc::new(WireCounters::default()),
    )
    .unwrap();

    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 1, 1, Some(7));
    let shares = client.compute_stat_shares().unwrap();

    let report = tokio::time::timeout(Duration::from_secs(45), sh.submit_stats(shares, ""))
        .await
        .expect("submit must time out and return, not hang on a blackholed shareholder");

    assert_eq!(report.attempted, 1);
    assert_eq!(
        report.accepted, 0,
        "an unresponsive shareholder cannot acknowledge a submission"
    );

    // Hold the blackholing listener open until the submit has resolved.
    drop(listener);
}

/// Two blackholed shareholders alongside one healthy one must not serialize.
/// A sequential fan-out pays each unresponsive shareholder's
/// `REQUEST_TIMEOUT` in turn (~2x20s here); a concurrent one is bounded by a
/// single timeout no matter how many shareholders are unresponsive. Unlike
/// `submit_to_unresponsive_shareholder_times_out_not_hangs` (one shareholder,
/// so sequential vs. concurrent take the same wall time), this pins the
/// scheduling change itself via elapsed time.
#[tokio::test]
async fn slow_shareholders_do_not_serialize_the_fan_out() {
    let blackhole_a = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let blackhole_b = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr_a = blackhole_a.local_addr().unwrap();
    let addr_b = blackhole_b.local_addr().unwrap();

    // The one healthy shareholder is x = 3, i.e. endpoints[2], matching the
    // load-bearing endpoint-order-is-x invariant.
    let RunningShareholder {
        client_addr,
        internal_addr,
        handle,
    } = serve(cfg(3)).await.unwrap();
    let endpoints = vec![
        format!("http://{addr_a}"),
        format!("http://{addr_b}"),
        format!("http://{client_addr}"),
    ];

    let token = mint(ISS, AUD, KID, "device-1", 300, PRIV).unwrap();
    let mut sh = Shareholders::connect_best_effort(
        &endpoints,
        token,
        None,
        Arc::new(WireCounters::default()),
    )
    .unwrap();

    let mut internal = ShareholderInternalClient::connect(format!("http://{internal_addr}"))
        .await
        .unwrap();
    internal
        .open_round(OpenRoundRequest {
            round_id: 0,
            depth: u32::MAX,
            phase: SharePhase::Stats as i32,
            session_id: String::new(),
        })
        .await
        .unwrap();

    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(7));
    let shares = client.compute_stat_shares().unwrap();

    let started = tokio::time::Instant::now();
    let report = tokio::time::timeout(Duration::from_secs(35), sh.submit_stats(shares, ""))
        .await
        .expect("a concurrent fan-out must not take as long as two serialized timeouts");
    let elapsed = started.elapsed();

    assert_eq!(report.attempted, 3);
    assert_eq!(
        report.accepted, 1,
        "only the one healthy shareholder can acknowledge"
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "elapsed {elapsed:?} suggests the two blackholed shareholders were awaited \
         sequentially instead of concurrently"
    );

    drop(blackhole_a);
    drop(blackhole_b);
    handle.shutdown();
}

/// With path hiding over 2 active nodes, each shareholder must receive the
/// client's WHOLE per-round contribution as one atomic batch: exactly one
/// commitment, with both node-ids present under it, on every shareholder.
#[tokio::test]
async fn path_hiding_delivers_one_batch_with_all_nodes_per_shareholder() {
    let mut addrs = Vec::new();
    let mut internals = Vec::new();
    let mut handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: a,
            internal_addr: i,
            handle: h,
        } = serve(cfg(x)).await.unwrap();
        addrs.push(format!("http://{a}"));
        internals.push(i);
        handles.push(h);
    }
    let token = mint(ISS, AUD, KID, "device-1", 300, PRIV).unwrap();
    let mut sh =
        Shareholders::connect_best_effort(&addrs, token, None, Arc::new(WireCounters::default()))
            .unwrap();

    let mut ints = Vec::new();
    for internal in &internals {
        ints.push(
            ShareholderInternalClient::connect(format!("http://{internal}"))
                .await
                .unwrap(),
        );
    }
    for int in &mut ints {
        int.open_round(OpenRoundRequest {
            round_id: 2,
            depth: 1,
            phase: SharePhase::Gradient as i32,
            session_id: String::new(),
        })
        .await
        .unwrap();
    }

    // Depth 1 with one split at the root: active nodes are {1, 2}.
    let split = SplitDecision {
        node_id: 0,
        feature_idx: 0,
        threshold: 1.0,
        gain: 0.0,
        left_child_id: 1,
        right_child_id: 2,
        g_left: 0.0,
        h_left: 0.0,
        g_right: 0.0,
        h_right: 0.0,
    };
    let ctx = RoundContext {
        bins: vec![
            BinConfiguration {
                feature_idx: 0,
                edges: vec![f64::NEG_INFINITY, 0.0, 2.0],
            },
            BinConfiguration {
                feature_idx: 1,
                edges: vec![f64::NEG_INFINITY, 0.0, 2.0],
            },
        ],
        model: Model::new(0.0, 0.15),
        splits: BTreeMap::from([(0, split)]),
        round_id: 2,
        depth: 1,
    };
    let mut client = Client::new("c0".into(), vec![1.0, 2.0], 1.0, 3, 2, Some(7));
    let shares = client
        .compute_gradient_shares(&ctx, &Loss::Logistic, true)
        .unwrap();
    // 2 active nodes x 3 shareholders, all under one per-round commitment.
    assert_eq!(shares.len(), 6);
    sh.submit_gradients(shares, "").await;

    for int in &mut ints {
        let grads = int
            .list_commitments(ListCommitmentsRequest {
                phase: SharePhase::Gradient as i32,
                depth: 1,
                session_id: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            grads.commitments.len(),
            1,
            "one commitment per client per round"
        );
        assert_eq!(
            grads.node_ids,
            vec![1, 2],
            "both active nodes under the one commitment"
        );
    }
    for h in handles {
        h.shutdown();
    }
}
