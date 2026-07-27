//! Wire byte counters: the `CountingIo` transport wrapper tallies socket bytes
//! into a shared `WireCounters`, and one `ClientSession`'s counters accumulate
//! traffic from the aggregator channel and the whole shareholder fan-out.

use pbr_client::driver::{ClientSession, SessionParams, StepOutcome};
use pbr_client::jwt::mint;
use pbr_client::wire_metrics::{CountingIo, WireCounters};
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{
    DatasetTable, RunningAggregator, SessionSpec, serve as serve_aggregator,
};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey, TlsConfig};
use pbr_server::shareholder::{RunningShareholder, serve as serve_shareholder};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Bytes written through a `CountingIo` land in `tx()`, bytes read land in
/// `rx()`, each counting exactly what crossed the wrapped stream.
#[tokio::test]
async fn counting_io_tallies_tx_and_rx() {
    let (a, mut b) = tokio::io::duplex(4096);
    let counters = Arc::new(WireCounters::default());
    let mut ca = CountingIo::new(a, counters.clone());

    ca.write_all(&[0u8; 128]).await.unwrap();
    ca.flush().await.unwrap();
    assert_eq!(counters.tx(), 128, "writes through CountingIo count as tx");

    b.write_all(&[0u8; 64]).await.unwrap();
    b.flush().await.unwrap();
    let mut buf = [0u8; 64];
    ca.read_exact(&mut buf).await.unwrap();
    assert_eq!(counters.rx(), 64, "reads through CountingIo count as rx");
}

/// Two `CountingIo` wrappers over one `Arc<WireCounters>` accumulate into a
/// single total: this is the shape of a session sharing one counter across the
/// aggregator channel and every shareholder channel, so a per-round delta sees
/// poll + submit bytes across all sockets at once.
#[tokio::test]
async fn counters_are_shared() {
    let (a, _b) = tokio::io::duplex(4096);
    let (c, _d) = tokio::io::duplex(4096);
    let counters = Arc::new(WireCounters::default());
    let mut ca = CountingIo::new(a, counters.clone());
    let mut cc = CountingIo::new(c, counters.clone());

    ca.write_all(&[0u8; 10]).await.unwrap();
    ca.flush().await.unwrap();
    cc.write_all(&[0u8; 7]).await.unwrap();
    cc.flush().await.unwrap();

    assert_eq!(
        counters.tx(),
        17,
        "both sockets tally into the one shared counter"
    );
}

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("../../pbr-server/tests/fixtures/test_key.pem");
/// The CA the fixtures' server cert chains to; pinned by the client below.
const CA_PEM: &[u8] = include_bytes!("../../pbr-server/tests/fixtures/tls/ca.crt");
const THRESHOLD: usize = 2;

fn auth_cfg() -> AuthConfig {
    AuthConfig {
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
    }
}

fn tls_cfg() -> TlsConfig {
    TlsConfig {
        cert_path: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pbr-server/tests/fixtures/tls/server.crt"
        )
        .into(),
        key_path: concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../pbr-server/tests/fixtures/tls/server.key"
        )
        .into(),
    }
}

fn shareholder_cfg(x: u64) -> ShareholderConfig {
    ShareholderConfig {
        x_coord: x,
        min_clients: 1,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: Some(tls_cfg()),
    }
}

/// A full session over the in-process TLS cluster moves real ciphertext: after
/// training to completion, the session's shared counters show nonzero tx AND
/// rx, proving the below-TLS connector is wired into both the aggregator
/// channel and the shareholder fan-out (a channel built the ordinary way would
/// leave the counters at zero).
#[tokio::test]
async fn session_counts_nonzero_wire_bytes() {
    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr,
            internal_addr,
            handle,
        } = serve_shareholder(shareholder_cfg(x)).await.unwrap();
        client_eps.push(format!("https://{client_addr}"));
        internal_eps.push(format!("http://{internal_addr}"));
        sh_handles.push(handle);
    }

    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: internal_eps,
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
        auth: auth_cfg(),
        fcm: None,
        tls: Some(tls_cfg()),
        datasets: DatasetTable::default(),
        admin_token: None,
        state_path: ":memory:".into(),
        eval: None,
    })
    .await
    .unwrap();

    agg_handle
        .create_session(SessionSpec {
            dataset_id: String::new(),
            title: "wire metrics test".into(),
            n_trees: 1,
            max_depth: 1,
            n_bins: 8,
            learning_rate: 0.3,
            lambda: 1.0,
            min_clients: 1,
            target_clients: 1,
            submission_window_ms: 2_000,
        })
        .expect("the session under test must be created");

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("https://{agg_addr}");

    let mut session = ClientSession::enroll(SessionParams {
        agg_endpoint: agg_url,
        shareholder_endpoints: Some(client_eps),
        token,
        records: vec![(vec![2.0, 1.0], 1.0)],
        threshold: Some(THRESHOLD),
        hide_path: true,
        ca_pem: Some(CA_PEM.to_vec()),
        session_id: None,
    })
    .await
    .expect("a pinned-CA client must enroll over TLS");

    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match session.step().await.expect("step must not error") {
                StepOutcome::Completed => break,
                StepOutcome::Failed => panic!("aggregator session failed"),
                StepOutcome::NothingNew { next_poll_after, .. } => {
                    tokio::time::sleep(next_poll_after).await;
                }
                StepOutcome::Submitted { .. } => {}
            }
        }
    })
    .await
    .expect("session must complete within the timeout");

    let tx = session.wire_counters().tx();
    let rx = session.wire_counters().rx();
    assert!(tx > 0, "session must have written ciphertext (tx={tx})");
    assert!(rx > 0, "session must have read ciphertext (rx={rx})");

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}
