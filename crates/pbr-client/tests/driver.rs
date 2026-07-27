//! End-to-end tests for `pbr_client::driver`: concurrent clients against a real
//! three-shareholder cluster, one daemon killed mid-session so best-effort
//! threshold delivery has to carry the round on 2-of-3.
//!
//! The kill lands the instant the session leaves `StatsPending`, when the sole
//! gradient round's context is published and strictly before that round's
//! window closes and its gather runs: a kill before a gather, not after the
//! session already completed.

use pbr_client::driver::{ClientSession, RoundKind, SessionParams, StepOutcome, run_to_completion, run_collecting};
use pbr_client::jwt::mint;
use pbr_proto::v1::admin_service_client::AdminServiceClient;
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_proto::v1::aggregator_service_server::{AggregatorService, AggregatorServiceServer};
use pbr_proto::v1::{
    Ack, CreateSessionRequest, EnrollRequest, ListSessionsRequest, PollSessionRequest,
    PollSessionResponse, RegisterDeviceRequest, SessionConfig, SessionList, SessionPhase,
};
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::aggregator::{DatasetTable, RunningAggregator, SessionSpec, serve as serve_aggregator};
use pbr_server::config::{AuthConfig, ShareholderConfig, StaticKey, TlsConfig};
use pbr_server::shareholder::{RunningShareholder, serve as serve_shareholder};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tonic::metadata::MetadataValue;
use tonic::{Request, Response, Status};

const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";
const PRIV: &[u8] = include_bytes!("../../pbr-server/tests/fixtures/test_key.pem");
/// The CA the fixtures' server cert chains to; pinned by the TLS client below.
const CA_PEM: &[u8] = include_bytes!("../../pbr-server/tests/fixtures/tls/ca.crt");

// At least the aggregator's min_clients, and no more, so the round loop can
// gather while the test stays fast.
const N_CLIENTS: usize = 3;
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

fn shareholder_cfg(x: u64) -> ShareholderConfig {
    ShareholderConfig {
        x_coord: x,
        min_clients: N_CLIENTS,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: None,
    }
}

/// The deployment cert and key, so the metrics test measures real ciphertext.
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

fn tls_shareholder_cfg(x: u64) -> ShareholderConfig {
    ShareholderConfig {
        x_coord: x,
        min_clients: 1,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: Some(tls_cfg()),
    }
}

fn bearer(token: &str) -> MetadataValue<tonic::metadata::Ascii> {
    format!("Bearer {token}").parse().unwrap()
}

/// The single-tree depth-1 session these tests train. The aggregator boots
/// hosting none, so each test creates this one; `dataset_id` is empty so an
/// empty-selector enroll resolves to it. Only the round-close floors vary.
fn training_spec(min_clients: usize, target_clients: usize) -> SessionSpec {
    SessionSpec {
        dataset_id: String::new(),
        title: "driver test".into(),
        n_trees: 1,
        max_depth: 1,
        n_bins: 8,
        learning_rate: 0.3,
        lambda: 1.0,
        min_clients,
        target_clients,
        submission_window_ms: 2_000,
    }
}

/// Synthetic rows with nonzero variance, so Gaussian bin definition is
/// well-defined.
fn synthetic_rows() -> Vec<(Vec<f64>, f64)> {
    (0..N_CLIENTS)
        .map(|i| {
            let f0 = i as f64 * 2.0;
            let label = if f0 > 2.0 { 1.0 } else { 0.0 };
            (vec![f0, 1.0], label)
        })
        .collect()
}

#[tokio::test]
async fn client_completes_despite_shareholder_killed_mid_session() {
    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles: Vec<Option<pbr_server::shareholder::ShutdownHandle>> = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: addr,
            internal_addr: internal,
            handle: h,
        } = serve_shareholder(shareholder_cfg(x)).await.unwrap();
        client_eps.push(format!("http://{addr}"));
        internal_eps.push(format!("http://{internal}"));
        sh_handles.push(Some(h));
    }

    // One tree at depth 1, so there is exactly one gradient round to kill a
    // shareholder ahead of.
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
        tls: None,
        datasets: DatasetTable::default(),
        admin_token: None,
        state_path: ":memory:".into(),
        eval: None,
    })
    .await
    .unwrap();

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(N_CLIENTS, N_CLIENTS))
        .expect("the session under test must be created");

    // Explicit shareholder endpoints and threshold: the non-fallback path.
    let mut tasks = Vec::new();
    for (features, label) in synthetic_rows() {
        let agg_url = agg_url.clone();
        let client_eps = client_eps.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: agg_url,
                    shareholder_endpoints: Some(client_eps),
                    token,
                    records: vec![(features, label)],
                    threshold: Some(THRESHOLD),
                    hide_path: true,
                    ca_pem: None,
                    session_id: None,
                },
                |_| {},
            )
            .await
        }));
    }

    // Watches phase transitions on its own connection and kills shareholder
    // x=2 the instant the session leaves StatsPending, when the sole gradient
    // round is published and before its CloseRound and gather.
    let mut observer = AggregatorServiceClient::connect(agg_url.clone())
        .await
        .unwrap();
    let watch_deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(
            Instant::now() < watch_deadline,
            "session never left StatsPending"
        );
        let mut req = Request::new(PollSessionRequest {
            last_seen_round_id: 0,
            session_id: String::new(),
        });
        req.metadata_mut().insert("authorization", bearer(&token));
        let resp = observer.poll_session(req).await.unwrap().into_inner();
        assert_ne!(
            resp.phase(),
            SessionPhase::Failed,
            "session failed before the gradient round was even published"
        );
        if resp.phase() == SessionPhase::Training {
            break;
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
    }
    // The gradient round's 2s window has not elapsed, so this precedes its
    // gather.
    sh_handles[1]
        .take()
        .expect("shareholder 2 handle present")
        .shutdown();

    // 2-of-3 tolerates the dead shareholder, so every client still completes.
    let results_deadline = Instant::now() + Duration::from_secs(60);
    for task in tasks {
        let remaining = results_deadline.saturating_duration_since(Instant::now());
        let outcome = tokio::time::timeout(remaining, task)
            .await
            .expect("client did not finish in time")
            .expect("client task panicked");
        outcome.expect("run_to_completion should succeed despite one dead shareholder");
    }

    agg_handle.shutdown();
    for h in sh_handles.into_iter().flatten() {
        h.shutdown();
    }
}

/// A client given only the aggregator endpoint and token must still bootstrap:
/// it learns the shareholder endpoints and threshold from the enroll response
/// and completes the session like one driven with explicit endpoints.
#[tokio::test]
async fn client_bootstraps_from_enroll_only() {
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

    // Advertises client_shareholder_endpoints, which EnrollSession hands to
    // clients verbatim.
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: internal_eps,
        client_shareholder_endpoints: client_eps,
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(N_CLIENTS, N_CLIENTS))
        .expect("the session under test must be created");

    // No shareholder URLs and no threshold: both must come from the enroll
    // response.
    let mut tasks = Vec::new();
    for (features, label) in synthetic_rows() {
        let agg_url = agg_url.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: agg_url,
                    shareholder_endpoints: None,
                    token,
                    records: vec![(features, label)],
                    threshold: None,
                    hide_path: true,
                    ca_pem: None,
                    session_id: None,
                },
                |_| {},
            )
            .await
        }));
    }

    let results_deadline = Instant::now() + Duration::from_secs(60);
    for task in tasks {
        let remaining = results_deadline.saturating_duration_since(Instant::now());
        let outcome = tokio::time::timeout(remaining, task)
            .await
            .expect("client did not finish in time")
            .expect("client task panicked");
        outcome.expect("run_to_completion should succeed when bootstrapping from enroll only");
    }

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// Driving a `ClientSession` one `step()` at a time from the test, rather than
/// through `run_to_completion`, must reach `Completed` having seen at least one
/// `Submitted` outcome and one `NothingNew` (a poll while another round's
/// window is still open).
#[tokio::test]
async fn step_api_drives_session_manually() {
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
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(N_CLIENTS, N_CLIENTS))
        .expect("the session under test must be created");

    // The other rows run normally, so the round loop reaches target_clients and
    // gathers early instead of waiting out the window.
    let mut rows = synthetic_rows();
    let (my_features, my_label) = rows.remove(0);
    let mut tasks = Vec::new();
    for (features, label) in rows {
        let agg_url = agg_url.clone();
        let client_eps = client_eps.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: agg_url,
                    shareholder_endpoints: Some(client_eps),
                    token,
                    records: vec![(features, label)],
                    threshold: Some(THRESHOLD),
                    hide_path: true,
                    ca_pem: None,
                    session_id: None,
                },
                |_| {},
            )
            .await
        }));
    }

    let mut session = ClientSession::enroll(SessionParams {
        agg_endpoint: agg_url,
        shareholder_endpoints: Some(client_eps),
        token,
        records: vec![(my_features, my_label)],
        threshold: Some(THRESHOLD),
        hide_path: true,
        ca_pem: None,
        session_id: None,
    })
    .await
    .unwrap();

    let mut saw_submitted = false;
    let mut saw_nothing_new = false;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        assert!(
            Instant::now() < deadline,
            "manual session did not complete in time"
        );
        match session.step().await.unwrap() {
            StepOutcome::Submitted { .. } => saw_submitted = true,
            StepOutcome::NothingNew { next_poll_after, .. } => {
                saw_nothing_new = true;
                tokio::time::sleep(next_poll_after).await;
            }
            StepOutcome::Completed => break,
            StepOutcome::Failed => panic!("aggregator session failed"),
        }
    }

    assert!(saw_submitted, "expected at least one Submitted outcome");
    assert!(saw_nothing_new, "expected at least one NothingNew outcome");

    for task in tasks {
        task.await
            .expect("client task panicked")
            .expect("background client should complete");
    }

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// A `Submitted` step carries the measurements the bridge turns into a
/// `RoundSummary`: compute microseconds (a single record's crypto floors to
/// 0 ms), a poll duration, and the step's wire bytes. Run against a TLS cluster
/// so the deltas are real ciphertext.
#[tokio::test]
async fn submitted_carries_per_round_metrics() {
    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr,
            internal_addr,
            handle,
        } = serve_shareholder(tls_shareholder_cfg(x)).await.unwrap();
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("https://{agg_addr}");

    agg_handle
        .create_session(training_spec(1, 1))
        .expect("the session under test must be created");

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

    let mut submitted = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        assert!(
            Instant::now() < deadline,
            "session did not complete in time"
        );
        match session.step().await.expect("step must not error") {
            StepOutcome::Submitted {
                round_id,
                session_id,
                round_kind,
                compute_us,
                poll_us,
                submit_us,
                tx_bytes,
                rx_bytes,
                below_threshold,
                report,
                n_records: _,
            } => {
                assert!(
                    !session_id.is_empty(),
                    "every submitted round carries the aggregator session id"
                );
                submitted.push((
                    round_id,
                    round_kind,
                    compute_us,
                    poll_us,
                    submit_us,
                    tx_bytes,
                    rx_bytes,
                    below_threshold,
                    report,
                ));
            }
            StepOutcome::NothingNew { next_poll_after, .. } => {
                tokio::time::sleep(next_poll_after).await;
            }
            StepOutcome::Completed => break,
            StepOutcome::Failed => panic!("aggregator session failed"),
        }
    }

    assert!(
        !submitted.is_empty(),
        "expected at least the stats + one gradient round"
    );

    // The first Submitted is the stats round (round_id 1).
    let (round_id, round_kind, compute_us, _poll_us, _submit_us, ..) = &submitted[0];
    assert_eq!(*round_id, 1, "the first submitted round is stats");
    assert_eq!(*round_kind, RoundKind::Stats);
    assert!(
        *compute_us > 0,
        "share computation must take some microseconds"
    );

    // Every round moved real ciphertext and cleared the 2-of-3 threshold.
    for (_id, _kind, compute_us, poll_us, _submit_us, tx_bytes, rx_bytes, below, _report) in
        &submitted
    {
        assert!(*compute_us > 0, "compute_us must be measured per round");
        assert!(*tx_bytes > 0, "the step must have written ciphertext");
        assert!(*rx_bytes > 0, "the step must have read ciphertext");
        assert!(
            *poll_us < 30_000_000,
            "poll_us must be a sane measured duration, not the RPC deadline"
        );
        assert!(
            !*below,
            "a 2-of-3 delivery to three live shareholders is not below threshold"
        );
    }

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// A device contributing several records as one batch client submits one
/// commitment per round, not one per record: what makes a phone's per-round
/// message size independent of how many records it holds.
#[tokio::test(flavor = "multi_thread")]
async fn a_batch_client_submits_one_contribution_per_round() {
    // With a single batch client, each shareholder's own min_clients gate must
    // also be 1, or it withholds sums awaiting commitments that never come.
    let solo_shareholder_cfg = |x: u64| ShareholderConfig {
        x_coord: x,
        min_clients: 1,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: None,
    };

    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: addr,
            internal_addr: internal,
            handle: h,
        } = serve_shareholder(solo_shareholder_cfg(x)).await.unwrap();
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
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(1, 1))
        .expect("the session under test must be created");

    let records: Vec<(Vec<f64>, f64)> = vec![
        (vec![1.0, 2.0], 1.0),
        (vec![3.0, 4.0], 0.0),
        (vec![5.0, 6.0], 1.0),
    ];

    let mut submitted = 0usize;
    let mut n_records_seen = 0u32;
    run_to_completion(
        SessionParams {
            agg_endpoint: agg_url,
            shareholder_endpoints: Some(client_eps),
            session_id: None,
            token,
            records: records.clone(),
            threshold: Some(THRESHOLD),
            hide_path: true,
            ca_pem: None,
        },
        |outcome| {
            if let StepOutcome::Submitted { n_records, .. } = outcome {
                submitted += 1;
                n_records_seen = *n_records;
            }
        },
    )
    .await
    .expect("a batch client completes a session");

    assert!(submitted >= 1, "at least the stats round is submitted");
    assert_eq!(
        n_records_seen, 3,
        "the summary reports the device's batch size, not a per-record index"
    );

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// The anonymity floor counts distinct contributors, and one client holding
/// many records counts as one. If a device's records counted individually, a
/// single phone could satisfy a two-client floor alone and the guarantee would
/// be void.
///
/// Shareholders enforce `min_clients: 2` while the aggregator's own coarser
/// gate stays at 1, so its round loop closes on the lone client and the
/// assertion isolates the shareholders' floor. The gather then asks each
/// shareholder for sums over the one commitment this batch client submitted;
/// every shareholder refuses with `InsufficientClients`, `define_bins` gets
/// nothing usable, and the session fails. A hard error, not a timeout: no
/// retry exists at this layer, so it fails within about one poll cycle.
#[tokio::test]
async fn a_multi_record_batch_client_does_not_satisfy_a_two_client_floor() {
    let sh_cfg = |x: u64| ShareholderConfig {
        x_coord: x,
        min_clients: 2,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: None,
    };

    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: addr,
            internal_addr: internal,
            handle: h,
        } = serve_shareholder(sh_cfg(x)).await.unwrap();
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
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    // The lone client satisfies the aggregator's gate, so what must refuse the
    // single commitment is the shareholders' min_clients=2.
    agg_handle
        .create_session(training_spec(1, 1))
        .expect("the session under test must be created");

    let records: Vec<(Vec<f64>, f64)> = vec![
        (vec![1.0, 2.0], 1.0),
        (vec![3.0, 4.0], 0.0),
        (vec![5.0, 6.0], 1.0),
        (vec![7.0, 8.0], 0.0),
        (vec![9.0, 10.0], 1.0),
    ];

    let result = tokio::time::timeout(
        Duration::from_secs(30),
        run_to_completion(
            SessionParams {
                agg_endpoint: agg_url,
                shareholder_endpoints: Some(client_eps),
                token,
                records,
                threshold: Some(THRESHOLD),
                hide_path: true,
                ca_pem: None,
                session_id: None,
            },
            |_| {},
        ),
    )
    .await
    .expect("the session must resolve, not hang past the timeout");

    let err = match result {
        Ok(()) => panic!(
            "one batch client, however many records it carries, must not satisfy a \
             two-client anonymity floor, but the session completed"
        ),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("aggregator session failed"),
        "must fail because the aggregator's session failed once the shareholders refused \
         to release sums for a single commitment against min_clients=2, not some other error: {err}"
    );

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// The counterpart to the test above: two distinct clients, one a multi-record
/// batch, do satisfy `min_clients: 2`, so the round releases. The pair pins the
/// floor from both sides: it counts clients, not records, and is satisfiable.
#[tokio::test]
async fn two_clients_one_multi_record_satisfy_a_two_client_floor() {
    let sh_cfg = |x: u64| ShareholderConfig {
        x_coord: x,
        min_clients: 2,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: None,
    };

    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: addr,
            internal_addr: internal,
            handle: h,
        } = serve_shareholder(sh_cfg(x)).await.unwrap();
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
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(2, 2))
        .expect("the session under test must be created");

    let multi_record: Vec<(Vec<f64>, f64)> = vec![
        (vec![1.0, 2.0], 1.0),
        (vec![3.0, 4.0], 0.0),
        (vec![5.0, 6.0], 1.0),
        (vec![7.0, 8.0], 0.0),
        (vec![9.0, 10.0], 1.0),
    ];
    let single_record: Vec<(Vec<f64>, f64)> = vec![(vec![2.0, 1.0], 0.0)];

    let mut tasks = Vec::new();
    for records in [multi_record, single_record] {
        let agg_url = agg_url.clone();
        let client_eps = client_eps.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: agg_url,
                    shareholder_endpoints: Some(client_eps),
                    token,
                    records,
                    threshold: Some(THRESHOLD),
                    hide_path: true,
                    ca_pem: None,
                    session_id: None,
                },
                |_| {},
            )
            .await
        }));
    }

    let deadline = Instant::now() + Duration::from_secs(60);
    for task in tasks {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let outcome = tokio::time::timeout(remaining, task)
            .await
            .expect("client did not finish in time")
            .expect("client task panicked");
        outcome.expect(
            "two distinct clients (one of them multi-record) must satisfy a two-client \
             floor and complete the session",
        );
    }

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// `run_to_completion`'s callback fires on step outcomes and the session still
/// completes.
#[tokio::test]
async fn run_to_completion_invokes_progress_callback() {
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
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(N_CLIENTS, N_CLIENTS))
        .expect("the session under test must be created");

    let mut rows = synthetic_rows();
    let (my_features, my_label) = rows.remove(0);

    let mut tasks = Vec::new();
    for (features, label) in rows {
        let agg_url = agg_url.clone();
        let client_eps = client_eps.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: agg_url,
                    shareholder_endpoints: Some(client_eps),
                    token,
                    records: vec![(features, label)],
                    threshold: Some(THRESHOLD),
                    hide_path: true,
                    ca_pem: None,
                    session_id: None,
                },
                |_| {},
            )
            .await
        }));
    }

    let progress_count = Arc::new(AtomicUsize::new(0));
    let counter = progress_count.clone();
    run_to_completion(
        SessionParams {
            agg_endpoint: agg_url,
            shareholder_endpoints: Some(client_eps),
            token,
            records: vec![(my_features, my_label)],
            threshold: Some(THRESHOLD),
            hide_path: true,
            ca_pem: None,
            session_id: None,
        },
        move |_outcome: &StepOutcome| {
            counter.fetch_add(1, Ordering::SeqCst);
        },
    )
    .await
    .expect("run_to_completion should complete");

    assert!(
        progress_count.load(Ordering::SeqCst) > 0,
        "progress callback should have been invoked at least once"
    );

    for task in tasks {
        task.await
            .expect("client task panicked")
            .expect("background client should complete");
    }

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn run_collecting_reports_session_wire_totals() {
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
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(N_CLIENTS, N_CLIENTS))
        .expect("the session under test must be created");

    let mut rows = synthetic_rows();
    let (my_features, my_label) = rows.remove(0);

    let mut tasks = Vec::new();
    for (features, label) in rows {
        let agg_url = agg_url.clone();
        let client_eps = client_eps.clone();
        let token = token.clone();
        tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: agg_url,
                    shareholder_endpoints: Some(client_eps),
                    token,
                    records: vec![(features, label)],
                    threshold: Some(THRESHOLD),
                    hide_path: true,
                    ca_pem: None,
                    session_id: None,
                },
                |_| {},
            )
            .await
        }));
    }

    let run = run_collecting(
        SessionParams {
            agg_endpoint: agg_url,
            shareholder_endpoints: Some(client_eps),
            token,
            records: vec![(my_features, my_label)],
            threshold: Some(THRESHOLD),
            hide_path: true,
            ca_pem: None,
            session_id: None,
        },
        |_| {},
    )
    .await
    .expect("run_collecting should complete");

    // A completed session polled and submitted at least the stats round.
    assert!(run.n_rounds >= 1, "expected >=1 submitted round");
    assert!(run.submit_tx > 0 && run.submit_rx > 0, "submit deltas must be non-zero");
    // The session total spans handshakes and polls too, so it is the larger.
    assert!(run.total_tx >= run.submit_tx, "total tx must include handshakes/polls");
    assert!(run.total_rx >= run.submit_rx, "total rx must include handshakes/polls");

    for task in tasks {
        task.await
            .expect("client task panicked")
            .expect("background client should complete");
    }

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// `ClientSession::enroll` rejects an empty batch before any network I/O:
/// `Client::new_batch` panics on one, and a panic crossing the Flutter bridge
/// is worse than a returned error. Pointing at a dead address proves the guard
/// runs first, since otherwise this would fail with a connection error.
#[tokio::test]
async fn enroll_rejects_empty_records_before_any_network_io() {
    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let result = ClientSession::enroll(SessionParams {
        agg_endpoint: "http://127.0.0.1:1".to_string(),
        shareholder_endpoints: None,
        token,
        records: vec![],
        threshold: None,
        hide_path: true,
        ca_pem: None,
        session_id: None,
    })
    .await;
    let err = match result {
        Ok(_) => panic!("an empty records batch must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("at least one record"),
        "must fail with the empty-records guard's message, not a connection error: {err}"
    );
}

/// `ClientSession::enroll` rejects a ragged batch before any network I/O: it
/// indexes out of bounds inside `pbr-core`'s share computation, and every
/// bridge entry point is documented infallible. The dead address proves the
/// guard runs first.
#[tokio::test]
async fn enroll_rejects_ragged_records_before_any_network_io() {
    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let result = ClientSession::enroll(SessionParams {
        agg_endpoint: "http://127.0.0.1:1".to_string(),
        shareholder_endpoints: None,
        token,
        records: vec![
            (vec![1.0, 2.0], 0.0),
            (vec![1.0, 2.0], 1.0),
            (vec![1.0], 1.0),
        ],
        threshold: None,
        hide_path: true,
        ca_pem: None,
        session_id: None,
    })
    .await;
    let err = match result {
        Ok(_) => panic!("a ragged records batch must be rejected"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("record 2"),
        "must name the offending index: {msg}"
    );
    assert!(
        msg.contains("has 1 features") && msg.contains("record 0 has 2 features"),
        "must name both widths: {msg}"
    );
}

/// Answers `EnrollSession` with an empty `session_id`, a malformed
/// response no real aggregator (its session store always assigns a concrete
/// id) produces, standing in for one so the client's guard against it can be
/// exercised. Every other method is unreachable: `ClientSession::enroll`
/// rejects the empty id before ever calling them.
struct EmptySessionIdAggregator;

#[tonic::async_trait]
impl AggregatorService for EmptySessionIdAggregator {
    async fn enroll_session(
        &self,
        _req: Request<EnrollRequest>,
    ) -> Result<Response<SessionConfig>, Status> {
        Ok(Response::new(SessionConfig {
            bin_edges: Vec::new(),
            phase: SessionPhase::StatsPending as i32,
            n_features: 0,
            session_id: String::new(),
            threshold: 1,
            n_parties: 1,
            shareholder_endpoints: vec!["http://127.0.0.1:1".to_string()],
        }))
    }

    async fn poll_session(
        &self,
        _req: Request<PollSessionRequest>,
    ) -> Result<Response<PollSessionResponse>, Status> {
        unimplemented!("enroll must reject the empty session_id before any poll")
    }

    async fn register_device(
        &self,
        _req: Request<RegisterDeviceRequest>,
    ) -> Result<Response<Ack>, Status> {
        unimplemented!("not exercised by this test")
    }

    async fn list_sessions(
        &self,
        _req: Request<ListSessionsRequest>,
    ) -> Result<Response<SessionList>, Status> {
        unimplemented!("not exercised by this test")
    }
}

/// `ClientSession::enroll` rejects an empty `session_id` in the server's
/// `EnrollSession` response: unlike the records guards above, this one
/// depends on a round trip actually completing, so it needs a server that
/// will produce that malformed response rather than a dead address.
#[tokio::test]
async fn enroll_rejects_empty_session_id_from_server() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(AggregatorServiceServer::new(EmptySessionIdAggregator))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                async {
                    let _ = rx.await;
                },
            ),
    );

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let result = ClientSession::enroll(SessionParams {
        agg_endpoint: format!("http://{addr}"),
        shareholder_endpoints: None,
        token,
        records: vec![(vec![1.0, 2.0], 0.0)],
        threshold: None,
        hide_path: true,
        ca_pem: None,
        session_id: None,
    })
    .await;
    let err = match result {
        Ok(_) => panic!("an empty session_id from the server must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("empty"),
        "must fail with the empty-session_id guard's message, not some other error: {err}"
    );

    let _ = tx.send(());
    let _ = server.await;
}

/// `pbr_client::driver::list_sessions` maps the wire `SessionSummary` to the
/// driver's own type field-by-field; neither `admin.rs` nor
/// `aggregator_loop.rs` in `pbr-server` exercises that mapping, since both
/// call `AggregatorServiceClient` directly. This is the only test that would
/// catch `dataset_id` being dropped or mismapped there: a dataset-less
/// session's must come through empty, and an admin-created session's must
/// echo the dataset it was created for.
#[tokio::test]
async fn list_sessions_carries_dataset_id_through_the_driver() {
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

    const ADMIN_TOKEN: &str = "test-admin-token";
    let RunningAggregator {
        addr: agg_addr,
        handle: agg_handle,
    } = serve_aggregator(AggregatorConfig {
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_shareholder_endpoints: internal_eps,
        client_shareholder_endpoints: client_eps,
        threshold: THRESHOLD,
        auth: auth_cfg(),
        fcm: None,
        tls: None,
        datasets: DatasetTable::from_pairs(vec![("heart_disease".into(), 2)]),
        admin_token: Some(ADMIN_TOKEN.into()),
        state_path: ":memory:".into(),
        eval: None,
    })
    .await
    .unwrap();
    let agg_url = format!("http://{agg_addr}");
    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();

    // A dataset-less session, so the empty `dataset_id` case is exercised
    // through the driver mapping before the admin-created one below.
    agg_handle
        .create_session(training_spec(N_CLIENTS, N_CLIENTS))
        .expect("the dataset-less session must be created");

    let before = pbr_client::driver::list_sessions(&agg_url, &token, None)
        .await
        .expect("list_sessions should succeed");
    assert_eq!(before.len(), 1, "only the dataset-less session is hosted yet");
    assert_eq!(
        before[0].dataset_id, "",
        "a dataset-less session names no dataset"
    );

    let mut admin = AdminServiceClient::connect(agg_url.clone()).await.unwrap();
    let mut req = Request::new(CreateSessionRequest {
        dataset_id: "heart_disease".into(),
        title: "t".into(),
        n_trees: 1,
        max_depth: 1,
        n_bins: 4,
        learning_rate: 0.1,
        lambda: 1.0,
        min_clients: 1,
        target_clients: 1,
        submission_window_ms: 2_000,
    });
    req.metadata_mut()
        .insert("authorization", bearer(ADMIN_TOKEN));
    let created = admin.create_session(req).await.unwrap().into_inner();

    let after = pbr_client::driver::list_sessions(&agg_url, &token, None)
        .await
        .expect("list_sessions should succeed");
    let summary = after
        .iter()
        .find(|s| s.session_id == created.session_id)
        .expect("the admin-created session must be listed");
    assert_eq!(summary.dataset_id, "heart_disease");

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}

/// A wake-driven caller advances a session one round at a time, persisting the
/// round watermark between wakes. Resuming from a stored watermark must not
/// re-submit a round the device already contributed to: the aggregator counts
/// distinct commitments, so a duplicate submission would inflate its client
/// count for that round.
#[tokio::test(flavor = "multi_thread")]
async fn a_resumed_session_does_not_repeat_a_round_it_already_submitted() {
    // A single batch client, so both the shareholders' own min_clients gate
    // and the aggregator's must be 1, or the round loop withholds sums/gather
    // waiting for a second contributor that will never arrive.
    let solo_shareholder_cfg = |x: u64| ShareholderConfig {
        x_coord: x,
        min_clients: 1,
        listen: "127.0.0.1:0".parse().unwrap(),
        internal_listen: "127.0.0.1:0".parse().unwrap(),
        auth: auth_cfg(),
        tls: None,
    };

    let mut client_eps = Vec::new();
    let mut internal_eps = Vec::new();
    let mut sh_handles = Vec::new();
    for x in 1..=3u64 {
        let RunningShareholder {
            client_addr: addr,
            internal_addr: internal,
            handle: h,
        } = serve_shareholder(solo_shareholder_cfg(x)).await.unwrap();
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
        client_shareholder_endpoints: client_eps.clone(),
        threshold: THRESHOLD,
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

    let token = mint(ISS, AUD, KID, "test-driver", 300, PRIV).unwrap();
    let agg_url = format!("http://{agg_addr}");

    agg_handle
        .create_session(training_spec(1, 1))
        .expect("the session under test must be created");

    let params = || SessionParams {
        agg_endpoint: agg_url.clone(),
        shareholder_endpoints: Some(client_eps.clone()),
        token: token.clone(),
        records: vec![(vec![2.0, 1.0], 1.0)],
        threshold: Some(THRESHOLD),
        hide_path: true,
        ca_pem: None,
        session_id: None,
    };

    // First wake: enroll fresh, then step until the round is actually
    // submitted, and remember the watermark. The aggregator's round loop
    // opens round 1 in a task spawned separately from where the server
    // starts accepting connections, so an early poll can legitimately see
    // NothingNew before that round is published; only a hang (not a single
    // NothingNew) is a failure here.
    let mut s1 = ClientSession::enroll(params()).await.unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let first_round = loop {
        assert!(
            Instant::now() < deadline,
            "first session did not submit a round in time"
        );
        match s1.step().await.unwrap() {
            StepOutcome::Submitted { round_id, .. } => break round_id,
            StepOutcome::NothingNew { next_poll_after, .. } => {
                tokio::time::sleep(next_poll_after).await;
            }
            other => panic!("expected a submitted round, got {other:?}"),
        }
    };
    let watermark = s1.last_seen();
    assert_eq!(watermark, first_round);

    // Second wake: a fresh session resumed at the watermark must not be handed
    // the same round again.
    let mut s2 = ClientSession::enroll_at(params(), watermark).await.unwrap();
    match s2.step().await.unwrap() {
        StepOutcome::Submitted { round_id, .. } => assert_ne!(
            round_id, first_round,
            "a resumed session must not re-submit the round its watermark covers"
        ),
        StepOutcome::NothingNew { .. } | StepOutcome::Completed => {}
        StepOutcome::Failed => panic!("session failed"),
    }

    agg_handle.shutdown();
    for h in sh_handles {
        h.shutdown();
    }
}
