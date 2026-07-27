//! Headless four-process end-to-end: three shareholder processes and an
//! aggregator, spawned from the committed `deploy/local/*.toml` configs
//! rendered onto freshly reserved ports, training the heart_disease reference
//! configuration (15 trees, depth 3, lr 0.15, lambda 2.0, 10 bins, 2-of-3
//! sharing) over real localhost gRPC.
//!
//! The aggregator boots hosting no session, so this test schedules the
//! reference session over the admin plane once the cluster is up, exactly as
//! an operator would.
//!
//! Clients run in-process through the production `run_to_completion`, so the
//! cross-process surfaces are the deployed ones: clients to aggregator,
//! clients to shareholders, and aggregator to shareholders. Both tests below
//! run the same session over the same data and differ only in how the train
//! split is packed into clients: 237 single-record clients, or the three-device
//! fleet shape the phones actually run.
//!
//! The final model arrives the way any client's does: one read-only poll at
//! `COMPLETED` carries the `ModelProto`. Scoring is on the held-out 20% split,
//! the same split `pbr-core`'s own heart_disease test uses, gated at mean
//! AUC >= 0.80. That sits below the ~0.88 the configuration reaches
//! single-process by just enough to absorb secure-aggregation noise, and not
//! enough to pass without genuine training.
//!
//! `#[ignore]` by default: spawning four release-built processes is heavy.
//! Run explicitly with `cargo test -p pbr-e2e --release -- --ignored`.

use pbr_client::driver::{SessionParams, run_to_completion};
use pbr_client::jwt::mint;
use pbr_core::{auc_roc, read_csv};
use pbr_proto::convert::model_from_proto;
use pbr_proto::v1::admin_service_client::AdminServiceClient;
use pbr_proto::v1::aggregator_service_client::AggregatorServiceClient;
use pbr_proto::v1::poll_session_response::Body;
use pbr_proto::v1::{CreateSessionRequest, PollSessionRequest, SessionPhase};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tonic::Request;

/// One training row: features and label.
type Record = (Vec<f64>, f64);

const THRESHOLD: usize = 2;
const N_TREES: usize = 15;

// The heart_disease reference session's remaining hyperparameters. The
// aggregator boots hosting no session, so this test schedules the session
// itself over the admin plane; these values live here rather than in
// `deploy/local/aggregator.toml`.
const MAX_DEPTH: u32 = 3;
const N_BINS: u32 = 10;
const LEARNING_RATE: f64 = 0.15;
const LAMBDA: f64 = 2.0;
const SUBMISSION_WINDOW_MS: u64 = 5_000;
const MIN_CLIENTS: u32 = 10;
const TARGET_CLIENTS: u32 = 237;

/// The dev admin bearer token `deploy/local/aggregator.toml` commits, for a
/// loopback cluster that already uses committed dev keys.
const ADMIN_TOKEN: &str = "dev-admin-token";

// Dev issuer; must match the [auth] blocks in deploy/local/*.toml.
const ISS: &str = "https://test-issuer.local";
const AUD: &str = "pbr";
const KID: &str = "test-1";

/// Whole-session budget. Healthy rounds early-close once all commitments land
/// everywhere, a second or two each, so this only bounds a wedged run.
const SESSION_DEADLINE: Duration = Duration::from_secs(600);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/pbr-e2e has a repo root two levels up")
        .to_path_buf()
}

fn target_dir(root: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"))
}

/// Build the release `pbr-server` and `pbr-client` binaries, returning the
/// server path. The client binary is built alongside so the run also proves
/// the released CLI compiles, though rows are driven through the driver
/// library; it sits behind pbr-client's `cli` feature, which the build must
/// select explicitly for `--bin pbr-client` to resolve.
fn build_release_binaries(root: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--bin",
            "pbr-server",
            "--bin",
            "pbr-client",
            "--features",
            "pbr-client/cli",
        ])
        .current_dir(root)
        .status()
        .expect("cargo is available");
    assert!(
        status.success(),
        "release build of pbr-server/pbr-client failed"
    );
    target_dir(root).join("release/pbr-server")
}

/// Bind a loopback listener on an OS-assigned port and return both. The
/// listener stays alive to hold the port; the caller drops every reservation
/// together just before spawning the servers, keeping the window in which the
/// OS could reassign a port as small as possible.
fn reserve_port() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve a free loopback port");
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

/// The loopback ports a rendered cluster binds: the aggregator's listener,
/// each shareholder's client-facing listener, and each shareholder's internal
/// gather listener. Index *i* of both arrays belongs to the shareholder at
/// Shamir evaluation point x = *i* + 1.
struct ClusterPorts {
    agg: u16,
    client: [u16; 3],
    internal: [u16; 3],
}

/// Parse a committed deploy config.
fn read_config(path: &Path) -> toml::Table {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Overwrite an existing top-level key, panicking rather than inserting a new
/// one. Every key rewritten here exists to displace a committed value, so a
/// rename in `deploy/local/*.toml` must fail loudly instead of silently
/// leaving a server on the committed port, session store, or anonymity floor.
fn set(cfg: &mut toml::Table, key: &str, value: impl Into<toml::Value>) {
    let slot = cfg.get_mut(key).unwrap_or_else(|| {
        panic!("deploy/local config must still carry `{key}` for this test to rewrite it")
    });
    *slot = value.into();
}

fn write_config(dir: &Path, name: &str, cfg: &toml::Table) -> PathBuf {
    let path = dir.join(name);
    let text = toml::to_string(cfg).unwrap_or_else(|e| panic!("serialize {name}: {e}"));
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    path
}

/// Render the committed `deploy/local/*.toml` files into `out_dir` onto
/// `ports`, with the shareholders' anonymity floor set to `min_clients` and
/// the aggregator's session store redirected into `out_dir`. Returns the
/// aggregator config and the three shareholder configs in x = 1, 2, 3 order.
///
/// The committed files stay the source of truth and everything else is carried
/// over untouched, key paths included: those are relative to the repo root,
/// which the servers are spawned in. The store has to move because the
/// committed path is shared, and a run that inherited a previous run's
/// sessions could resolve the wrong one.
fn render_configs(
    root: &Path,
    out_dir: &Path,
    ports: &ClusterPorts,
    min_clients: usize,
) -> (PathBuf, [PathBuf; 3]) {
    let url = |port: &u16| toml::Value::from(format!("http://127.0.0.1:{port}"));

    let mut agg = read_config(&root.join("deploy/local/aggregator.toml"));
    set(&mut agg, "listen", format!("127.0.0.1:{}", ports.agg));
    set(
        &mut agg,
        "internal_shareholder_endpoints",
        ports.internal.iter().map(url).collect::<Vec<_>>(),
    );
    set(
        &mut agg,
        "client_shareholder_endpoints",
        ports.client.iter().map(url).collect::<Vec<_>>(),
    );
    set(
        &mut agg,
        "state_path",
        out_dir.join("sessions.sqlite").display().to_string(),
    );
    let agg_path = write_config(out_dir, "aggregator.toml", &agg);

    let shareholders = std::array::from_fn(|i| {
        let name = format!("shareholder-{}.toml", i + 1);
        let mut cfg = read_config(&root.join("deploy/local").join(&name));
        set(&mut cfg, "listen", format!("127.0.0.1:{}", ports.client[i]));
        set(
            &mut cfg,
            "internal_listen",
            format!("127.0.0.1:{}", ports.internal[i]),
        );
        set(&mut cfg, "min_clients", min_clients as i64);
        write_config(out_dir, &name, &cfg)
    });

    (agg_path, shareholders)
}

/// Kills the spawned server when the test ends, panic included, so a failed
/// assertion cannot leak a four-process cluster.
struct ChildGuard {
    name: &'static str,
    child: Child,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_server(
    server_bin: &Path,
    root: &Path,
    log_dir: &Path,
    name: &'static str,
    role: &str,
    config: &Path,
) -> ChildGuard {
    let stdout = std::fs::File::create(log_dir.join(format!("{name}.stdout.log"))).unwrap();
    let stderr = std::fs::File::create(log_dir.join(format!("{name}.stderr.log"))).unwrap();
    let child = Command::new(server_bin)
        .arg("--role")
        .arg(role)
        .arg("--config")
        .arg(config)
        // The deploy configs resolve key paths relative to the repo root.
        .current_dir(root)
        .env("RUST_LOG", "info")
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {name}: {e}"));
    ChildGuard { name, child }
}

/// The tail of every server log, printed when a cluster is dropped while
/// panicking so a red run carries its own diagnosis.
fn server_logs(log_dir: &Path) -> String {
    let mut out = String::new();
    let Ok(entries) = std::fs::read_dir(log_dir) else {
        return out;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        let tail: Vec<&str> = content.lines().rev().take(25).collect();
        out.push_str(&format!(
            "\n---- {} (last {} lines) ----\n",
            path.display(),
            tail.len()
        ));
        for line in tail.into_iter().rev() {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// A live four-process cluster and the URLs a client needs to reach it.
///
/// Dropping it kills every server. A drop while panicking also prints the tail
/// of each server log and leaves the run directory behind, so a failure can be
/// diagnosed from the test output and the ports it used inspected afterwards;
/// a clean drop removes the directory.
struct Cluster {
    agg_url: String,
    shareholder_urls: Vec<String>,
    run_dir: PathBuf,
    log_dir: PathBuf,
    guards: Vec<ChildGuard>,
}

impl Cluster {
    /// Build the binaries, render the deploy configs onto fresh loopback
    /// ports, spawn all four servers, and return once every listener accepts.
    ///
    /// `tag` separates this cluster's run directory from the other test's, so
    /// the two can run concurrently under plain `cargo test` as well as
    /// serially under nextest. `min_clients` displaces the committed anonymity
    /// floor and must match the floors in the `CreateSession` request, or a
    /// round closes with fewer commitments than a shareholder will
    /// reconstruct from.
    fn start(tag: &str, min_clients: usize) -> Cluster {
        let root = repo_root();
        let server_bin = build_release_binaries(&root);

        let run_dir = target_dir(&root).join(format!("e2e-{tag}-{}", std::process::id()));
        let log_dir = run_dir.join("logs");
        let config_dir = run_dir.join("configs");
        std::fs::create_dir_all(&log_dir).unwrap();
        std::fs::create_dir_all(&config_dir).unwrap();

        // Fresh loopback ports, so a stale demo cluster or a second e2e run on
        // the committed 42800-42803/42811-42813 cannot make these fail to bind.
        let reservations: Vec<(TcpListener, u16)> = (0..7).map(|_| reserve_port()).collect();
        let ports = ClusterPorts {
            agg: reservations[0].1,
            client: [reservations[1].1, reservations[2].1, reservations[3].1],
            internal: [reservations[4].1, reservations[5].1, reservations[6].1],
        };
        let (agg_cfg, sh_cfgs) = render_configs(&root, &config_dir, &ports, min_clients);

        // Release the reserved ports immediately before spawning so the
        // servers can bind them.
        drop(reservations);

        let names = ["shareholder-1", "shareholder-2", "shareholder-3"];
        let mut guards: Vec<ChildGuard> = names
            .iter()
            .zip(&sh_cfgs)
            .map(|(name, cfg)| spawn_server(&server_bin, &root, &log_dir, name, "shareholder", cfg))
            .collect();
        guards.push(spawn_server(
            &server_bin,
            &root,
            &log_dir,
            "aggregator",
            "aggregator",
            &agg_cfg,
        ));

        let mut cluster = Cluster {
            agg_url: format!("http://127.0.0.1:{}", ports.agg),
            shareholder_urls: ports
                .client
                .iter()
                .map(|p| format!("http://127.0.0.1:{p}"))
                .collect(),
            run_dir,
            log_dir,
            guards,
        };
        let addrs: Vec<String> = ports
            .client
            .iter()
            .chain(std::iter::once(&ports.agg))
            .map(|p| format!("127.0.0.1:{p}"))
            .collect();
        cluster.wait_for_listeners(&addrs);
        cluster
    }

    fn wait_for_listeners(&mut self, addrs: &[String]) {
        let deadline = Instant::now() + Duration::from_secs(30);
        for addr in addrs {
            loop {
                // A server that already died will never open its port; fail
                // now instead of spinning out the full 30 s.
                for g in self.guards.iter_mut() {
                    if let Ok(Some(status)) = g.child.try_wait() {
                        panic!(
                            "{} exited early ({status}) before the cluster was up",
                            g.name
                        );
                    }
                }
                match TcpStream::connect(addr) {
                    Ok(_) => break,
                    Err(_) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(100))
                    }
                    Err(e) => panic!("{addr} never came up: {e}"),
                }
            }
        }
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        // Kill the servers before reading their logs, so whatever they
        // buffered at the moment of failure has been flushed.
        self.guards.clear();
        if std::thread::panicking() {
            eprintln!(
                "cluster left at {}{}",
                self.run_dir.display(),
                server_logs(&self.log_dir)
            );
        } else {
            let _ = std::fs::remove_dir_all(&self.run_dir);
        }
    }
}

/// The heart_disease 80/20 split, the same split rule `pbr-core`'s own
/// heart_disease integration test uses (pre-shuffled CSV, first 80% train).
///
/// Reads `pbr-core`'s copy, not `data/heart_disease.csv`: the two hold the same
/// 297 rows in different order, and only the pre-shuffled one makes a split by
/// row position representative. Pointing this at `data/` instead, or collapsing
/// the two files into one, leaves the row count intact and the held-out 20%
/// silently unrepresentative.
struct Split {
    train: Vec<Record>,
    test_features: Vec<Vec<f64>>,
    test_targets: Vec<f64>,
}

impl Split {
    fn load() -> Split {
        let csv =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../pbr-core/tests/data/heart_disease.csv");
        let dataset = read_csv(&csv, "target").expect("read heart_disease.csv");
        let split_idx = (dataset.features.len() as f64 * 0.8) as usize;
        let split = Split {
            train: dataset.features[..split_idx]
                .iter()
                .cloned()
                .zip(dataset.targets[..split_idx].iter().copied())
                .collect(),
            test_features: dataset.features[split_idx..].to_vec(),
            test_targets: dataset.targets[split_idx..].to_vec(),
        };
        assert_eq!(
            split.train.len(),
            237,
            "the heart_disease 80% train split (crates/pbr-core/tests/data/heart_disease.csv) must have exactly 237 rows"
        );
        split
    }

    /// The train split as `n` equal contiguous batches, one per client.
    fn batches(&self, n: usize) -> Vec<Vec<Record>> {
        assert_eq!(
            self.train.len() % n,
            0,
            "{} train rows must split into {n} equal batches",
            self.train.len()
        );
        self.train
            .chunks(self.train.len() / n)
            .map(<[_]>::to_vec)
            .collect()
    }
}

/// Schedule the reference session over the admin plane, as an operator would;
/// the aggregator boots hosting none, so this is the only way it gets one. The
/// floors are parameters: the 237-record run uses the full-scale values, the
/// three-batch-client run closes on 3.
async fn create_heart_disease_session(
    agg_url: &str,
    min_clients: u32,
    target_clients: u32,
) -> String {
    let mut admin = AdminServiceClient::connect(agg_url.to_string())
        .await
        .unwrap_or_else(|e| panic!("connect to admin plane at {agg_url}: {e}"));
    let mut req = Request::new(CreateSessionRequest {
        dataset_id: "heart_disease".into(),
        title: "pbr-e2e heart_disease".into(),
        n_trees: N_TREES as u32,
        max_depth: MAX_DEPTH,
        n_bins: N_BINS,
        learning_rate: LEARNING_RATE,
        lambda: LAMBDA,
        min_clients,
        target_clients,
        submission_window_ms: SUBMISSION_WINDOW_MS,
    });
    req.metadata_mut().insert(
        "authorization",
        format!("Bearer {ADMIN_TOKEN}").parse().unwrap(),
    );
    admin
        .create_session(req)
        .await
        .unwrap_or_else(|e| panic!("CreateSession failed: {e}"))
        .into_inner()
        .session_id
}

/// Drive `batches` through the real gRPC surfaces, one client task each, then
/// take the final model the way a client does and gate it on held-out AUC.
async fn train_and_gate(
    cluster: &Cluster,
    session_id: &str,
    subject: &str,
    batches: Vec<Vec<Record>>,
    split: &Split,
) {
    let priv_pem =
        std::fs::read(repo_root().join("crates/pbr-server/tests/fixtures/test_key.pem")).unwrap();
    let token = mint(ISS, AUD, KID, subject, 3600, &priv_pem).unwrap();

    let n_clients = batches.len();
    let started = Instant::now();
    let mut tasks = Vec::with_capacity(n_clients);
    for records in batches {
        let urls = cluster.shareholder_urls.clone();
        let token = token.clone();
        let agg_url = cluster.agg_url.clone();
        tasks.push(tokio::spawn(async move {
            run_to_completion(
                SessionParams {
                    agg_endpoint: agg_url,
                    shareholder_endpoints: Some(urls),
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
    let deadline = started + SESSION_DEADLINE;
    for (i, task) in tasks.into_iter().enumerate() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let outcome = tokio::time::timeout(remaining, task)
            .await
            .unwrap_or_else(|_| {
                panic!("client {i} did not reach COMPLETED within {SESSION_DEADLINE:?}")
            })
            .expect("client task panicked");
        outcome.unwrap_or_else(|e| panic!("client {i} failed: {e:#}"));
    }
    println!(
        "all {n_clients} clients completed after {:?}",
        started.elapsed()
    );

    // The final model, exactly as a client obtains it: one PollSession at
    // COMPLETED returns the final RoundContext with the ModelProto.
    // Polled by the concrete id CreateSession returned: the empty selector
    // only resolves a sole *live* session and this one is now terminal.
    let mut agg = AggregatorServiceClient::connect(cluster.agg_url.clone())
        .await
        .unwrap();
    let mut req = Request::new(PollSessionRequest {
        last_seen_round_id: 0,
        session_id: session_id.to_string(),
    });
    req.metadata_mut()
        .insert("authorization", format!("Bearer {token}").parse().unwrap());
    let resp = agg.poll_session(req).await.unwrap().into_inner();
    assert_eq!(
        resp.phase(),
        SessionPhase::Completed,
        "session must be COMPLETED"
    );
    let Some(Body::Ctx(ctx)) = resp.body else {
        panic!("completed session must publish a final round context");
    };
    let model = model_from_proto(
        ctx.model
            .expect("completed context carries the final model"),
    )
    .unwrap();
    assert_eq!(
        model.trees.len(),
        N_TREES,
        "session must train all {N_TREES} trees"
    );

    // AUC gate on the held-out split. The reference configuration reaches
    // ~0.88 on this split single-process; 0.80 fails on any real integration
    // bug (round ordering, gather corruption, config drift) while absorbing
    // run-to-run wobble from which submissions land in each round's window.
    let predictions = model.predict(&split.test_features);
    let auc = auc_roc(&predictions, &split.test_targets);
    println!(
        "held-out AUC = {auc:.4} over {} rows ({n_clients} clients)",
        split.test_targets.len()
    );
    assert!(
        auc >= 0.80,
        "held-out AUC {auc:.4} below the 0.80 gate — this indicates a real \
         integration bug, not a flaky threshold"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "spawns a real 4-process cluster; run with: cargo test -p pbr-e2e --release -- --ignored"]
async fn four_process_cluster_trains_heart_disease_to_auc_gate() {
    let split = Split::load();
    let cluster = Cluster::start("single", MIN_CLIENTS as usize);
    let session_id =
        create_heart_disease_session(&cluster.agg_url, MIN_CLIENTS, TARGET_CLIENTS).await;
    let batches = split.batches(TARGET_CLIENTS as usize);
    train_and_gate(&cluster, &session_id, "e2e-heart-disease", batches, &split).await;
}

/// The fleet configuration: three devices, each holding a contiguous slice of
/// the train split as one batch client. This is the shape the phone fleet
/// actually runs, and it must reach the same model quality as 237
/// single-record clients: the aggregator sums the same gradients either way,
/// it just receives them in three messages instead of 237.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "spawns a real 4-process cluster; run with: cargo test -p pbr-e2e --release -- --ignored"]
async fn three_batch_clients_train_heart_disease_to_auc_gate() {
    let split = Split::load();
    // Three contributors, not 237: each shareholder's own anonymity floor and
    // the aggregator's round-close policy have to be lowered together.
    let cluster = Cluster::start("batch", 3);
    let session_id = create_heart_disease_session(&cluster.agg_url, 3, 3).await;
    let batches = split.batches(3);
    train_and_gate(
        &cluster,
        &session_id,
        "e2e-heart-disease-batch",
        batches,
        &split,
    )
    .await;
}
