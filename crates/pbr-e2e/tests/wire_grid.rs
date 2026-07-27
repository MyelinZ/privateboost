use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

/// Absolute CSV path for a dataset id (heart_disease lives under pbr-core tests).
fn dataset_csv(root: &Path, id: &str) -> PathBuf {
    if id == "heart_disease" {
        root.join("crates/pbr-core/tests/data/heart_disease.csv")
    } else {
        root.join(format!("data/{id}.csv"))
    }
}

/// (grower_batch = all rows but the last, measured_record = last row, n_features).
type DatasetSplit = (Vec<(Vec<f64>, f64)>, (Vec<f64>, f64), usize);

fn load_split(dataset_id: &str) -> DatasetSplit {
    let root = repo_root();
    let ds = pbr_core::read_csv(&dataset_csv(&root, dataset_id), "target")
        .unwrap_or_else(|e| panic!("read {dataset_id}: {e}"));
    let mut records: Vec<(Vec<f64>, f64)> = ds
        .features
        .iter()
        .cloned()
        .zip(ds.targets.iter().copied())
        .collect();
    let n_features = records[0].0.len();
    let measured = records.pop().expect("dataset non-empty");
    (records, measured, n_features)
}

#[test]
fn load_split_feature_counts_match_the_spec() {
    for (id, f) in [
        ("heart_disease", 13usize),
        ("pima_diabetes", 8),
        ("breast_cancer", 30),
        ("cdc_diabetes", 21),
    ] {
        let (grower, measured, n) = load_split(id);
        assert_eq!(n, f, "{id} feature count");
        assert_eq!(measured.0.len(), f, "{id} measured record width");
        assert!(!grower.is_empty(), "{id} grower non-empty");
        assert!(grower.len() >= 100, "{id} grower non-trivial");
    }
}

const TLS_CRT: &str = "crates/pbr-server/tests/fixtures/tls/server.crt";
const TLS_KEY: &str = "crates/pbr-server/tests/fixtures/tls/server.key";
const PUBKEY: &str = "crates/pbr-server/tests/fixtures/test_key.pub.pem";
const CA_CRT: &str = "crates/pbr-server/tests/fixtures/tls/ca.crt";
const PRIVKEY: &str = "crates/pbr-server/tests/fixtures/test_key.pem";

fn auth_block() -> String {
    format!(
        "[auth]\nissuer = \"https://test-issuer.local\"\naudience = \"pbr\"\n\
         static_keys = [{{ kid = \"test-1\", public_key_pem_path = \"{PUBKEY}\" }}]\n"
    )
}

fn tls_block() -> String {
    format!("[tls]\ncert_path = \"{TLS_CRT}\"\nkey_path = \"{TLS_KEY}\"\n")
}

/// Aggregator TOML: N shareholders on the given ports, cluster `threshold`,
/// all four datasets whitelisted, TLS on. `internal_ports` stay `http://`
/// (internal gather plane, plaintext loopback); `client_ports` are `https://`
/// since clients receive these endpoints verbatim via `EnrollSession`.
fn aggregator_toml(
    agg_port: u16,
    client_ports: &[u16],
    internal_ports: &[u16],
    threshold: usize,
) -> String {
    let internal = internal_ports
        .iter()
        .map(|p| format!("\"http://127.0.0.1:{p}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let client = client_ports
        .iter()
        .map(|p| format!("\"https://127.0.0.1:{p}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "listen = \"127.0.0.1:{agg_port}\"\n\
         internal_shareholder_endpoints = [{internal}]\n\
         client_shareholder_endpoints = [{client}]\n\
         threshold = {threshold}\n\
         admin_token = \"dev-admin-token\"\n\
         state_path = \":memory:\"\n\
         [datasets]\n\
         heart_disease = 13\npima_diabetes = 8\nbreast_cancer = 30\ncdc_diabetes = 21\n\
         {}{}",
        auth_block(),
        tls_block(),
    )
}

/// Shareholder TOML for Shamir point `x_coord`. `min_clients = 1` (functional
/// floor, not a privacy config, matches the emulator; per-client byte counts
/// are cohort-size independent). Internal listener stays plaintext loopback.
fn shareholder_toml(x_coord: usize, client_port: u16, internal_port: u16) -> String {
    format!(
        "x_coord = {x_coord}\nmin_clients = 1\n\
         listen = \"127.0.0.1:{client_port}\"\n\
         internal_listen = \"127.0.0.1:{internal_port}\"\n\
         {}{}",
        auth_block(),
        tls_block(),
    )
}

/// Bind a loopback listener on an OS-assigned port and return it with its
/// port. The listener is kept alive so the port stays reserved; the caller
/// drops all reservations together immediately before spawning the servers,
/// keeping the window in which the OS could reassign the port as small as
/// possible.
fn reserve_port() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve a free loopback port");
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

/// Kills the spawned server process when the test ends, including on
/// panic, so a failed assertion cannot leak a running cluster.
struct ChildGuard {
    name: String,
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
    name: String,
    role: &str,
    config: &str,
) -> ChildGuard {
    let stdout = std::fs::File::create(log_dir.join(format!("{name}.stdout.log"))).unwrap();
    let stderr = std::fs::File::create(log_dir.join(format!("{name}.stderr.log"))).unwrap();
    let child = Command::new(server_bin)
        .args(["--role", role, "--config", config])
        // The rendered configs resolve key paths relative to the repo root.
        .current_dir(root)
        .env("RUST_LOG", "info")
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {name}: {e}"));
    ChildGuard { name, child }
}

/// Last portion of every server log, appended to failure panics so a red
/// run carries its own diagnosis.
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

fn wait_for_listeners(addrs: &[String], guards: &mut [ChildGuard], log_dir: &Path) {
    let deadline = Instant::now() + Duration::from_secs(30);
    for addr in addrs {
        loop {
            // A server that already died will never open its port; surface
            // its logs instead of spinning out the full 30 s.
            for g in guards.iter_mut() {
                if let Ok(Some(status)) = g.child.try_wait() {
                    panic!(
                        "{} exited early ({status}) before the cluster was up{}",
                        g.name,
                        server_logs(log_dir)
                    );
                }
            }
            match TcpStream::connect(addr) {
                Ok(_) => break,
                Err(_) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100))
                }
                Err(e) => panic!("{addr} never came up: {e}{}", server_logs(log_dir)),
            }
        }
    }
}

/// Builds the real `pbr-server` binary (cheap when already cached). Clients
/// are driven in-process via the `pbr-client` library in later tasks, so no
/// client binary is needed here.
fn build_binaries() {
    let root = repo_root();
    let status = Command::new("cargo")
        .args(["build", "--release", "--bin", "pbr-server"])
        .current_dir(&root)
        .status()
        .expect("cargo is available");
    assert!(status.success(), "release build of pbr-server failed");
}

fn server_bin_path() -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target"));
    target_dir.join("release/pbr-server")
}

/// Removes the rendered-config temp dir on drop, so a cluster's configs
/// don't accumulate under `target/` across repeated test runs.
struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A live cluster: one aggregator + `n_shareholders` shareholders, all
/// TLS on their client-facing listeners, torn down when dropped.
struct Cluster {
    agg_url: String,
    shareholder_urls: Vec<String>,
    _guards: Vec<ChildGuard>,
    _cfg_dir: TempDirGuard,
}

/// Renders a fresh `n_shareholders`-of-`threshold` TLS cluster into a temp
/// dir under `target/`, spawns the real `pbr-server` binaries (shareholders
/// first, then the aggregator), and waits for all `n_shareholders + 1`
/// client-facing listeners to accept TCP connections before returning.
async fn start_cluster(n_shareholders: usize, threshold: usize) -> Cluster {
    let root = repo_root();
    let server_bin = server_bin_path();
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"));

    let log_dir = target_dir.join("wire-grid-logs");
    std::fs::create_dir_all(&log_dir).unwrap();

    // Reserve 1 + 2n loopback ports: aggregator, n client-facing shareholder
    // ports, n internal gather ports. Held alive until right before spawn.
    let reservations: Vec<(TcpListener, u16)> = (0..1 + 2 * n_shareholders)
        .map(|_| reserve_port())
        .collect();
    let agg_port = reservations[0].1;
    let client_ports: Vec<u16> = reservations[1..=n_shareholders]
        .iter()
        .map(|(_, p)| *p)
        .collect();
    let internal_ports: Vec<u16> = reservations[1 + n_shareholders..]
        .iter()
        .map(|(_, p)| *p)
        .collect();

    let cfg_dir = target_dir.join(format!(
        "wire-configs-{}-{}of",
        std::process::id(),
        n_shareholders
    ));
    std::fs::create_dir_all(&cfg_dir).unwrap();

    let agg_cfg = cfg_dir.join("aggregator.toml");
    std::fs::write(
        &agg_cfg,
        aggregator_toml(agg_port, &client_ports, &internal_ports, threshold),
    )
    .unwrap();

    let sh_cfgs: Vec<PathBuf> = (0..n_shareholders)
        .map(|i| {
            let path = cfg_dir.join(format!("shareholder-{}.toml", i + 1));
            std::fs::write(
                &path,
                shareholder_toml(i + 1, client_ports[i], internal_ports[i]),
            )
            .unwrap();
            path
        })
        .collect();

    // Release the reserved ports immediately before spawning so the servers
    // can bind them.
    drop(reservations);

    let mut guards: Vec<ChildGuard> = (0..n_shareholders)
        .map(|i| {
            spawn_server(
                &server_bin,
                &root,
                &log_dir,
                format!("shareholder-{}", i + 1),
                "shareholder",
                sh_cfgs[i].to_str().unwrap(),
            )
        })
        .collect();
    guards.push(spawn_server(
        &server_bin,
        &root,
        &log_dir,
        "aggregator".to_string(),
        "aggregator",
        agg_cfg.to_str().unwrap(),
    ));

    let listener_addrs: Vec<String> = client_ports
        .iter()
        .map(|p| format!("127.0.0.1:{p}"))
        .chain(std::iter::once(format!("127.0.0.1:{agg_port}")))
        .collect();
    wait_for_listeners(&listener_addrs, &mut guards, &log_dir);

    Cluster {
        agg_url: format!("https://127.0.0.1:{agg_port}"),
        shareholder_urls: client_ports
            .iter()
            .map(|p| format!("https://127.0.0.1:{p}"))
            .collect(),
        _guards: guards,
        _cfg_dir: TempDirGuard(cfg_dir),
    }
}

/// Installs the ring provider as rustls's process-wide default, exactly
/// once. The admin channel below is built directly against tonic/rustls
/// rather than through `pbr_client::rpc` (which installs this on the
/// grower/measured clients' own channels), so it needs the same install
/// before its first handshake or `ClientTlsConfig` construction panics with
/// "no process-level CryptoProvider".
fn ensure_crypto_provider() {
    static INSTALL_CRYPTO: std::sync::Once = std::sync::Once::new();
    INSTALL_CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Schedules one session on the live aggregator via the admin RPC plane
/// (TLS pinned to the test CA, authenticated with the cluster's static admin
/// token). `min_clients = target_clients = 2` so the round loop needs both
/// the grower and measured clients to submit before a round closes.
async fn create_session(agg_url: &str, dataset: &str, depth: u32) {
    use pbr_proto::v1::CreateSessionRequest;
    use pbr_proto::v1::admin_service_client::AdminServiceClient;

    ensure_crypto_provider();
    let ca = std::fs::read(repo_root().join(CA_CRT)).unwrap();
    let tls = tonic::transport::ClientTlsConfig::new()
        .ca_certificate(tonic::transport::Certificate::from_pem(ca));
    let channel = tonic::transport::Channel::from_shared(agg_url.to_string())
        .unwrap()
        .tls_config(tls)
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut admin = AdminServiceClient::new(channel);
    let mut req = tonic::Request::new(CreateSessionRequest {
        dataset_id: dataset.into(),
        title: format!("wire {dataset} d{depth}"),
        n_trees: 15,
        max_depth: depth,
        n_bins: 10,
        learning_rate: 0.15,
        lambda: 2.0,
        min_clients: 2,
        target_clients: 2,
        submission_window_ms: 5000,
    });
    req.metadata_mut()
        .insert("authorization", "Bearer dev-admin-token".parse().unwrap());
    admin.create_session(req).await.expect("create_session");
}

/// One wire-cost grid row: the measured client's session totals for one
/// (dataset, depth, threshold, hide_path) configuration.
struct WireRow {
    dataset: String,
    depth: u32,
    threshold: usize,
    hide_path: bool,
    total_tx: u64,
    total_rx: u64,
    submit_tx: u64,
    submit_rx: u64,
    n_rounds: u64,
}

/// Measures one wire-cost configuration end to end: schedule a session, then
/// run two clients against it concurrently, a grower (all-but-one training
/// row, whose batch forces the aggregator through every real split of the
/// tree) and a measured client (the single held-out record). Only the
/// measured client's `WireRun` is snapshotted into the returned row.
async fn measure_config(
    cluster: &Cluster,
    dataset: &str,
    depth: u32,
    threshold: usize,
    hide_path: bool,
) -> WireRow {
    create_session(&cluster.agg_url, dataset, depth).await;
    let (grower_batch, measured_record, _n_features) = load_split(dataset);
    let root = repo_root();
    let priv_pem = std::fs::read(root.join(PRIVKEY)).unwrap();
    let ca_pem = std::fs::read(root.join(CA_CRT)).unwrap();
    let mk = |records: Vec<(Vec<f64>, f64)>| pbr_client::driver::SessionParams {
        agg_endpoint: cluster.agg_url.clone(),
        shareholder_endpoints: Some(cluster.shareholder_urls.clone()),
        token: pbr_client::jwt::mint(
            "https://test-issuer.local",
            "pbr",
            "test-1",
            "wire-grid",
            3600,
            &priv_pem,
        )
        .unwrap(),
        records,
        threshold: Some(threshold),
        hide_path,
        ca_pem: Some(ca_pem.clone()),
        session_id: None,
    };
    // The grower runs concurrently, forcing the aggregator through the real
    // tree; only the measured client's WireRun is snapshotted below.
    let grower = tokio::spawn(pbr_client::driver::run_to_completion(
        mk(grower_batch),
        |_| {},
    ));
    let run = pbr_client::driver::run_collecting(mk(vec![measured_record]), |_| {})
        .await
        .expect("measured run");
    grower.await.unwrap().expect("grower run");
    WireRow {
        dataset: dataset.into(),
        depth,
        threshold,
        hide_path,
        total_tx: run.total_tx,
        total_rx: run.total_rx,
        submit_tx: run.submit_tx,
        submit_rx: run.submit_rx,
        n_rounds: run.n_rounds,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "spawns real pbr-server binaries over TLS; run with --ignored"]
async fn cluster_of_three_comes_up_over_tls() {
    build_binaries();
    let cluster = start_cluster(3, 2).await;
    assert_eq!(cluster.shareholder_urls.len(), 3);
    assert!(cluster.agg_url.starts_with("https://"));
    // Readiness already asserted inside start_cluster (wait_for_listeners).
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "spawns a real TLS cluster; run with --ignored"]
async fn measure_one_config_end_to_end() {
    build_binaries();
    let cluster = start_cluster(3, 2).await;
    let row = measure_config(&cluster, "heart_disease", 1, 2, false).await;
    assert_eq!(row.dataset, "heart_disease");
    assert_eq!(row.depth, 1);
    assert_eq!(row.threshold, 2);
    assert!(!row.hide_path);
    // depth 1 => 1 gradient round per tree; 15 trees => 15 gradient rounds + 1 stats.
    assert_eq!(row.n_rounds, 16, "stats + 15 depth-1 gradient rounds");
    assert!(row.total_tx >= row.submit_tx && row.submit_tx > 0);
    assert!(row.total_rx >= row.submit_rx && row.submit_rx > 0);
}

const SWEEP_DATASETS: [&str; 4] = [
    "heart_disease",
    "pima_diabetes",
    "breast_cancer",
    "cdc_diabetes",
];

/// The full wire-cost grid: every dataset over depths 1..=6 under a 2-of-3
/// cluster, plus the depth-3 cross-check under a 3-of-5 cluster, each
/// crossed with both hide-path arms. 4*6*2 + 4*2 = 56 configurations.
fn sweep_configs() -> Vec<(&'static str, u32, usize, bool)> {
    let mut configs = Vec::with_capacity(56);
    for dataset in SWEEP_DATASETS {
        for depth in 1u32..=6 {
            for hide in [false, true] {
                configs.push((dataset, depth, 2usize, hide));
            }
        }
    }
    for dataset in SWEEP_DATASETS {
        for hide in [false, true] {
            configs.push((dataset, 3u32, 3usize, hide));
        }
    }
    configs
}

#[test]
fn sweep_configs_cover_the_full_grid() {
    let configs = sweep_configs();
    assert_eq!(configs.len(), 56, "56 total configs");

    let threshold_2: Vec<_> = configs.iter().copied().filter(|&(_, _, t, _)| t == 2).collect();
    let threshold_3: Vec<_> = configs.iter().copied().filter(|&(_, _, t, _)| t == 3).collect();
    assert_eq!(threshold_2.len(), 48, "2-of-3 subset");
    assert_eq!(threshold_3.len(), 8, "3-of-5 subset");

    for dataset in SWEEP_DATASETS {
        for depth in 1u32..=6 {
            for hide in [false, true] {
                assert!(
                    configs.contains(&(dataset, depth, 2, hide)),
                    "missing 2-of-3 config: {dataset} depth {depth} hide {hide}"
                );
            }
        }
    }
    // The 3-of-5 subset is depth 3 only, for every dataset and hide arm.
    for &(dataset, depth, threshold, _) in &threshold_3 {
        assert_eq!(depth, 3, "3-of-5 subset is depth 3 only ({dataset})");
        assert_eq!(threshold, 3);
    }
    for dataset in SWEEP_DATASETS {
        for hide in [false, true] {
            assert!(configs.contains(&(dataset, 3, 3, hide)));
        }
    }
}

const WIRE_CSV_HEADER: &str =
    "dataset,depth,threshold,hide_path,total_tx,total_rx,submit_tx,submit_rx,n_rounds";

fn wire_csv_row(row: &WireRow) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{}",
        row.dataset,
        row.depth,
        row.threshold,
        row.hide_path,
        row.total_tx,
        row.total_rx,
        row.submit_tx,
        row.submit_rx,
        row.n_rounds,
    )
}

/// Appends `WireRow`s to a CSV file one at a time, flushing after every
/// write. The 56-config sweep runs for tens of minutes; flushing per row
/// means a mid-sweep failure still leaves every config measured so far on
/// disk, and `tail -f` shows live progress.
struct WireCsvWriter {
    file: std::fs::File,
}

impl WireCsvWriter {
    fn create(path: &Path) -> Self {
        use std::io::Write;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut file =
            std::fs::File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
        writeln!(file, "{WIRE_CSV_HEADER}").unwrap();
        file.flush().unwrap();
        WireCsvWriter { file }
    }

    fn append(&mut self, row: &WireRow) {
        use std::io::Write;
        writeln!(self.file, "{}", wire_csv_row(row)).unwrap();
        self.file.flush().unwrap();
    }
}

#[test]
fn wire_csv_writer_flushes_header_then_each_row() {
    let path = std::env::temp_dir().join(format!(
        "wire_csv_writer_test_{}_{}.csv",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    {
        let mut writer = WireCsvWriter::create(&path);
        writer.append(&WireRow {
            dataset: "heart_disease".into(),
            depth: 1,
            threshold: 2,
            hide_path: false,
            total_tx: 100,
            total_rx: 200,
            submit_tx: 50,
            submit_rx: 60,
            n_rounds: 16,
        });
        writer.append(&WireRow {
            dataset: "pima_diabetes".into(),
            depth: 3,
            threshold: 3,
            hide_path: true,
            total_tx: 999,
            total_rx: 888,
            submit_tx: 77,
            submit_rx: 66,
            n_rounds: 46,
        });
    }
    let content = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(
        lines,
        vec![
            "dataset,depth,threshold,hide_path,total_tx,total_rx,submit_tx,submit_rx,n_rounds",
            "heart_disease,1,2,false,100,200,50,60,16",
            "pima_diabetes,3,3,true,999,888,77,66,46",
        ]
    );
}

/// Measures every config in `sweep_configs()` and appends each row to
/// `results/wire_measured.csv` as it completes. Each config gets its own
/// fresh cluster (`start_cluster` inside the loop) rather than sharing one
/// aggregator across all 56 sessions, so every measurement runs on the exact
/// single-session path `measure_one_config_end_to_end` already proves, with no
/// cross-session enrollment or live-session-cap interaction to reason about.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "full 56-config wire sweep; run with --ignored; writes results/wire_measured.csv"]
async fn wire_grid_sweep() {
    build_binaries();
    let configs = sweep_configs();
    let total = configs.len();
    let mut csv = WireCsvWriter::create(&repo_root().join("results/wire_measured.csv"));
    let mut n_written = 0usize;
    for (i, (dataset, depth, threshold, hide)) in configs.into_iter().enumerate() {
        let n_shareholders = if threshold == 2 { 3 } else { 5 };
        let cluster = start_cluster(n_shareholders, threshold).await;
        let row = measure_config(&cluster, dataset, depth, threshold, hide).await;
        eprintln!(
            "[{}/{total}] {dataset} depth={depth} threshold={threshold} hide={hide} -> n_rounds={}",
            i + 1,
            row.n_rounds
        );
        csv.append(&row);
        n_written += 1;
    }
    assert_eq!(n_written, 56, "expected 56 measured configs");
}
