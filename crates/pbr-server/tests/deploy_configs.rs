//! The committed cluster configs under `deploy/` must keep parsing and
//! validating. The heavy `pbr-e2e` test boots `deploy/local` for real but is
//! `#[ignore]`d, and `deploy/emulator` is never booted in CI at all, so
//! nothing in the default `cargo test` run exercises the config loaders
//! against these files. A field rename or a validation-breaking edit would
//! otherwise surface only when someone runs the e2e or the demo by hand.

use pbr_server::agg_config::AggregatorConfig;
use pbr_server::config::ShareholderConfig;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/pbr-server has a repo root two levels up")
        .to_path_buf()
}

fn assert_cluster_loads(dir: &str) {
    let root = repo_root().join(dir);
    AggregatorConfig::load(&root.join("aggregator.toml"))
        .unwrap_or_else(|e| panic!("{dir}/aggregator.toml must load and validate: {e:#}"));
    for name in [
        "shareholder-1.toml",
        "shareholder-2.toml",
        "shareholder-3.toml",
    ] {
        ShareholderConfig::load(&root.join(name))
            .unwrap_or_else(|e| panic!("{dir}/{name} must load and validate: {e:#}"));
    }
}

#[test]
fn deploy_local_configs_load_and_validate() {
    assert_cluster_loads("deploy/local");
}

#[test]
fn deploy_emulator_configs_load_and_validate() {
    assert_cluster_loads("deploy/emulator");
}

#[test]
fn deploy_hetzner_configs_load_and_validate() {
    assert_cluster_loads("deploy/hetzner/configs");
}

#[test]
fn deploy_hetzner_smoke_configs_load_and_validate() {
    assert_cluster_loads("deploy/hetzner/configs/smoke");
}
