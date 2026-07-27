//! CLI wrapper around `pbr_client::driver::run_collecting`: drives one
//! training row through an entire aggregator session over real gRPC.

use clap::Parser;
use pbr_client::driver::{SessionParams, WireRun, run_collecting};
use pbr_client::jwt::mint;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "pbr-client")]
struct Args {
    /// Aggregator base URL, e.g. http://127.0.0.1:7000
    #[arg(long)]
    aggregator: String,

    /// Comma-separated shareholder client-facing URLs, ordered by Shamir
    /// evaluation point x = 1, 2, 3, ... . Optional: if omitted, the client
    /// learns them from EnrollSession.
    #[arg(long, value_delimiter = ',')]
    shareholders: Option<Vec<String>>,

    /// Feature row, comma-separated floats.
    #[arg(long, value_delimiter = ',')]
    features: Vec<f64>,

    /// Training label for this row.
    #[arg(long)]
    label: f64,

    /// Shamir reconstruction threshold; must match the shareholder/aggregator
    /// deployment's configured threshold. Optional: if omitted, the client
    /// learns it from EnrollSession.
    #[arg(long)]
    threshold: Option<usize>,

    /// Hide the client's true tree path among all active nodes at each depth,
    /// instead of only submitting for its own node. Takes an explicit value
    /// (`--hide-path false` disables it); defaults to enabled.
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    hide_path: bool,

    /// Pre-minted bearer token. Required unless --mint is set.
    #[arg(long)]
    token: Option<String>,

    /// Dev flag: mint a fresh bearer token locally instead of --token,
    /// using --mint-key (and the other --mint-* fields, which default to
    /// the dev issuer/audience/kid used across this workspace's test
    /// fixtures).
    #[arg(long)]
    mint: bool,
    #[arg(long, default_value = "https://test-issuer.local")]
    mint_issuer: String,
    #[arg(long, default_value = "pbr")]
    mint_audience: String,
    #[arg(long, default_value = "test-1")]
    mint_kid: String,
    /// Path to the RSA private key PEM used to sign the minted token.
    #[arg(long)]
    mint_key: Option<PathBuf>,

    /// Path to a PEM CA certificate to pin for TLS. When set, the aggregator
    /// and shareholder connections are made over TLS verified against this CA
    /// (use https:// URLs). Omitted leaves every connection plaintext.
    #[arg(long)]
    ca_cert: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let token = if args.mint {
        let key_path = args
            .mint_key
            .ok_or_else(|| anyhow::anyhow!("--mint requires --mint-key <RSA private key PEM>"))?;
        let pem = std::fs::read(&key_path)?;
        mint(
            &args.mint_issuer,
            &args.mint_audience,
            &args.mint_kid,
            "pbr-client-cli",
            3600,
            &pem,
        )?
    } else {
        args.token
            .ok_or_else(|| anyhow::anyhow!("either --token or --mint must be provided"))?
    };

    let ca_pem = args.ca_cert.map(std::fs::read).transpose()?;

    let WireRun { total_tx, total_rx, submit_tx, submit_rx, n_rounds } = run_collecting(
        SessionParams {
            agg_endpoint: args.aggregator,
            shareholder_endpoints: args.shareholders,
            token,
            records: vec![(args.features, args.label)],
            threshold: args.threshold,
            hide_path: args.hide_path,
            ca_pem,
            session_id: None,
        },
        |_| {},
    )
    .await?;

    eprintln!(
        "wire totals: total_tx={total_tx} total_rx={total_rx} submit_tx={submit_tx} submit_rx={submit_rx} n_rounds={n_rounds}"
    );

    Ok(())
}
