use std::sync::Arc;

use privateboost::domain::aggregator::Aggregator;
use privateboost::grpc::aggregator_service::run_aggregator_loop;
use privateboost::grpc::remote_shareholder::RemoteShareHolder;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let aggregator_id: i32 = std::env::var("AGGREGATOR_ID")
        .expect("AGGREGATOR_ID env var required")
        .parse()?;

    let shareholders_str = std::env::var("SHAREHOLDERS")
        .expect("SHAREHOLDERS env var required (comma-separated host:port list)");
    let sh_addresses: Vec<String> = shareholders_str
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let threshold: usize = std::env::var("THRESHOLD")
        .unwrap_or_else(|_| "2".into())
        .parse()?;
    let n_bins: usize = std::env::var("N_BINS")
        .unwrap_or_else(|_| "10".into())
        .parse()?;
    let n_trees: usize = std::env::var("N_TREES")
        .unwrap_or_else(|_| "15".into())
        .parse()?;
    let max_depth: usize = std::env::var("MAX_DEPTH")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    let learning_rate: f64 = std::env::var("LEARNING_RATE")
        .unwrap_or_else(|_| "0.15".into())
        .parse()?;
    let lambda_reg: f64 = std::env::var("LAMBDA_REG")
        .unwrap_or_else(|_| "2.0".into())
        .parse()?;
    let min_clients: usize = std::env::var("MIN_CLIENTS")
        .unwrap_or_else(|_| "10".into())
        .parse()?;
    let run_id = std::env::var("RUN_ID")
        .expect("RUN_ID env var required");

    info!(
        aggregator_id,
        ?sh_addresses,
        threshold,
        n_trees,
        max_depth,
        %run_id,
        "Aggregator starting"
    );

    // Connect to shareholders
    let mut remote_shareholders = Vec::new();
    let mut sh_clients: Vec<Box<dyn privateboost::domain::aggregator::ShareHolderClient>> = Vec::new();

    for (i, addr) in sh_addresses.iter().enumerate() {
        let rsh = RemoteShareHolder::connect(addr, (i + 1) as i32, run_id.clone()).await?;
        let rsh = Arc::new(rsh);
        remote_shareholders.push(Arc::clone(&rsh));
        sh_clients.push(Box::new(rsh.clone()));
    }

    // Feature scales and gradient scale will be populated from the training config.
    // For now, use environment variables with sensible defaults.
    let feature_scales: Vec<f64> = std::env::var("FEATURE_SCALES")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<f64>().unwrap_or(1e6))
        .collect();
    let gradient_scale: f64 = std::env::var("GRADIENT_SCALE")
        .unwrap_or_else(|_| "1000000".into())
        .parse()?;

    let aggregator = Aggregator::new(
        sh_clients,
        n_bins,
        threshold,
        min_clients,
        learning_rate,
        lambda_reg,
        feature_scales,
        gradient_scale,
    )?;

    run_aggregator_loop(
        aggregator_id,
        aggregator,
        remote_shareholders,
        run_id,
        n_trees,
        max_depth,
    )
    .await?;

    info!(aggregator_id, "Aggregator finished");
    Ok(())
}
