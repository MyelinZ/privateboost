use privateboost::grpc::aggregator_service::{AggregatorConfig, AggregatorServiceImpl};
use privateboost::proto::aggregator_service_server::AggregatorServiceServer;
use privateboost::proto::FeatureSpec;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port = std::env::var("PORT").unwrap_or_else(|_| "50052".into());

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
    let loss = std::env::var("LOSS").unwrap_or_else(|_| "squared".into());
    let target_column = std::env::var("TARGET_COLUMN").unwrap_or_else(|_| "target".into());

    let target_count: Option<usize> = std::env::var("TARGET_COUNT")
        .ok()
        .and_then(|s| if s.is_empty() { None } else { s.parse().ok() });
    let target_fraction: Option<f64> = std::env::var("TARGET_FRACTION")
        .ok()
        .and_then(|s| if s.is_empty() { None } else { s.parse().ok() });

    let features_str = std::env::var("FEATURES").unwrap_or_default();
    let features: Vec<FeatureSpec> = features_str
        .split(',')
        .enumerate()
        .filter(|(_, name)| !name.trim().is_empty())
        .map(|(i, name)| FeatureSpec {
            index: i as i32,
            name: name.trim().to_string(),
        })
        .collect();

    let config = AggregatorConfig {
        sh_addresses: sh_addresses.clone(),
        n_bins,
        threshold,
        min_clients,
        learning_rate,
        lambda_reg,
        n_trees,
        max_depth,
        loss,
        target_count,
        target_fraction,
        features,
        target_column,
    };

    let addr = format!("0.0.0.0:{port}").parse()?;
    let service = AggregatorServiceImpl::new(config);

    eprintln!("Aggregator server listening on {addr}");
    eprintln!("  Shareholders: {sh_addresses:?}");
    eprintln!("  Threshold: {threshold}, Trees: {n_trees}, Depth: {max_depth}");
    Server::builder()
        .add_service(AggregatorServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
