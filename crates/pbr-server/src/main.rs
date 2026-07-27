use clap::Parser;
use pbr_server::agg_config::AggregatorConfig;
use pbr_server::config::ShareholderConfig;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "pbr-server")]
struct Args {
    /// Role: "shareholder" or "aggregator".
    #[arg(long)]
    role: String,
    #[arg(long)]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    match args.role.as_str() {
        "shareholder" => {
            let cfg = ShareholderConfig::load(&args.config)?;
            // The handle owns the servers' shutdown channels: it must stay
            // alive until ctrl-c, or the listeners tear down immediately.
            let pbr_server::shareholder::RunningShareholder {
                client_addr: addr,
                internal_addr: internal,
                handle: _handle,
            } = pbr_server::shareholder::serve(cfg).await?;
            tracing::info!(%addr, %internal, "shareholder up");
            tokio::signal::ctrl_c().await?;
            Ok(())
        }
        "aggregator" => {
            let cfg = AggregatorConfig::load(&args.config)?;
            let pbr_server::aggregator::RunningAggregator {
                addr,
                handle: _handle,
            } = pbr_server::aggregator::serve(cfg).await?;
            tracing::info!(%addr, "aggregator up");
            tokio::signal::ctrl_c().await?;
            Ok(())
        }
        other => anyhow::bail!("unknown role: {other}"),
    }
}
