use privateboost::grpc::shareholder_service::ShareholderServiceImpl;
use privateboost::proto::shareholder_service_server::ShareholderServiceServer;
use tonic::transport::Server;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "50051".into());
    let min_clients: usize = std::env::var("MIN_CLIENTS")
        .unwrap_or_else(|_| "10".into())
        .parse()?;
    let expected_aggregators: usize = std::env::var("EXPECTED_AGGREGATORS")
        .unwrap_or_else(|_| "1".into())
        .parse()?;

    let addr = format!("0.0.0.0:{port}").parse()?;
    let service = ShareholderServiceImpl::new(min_clients, expected_aggregators);

    info!(%addr, min_clients, expected_aggregators, "Shareholder server listening");
    Server::builder()
        .add_service(ShareholderServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
