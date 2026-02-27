use std::sync::Arc;

use privateboost::coordinator::grpc_service::CoordinatorGrpcService;
use privateboost::coordinator::service::Coordinator;
use privateboost::proto::coordinator_service_server::CoordinatorServiceServer;
use tonic::transport::Server;
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "50053".into());
    let addr = format!("0.0.0.0:{port}").parse()?;

    let coordinator = Arc::new(Coordinator::new());
    let service = CoordinatorGrpcService::new(coordinator);

    info!(%addr, "Coordinator server listening");
    Server::builder()
        .add_service(CoordinatorServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
