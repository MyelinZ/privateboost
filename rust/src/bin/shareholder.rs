use privateboost::grpc::shareholder_service::ShareholderServiceImpl;
use privateboost::proto::shareholder_service_server::ShareholderServiceServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port = std::env::var("PORT").unwrap_or_else(|_| "50051".into());
    let min_clients: usize = std::env::var("MIN_CLIENTS")
        .unwrap_or_else(|_| "10".into())
        .parse()?;

    let addr = format!("0.0.0.0:{port}").parse()?;
    let service = ShareholderServiceImpl::new(min_clients);

    eprintln!("Shareholder server listening on {addr} (min_clients={min_clients})");
    Server::builder()
        .add_service(ShareholderServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
