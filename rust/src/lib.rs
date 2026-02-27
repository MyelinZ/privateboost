pub mod coordinator;
pub mod crypto;
pub mod domain;
pub mod grpc;

pub mod proto {
    tonic::include_proto!("privateboost");
}
