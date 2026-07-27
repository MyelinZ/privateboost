//! Test-only doubles shared by this crate's unit-test modules.

use pbr_proto::v1::shareholder_internal_server::{ShareholderInternal, ShareholderInternalServer};
use pbr_proto::v1::{
    Ack, CloseRoundRequest, CommitmentList, EndSessionRequest, GetSumsRequest,
    ListCommitmentsRequest, OpenRoundRequest, SumShare,
};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

/// A shareholder whose HTTP/2 connection is healthy but whose read RPCs never
/// answer (SIGSTOP, a poisoned lock, a black-holed link): it acks the round
/// lifecycle but wedges forever on `list_commitments`/`get_sums`. Exercises
/// the internal-plane per-RPC deadline in both the round loop's and the
/// gather snapshot's timeout tests.
pub(crate) struct WedgedShareholder;

#[tonic::async_trait]
impl ShareholderInternal for WedgedShareholder {
    async fn list_commitments(
        &self,
        _req: Request<ListCommitmentsRequest>,
    ) -> Result<Response<CommitmentList>, Status> {
        std::future::pending::<()>().await;
        unreachable!()
    }
    async fn get_sums(&self, _req: Request<GetSumsRequest>) -> Result<Response<SumShare>, Status> {
        std::future::pending::<()>().await;
        unreachable!()
    }
    async fn open_round(&self, _req: Request<OpenRoundRequest>) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {}))
    }
    async fn close_round(&self, _req: Request<CloseRoundRequest>) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {}))
    }
    async fn end_session(
        &self,
        _req: Request<EndSessionRequest>,
    ) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {}))
    }
}

/// A [`WedgedShareholder`] served on a fresh loopback port.
pub(crate) struct WedgedServer {
    pub(crate) addr: SocketAddr,
    shutdown: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

impl WedgedServer {
    pub(crate) async fn spawn() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (shutdown, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(ShareholderInternalServer::new(WedgedShareholder))
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    async {
                        let _ = rx.await;
                    },
                ),
        );
        Self {
            addr,
            shutdown,
            task,
        }
    }

    pub(crate) async fn shutdown(self) {
        let _ = self.shutdown.send(());
        let _ = self.task.await;
    }
}

/// The control RPCs a [`RecordingShareholder`] has received. `close_round` and
/// `end_session` end up producing identical downstream pool state on this
/// fake's real counterpart (an `end_session` frees the pool as completely as
/// a `close_round` leaves it), so a caller that must tell the two RPCs apart
/// needs the RPC traffic itself, not the state it left behind.
#[derive(Default)]
pub(crate) struct RecordedCalls {
    pub(crate) close_round: Option<CloseRoundRequest>,
    pub(crate) end_session: Option<EndSessionRequest>,
}

/// A shareholder that Acks every control RPC and records each `close_round`/
/// `end_session` request it receives into a shared [`RecordedCalls`] log.
pub(crate) struct RecordingShareholder {
    calls: Arc<Mutex<RecordedCalls>>,
}

#[tonic::async_trait]
impl ShareholderInternal for RecordingShareholder {
    async fn list_commitments(
        &self,
        _req: Request<ListCommitmentsRequest>,
    ) -> Result<Response<CommitmentList>, Status> {
        Ok(Response::new(CommitmentList {
            commitments: Vec::new(),
            node_ids: Vec::new(),
        }))
    }
    async fn get_sums(&self, _req: Request<GetSumsRequest>) -> Result<Response<SumShare>, Status> {
        Err(Status::unimplemented(
            "RecordingShareholder does not serve get_sums",
        ))
    }
    async fn open_round(&self, _req: Request<OpenRoundRequest>) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {}))
    }
    async fn close_round(&self, req: Request<CloseRoundRequest>) -> Result<Response<Ack>, Status> {
        self.calls.lock().unwrap().close_round = Some(req.into_inner());
        Ok(Response::new(Ack {}))
    }
    async fn end_session(
        &self,
        req: Request<EndSessionRequest>,
    ) -> Result<Response<Ack>, Status> {
        self.calls.lock().unwrap().end_session = Some(req.into_inner());
        Ok(Response::new(Ack {}))
    }
}

/// A [`RecordingShareholder`] served on a fresh loopback port, with a shared
/// handle to its recorded calls the caller can read at any time (including
/// after `shutdown`).
pub(crate) struct RecordingServer {
    pub(crate) addr: SocketAddr,
    pub(crate) calls: Arc<Mutex<RecordedCalls>>,
    shutdown: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

impl RecordingServer {
    pub(crate) async fn spawn() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let calls = Arc::new(Mutex::new(RecordedCalls::default()));
        let (shutdown, rx) = oneshot::channel::<()>();
        let task = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(ShareholderInternalServer::new(RecordingShareholder {
                    calls: calls.clone(),
                }))
                .serve_with_incoming_shutdown(
                    tokio_stream::wrappers::TcpListenerStream::new(listener),
                    async {
                        let _ = rx.await;
                    },
                ),
        );
        Self {
            addr,
            calls,
            shutdown,
            task,
        }
    }

    pub(crate) async fn shutdown(self) {
        let _ = self.shutdown.send(());
        let _ = self.task.await;
    }
}
