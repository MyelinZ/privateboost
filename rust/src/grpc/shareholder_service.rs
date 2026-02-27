use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};
use tonic::{Request, Response, Status};

use crate::crypto::shamir::Share;
use crate::domain::shareholder::ShareHolder;
use crate::grpc::{ndarray_to_vec, vec_to_ndarray};
use crate::proto;

pub struct ShareholderServiceImpl {
    min_clients: usize,
    sessions: Arc<Mutex<HashMap<String, Arc<RwLock<ShareHolder>>>>>,
    cancelled: Arc<Mutex<HashSet<String>>>,
}

impl ShareholderServiceImpl {
    pub fn new(min_clients: usize) -> Self {
        Self {
            min_clients,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            cancelled: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    async fn get_or_create_session(
        &self,
        session_id: &str,
    ) -> Result<Arc<RwLock<ShareHolder>>, Status> {
        let cancelled = self.cancelled.lock().await;
        if cancelled.contains(session_id) {
            return Err(Status::not_found(format!(
                "Session {session_id} cancelled"
            )));
        }
        drop(cancelled);

        let mut sessions = self.sessions.lock().await;
        let sh = sessions
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(RwLock::new(ShareHolder::new(self.min_clients))));
        Ok(Arc::clone(sh))
    }

    async fn get_session(
        &self,
        session_id: &str,
    ) -> Result<Arc<RwLock<ShareHolder>>, Status> {
        let cancelled = self.cancelled.lock().await;
        if cancelled.contains(session_id) {
            return Err(Status::not_found(format!(
                "Session {session_id} cancelled"
            )));
        }
        drop(cancelled);

        let sessions = self.sessions.lock().await;
        sessions
            .get(session_id)
            .map(Arc::clone)
            .ok_or_else(|| Status::not_found(format!("Session {session_id} not found")))
    }
}

#[allow(clippy::result_large_err)]
fn share_from_proto(proto_share: &proto::Share) -> Result<Share, Status> {
    let y_arr = proto_share
        .y
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("Share must have y field"))?;
    let y = ndarray_to_vec(y_arr);
    Ok(Share {
        x: proto_share.x,
        y,
    })
}

#[allow(clippy::result_large_err)]
fn bytes_to_commitment(bytes: &[u8]) -> Result<[u8; 32], Status> {
    bytes
        .try_into()
        .map_err(|_| Status::invalid_argument("commitment must be 32 bytes"))
}

#[tonic::async_trait]
impl proto::shareholder_service_server::ShareholderService for ShareholderServiceImpl {
    async fn submit_stats(
        &self,
        request: Request<proto::SubmitStatsRequest>,
    ) -> Result<Response<proto::SubmitStatsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_or_create_session(&req.session_id).await?;
        let share = share_from_proto(
            req.share
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("missing share"))?,
        )?;
        let commitment = bytes_to_commitment(&req.commitment)?;
        let mut sh = sh.write().await;
        sh.receive_stats(commitment, share);
        Ok(Response::new(proto::SubmitStatsResponse {}))
    }

    async fn submit_gradients(
        &self,
        request: Request<proto::SubmitGradientsRequest>,
    ) -> Result<Response<proto::SubmitGradientsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_or_create_session(&req.session_id).await?;
        let share = share_from_proto(
            req.share
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("missing share"))?,
        )?;
        let commitment = bytes_to_commitment(&req.commitment)?;
        let mut sh = sh.write().await;
        sh.receive_gradients(req.round_id, req.depth, commitment, share, req.node_id);
        Ok(Response::new(proto::SubmitGradientsResponse {}))
    }

    async fn get_stats_commitments(
        &self,
        request: Request<proto::GetStatsCommitmentsRequest>,
    ) -> Result<Response<proto::GetStatsCommitmentsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_session(&req.session_id).await?;
        let sh = sh.read().await;
        let commitments = sh.get_stats_commitments();
        Ok(Response::new(proto::GetStatsCommitmentsResponse {
            commitments: commitments.into_iter().map(|c| c.to_vec()).collect(),
        }))
    }

    async fn get_gradient_commitments(
        &self,
        request: Request<proto::GetGradientCommitmentsRequest>,
    ) -> Result<Response<proto::GetGradientCommitmentsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_session(&req.session_id).await?;
        let sh = sh.read().await;
        if sh.current_round_id() != req.round_id {
            return Ok(Response::new(proto::GetGradientCommitmentsResponse {
                commitments: vec![],
            }));
        }
        let commitments = sh.get_gradient_commitments(req.depth);
        Ok(Response::new(proto::GetGradientCommitmentsResponse {
            commitments: commitments.into_iter().map(|c| c.to_vec()).collect(),
        }))
    }

    async fn get_gradient_node_ids(
        &self,
        request: Request<proto::GetGradientNodeIdsRequest>,
    ) -> Result<Response<proto::GetGradientNodeIdsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_session(&req.session_id).await?;
        let sh = sh.read().await;
        if sh.current_round_id() != req.round_id {
            return Ok(Response::new(proto::GetGradientNodeIdsResponse {
                node_ids: vec![],
            }));
        }
        let node_ids = sh.get_gradient_node_ids(req.depth);
        Ok(Response::new(proto::GetGradientNodeIdsResponse {
            node_ids: node_ids.into_iter().collect(),
        }))
    }

    async fn get_stats_sum(
        &self,
        request: Request<proto::GetStatsSumRequest>,
    ) -> Result<Response<proto::GetStatsSumResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_session(&req.session_id).await?;
        let commitments: Vec<[u8; 32]> = req
            .commitments
            .iter()
            .map(|c| bytes_to_commitment(c))
            .collect::<Result<_, _>>()?;
        let sh = sh.read().await;
        let total = sh
            .get_stats_sum(&commitments)
            .map_err(|e| Status::failed_precondition(e.to_string()))?;
        Ok(Response::new(proto::GetStatsSumResponse {
            sum: Some(vec_to_ndarray(&total)),
        }))
    }

    async fn get_gradients_sum(
        &self,
        request: Request<proto::GetGradientsSumRequest>,
    ) -> Result<Response<proto::GetGradientsSumResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_session(&req.session_id).await?;
        let commitments: Vec<[u8; 32]> = req
            .commitments
            .iter()
            .map(|c| bytes_to_commitment(c))
            .collect::<Result<_, _>>()?;
        let sh = sh.read().await;
        let total = sh
            .get_gradients_sum(req.depth, &commitments, req.node_id)
            .map_err(|e| Status::failed_precondition(e.to_string()))?;
        Ok(Response::new(proto::GetGradientsSumResponse {
            sum: Some(vec_to_ndarray(&total)),
        }))
    }

    async fn cancel_session(
        &self,
        request: Request<proto::CancelSessionRequest>,
    ) -> Result<Response<proto::CancelSessionResponse>, Status> {
        let req = request.into_inner();
        let mut cancelled = self.cancelled.lock().await;
        cancelled.insert(req.session_id.clone());
        drop(cancelled);
        let mut sessions = self.sessions.lock().await;
        sessions.remove(&req.session_id);
        Ok(Response::new(proto::CancelSessionResponse {}))
    }

    async fn reset(
        &self,
        request: Request<proto::ResetRequest>,
    ) -> Result<Response<proto::ResetResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_session(&req.session_id).await?;
        let mut sh = sh.write().await;
        sh.reset();
        Ok(Response::new(proto::ResetResponse {}))
    }
}
