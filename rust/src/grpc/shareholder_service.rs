use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use crate::crypto::shamir::Share;
use crate::domain::model::StepId;
use crate::domain::shareholder::{Phase, ShareHolder, VoteStatus};
use crate::grpc::{proto_share_to_scalars, scalars_to_bytes};
use crate::proto;

pub struct ShareholderServiceImpl {
    min_clients: usize,
    runs: Arc<tokio::sync::Mutex<HashMap<String, Arc<RwLock<ShareHolder>>>>>,
    expected_aggregators: usize,
}

impl ShareholderServiceImpl {
    pub fn new(min_clients: usize, expected_aggregators: usize) -> Self {
        Self {
            min_clients,
            runs: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            expected_aggregators,
        }
    }

    /// Register a run (called when coordinator notifies of a new active run).
    pub async fn register_run(
        &self,
        run_id: &str,
        target: usize,
        n_trees: usize,
        max_depth: usize,
    ) {
        let mut runs = self.runs.lock().await;
        if !runs.contains_key(run_id) {
            let mut sh = ShareHolder::new(self.min_clients);
            sh.set_target(target);
            sh.set_expected_aggregators(self.expected_aggregators);
            sh.set_training_params(n_trees, max_depth);
            runs.insert(run_id.to_string(), Arc::new(RwLock::new(sh)));
        }
    }

    /// Remove a run (called when coordinator says cancelled/complete).
    pub async fn remove_run(&self, run_id: &str) {
        let mut runs = self.runs.lock().await;
        runs.remove(run_id);
    }

    async fn get_run(&self, run_id: &str) -> Result<Arc<RwLock<ShareHolder>>, Status> {
        let runs = self.runs.lock().await;
        runs.get(run_id)
            .map(Arc::clone)
            .ok_or_else(|| Status::not_found(format!("Run {run_id} not found")))
    }
}

#[allow(clippy::result_large_err)]
fn share_from_proto(proto_share: &proto::Share) -> Result<Share, Status> {
    let (_x, scalars) =
        proto_share_to_scalars(proto_share).map_err(Status::invalid_argument)?;
    Ok(Share {
        x: proto_share.x,
        y: scalars,
    })
}

#[allow(clippy::result_large_err)]
fn bytes_to_commitment(bytes: &[u8]) -> Result<[u8; 32], Status> {
    bytes
        .try_into()
        .map_err(|_| Status::invalid_argument("commitment must be 32 bytes"))
}

fn proto_step_to_domain(step: &proto::StepId) -> StepId {
    match proto::StepType::try_from(step.step_type) {
        Ok(proto::StepType::Stats) => StepId::stats(),
        Ok(proto::StepType::Gradients) | Err(_) => StepId::gradients(step.round_id, step.depth),
    }
}

fn serialize_result(result: &proto::submit_result_request::Result) -> Vec<u8> {
    use prost::Message;
    match result {
        proto::submit_result_request::Result::BinsResult(br) => {
            let mut buf = Vec::new();
            br.encode(&mut buf).unwrap();
            buf
        }
        proto::submit_result_request::Result::SplitsResult(sr) => {
            let mut buf = Vec::new();
            sr.encode(&mut buf).unwrap();
            buf
        }
        proto::submit_result_request::Result::TreeResult(tr) => {
            let mut buf = Vec::new();
            tr.encode(&mut buf).unwrap();
            buf
        }
    }
}

#[tonic::async_trait]
impl proto::shareholder_service_server::ShareholderService for ShareholderServiceImpl {
    async fn submit_stats(
        &self,
        request: Request<proto::SubmitStatsRequest>,
    ) -> Result<Response<proto::SubmitStatsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
        let share = share_from_proto(
            req.share
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("missing share"))?,
        )?;
        let commitment = bytes_to_commitment(&req.commitment)?;
        let mut sh = sh.write().await;
        let accepted = sh.receive_stats(commitment, share);
        if !accepted {
            tracing::warn!(run_id = %req.run_id, "stats share rejected (frozen)");
        }
        Ok(Response::new(proto::SubmitStatsResponse {}))
    }

    async fn submit_gradients(
        &self,
        request: Request<proto::SubmitGradientsRequest>,
    ) -> Result<Response<proto::SubmitGradientsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
        let share = share_from_proto(
            req.share
                .as_ref()
                .ok_or_else(|| Status::invalid_argument("missing share"))?,
        )?;
        let commitment = bytes_to_commitment(&req.commitment)?;
        let mut sh = sh.write().await;
        let accepted =
            sh.receive_gradients(req.round_id, req.depth, commitment, share, req.node_id);
        if !accepted {
            tracing::warn!(
                run_id = %req.run_id,
                round_id = req.round_id,
                depth = req.depth,
                "gradient share rejected (frozen)"
            );
        }
        Ok(Response::new(proto::SubmitGradientsResponse {}))
    }

    async fn get_stats_commitments(
        &self,
        request: Request<proto::GetStatsCommitmentsRequest>,
    ) -> Result<Response<proto::GetStatsCommitmentsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
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
        let sh = self.get_run(&req.run_id).await?;
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
        let sh = self.get_run(&req.run_id).await?;
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
        let sh = self.get_run(&req.run_id).await?;
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
            sum: Some(proto::NdArray {
                dtype: 0,
                shape: vec![total.len() as i64],
                data: scalars_to_bytes(&total),
            }),
        }))
    }

    async fn get_gradients_sum(
        &self,
        request: Request<proto::GetGradientsSumRequest>,
    ) -> Result<Response<proto::GetGradientsSumResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
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
            sum: Some(proto::NdArray {
                dtype: 0,
                shape: vec![total.len() as i64],
                data: scalars_to_bytes(&total),
            }),
        }))
    }

    async fn submit_result(
        &self,
        request: Request<proto::SubmitResultRequest>,
    ) -> Result<Response<proto::SubmitResultResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;

        let step = req
            .step
            .ok_or_else(|| Status::invalid_argument("missing step"))?;
        let step_id = proto_step_to_domain(&step);

        let result_bytes = match &req.result {
            Some(r) => serialize_result(r),
            None => return Err(Status::invalid_argument("missing result")),
        };

        let mut sh = sh.write().await;
        let status = sh.submit_vote(step_id, req.aggregator_id, req.result_hash, result_bytes);

        Ok(Response::new(proto::SubmitResultResponse {
            consensus_reached: status == VoteStatus::Consensus,
        }))
    }

    async fn get_consensus_bins(
        &self,
        request: Request<proto::GetConsensusBinsRequest>,
    ) -> Result<Response<proto::GetConsensusBinsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
        let sh = sh.read().await;

        let step = StepId::stats();
        match sh.get_consensus(step) {
            Some(data) => {
                let bins_result: proto::BinsResult =
                    prost::Message::decode(data.as_slice()).map_err(|e| {
                        Status::internal(format!("Failed to decode bins: {e}"))
                    })?;
                Ok(Response::new(proto::GetConsensusBinsResponse {
                    bins: bins_result.bins,
                    initial_prediction: bins_result.initial_prediction,
                    ready: true,
                }))
            }
            None => Ok(Response::new(proto::GetConsensusBinsResponse {
                bins: vec![],
                initial_prediction: 0.0,
                ready: false,
            })),
        }
    }

    async fn get_consensus_splits(
        &self,
        request: Request<proto::GetConsensusSplitsRequest>,
    ) -> Result<Response<proto::GetConsensusSplitsResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
        let sh = sh.read().await;

        let step = StepId::gradients(req.round_id, req.depth);
        match sh.get_consensus(step) {
            Some(data) => {
                let splits_result: proto::SplitsResult =
                    prost::Message::decode(data.as_slice()).map_err(|e| {
                        Status::internal(format!("Failed to decode splits: {e}"))
                    })?;
                Ok(Response::new(proto::GetConsensusSplitsResponse {
                    splits: splits_result.splits,
                    ready: true,
                }))
            }
            None => Ok(Response::new(proto::GetConsensusSplitsResponse {
                splits: HashMap::new(),
                ready: false,
            })),
        }
    }

    async fn get_consensus_model(
        &self,
        request: Request<proto::GetConsensusModelRequest>,
    ) -> Result<Response<proto::GetConsensusModelResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
        let sh = sh.read().await;

        if sh.phase() != Phase::TrainingComplete {
            return Ok(Response::new(proto::GetConsensusModelResponse {
                model: None,
            }));
        }

        // Model assembly from individual tree results will come later.
        // For now, return empty model placeholder when training is complete.
        Ok(Response::new(proto::GetConsensusModelResponse {
            model: None,
        }))
    }

    async fn get_run_state(
        &self,
        request: Request<proto::GetRunStateRequest>,
    ) -> Result<Response<proto::GetRunStateResponse>, Status> {
        let req = request.into_inner();
        let sh = self.get_run(&req.run_id).await?;
        let sh = sh.read().await;

        let phase = match sh.phase() {
            Phase::CollectingStats => proto::Phase::CollectingStats,
            Phase::FrozenStats => proto::Phase::FrozenStats,
            Phase::CollectingGradients => proto::Phase::CollectingGradients,
            Phase::FrozenGradients => proto::Phase::FrozenGradients,
            Phase::TrainingComplete => proto::Phase::TrainingComplete,
        };

        Ok(Response::new(proto::GetRunStateResponse {
            phase: phase as i32,
            round_id: sh.current_round_id(),
            depth: sh.current_depth(),
            expected_aggregators: self.expected_aggregators as i32,
            received_results: 0,
        }))
    }
}
