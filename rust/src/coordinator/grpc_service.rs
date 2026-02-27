use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::coordinator::service::{Coordinator, RunStatus};
use crate::domain::model::RunConfig;
use crate::proto;

pub struct CoordinatorGrpcService {
    coordinator: Arc<Coordinator>,
}

impl CoordinatorGrpcService {
    pub fn new(coordinator: Arc<Coordinator>) -> Self {
        Self { coordinator }
    }
}

#[tonic::async_trait]
impl proto::coordinator_service_server::CoordinatorService for CoordinatorGrpcService {
    async fn create_run(
        &self,
        request: Request<proto::CreateRunRequest>,
    ) -> Result<Response<proto::CreateRunResponse>, Status> {
        let req = request.into_inner();
        let config = req.config.ok_or_else(|| Status::invalid_argument("config required"))?;
        let run_config = training_config_to_domain(&config)?;
        let run_id = self.coordinator.create_run(run_config);
        Ok(Response::new(proto::CreateRunResponse { run_id }))
    }

    async fn cancel_run(
        &self,
        request: Request<proto::CancelRunRequest>,
    ) -> Result<Response<proto::CancelRunResponse>, Status> {
        let req = request.into_inner();
        self.coordinator.cancel_run(&req.run_id);
        Ok(Response::new(proto::CancelRunResponse {}))
    }

    async fn list_runs(
        &self,
        _request: Request<proto::ListRunsRequest>,
    ) -> Result<Response<proto::ListRunsResponse>, Status> {
        let runs = self.coordinator.list_runs();
        let run_infos = runs
            .into_iter()
            .map(|(run_id, status)| proto::RunInfo {
                run_id,
                status: match status {
                    RunStatus::Active => proto::RunStatus::RunActive as i32,
                    RunStatus::Cancelled => proto::RunStatus::RunCancelled as i32,
                    RunStatus::Complete => proto::RunStatus::RunComplete as i32,
                },
            })
            .collect();
        Ok(Response::new(proto::ListRunsResponse { runs: run_infos }))
    }

    async fn get_run_config(
        &self,
        request: Request<proto::GetRunConfigRequest>,
    ) -> Result<Response<proto::GetRunConfigResponse>, Status> {
        let req = request.into_inner();
        let config = self
            .coordinator
            .get_run_config(&req.run_id)
            .ok_or_else(|| Status::not_found(format!("Run {} not found", req.run_id)))?;
        Ok(Response::new(proto::GetRunConfigResponse {
            config: Some(domain_to_training_config(&config)),
        }))
    }
}

#[allow(clippy::result_large_err)]
fn training_config_to_domain(config: &proto::TrainingConfig) -> Result<RunConfig, Status> {
    let target_count = match &config.target {
        Some(proto::training_config::Target::TargetCount(tc)) => *tc as usize,
        _ => config.min_clients as usize,
    };
    Ok(RunConfig {
        run_id: String::new(),
        n_bins: config.n_bins as usize,
        threshold: 0,
        min_clients: config.min_clients as usize,
        learning_rate: config.learning_rate,
        lambda_reg: config.lambda_reg,
        n_trees: config.n_trees as usize,
        max_depth: config.max_depth as usize,
        loss: config.loss.clone(),
        target_count,
        features: config.features.iter().map(|f| f.name.clone()).collect(),
        target_column: config.target_column.clone(),
        feature_scales: config.features.iter().map(|f| f.scale).collect(),
        gradient_scale: config.gradient_scale,
    })
}

fn domain_to_training_config(config: &RunConfig) -> proto::TrainingConfig {
    proto::TrainingConfig {
        features: config
            .features
            .iter()
            .enumerate()
            .map(|(i, name)| proto::FeatureSpec {
                index: i as i32,
                name: name.clone(),
                scale: config.feature_scales.get(i).copied().unwrap_or(1.0),
            })
            .collect(),
        target_column: config.target_column.clone(),
        loss: config.loss.clone(),
        n_bins: config.n_bins as i32,
        n_trees: config.n_trees as i32,
        max_depth: config.max_depth as i32,
        learning_rate: config.learning_rate,
        lambda_reg: config.lambda_reg,
        min_clients: config.min_clients as i32,
        target: Some(proto::training_config::Target::TargetCount(
            config.target_count as i32,
        )),
        gradient_scale: config.gradient_scale,
    }
}
