use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};
use tonic::{Request, Response, Status};

use crate::domain::aggregator::{Aggregator, ShareHolderClient};
use crate::domain::model::*;
use crate::grpc::remote_shareholder::RemoteShareHolder;
use crate::grpc::vec_to_ndarray;
use crate::proto;

// --- Session and config types ---

pub struct SessionState {
    pub phase: AtomicI32,
    pub round_id: AtomicI32,
    pub depth: AtomicI32,
    pub cancelled: AtomicBool,
    pub aggregator: RwLock<Aggregator>,
    pub remote_shareholders: Vec<Arc<RemoteShareHolder>>,
    training_started: AtomicBool,
}

pub struct AggregatorConfig {
    pub sh_addresses: Vec<String>,
    pub n_bins: usize,
    pub threshold: usize,
    pub min_clients: usize,
    pub learning_rate: f64,
    pub lambda_reg: f64,
    pub n_trees: usize,
    pub max_depth: usize,
    pub loss: String,
    pub target_count: Option<usize>,
    pub target_fraction: Option<f64>,
    pub features: Vec<proto::FeatureSpec>,
    pub target_column: String,
}

pub struct AggregatorServiceImpl {
    sessions: Arc<Mutex<HashMap<String, Arc<SessionState>>>>,
    config: Arc<AggregatorConfig>,
}

impl AggregatorServiceImpl {
    pub fn new(config: AggregatorConfig) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            config: Arc::new(config),
        }
    }

    async fn get_or_create_session(
        &self,
        session_id: &str,
    ) -> Result<Arc<SessionState>, Status> {
        let mut sessions = self.sessions.lock().await;
        if let Some(state) = sessions.get(session_id) {
            return Ok(Arc::clone(state));
        }

        let mut remote_shareholders = Vec::new();
        let mut sh_clients: Vec<Box<dyn ShareHolderClient>> = Vec::new();

        for (i, addr) in self.config.sh_addresses.iter().enumerate() {
            let rsh = RemoteShareHolder::connect(addr, (i + 1) as i32, session_id.to_string())
                .await
                .map_err(|e| {
                    Status::internal(format!("Failed to connect to shareholder {addr}: {e}"))
                })?;
            let rsh = Arc::new(rsh);
            remote_shareholders.push(Arc::clone(&rsh));
            sh_clients.push(Box::new(rsh.clone()));
        }

        let aggregator = Aggregator::new(
            sh_clients,
            self.config.n_bins,
            self.config.threshold,
            self.config.min_clients,
            self.config.learning_rate,
            self.config.lambda_reg,
        )
        .map_err(|e| Status::internal(format!("Failed to create aggregator: {e}")))?;

        let state = Arc::new(SessionState {
            phase: AtomicI32::new(proto::Phase::WaitingForClients as i32),
            round_id: AtomicI32::new(0),
            depth: AtomicI32::new(0),
            cancelled: AtomicBool::new(false),
            aggregator: RwLock::new(aggregator),
            remote_shareholders,
            training_started: AtomicBool::new(false),
        });

        sessions.insert(session_id.to_string(), Arc::clone(&state));
        Ok(state)
    }

    async fn get_session(&self, session_id: &str) -> Result<Arc<SessionState>, Status> {
        let sessions = self.sessions.lock().await;
        sessions
            .get(session_id)
            .map(Arc::clone)
            .ok_or_else(|| Status::not_found(format!("Session {session_id} not found")))
    }
}

// --- Training loop ---

async fn run_training(state: Arc<SessionState>, config: Arc<AggregatorConfig>) {
    if let Err(e) = run_training_inner(&state, &config).await {
        tracing::error!("Training error: {e}");
    }
}

async fn run_training_inner(
    state: &SessionState,
    config: &AggregatorConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let target = config.target_count.unwrap_or(config.min_clients);

    // Phase 1: Poll stats commitments until target reached
    loop {
        if state.cancelled.load(Ordering::Relaxed) {
            return Ok(());
        }
        let result = {
            let agg = state.aggregator.read().await;
            agg.shareholders[0].get_stats_commitments().await
        };
        match result {
            Ok(commitments) if commitments.len() >= target => break,
            _ => {}
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    if state.cancelled.load(Ordering::Relaxed) {
        return Ok(());
    }

    // Phase 2: Define bins
    {
        let mut agg = state.aggregator.write().await;
        agg.define_bins().await?;
    }

    // Phase 3: Training rounds
    for round_id in 0..config.n_trees {
        if state.cancelled.load(Ordering::Relaxed) {
            return Ok(());
        }

        state
            .phase
            .store(proto::Phase::CollectingGradients as i32, Ordering::Relaxed);
        state.round_id.store(round_id as i32, Ordering::Relaxed);
        state.depth.store(0, Ordering::Relaxed);

        for rsh in &state.remote_shareholders {
            rsh.set_round_id(round_id as i32);
        }

        for depth in 0..config.max_depth as i32 {
            if state.cancelled.load(Ordering::Relaxed) {
                return Ok(());
            }

            state.depth.store(depth, Ordering::Relaxed);

            // Poll gradient commitments until target reached
            loop {
                if state.cancelled.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let result = {
                    let agg = state.aggregator.read().await;
                    agg.shareholders[0].get_gradient_commitments(depth).await
                };
                match result {
                    Ok(commitments) if commitments.len() >= target => break,
                    _ => {}
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }

            if state.cancelled.load(Ordering::Relaxed) {
                return Ok(());
            }

            let found_splits = {
                let mut agg = state.aggregator.write().await;
                agg.compute_splits(depth, 1).await?
            };
            if !found_splits {
                break;
            }
        }

        {
            let mut agg = state.aggregator.write().await;
            agg.finish_round();
        }
    }

    state
        .phase
        .store(proto::Phase::TrainingComplete as i32, Ordering::Relaxed);
    Ok(())
}

// --- Proto conversion helpers ---

fn model_to_proto(model: &Model) -> proto::Model {
    proto::Model {
        initial_prediction: model.initial_prediction,
        learning_rate: model.learning_rate,
        trees: model
            .trees
            .iter()
            .map(|t| proto::Tree {
                root: Some(tree_node_to_proto(&t.root)),
            })
            .collect(),
    }
}

fn tree_node_to_proto(node: &TreeNode) -> proto::TreeNode {
    match node {
        TreeNode::Split(s) => proto::TreeNode {
            node: Some(proto::tree_node::Node::Split(Box::new(proto::SplitNode {
                feature_idx: s.feature_idx as i32,
                threshold: s.threshold,
                left: Some(Box::new(tree_node_to_proto(&s.left))),
                right: Some(Box::new(tree_node_to_proto(&s.right))),
            }))),
        },
        TreeNode::Leaf(l) => proto::TreeNode {
            node: Some(proto::tree_node::Node::Leaf(proto::LeafNode {
                value: l.value,
            })),
        },
    }
}

fn bin_config_to_proto(bc: &BinConfiguration) -> proto::BinConfiguration {
    proto::BinConfiguration {
        feature_idx: bc.feature_idx as i32,
        edges: Some(vec_to_ndarray(&bc.edges)),
        inner_edges: Some(vec_to_ndarray(&bc.inner_edges)),
        n_bins: bc.n_bins as i32,
    }
}

fn split_decision_to_proto(sd: &SplitDecision) -> proto::SplitDecision {
    proto::SplitDecision {
        node_id: sd.node_id,
        feature_idx: sd.feature_idx as i32,
        threshold: sd.threshold,
        gain: sd.gain,
        left_child_id: sd.left_child_id,
        right_child_id: sd.right_child_id,
    }
}

// --- gRPC trait implementation ---

#[tonic::async_trait]
impl proto::aggregator_service_server::AggregatorService for AggregatorServiceImpl {
    async fn join_session(
        &self,
        request: Request<proto::JoinSessionRequest>,
    ) -> Result<Response<proto::JoinSessionResponse>, Status> {
        let req = request.into_inner();
        let state = self.get_or_create_session(&req.session_id).await?;

        let shareholders: Vec<proto::ShareholderInfo> = self
            .config
            .sh_addresses
            .iter()
            .enumerate()
            .map(|(i, addr)| proto::ShareholderInfo {
                id: i.to_string(),
                address: addr.clone(),
                x_coord: (i + 1) as i32,
            })
            .collect();

        let target_oneof = if let Some(tc) = self.config.target_count {
            Some(proto::training_config::Target::TargetCount(tc as i32))
        } else {
            self.config
                .target_fraction
                .map(|tf| proto::training_config::Target::TargetFraction(tf as f32))
        };

        let config = proto::TrainingConfig {
            features: self.config.features.clone(),
            target_column: self.config.target_column.clone(),
            loss: self.config.loss.clone(),
            n_bins: self.config.n_bins as i32,
            n_trees: self.config.n_trees as i32,
            max_depth: self.config.max_depth as i32,
            learning_rate: self.config.learning_rate,
            lambda_reg: self.config.lambda_reg,
            min_clients: self.config.min_clients as i32,
            target: target_oneof,
        };

        // Spawn training if not already started
        if !state.training_started.swap(true, Ordering::AcqRel) {
            let state_clone = Arc::clone(&state);
            let config_clone = Arc::clone(&self.config);
            tokio::spawn(run_training(state_clone, config_clone));
        }

        Ok(Response::new(proto::JoinSessionResponse {
            session_id: req.session_id,
            shareholders,
            threshold: self.config.threshold as i32,
            config: Some(config),
        }))
    }

    async fn cancel_session(
        &self,
        request: Request<proto::CancelSessionRequest>,
    ) -> Result<Response<proto::CancelSessionResponse>, Status> {
        let req = request.into_inner();

        let state = {
            let mut sessions = self.sessions.lock().await;
            sessions.remove(&req.session_id)
        };

        if let Some(state) = state {
            state.cancelled.store(true, Ordering::Relaxed);
            for rsh in &state.remote_shareholders {
                let _ = rsh.cancel_session().await;
            }
        }

        Ok(Response::new(proto::CancelSessionResponse {}))
    }

    async fn get_round_state(
        &self,
        request: Request<proto::GetRoundStateRequest>,
    ) -> Result<Response<proto::GetRoundStateResponse>, Status> {
        let req = request.into_inner();
        let state = self.get_session(&req.session_id).await?;

        Ok(Response::new(proto::GetRoundStateResponse {
            phase: state.phase.load(Ordering::Relaxed),
            round_id: state.round_id.load(Ordering::Relaxed),
            depth: state.depth.load(Ordering::Relaxed),
        }))
    }

    async fn get_training_state(
        &self,
        request: Request<proto::GetTrainingStateRequest>,
    ) -> Result<Response<proto::GetTrainingStateResponse>, Status> {
        let req = request.into_inner();
        let state = self.get_session(&req.session_id).await?;

        let agg = state.aggregator.read().await;
        let model = model_to_proto(&agg.model);
        let current_splits: HashMap<i32, proto::SplitDecision> = agg
            .splits
            .iter()
            .map(|(&nid, sd)| (nid, split_decision_to_proto(sd)))
            .collect();
        let bins: Vec<proto::BinConfiguration> =
            agg.bin_configs.iter().map(bin_config_to_proto).collect();
        let round_id = state.round_id.load(Ordering::Relaxed);
        let depth = state.depth.load(Ordering::Relaxed);

        Ok(Response::new(proto::GetTrainingStateResponse {
            model: Some(model),
            current_splits,
            bins,
            round_id,
            depth,
        }))
    }

    async fn get_model(
        &self,
        request: Request<proto::GetModelRequest>,
    ) -> Result<Response<proto::GetModelResponse>, Status> {
        let req = request.into_inner();
        let state = self.get_session(&req.session_id).await?;

        let agg = state.aggregator.read().await;
        let model = model_to_proto(&agg.model);

        Ok(Response::new(proto::GetModelResponse {
            model: Some(model),
        }))
    }
}
