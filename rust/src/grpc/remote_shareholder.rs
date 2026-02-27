use std::collections::HashSet;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use tonic::transport::Channel;

use crate::crypto::commitment::Commitment;
use crate::domain::aggregator::ShareHolderClient;
use crate::domain::model::{Depth, NodeId};
use crate::grpc::ndarray_to_vec;
use crate::proto;
use crate::proto::shareholder_service_client::ShareholderServiceClient;

pub struct RemoteShareHolder {
    x: i32,
    round_id: AtomicI32,
    session_id: String,
    client: tokio::sync::Mutex<ShareholderServiceClient<Channel>>,
}

impl RemoteShareHolder {
    pub async fn connect(address: &str, x_coord: i32, session_id: String) -> Result<Self> {
        let url = if address.starts_with("http://") || address.starts_with("https://") {
            address.to_string()
        } else {
            format!("http://{address}")
        };
        let client = ShareholderServiceClient::connect(url).await?;
        Ok(Self {
            x: x_coord,
            round_id: AtomicI32::new(0),
            session_id,
            client: tokio::sync::Mutex::new(client),
        })
    }

    pub fn set_round_id(&self, round_id: i32) {
        self.round_id.store(round_id, Ordering::Relaxed);
    }

    pub async fn cancel_session(&self) -> Result<()> {
        let mut client = self.client.lock().await;
        client
            .cancel_session(crate::proto::CancelSessionRequest {
                session_id: self.session_id.clone(),
            })
            .await?;
        Ok(())
    }
}

fn bytes_to_commitment(b: &[u8]) -> Result<Commitment> {
    b.try_into()
        .map_err(|_| anyhow!("invalid commitment length"))
}

#[async_trait]
impl ShareHolderClient for Arc<RemoteShareHolder> {
    fn x_coord(&self) -> i32 {
        (**self).x_coord()
    }

    async fn get_stats_commitments(&self) -> Result<HashSet<Commitment>> {
        (**self).get_stats_commitments().await
    }

    async fn get_gradient_commitments(&self, depth: Depth) -> Result<HashSet<Commitment>> {
        (**self).get_gradient_commitments(depth).await
    }

    async fn get_gradient_node_ids(&self, depth: Depth) -> Result<HashSet<NodeId>> {
        (**self).get_gradient_node_ids(depth).await
    }

    async fn get_stats_sum(&self, commitments: &[Commitment]) -> Result<(i32, Vec<f64>)> {
        (**self).get_stats_sum(commitments).await
    }

    async fn get_gradients_sum(
        &self,
        depth: Depth,
        commitments: &[Commitment],
        node_id: NodeId,
    ) -> Result<(i32, Vec<f64>)> {
        (**self).get_gradients_sum(depth, commitments, node_id).await
    }
}

#[async_trait]
impl ShareHolderClient for RemoteShareHolder {
    fn x_coord(&self) -> i32 {
        self.x
    }

    async fn get_stats_commitments(&self) -> Result<HashSet<Commitment>> {
        let mut client = self.client.lock().await;
        let resp = client
            .get_stats_commitments(proto::GetStatsCommitmentsRequest {
                session_id: self.session_id.clone(),
            })
            .await?
            .into_inner();

        resp.commitments
            .iter()
            .map(|c| bytes_to_commitment(c))
            .collect()
    }

    async fn get_gradient_commitments(&self, depth: Depth) -> Result<HashSet<Commitment>> {
        let round_id = self.round_id.load(Ordering::Relaxed);
        let mut client = self.client.lock().await;
        let resp = client
            .get_gradient_commitments(proto::GetGradientCommitmentsRequest {
                session_id: self.session_id.clone(),
                round_id,
                depth,
            })
            .await?
            .into_inner();

        resp.commitments
            .iter()
            .map(|c| bytes_to_commitment(c))
            .collect()
    }

    async fn get_gradient_node_ids(&self, depth: Depth) -> Result<HashSet<NodeId>> {
        let round_id = self.round_id.load(Ordering::Relaxed);
        let mut client = self.client.lock().await;
        let resp = client
            .get_gradient_node_ids(proto::GetGradientNodeIdsRequest {
                session_id: self.session_id.clone(),
                round_id,
                depth,
            })
            .await?
            .into_inner();

        Ok(resp.node_ids.into_iter().collect())
    }

    async fn get_stats_sum(&self, commitments: &[Commitment]) -> Result<(i32, Vec<f64>)> {
        let proto_commitments: Vec<Vec<u8>> = commitments.iter().map(|c| c.to_vec()).collect();
        let mut client = self.client.lock().await;
        let resp = client
            .get_stats_sum(proto::GetStatsSumRequest {
                session_id: self.session_id.clone(),
                commitments: proto_commitments,
            })
            .await?
            .into_inner();

        let values = resp
            .sum
            .as_ref()
            .map(ndarray_to_vec)
            .unwrap_or_default();

        Ok((self.x, values))
    }

    async fn get_gradients_sum(
        &self,
        depth: Depth,
        commitments: &[Commitment],
        node_id: NodeId,
    ) -> Result<(i32, Vec<f64>)> {
        let round_id = self.round_id.load(Ordering::Relaxed);
        let proto_commitments: Vec<Vec<u8>> = commitments.iter().map(|c| c.to_vec()).collect();
        let mut client = self.client.lock().await;
        let resp = client
            .get_gradients_sum(proto::GetGradientsSumRequest {
                session_id: self.session_id.clone(),
                round_id,
                depth,
                commitments: proto_commitments,
                node_id,
            })
            .await?
            .into_inner();

        let values = resp
            .sum
            .as_ref()
            .map(ndarray_to_vec)
            .unwrap_or_default();

        Ok((self.x, values))
    }
}
