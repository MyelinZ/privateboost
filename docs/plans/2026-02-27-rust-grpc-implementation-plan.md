# Rust gRPC Servers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port ShareholderService and AggregatorService gRPC servers to Rust, wire-compatible with existing Python clients.

**Architecture:** Single Rust crate (`rust/`) with two binaries. Shares `proto/privateboost.proto`. tonic for gRPC, tokio for async. Python simulation validates correctness.

**Tech Stack:** Rust 1.85+, tonic, prost, tokio, sha2, rand

**Reference:** Port algorithms from Python sources in `src/privateboost/`. The proto file at `proto/privateboost.proto` is the wire contract — do not modify it.

---

### Task 1: Scaffold Cargo project and proto codegen

**Files:**
- Create: `rust/Cargo.toml`
- Create: `rust/build.rs`
- Create: `rust/src/lib.rs`
- Create: `rust/src/bin/shareholder.rs` (placeholder)
- Create: `rust/src/bin/aggregator.rs` (placeholder)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "privateboost"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "shareholder"
path = "src/bin/shareholder.rs"

[[bin]]
name = "aggregator"
path = "src/bin/aggregator.rs"

[dependencies]
tonic = "0.13"
prost = "0.13"
tokio = { version = "1", features = ["full"] }
sha2 = "0.10"
rand = "0.9"
async-trait = "0.1"
anyhow = "1"

[build-dependencies]
tonic-build = "0.13"

[dev-dependencies]
tonic = "0.13"
tokio = { version = "1", features = ["full"] }
```

**Step 2: Create build.rs**

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["../proto/privateboost.proto"], &["../proto"])?;
    Ok(())
}
```

**Step 3: Create src/lib.rs**

```rust
pub mod crypto;
pub mod domain;
pub mod grpc;

pub mod proto {
    tonic::include_proto!("privateboost");
}
```

**Step 4: Create placeholder modules**

Create empty module files so it compiles:
- `rust/src/crypto/mod.rs` — `pub mod shamir; pub mod commitment;`
- `rust/src/crypto/shamir.rs` — empty
- `rust/src/crypto/commitment.rs` — empty
- `rust/src/domain/mod.rs` — `pub mod model; pub mod shareholder; pub mod aggregator;`
- `rust/src/domain/model.rs` — empty
- `rust/src/domain/shareholder.rs` — empty
- `rust/src/domain/aggregator.rs` — empty
- `rust/src/grpc/mod.rs` — `pub mod shareholder_service; pub mod aggregator_service; pub mod remote_shareholder;`
- `rust/src/grpc/shareholder_service.rs` — empty
- `rust/src/grpc/aggregator_service.rs` — empty
- `rust/src/grpc/remote_shareholder.rs` — empty
- `rust/src/bin/shareholder.rs` — `fn main() {}`
- `rust/src/bin/aggregator.rs` — `fn main() {}`

**Step 5: Verify it compiles**

```bash
cd rust && cargo build 2>&1
```

Expected: successful build, proto stubs generated.

**Step 6: Commit**

```bash
git add rust/
git commit -m "feat(rust): scaffold Cargo project with proto codegen"
```

---

### Task 2: Crypto — Shamir secret sharing

**Files:**
- Create: `rust/src/crypto/shamir.rs`
- Modify: `rust/src/crypto/mod.rs`

**Reference:** Port from `src/privateboost/crypto/shamir.py`

**What to implement:**

```rust
// rust/src/crypto/shamir.rs

#[derive(Debug, Clone)]
pub struct Share {
    pub x: i32,
    pub y: Vec<f64>,
}

/// Split values into n Shamir shares with given threshold.
///
/// Algorithm:
/// 1. For each element, create random polynomial of degree (threshold-1)
///    with the element as the constant term
/// 2. Evaluate polynomial at x = 1, 2, ..., n_parties
///
/// Uses Vandermonde matrix multiplication:
///   coeffs = [values; random_rows]  (threshold x n_values)
///   vander[i][j] = x_i^j           (n_parties x threshold)
///   y_matrix = vander @ coeffs      (n_parties x n_values)
pub fn share(values: &[f64], n_parties: usize, threshold: usize) -> Vec<Share>

/// Reconstruct values from threshold shares using Lagrange interpolation at x=0.
///
/// L_j(0) = product_{i!=j} (-x_i) / (x_j - x_i)
/// result = sum_j( L_j(0) * y_j )
pub fn reconstruct(shares: &[Share], threshold: usize) -> Vec<f64>
```

Key details:
- Random coefficients: `Uniform(-1e10, 1e10)` — use `rand::Rng::gen_range`
- Vandermonde: `vander[i][j] = (i+1).pow(j)` where i is 0-indexed, x = i+1
- Matrix multiply: manual nested loops (no numpy), `y[party][val] = sum_k(vander[party][k] * coeffs[k][val])`
- Lagrange: take first `threshold` shares, compute coefficients at x=0, weighted sum

**Tests** (in same file or `rust/tests/crypto_test.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shamir_reconstruction() {
        let values = vec![42.0, -17.5, 100.0];
        let shares = share(&values, 3, 2);
        let result = reconstruct(&shares[..2], 2);
        for (a, b) in values.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn test_shamir_linearity() {
        // Sum of shares = share of sum
        let v1 = vec![10.0, 20.0];
        let v2 = vec![30.0, 40.0];
        let s1 = share(&v1, 3, 2);
        let s2 = share(&v2, 3, 2);
        // Sum shares element-wise per party
        let summed: Vec<Share> = s1.iter().zip(s2.iter()).map(|(a, b)| {
            Share { x: a.x, y: a.y.iter().zip(b.y.iter()).map(|(x, y)| x + y).collect() }
        }).collect();
        let result = reconstruct(&summed[..2], 2);
        assert!((result[0] - 40.0).abs() < 1e-6);
        assert!((result[1] - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_3_of_5_threshold() {
        let values = vec![99.0];
        let shares = share(&values, 5, 3);
        let result = reconstruct(&shares[1..4], 3); // shares 2,3,4
        assert!((result[0] - 99.0).abs() < 1e-6);
    }
}
```

**Verify:** `cd rust && cargo test`

**Commit:** `git commit -m "feat(rust): implement Shamir secret sharing"`

---

### Task 3: Crypto — Commitments

**Files:**
- Create: `rust/src/crypto/commitment.rs`

**Reference:** Port from `src/privateboost/crypto/commitment.py`

```rust
use sha2::{Sha256, Digest};

pub type Commitment = [u8; 32];

/// SHA256(round_id_be_bytes || client_id_utf8 || nonce)
pub fn compute_commitment(round_id: i32, client_id: &str, nonce: &[u8]) -> Commitment {
    let mut hasher = Sha256::new();
    hasher.update(&round_id.to_be_bytes());
    hasher.update(client_id.as_bytes());
    hasher.update(nonce);
    hasher.finalize().into()
}

/// 32 random bytes
pub fn generate_nonce() -> [u8; 32] {
    rand::random()
}
```

**CRITICAL:** Python uses `round_id.to_bytes(8, "big")` (8 bytes, big-endian i64). Rust must match: use `(round_id as i64).to_be_bytes()` to get 8 bytes.

**Tests:**

```rust
#[test]
fn test_commitment_consistency() {
    let nonce = [0u8; 32];
    let c1 = compute_commitment(0, "client_1", &nonce);
    let c2 = compute_commitment(0, "client_1", &nonce);
    assert_eq!(c1, c2);
}

#[test]
fn test_commitment_round_separation() {
    let nonce = [0u8; 32];
    let c1 = compute_commitment(0, "client_1", &nonce);
    let c2 = compute_commitment(1, "client_1", &nonce);
    assert_ne!(c1, c2);
}
```

**Verify:** `cd rust && cargo test`

**Commit:** `git commit -m "feat(rust): implement commitment hashing"`

---

### Task 4: Domain types — Model, Tree, BinConfiguration, SplitDecision

**Files:**
- Create: `rust/src/domain/model.rs`

**Reference:** Port from `src/privateboost/tree.py` and `src/privateboost/messages.py`

```rust
pub type NodeId = i32;
pub type Depth = i32;

#[derive(Debug, Clone)]
pub enum TreeNode {
    Split(SplitNode),
    Leaf(LeafNode),
}

#[derive(Debug, Clone)]
pub struct SplitNode {
    pub feature_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left: Box<TreeNode>,
    pub right: Box<TreeNode>,
}

#[derive(Debug, Clone)]
pub struct LeafNode {
    pub value: f64,
    pub n_samples: usize,
}

#[derive(Debug, Clone)]
pub struct Tree {
    pub root: TreeNode,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub initial_prediction: f64,
    pub learning_rate: f64,
    pub trees: Vec<Tree>,
}

impl Model {
    pub fn new(initial_prediction: f64, learning_rate: f64) -> Self {
        Self { initial_prediction, learning_rate, trees: Vec::new() }
    }

    pub fn add_tree(&mut self, tree: Tree) {
        self.trees.push(tree);
    }

    /// Predict for a single sample (used in tests / reference)
    pub fn predict_one(&self, features: &[f64]) -> f64 {
        let mut pred = self.initial_prediction;
        for tree in &self.trees {
            pred += self.learning_rate * tree_predict(&tree.root, features);
        }
        pred
    }
}

fn tree_predict(node: &TreeNode, features: &[f64]) -> f64 {
    match node {
        TreeNode::Leaf(leaf) => leaf.value,
        TreeNode::Split(split) => {
            if features[split.feature_idx] <= split.threshold {
                tree_predict(&split.left, features)
            } else {
                tree_predict(&split.right, features)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinConfiguration {
    pub feature_idx: usize,
    pub edges: Vec<f64>,       // [-inf, ..., +inf]
    pub inner_edges: Vec<f64>, // no infinities
    pub n_bins: usize,
}

#[derive(Debug, Clone)]
pub struct SplitDecision {
    pub node_id: NodeId,
    pub feature_idx: usize,
    pub threshold: f64,
    pub gain: f64,
    pub left_child_id: NodeId,
    pub right_child_id: NodeId,
    pub g_left: f64,
    pub h_left: f64,
    pub g_right: f64,
    pub h_right: f64,
}

/// Aggregated gradient/hessian totals for a node
#[derive(Debug, Clone)]
pub struct NodeTotals {
    pub gradient_sum: f64,
    pub hessian_sum: f64,
}
```

No tests needed — these are pure data types. Tested implicitly via ShareHolder/Aggregator tests.

**Verify:** `cd rust && cargo build`

**Commit:** `git commit -m "feat(rust): add domain types (Model, Tree, BinConfig, SplitDecision)"`

---

### Task 5: ShareHolder

**Files:**
- Create: `rust/src/domain/shareholder.rs`

**Reference:** Port from `src/privateboost/shareholder.py` — same storage logic.

**What to implement:**

```rust
use std::collections::{HashMap, HashSet};
use crate::crypto::shamir::Share;
use crate::crypto::commitment::Commitment;
use crate::domain::model::{NodeId, Depth};

pub struct ShareHolder {
    pub min_clients: usize,
    stats: HashMap<Commitment, Share>,
    gradients: HashMap<Depth, HashMap<Commitment, HashMap<NodeId, Share>>>,
    current_round_id: i32,
}
```

Methods to implement (same logic as Python):
- `new(min_clients) -> Self`
- `current_round_id(&self) -> i32`
- `receive_stats(commitment, share)` — insert into stats map
- `receive_gradients(round_id, depth, commitment, share, node_id)` — clear gradients if new round, insert
- `get_stats_commitments() -> HashSet<Commitment>`
- `get_gradient_commitments(depth) -> HashSet<Commitment>`
- `get_gradient_node_ids(depth) -> HashSet<NodeId>`
- `get_stats_sum(commitments) -> Result<Vec<f64>>` — sum share.y vectors, enforce min_clients
- `get_gradients_sum(depth, commitments, node_id) -> Result<Vec<f64>>` — sum gradient shares
- `reset()` — clear all

Key detail for summing shares: iterate commitments, look up share, accumulate `y` vectors element-wise. Return the summed `Vec<f64>` (the x_coord is NOT included — the gRPC layer adds it).

**Tests:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_share(x: i32, values: &[f64]) -> Share {
        Share { x, y: values.to_vec() }
    }

    #[test]
    fn test_receive_and_get_stats() {
        let mut sh = ShareHolder::new(1);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_stats(c1, make_share(1, &[10.0, 20.0]));
        sh.receive_stats(c2, make_share(1, &[30.0, 40.0]));
        let commitments = sh.get_stats_commitments();
        assert_eq!(commitments.len(), 2);
    }

    #[test]
    fn test_stats_sum() {
        let mut sh = ShareHolder::new(2);
        let c1 = [1u8; 32];
        let c2 = [2u8; 32];
        sh.receive_stats(c1, make_share(1, &[10.0, 20.0]));
        sh.receive_stats(c2, make_share(1, &[30.0, 40.0]));
        let sum = sh.get_stats_sum(&[c1, c2]).unwrap();
        assert_eq!(sum, vec![40.0, 60.0]);
    }

    #[test]
    fn test_min_clients_enforcement() {
        let mut sh = ShareHolder::new(5);
        let c1 = [1u8; 32];
        sh.receive_stats(c1, make_share(1, &[1.0]));
        assert!(sh.get_stats_sum(&[c1]).is_err());
    }

    #[test]
    fn test_gradient_round_reset() {
        let mut sh = ShareHolder::new(1);
        let c1 = [1u8; 32];
        sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 1);
        // New round clears
        let c2 = [2u8; 32];
        sh.receive_gradients(1, 0, c2, make_share(1, &[2.0]), 0);
        assert_eq!(sh.get_gradient_commitments(0).len(), 1);
        assert!(sh.get_gradient_commitments(0).contains(&c2));
    }
}
```

**Verify:** `cd rust && cargo test`

**Commit:** `git commit -m "feat(rust): implement ShareHolder storage and aggregation"`

---

### Task 6: Aggregator

**Files:**
- Create: `rust/src/domain/aggregator.rs`

**Reference:** Port from `src/privateboost/aggregator.py` — the most complex file.

**What to implement:**

```rust
use async_trait::async_trait;

pub type Commitment = crate::crypto::commitment::Commitment;

#[async_trait]
pub trait ShareHolderClient: Send + Sync {
    fn x_coord(&self) -> i32;
    async fn get_stats_commitments(&self) -> anyhow::Result<HashSet<Commitment>>;
    async fn get_gradient_commitments(&self, depth: Depth) -> anyhow::Result<HashSet<Commitment>>;
    async fn get_gradient_node_ids(&self, depth: Depth) -> anyhow::Result<HashSet<NodeId>>;
    async fn get_stats_sum(&self, commitments: &[Commitment]) -> anyhow::Result<(i32, Vec<f64>)>;
    async fn get_gradients_sum(&self, depth: Depth, commitments: &[Commitment], node_id: NodeId) -> anyhow::Result<(i32, Vec<f64>)>;
}

pub struct Aggregator {
    shareholders: Vec<Box<dyn ShareHolderClient>>,
    n_bins: usize,
    threshold: usize,
    min_clients: usize,
    learning_rate: f64,
    lambda_reg: f64,
    // Internal state
    n_clients: usize,
    n_features: usize,
    bin_configs: Vec<BinConfiguration>,
    model: Model,
    next_node_id: NodeId,
    node_totals: HashMap<NodeId, NodeTotals>,
    splits: HashMap<NodeId, SplitDecision>,
}
```

Methods to port from Python `Aggregator`:

1. `async fn select_shareholders(&self, get_commitments)` — try all threshold-sized combos, pick max overlap. Use `itertools` or manual combinations. Since threshold is small (2-3), brute force is fine.

2. `async fn define_bins(&mut self)` — select shareholders, get stats sums, reconstruct via Lagrange, compute means/variances/stds, create BinConfigurations with edges `[-inf, linspace(μ-3σ, μ+3σ, n_bins+1), +inf]`.

3. `async fn compute_splits(&mut self, depth, min_samples) -> bool` — for each active node, reconstruct gradient histograms, scan bins for best split (gain = G²/(H+λ)). Store SplitDecision. Return true if any new splits found.

4. `fn finish_round(&mut self)` — build Tree from splits using recursive `build_node`. Leaf value = -G/(H+λ). Add tree to model.

Constants: `BIN_RANGE_STDS = 3`, `MIN_HESSIAN_THRESHOLD = 0.1`.

Port the algorithms line-by-line from the Python. The math is identical — just use `f64` instead of numpy.

For generating combinations of shareholders (select_shareholders), implement a simple combinations iterator or use the `itertools` crate. Since n is typically 3 and threshold 2, a manual nested loop works fine.

**Tests:** No unit tests for Aggregator in isolation — it requires ShareHolderClients. Tested via integration test (Task 10).

**Verify:** `cd rust && cargo build`

**Commit:** `git commit -m "feat(rust): implement Aggregator with ShareHolderClient trait"`

---

### Task 7: gRPC ShareholderService

**Files:**
- Create: `rust/src/grpc/shareholder_service.rs`

**Reference:** Port from `src/privateboost/grpc/shareholder_server.py`

**What to implement:**

```rust
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tonic::{Request, Response, Status};

use crate::proto::shareholder_service_server::ShareholderService;
use crate::domain::shareholder::ShareHolder;

pub struct ShareholderServiceImpl {
    min_clients: usize,
    sessions: Arc<Mutex<HashMap<String, Arc<RwLock<ShareHolder>>>>>,
    cancelled: Arc<Mutex<HashSet<String>>>,
}
```

Implement `ShareholderService` trait (generated by tonic):
- `submit_stats` — get_or_create session, convert proto Share to domain Share, call `sh.receive_stats`
- `submit_gradients` — same pattern, call `sh.receive_gradients`
- `get_stats_commitments` — get session, return commitments as `Vec<bytes>`
- `get_gradient_commitments` — check `current_round_id` matches request, return empty if not
- `get_gradient_node_ids` — same round_id check
- `get_stats_sum` — call `sh.get_stats_sum`, convert result to NdArray proto (f64 bytes, little-endian)
- `get_gradients_sum` — same pattern
- `cancel_session` — add to cancelled set, remove session
- `reset` — call `sh.reset()`

**Proto conversion helpers** (private functions in this file or a separate converters module):
- `fn share_from_proto(proto: proto::Share) -> Share` — parse NdArray bytes as `Vec<f64>`
- `fn vec_to_ndarray(values: &[f64]) -> proto::NdArray` — encode as f64 little-endian bytes

**Critical NdArray format:**
```rust
fn vec_to_ndarray(values: &[f64]) -> proto::NdArray {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    proto::NdArray {
        dtype: proto::DType::Float64 as i32,
        shape: vec![values.len() as i64],
        data: bytes,
    }
}

fn ndarray_to_vec(arr: &proto::NdArray) -> Vec<f64> {
    arr.data.chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}
```

**No separate test** — tested via integration test. The Python client is the test harness.

**Verify:** `cd rust && cargo build`

**Commit:** `git commit -m "feat(rust): implement ShareholderService gRPC server"`

---

### Task 8: RemoteShareHolder (gRPC adapter)

**Files:**
- Create: `rust/src/grpc/remote_shareholder.rs`

**Reference:** Port from `RemoteShareHolder` in `src/privateboost/grpc/aggregator_server.py`

**What to implement:**

```rust
use async_trait::async_trait;
use crate::domain::aggregator::ShareHolderClient;
use crate::proto::shareholder_service_client::ShareholderServiceClient;

pub struct RemoteShareHolder {
    x: i32,
    round_id: std::sync::atomic::AtomicI32,
    session_id: String,
    client: ShareholderServiceClient<tonic::transport::Channel>,
}
```

Implement `ShareHolderClient` trait: each method makes a gRPC call to the remote shareholder, converts proto responses to domain types.

`round_id` is set by the aggregator service before each training round (via `set_round_id` method or direct atomic store). Used in `get_gradient_commitments`, `get_gradient_node_ids`, `get_gradients_sum`.

**Verify:** `cd rust && cargo build`

**Commit:** `git commit -m "feat(rust): implement RemoteShareHolder gRPC adapter"`

---

### Task 9: gRPC AggregatorService

**Files:**
- Create: `rust/src/grpc/aggregator_service.rs`

**Reference:** Port from `src/privateboost/grpc/aggregator_server.py`

**What to implement:**

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{Mutex, RwLock};

pub struct SessionState {
    pub phase: i32,
    pub round_id: i32,
    pub depth: i32,
    pub cancelled: AtomicBool,
    pub aggregator: Aggregator,
    pub remote_shareholders: Vec<Arc<RemoteShareHolder>>,
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
    pub features: Vec<proto::FeatureSpec>,
    pub target_column: String,
}

pub struct AggregatorServiceImpl {
    sessions: Arc<Mutex<HashMap<String, Arc<RwLock<SessionState>>>>>,
    config: Arc<AggregatorConfig>,
}
```

Implement `AggregatorService` trait:
- `join_session` — create session + RemoteShareHolders, spawn `tokio::spawn(run_training(...))`
- `get_round_state` — read lock, return phase/round_id/depth
- `get_training_state` — read lock, convert model/splits/bins to proto
- `get_model` — read lock, convert model to proto
- `cancel_session` — set cancelled flag, forward cancel to shareholders

**Proto model conversion** (needed for GetTrainingState/GetModel):
- `fn model_to_proto(model: &Model) -> proto::Model`
- `fn tree_node_to_proto(node: &TreeNode) -> proto::TreeNode`
- `fn bin_config_to_proto(bc: &BinConfiguration) -> proto::BinConfiguration`
- `fn split_decision_to_proto(sd: &SplitDecision) -> proto::SplitDecision`

**Training loop** (`async fn run_training`):
Same flow as Python's `_run_training`:
1. Poll stats commitments until target reached (check cancelled between polls)
2. Write-lock, `aggregator.define_bins()`
3. For each round 0..n_trees:
   - Set round_id on RemoteShareHolders
   - For each depth 0..max_depth:
     - Poll gradient commitments until target reached
     - Write-lock, `aggregator.compute_splits(depth)`
   - Write-lock, `aggregator.finish_round()`
4. Set phase = TRAINING_COMPLETE

**Verify:** `cd rust && cargo build`

**Commit:** `git commit -m "feat(rust): implement AggregatorService gRPC server"`

---

### Task 10: Server binaries

**Files:**
- Create: `rust/src/bin/shareholder.rs`
- Create: `rust/src/bin/aggregator.rs`

**Reference:** Port from `src/privateboost/grpc/serve.py`

**shareholder.rs:**

```rust
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

    eprintln!("Shareholder server listening on {addr}");
    Server::builder()
        .add_service(ShareholderServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
```

**aggregator.rs:**

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port = std::env::var("PORT").unwrap_or_else(|_| "50052".into());
    let shareholders: Vec<String> = std::env::var("SHAREHOLDERS")
        .expect("SHAREHOLDERS env var required")
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    // Parse all other env vars: THRESHOLD, N_BINS, N_TREES, MAX_DEPTH,
    // LEARNING_RATE, LAMBDA_REG, MIN_CLIENTS, LOSS, TARGET_COUNT,
    // FEATURES, TARGET_COLUMN
    // Same defaults as docker-compose.yml

    let config = AggregatorConfig { /* ... */ };
    let service = AggregatorServiceImpl::new(config);

    eprintln!("Aggregator server listening on {addr}");
    Server::builder()
        .add_service(AggregatorServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}
```

**Verify:** `cd rust && cargo build --release`

**Commit:** `git commit -m "feat(rust): add shareholder and aggregator server binaries"`

---

### Task 11: Dockerfile and docker-compose override

**Files:**
- Create: `Dockerfile.rust`
- Create: `docker-compose.rust.yml`

**Dockerfile.rust:**

```dockerfile
FROM rust:1.85 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*
COPY proto/ proto/
COPY rust/ rust/
RUN cargo build --release --manifest-path rust/Cargo.toml

FROM debian:bookworm-slim
COPY --from=builder /app/rust/target/release/shareholder /usr/local/bin/
COPY --from=builder /app/rust/target/release/aggregator /usr/local/bin/
```

**docker-compose.rust.yml:**

```yaml
services:
  shareholder-1:
    build:
      context: .
      dockerfile: Dockerfile.rust
    command: []
    entrypoint: ["shareholder"]

  shareholder-2:
    build:
      context: .
      dockerfile: Dockerfile.rust
    command: []
    entrypoint: ["shareholder"]

  shareholder-3:
    build:
      context: .
      dockerfile: Dockerfile.rust
    command: []
    entrypoint: ["shareholder"]

  aggregator:
    build:
      context: .
      dockerfile: Dockerfile.rust
    command: []
    entrypoint: ["aggregator"]
```

Note: `command: []` clears the `command: shareholder` / `command: aggregator` from the base docker-compose.yml since the Rust images use entrypoint directly.

**Verify:**

```bash
docker compose -f docker-compose.yml -f docker-compose.rust.yml build
```

**Commit:** `git commit -m "feat(rust): add Dockerfile and docker-compose override"`

---

### Task 12: Integration test — Python simulation against Rust servers

**Step 1: Start Rust servers via Docker**

```bash
docker compose -f docker-compose.yml -f docker-compose.rust.yml up -d
docker compose ps  # verify all 4 running
```

**Step 2: Run Python simulation inside Docker network**

```bash
docker run --rm --network privateboost_default \
  -e AGGREGATOR=aggregator:50052 \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python privateboost-aggregator:latest \
  -m privateboost.grpc.simulate
```

Wait — this uses the Python image which may not exist if we only built Rust. Build Python image separately:

```bash
docker build -t privateboost-python -f Dockerfile .
docker run --rm --network privateboost_default \
  -e AGGREGATOR=aggregator:50052 \
  -e PYTHONUNBUFFERED=1 \
  privateboost-python \
  python -m privateboost.grpc.simulate
```

**Expected output:**
- All 15 rounds complete
- Test accuracy > 75%

**Step 3: Stop containers**

```bash
docker compose -f docker-compose.yml -f docker-compose.rust.yml down
```

**Step 4: Also verify Python unit tests still pass**

```bash
uv run pytest -v
```

**Commit:** `git commit -m "test(rust): verify wire compatibility via Python simulation"`

---

### Task 13: Final verification

**Step 1:** Run Rust tests: `cd rust && cargo test`

**Step 2:** Run Rust clippy: `cd rust && cargo clippy -- -D warnings`

**Step 3:** Run Python tests: `uv run pytest -v`

**Step 4:** Run Python lint: `make lint`

**Step 5:** Verify Docker integration (Task 12 steps) one final time.

All green = done.
