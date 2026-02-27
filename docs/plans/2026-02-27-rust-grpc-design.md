# Rust gRPC Servers Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port the ShareholderService and AggregatorService gRPC servers to Rust, keeping the Python clients unchanged.

**Architecture:** Single Rust crate in `rust/` with two binaries (`shareholder`, `aggregator`). Shares `proto/privateboost.proto` with the Python implementation. Python simulation script validates wire compatibility.

**Tech Stack:** tonic + prost (gRPC), tokio (async), sha2 (commitments), rand (nonce/polynomials)

---

## Project Structure

```
rust/
├── Cargo.toml
├── build.rs                       # tonic-build compiles ../proto/privateboost.proto
├── src/
│   ├── lib.rs                     # shared types, re-exports
│   ├── proto.rs                   # generated protobuf module
│   ├── crypto/
│   │   ├── mod.rs
│   │   ├── shamir.rs              # share(), reconstruct(), Share
│   │   └── commitment.rs          # compute_commitment(), generate_nonce()
│   ├── domain/
│   │   ├── mod.rs
│   │   ├── shareholder.rs         # ShareHolder (storage + sum)
│   │   ├── aggregator.rs          # Aggregator (reconstruction, bins, splits, trees)
│   │   └── model.rs               # Tree, Model, BinConfiguration, SplitDecision
│   ├── grpc/
│   │   ├── mod.rs
│   │   ├── shareholder_service.rs # tonic service impl
│   │   ├── aggregator_service.rs  # tonic service impl
│   │   └── remote_shareholder.rs  # ShareHolderClient impl over gRPC
│   └── bin/
│       ├── shareholder.rs         # main() for shareholder server
│       └── aggregator.rs          # main() for aggregator server
├── Dockerfile.rust
└── tests/
    ├── crypto_test.rs
    └── integration_test.rs
```

## Type Aliases

```rust
pub type Commitment = [u8; 32];
pub type NodeId = i32;
pub type Depth = i32;
```

## Core Domain Types

```rust
// crypto
pub struct Share { pub x: i32, pub y: Vec<f64> }
pub fn share(values: &[f64], n_parties: usize, threshold: usize) -> Vec<Share>;
pub fn reconstruct(shares: &[Share], threshold: usize) -> Vec<f64>;
pub fn compute_commitment(round_id: i32, client_id: &str, nonce: &[u8]) -> Commitment;
pub fn generate_nonce() -> Commitment;

// model
pub enum TreeNode { Split(SplitNode), Leaf(LeafNode) }
pub struct SplitNode { feature_idx, threshold, gain, left: Box<TreeNode>, right: Box<TreeNode> }
pub struct LeafNode { value: f64, n_samples: usize }
pub struct Tree { pub root: TreeNode }
pub struct Model { initial_prediction, learning_rate, trees: Vec<Tree> }
pub struct BinConfiguration { feature_idx, edges: Vec<f64>, inner_edges: Vec<f64>, n_bins }
pub struct SplitDecision { node_id, feature_idx, threshold, gain, left_child_id, right_child_id, g_left, h_left, g_right, h_right }
```

## ShareHolder

```rust
pub struct ShareHolder {
    min_clients: usize,
    stats: HashMap<Commitment, Share>,
    gradients: HashMap<Depth, HashMap<Commitment, HashMap<NodeId, Share>>>,
    current_round_id: i32,
}
```

Same logic as Python: `receive_stats`, `receive_gradients`, `get_stats_sum`, `get_gradients_sum`, etc. Auto-clears gradient store when `round_id` advances.

## Aggregator

Uses a trait for shareholder communication:

```rust
#[async_trait]
pub trait ShareHolderClient: Send + Sync {
    fn x_coord(&self) -> i32;
    async fn get_stats_commitments(&self) -> Result<HashSet<Commitment>>;
    async fn get_stats_sum(&self, commitments: &[Commitment]) -> Result<(i32, Vec<f64>)>;
    async fn get_gradient_commitments(&self, depth: Depth) -> Result<HashSet<Commitment>>;
    async fn get_gradient_node_ids(&self, depth: Depth) -> Result<HashSet<NodeId>>;
    async fn get_gradients_sum(&self, depth: Depth, commitments: &[Commitment], node_id: NodeId) -> Result<(i32, Vec<f64>)>;
}
```

`RemoteShareHolder` implements this trait via tonic gRPC stubs. `Aggregator` is generic over the trait. Same algorithms: `define_bins`, `compute_splits`, `finish_round`, `_select_shareholders`.

## gRPC Services

**ShareholderService:**
```rust
pub struct ShareholderService {
    min_clients: usize,
    sessions: Arc<Mutex<HashMap<String, Arc<RwLock<ShareHolder>>>>>,
    cancelled: Arc<Mutex<HashSet<String>>>,
}
```

**AggregatorService:**
```rust
pub struct SessionState {
    phase: Phase,
    round_id: i32,
    depth: i32,
    cancelled: AtomicBool,
    aggregator: Aggregator,
    remote_shareholders: Vec<RemoteShareHolder>,
}

pub struct AggregatorService {
    sessions: Arc<Mutex<HashMap<String, Arc<RwLock<SessionState>>>>>,
    config: AggregatorConfig,
}
```

`JoinSession` spawns `tokio::spawn(run_training(...))`. Training loop polls shareholders, computes bins/splits/trees. `RwLock` write held only during mutations; read lock for `GetTrainingState`/`GetModel`.

## Docker

**Dockerfile.rust** — multi-stage build:
```dockerfile
FROM rust:1.85 AS builder
WORKDIR /app
COPY rust/ rust/
COPY proto/ proto/
RUN apt-get update && apt-get install -y protobuf-compiler
RUN cargo build --release --manifest-path rust/Cargo.toml

FROM debian:bookworm-slim
COPY --from=builder /app/rust/target/release/shareholder /usr/local/bin/
COPY --from=builder /app/rust/target/release/aggregator /usr/local/bin/
ENTRYPOINT ["shareholder"]
```

**docker-compose.rust.yml** — override file that switches to Rust binaries:
```yaml
services:
  shareholder-1:
    build: { context: ., dockerfile: Dockerfile.rust }
    entrypoint: ["shareholder"]
  aggregator:
    build: { context: ., dockerfile: Dockerfile.rust }
    entrypoint: ["aggregator"]
```

Usage: `docker compose -f docker-compose.yml -f docker-compose.rust.yml up -d`

## Binaries

Both configured via the same env vars as docker-compose.yml already defines:
- Shareholder: `PORT`, `MIN_CLIENTS`
- Aggregator: `PORT`, `SHAREHOLDERS`, `THRESHOLD`, `N_BINS`, `N_TREES`, `MAX_DEPTH`, `LEARNING_RATE`, `LAMBDA_REG`, `MIN_CLIENTS`, `LOSS`, `TARGET_COUNT`

## Testing

**Unit tests** (Rust): Shamir reconstruction, additive homomorphism, commitment consistency.

**Integration test**: Run Python simulation (`uv run python -m privateboost.grpc.simulate`) against Rust servers. If it completes with >75% accuracy, the port is wire-compatible.

## Key Constraints

- **f64 everywhere** — Lagrange interpolation is numerically sensitive
- **NdArray endianness** — must match numpy's native byte order (little-endian on x86)
- **Proto compatibility** — same `proto/privateboost.proto`, no changes
- **Same env vars** — drop-in replacement for Python servers
