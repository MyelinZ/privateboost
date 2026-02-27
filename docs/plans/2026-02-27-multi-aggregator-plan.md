# Multi-Aggregator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single trusted aggregator with 2f+1 untrusted aggregators that independently compute results, with shareholders running majority vote for consensus.

**Architecture:** Shareholders become the central hub — they store shares, drive round progression via a freeze-then-vote state machine, and serve consensus results to clients. Aggregators become stateless compute nodes. A new coordinator service manages run lifecycle. Clients no longer talk to aggregators.

**Tech Stack:** Rust, tonic/prost (gRPC), tokio (async), SHA-256 (result hashing for vote comparison)

**Design doc:** `docs/plans/2026-02-27-multi-aggregator-design.md`

---

## Overview

| Task | Description | Key files |
|------|-------------|-----------|
| 1 | Proto schema — new services, messages, enums | `proto/privateboost.proto` |
| 2 | Domain types — RunConfig, StepId, ConsensusResult | `rust/src/domain/model.rs` |
| 3 | Coordinator service | `rust/src/coordinator/` (new) |
| 4 | ShareHolder freeze mechanism | `rust/src/domain/shareholder.rs` |
| 5 | ShareHolder consensus voting | `rust/src/domain/shareholder.rs` |
| 6 | ShareHolder state machine | `rust/src/domain/shareholder.rs` |
| 7 | ShareholderService rewrite | `rust/src/grpc/shareholder_service.rs` |
| 8 | Aggregator stateless compute | `rust/src/domain/aggregator.rs` |
| 9 | AggregatorService rewrite | `rust/src/grpc/aggregator_service.rs` |
| 10 | Binaries — coordinator, updated shareholder/aggregator | `rust/src/bin/` |
| 11 | Docker Compose with multi-aggregator topology | `Dockerfile.rust`, `docker-compose.rust.yml` |
| 12 | Integration test | Docker-based end-to-end |
| 13 | Remove Python aggregator and shareholder servers | `src/privateboost/grpc/` |

---

### Task 1: Proto Schema

Update `proto/privateboost.proto` with new services, messages, and enums for the multi-aggregator protocol.

**Files:**
- Modify: `proto/privateboost.proto`

**Step 1: Add CoordinatorService**

Add after the existing services:

```protobuf
service CoordinatorService {
  rpc CreateRun(CreateRunRequest) returns (CreateRunResponse);
  rpc CancelRun(CancelRunRequest) returns (CancelRunResponse);
  rpc ListRuns(ListRunsRequest) returns (ListRunsResponse);
  rpc GetRunConfig(GetRunConfigRequest) returns (GetRunConfigResponse);
}

message CreateRunRequest {
  TrainingConfig config = 1;
}
message CreateRunResponse {
  string run_id = 1;
}
message CancelRunRequest {
  string run_id = 1;
}
message CancelRunResponse {}
message ListRunsRequest {}
message ListRunsResponse {
  repeated RunInfo runs = 1;
}
message RunInfo {
  string run_id = 1;
  RunStatus status = 2;
}
enum RunStatus {
  RUN_ACTIVE = 0;
  RUN_CANCELLED = 1;
  RUN_COMPLETE = 2;
}
message GetRunConfigRequest {
  string run_id = 1;
}
message GetRunConfigResponse {
  TrainingConfig config = 1;
}
```

**Step 2: Update Phase enum and add StepType**

Replace the existing Phase enum:

```protobuf
enum Phase {
  WAITING_FOR_CLIENTS = 0;
  COLLECTING_STATS = 1;
  FROZEN_STATS = 2;
  COLLECTING_GRADIENTS = 3;
  FROZEN_GRADIENTS = 4;
  TRAINING_COMPLETE = 5;
}

message StepId {
  StepType step_type = 1;
  int32 round_id = 2;
  int32 depth = 3;
}

enum StepType {
  STATS = 0;
  GRADIENTS = 1;
}
```

**Step 3: Add new ShareholderService RPCs**

Add to the existing `ShareholderService`:

```protobuf
// Aggregator submits computed result for voting
rpc SubmitResult(SubmitResultRequest) returns (SubmitResultResponse);

// Client fetches consensus bins
rpc GetConsensusBins(GetConsensusBinsRequest) returns (GetConsensusBinsResponse);

// Client fetches consensus splits for a given round/depth
rpc GetConsensusSplits(GetConsensusSplitsRequest) returns (GetConsensusSplitsResponse);

// Client fetches consensus model (after training complete)
rpc GetConsensusModel(GetConsensusModelRequest) returns (GetConsensusModelResponse);

// Anyone polls the current run state
rpc GetRunState(GetRunStateRequest) returns (GetRunStateResponse);
```

With messages:

```protobuf
message SubmitResultRequest {
  string run_id = 1;
  StepId step = 2;
  int32 aggregator_id = 3;
  bytes result_hash = 4;           // SHA-256 of serialized result
  oneof result {
    BinsResult bins_result = 5;
    SplitsResult splits_result = 6;
    TreeResult tree_result = 7;
  }
}
message SubmitResultResponse {
  bool consensus_reached = 1;
}

message BinsResult {
  repeated BinConfiguration bins = 1;
  double initial_prediction = 2;
}
message SplitsResult {
  map<int32, SplitDecision> splits = 1;
}
message TreeResult {
  Tree tree = 1;
}

message GetConsensusBinsRequest {
  string run_id = 1;
}
message GetConsensusBinsResponse {
  repeated BinConfiguration bins = 1;
  double initial_prediction = 2;
  bool ready = 3;
}

message GetConsensusSplitsRequest {
  string run_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
}
message GetConsensusSplitsResponse {
  map<int32, SplitDecision> splits = 1;
  bool ready = 2;
}

message GetConsensusModelRequest {
  string run_id = 1;
}
message GetConsensusModelResponse {
  Model model = 1;
}

message GetRunStateRequest {
  string run_id = 1;
}
message GetRunStateResponse {
  Phase phase = 1;
  int32 round_id = 2;
  int32 depth = 3;
  int32 expected_aggregators = 4;
  int32 received_results = 5;
}
```

**Step 4: Remove AggregatorService RPCs that clients used**

Remove `GetRoundState`, `GetTrainingState`, `GetModel` from `AggregatorService`. Replace with:

```protobuf
service AggregatorService {
  // Aggregator polls this to discover runs (from coordinator)
  // No client-facing RPCs — aggregators talk to shareholders only
}
```

Actually, the aggregator doesn't need a gRPC service at all in the new design — it's a client of shareholders. Remove `AggregatorService` entirely.

**Step 5: Rebuild proto**

Run: `cd rust && cargo build 2>&1`
Expected: compiles (Rust code will have errors referencing removed types — that's OK, we fix those in later tasks)

**Step 6: Commit**

```bash
git add proto/privateboost.proto
git commit -m "feat(proto): add CoordinatorService, consensus RPCs, remove AggregatorService"
```

---

### Task 2: Domain Types

Add new domain types for runs, steps, and consensus results.

**Files:**
- Modify: `rust/src/domain/model.rs`

**Step 1: Add RunConfig and StepId types**

Add to `model.rs`:

```rust
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub run_id: String,
    pub n_bins: usize,
    pub threshold: usize,
    pub min_clients: usize,
    pub learning_rate: f64,
    pub lambda_reg: f64,
    pub n_trees: usize,
    pub max_depth: usize,
    pub loss: String,
    pub target_count: usize,
    pub features: Vec<String>,
    pub target_column: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepType {
    Stats,
    Gradients,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StepId {
    pub step_type: StepType,
    pub round_id: i32,
    pub depth: i32,
}

impl StepId {
    pub fn stats() -> Self {
        Self { step_type: StepType::Stats, round_id: 0, depth: 0 }
    }

    pub fn gradients(round_id: i32, depth: i32) -> Self {
        Self { step_type: StepType::Gradients, round_id, depth }
    }
}
```

**Step 2: Add ConsensusResult enum**

```rust
#[derive(Debug, Clone)]
pub enum ConsensusResult {
    Bins {
        bins: Vec<BinConfiguration>,
        initial_prediction: f64,
    },
    Splits {
        splits: HashMap<NodeId, SplitDecision>,
    },
    Tree {
        tree: Tree,
    },
}
```

**Step 3: Run tests**

Run: `cd rust && cargo test 2>&1`
Expected: 16 passed (new types are just data, no logic yet)

**Step 4: Commit**

```bash
git add rust/src/domain/model.rs
git commit -m "feat(domain): add RunConfig, StepId, and ConsensusResult types"
```

---

### Task 3: Coordinator Service

New lightweight service for run lifecycle management.

**Files:**
- Create: `rust/src/coordinator/mod.rs`
- Create: `rust/src/coordinator/service.rs`
- Modify: `rust/src/lib.rs` (add `pub mod coordinator`)
- Create: `rust/src/bin/coordinator.rs`

**Step 1: Write tests for Coordinator**

In `rust/src/coordinator/service.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_run() {
        let coord = Coordinator::new();
        let config = make_test_config();
        let run_id = coord.create_run(config.clone());
        assert!(!run_id.is_empty());
        let fetched = coord.get_run_config(&run_id).unwrap();
        assert_eq!(fetched.n_bins, config.n_bins);
    }

    #[test]
    fn test_list_runs() {
        let coord = Coordinator::new();
        let config = make_test_config();
        let run_id = coord.create_run(config);
        let runs = coord.list_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].0, run_id);
        assert_eq!(runs[0].1, RunStatus::Active);
    }

    #[test]
    fn test_cancel_run() {
        let coord = Coordinator::new();
        let config = make_test_config();
        let run_id = coord.create_run(config);
        coord.cancel_run(&run_id);
        let runs = coord.list_runs();
        assert_eq!(runs[0].1, RunStatus::Cancelled);
    }

    #[test]
    fn test_get_unknown_run() {
        let coord = Coordinator::new();
        assert!(coord.get_run_config("unknown").is_none());
    }

    fn make_test_config() -> RunConfig {
        RunConfig {
            run_id: String::new(),
            n_bins: 10,
            threshold: 2,
            min_clients: 5,
            learning_rate: 0.15,
            lambda_reg: 2.0,
            n_trees: 3,
            max_depth: 3,
            loss: "squared".into(),
            target_count: 5,
            features: vec!["f1".into(), "f2".into()],
            target_column: "target".into(),
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cd rust && cargo test coordinator 2>&1`
Expected: FAIL (module doesn't exist yet)

**Step 3: Implement Coordinator**

`rust/src/coordinator/mod.rs`:
```rust
pub mod service;
```

`rust/src/coordinator/service.rs`:
```rust
use std::collections::HashMap;
use std::sync::Mutex;
use uuid::Uuid;

use crate::domain::model::RunConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Active,
    Cancelled,
    Complete,
}

pub struct Coordinator {
    runs: Mutex<HashMap<String, (RunConfig, RunStatus)>>,
}

impl Coordinator {
    pub fn new() -> Self {
        Self {
            runs: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_run(&self, mut config: RunConfig) -> String {
        let run_id = Uuid::new_v4().to_string();
        config.run_id = run_id.clone();
        let mut runs = self.runs.lock().unwrap();
        runs.insert(run_id.clone(), (config, RunStatus::Active));
        run_id
    }

    pub fn cancel_run(&self, run_id: &str) {
        let mut runs = self.runs.lock().unwrap();
        if let Some(entry) = runs.get_mut(run_id) {
            entry.1 = RunStatus::Cancelled;
        }
    }

    pub fn complete_run(&self, run_id: &str) {
        let mut runs = self.runs.lock().unwrap();
        if let Some(entry) = runs.get_mut(run_id) {
            entry.1 = RunStatus::Complete;
        }
    }

    pub fn list_runs(&self) -> Vec<(String, RunStatus)> {
        let runs = self.runs.lock().unwrap();
        runs.iter()
            .map(|(id, (_, status))| (id.clone(), *status))
            .collect()
    }

    pub fn get_run_config(&self, run_id: &str) -> Option<RunConfig> {
        let runs = self.runs.lock().unwrap();
        runs.get(run_id).map(|(config, _)| config.clone())
    }

    pub fn get_run_status(&self, run_id: &str) -> Option<RunStatus> {
        let runs = self.runs.lock().unwrap();
        runs.get(run_id).map(|(_, status)| *status)
    }
}
```

Add `uuid` dependency to `Cargo.toml`:
```toml
uuid = { version = "1", features = ["v4"] }
```

Add `pub mod coordinator` to `lib.rs`.

**Step 4: Run tests**

Run: `cd rust && cargo test coordinator 2>&1`
Expected: 4 passed

**Step 5: Implement CoordinatorService gRPC wrapper**

This is a thin gRPC layer over the `Coordinator` struct. Implement all four RPCs from the proto (`CreateRun`, `CancelRun`, `ListRuns`, `GetRunConfig`) by delegating to the `Coordinator` methods and converting between proto and domain types.

**Step 6: Implement coordinator binary**

`rust/src/bin/coordinator.rs`:
```rust
use privateboost::coordinator::service::Coordinator;
// ... gRPC server setup on PORT env var (default 50053)
```

Add `[[bin]]` section to `Cargo.toml`.

**Step 7: Run all tests**

Run: `cd rust && cargo test 2>&1`
Expected: 20 passed (16 existing + 4 new coordinator tests)

**Step 8: Commit**

```bash
git add rust/src/coordinator/ rust/src/lib.rs rust/src/bin/coordinator.rs rust/Cargo.toml
git commit -m "feat: add coordinator service for run lifecycle management"
```

---

### Task 4: ShareHolder Freeze Mechanism

Add the ability for a ShareHolder to freeze its commitment set once enough shares arrive.

**Files:**
- Modify: `rust/src/domain/shareholder.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn test_stats_freeze() {
    let mut sh = ShareHolder::new(2);
    assert!(!sh.is_stats_frozen());
    let c1 = [1u8; 32];
    let c2 = [2u8; 32];
    sh.receive_stats(c1, make_share(1, &[1.0]));
    assert!(!sh.is_stats_frozen());
    sh.receive_stats(c2, make_share(1, &[2.0]));
    // 2 commitments >= min_clients=2, should auto-freeze
    assert!(sh.is_stats_frozen());
    // Subsequent submissions rejected
    let c3 = [3u8; 32];
    assert!(!sh.receive_stats(c3, make_share(1, &[3.0])));
    assert_eq!(sh.get_stats_commitments().len(), 2);
}

#[test]
fn test_gradients_freeze() {
    let mut sh = ShareHolder::new(2);
    let c1 = [1u8; 32];
    let c2 = [2u8; 32];
    let c3 = [3u8; 32];
    sh.receive_gradients(0, 0, c1, make_share(1, &[1.0]), 0);
    assert!(!sh.is_gradients_frozen(0, 0));
    sh.receive_gradients(0, 0, c2, make_share(1, &[2.0]), 0);
    assert!(sh.is_gradients_frozen(0, 0));
    // Rejected after freeze
    assert!(!sh.receive_gradients(0, 0, c3, make_share(1, &[3.0]), 0));
}

#[test]
fn test_freeze_with_target() {
    let mut sh = ShareHolder::new(2);
    sh.set_target(5); // freeze at 5, not min_clients
    for i in 0..4 {
        sh.receive_stats([i; 32], make_share(1, &[1.0]));
    }
    assert!(!sh.is_stats_frozen());
    sh.receive_stats([4u8; 32], make_share(1, &[5.0]));
    assert!(sh.is_stats_frozen());
}
```

**Step 2: Run tests to verify they fail**

Run: `cd rust && cargo test shareholder 2>&1`
Expected: FAIL (methods don't exist)

**Step 3: Implement freeze mechanism**

Add to `ShareHolder`:
- `target: usize` field (defaults to `min_clients`, can be overridden with `set_target`)
- `stats_frozen: bool` field
- `gradients_frozen: HashMap<(i32, i32), bool>` field (keyed by `(round_id, depth)`)
- `is_stats_frozen(&self) -> bool`
- `is_gradients_frozen(&self, round_id, depth) -> bool`
- `set_target(&mut self, target: usize)`
- Change `receive_stats` return type to `bool` (false if frozen)
- Change `receive_gradients` return type to `bool` (false if frozen)
- Auto-freeze logic: after inserting, check if `len(commitments) >= target`

**Step 4: Run tests**

Run: `cd rust && cargo test shareholder 2>&1`
Expected: all pass (old tests still pass — freeze threshold not reached in existing tests with min_clients enforcement, update existing tests as needed)

**Step 5: Commit**

```bash
git add rust/src/domain/shareholder.rs
git commit -m "feat(domain): add freeze mechanism to ShareHolder"
```

---

### Task 5: ShareHolder Consensus Voting

Add the ability for a ShareHolder to collect results from multiple aggregators and determine consensus via majority vote.

**Files:**
- Modify: `rust/src/domain/shareholder.rs`
- Modify: `rust/src/domain/model.rs` (if needed for hashing)

**Step 1: Write failing tests**

```rust
#[test]
fn test_consensus_majority() {
    let mut sh = ShareHolder::new(1);
    let step = StepId::stats();
    let result_a = vec![1u8; 32]; // hash A
    let result_b = vec![2u8; 32]; // hash B

    // 3 aggregators: 2 agree on A, 1 says B
    // expected_aggregators=3, threshold f: can tolerate 1 malicious (2f+1=3, f=1)
    sh.set_expected_aggregators(3);
    assert_eq!(sh.submit_vote(step, 0, result_a.clone(), "data_a".into()), VoteStatus::Pending);
    assert_eq!(sh.submit_vote(step, 1, result_b.clone(), "data_b".into()), VoteStatus::Pending);
    assert_eq!(sh.submit_vote(step, 2, result_a.clone(), "data_a".into()), VoteStatus::Consensus);

    let consensus = sh.get_consensus(step).unwrap();
    assert_eq!(consensus, "data_a");
}

#[test]
fn test_consensus_no_majority() {
    let mut sh = ShareHolder::new(1);
    let step = StepId::stats();
    sh.set_expected_aggregators(3);
    // All 3 disagree
    sh.submit_vote(step, 0, vec![1u8; 32], "a".into());
    sh.submit_vote(step, 1, vec![2u8; 32], "b".into());
    let status = sh.submit_vote(step, 2, vec![3u8; 32], "c".into());
    assert_eq!(status, VoteStatus::Disputed);
}

#[test]
fn test_consensus_duplicate_aggregator() {
    let mut sh = ShareHolder::new(1);
    let step = StepId::stats();
    sh.set_expected_aggregators(3);
    sh.submit_vote(step, 0, vec![1u8; 32], "a".into());
    // Same aggregator_id again — should be rejected
    sh.submit_vote(step, 0, vec![1u8; 32], "a".into());
    // Still only 1 vote counted
    assert!(sh.get_consensus(step).is_none());
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement voting**

Add to `ShareHolder`:
- `expected_aggregators: usize` field
- `votes: HashMap<StepId, HashMap<i32, (Vec<u8>, Vec<u8>)>>` — step → aggregator_id → (hash, serialized_result)
- `consensus: HashMap<StepId, Vec<u8>>` — step → winning serialized result

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteStatus {
    Pending,
    Consensus,
    Disputed,
}
```

`submit_vote` logic:
1. Reject if aggregator already voted for this step
2. Insert vote
3. If `votes.len() >= expected_aggregators`:
   - Group by hash, find group with most votes
   - If majority > expected_aggregators / 2 → `Consensus`, store result
   - Else → `Disputed`
4. Else → `Pending`

**Step 4: Run tests**

Run: `cd rust && cargo test shareholder 2>&1`
Expected: all pass

**Step 5: Commit**

```bash
git add rust/src/domain/shareholder.rs
git commit -m "feat(domain): add consensus voting to ShareHolder"
```

---

### Task 6: ShareHolder State Machine

Add phase tracking so shareholders drive round progression.

**Files:**
- Modify: `rust/src/domain/shareholder.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn test_state_machine_stats_flow() {
    let mut sh = ShareHolder::new(2);
    sh.set_target(2);
    sh.set_expected_aggregators(1);

    assert_eq!(sh.phase(), Phase::CollectingStats);

    // Submit stats until frozen
    sh.receive_stats([1u8; 32], make_share(1, &[1.0]));
    sh.receive_stats([2u8; 32], make_share(1, &[2.0]));
    assert_eq!(sh.phase(), Phase::FrozenStats);

    // Aggregator submits bins consensus
    let step = StepId::stats();
    sh.submit_vote(step, 0, vec![1u8; 32], "bins".into());
    assert_eq!(sh.phase(), Phase::CollectingGradients);
    assert_eq!(sh.current_round(), 0);
    assert_eq!(sh.current_depth(), 0);
}

#[test]
fn test_state_machine_gradient_advance() {
    // Set up a shareholder already in gradient collection
    let mut sh = make_sh_in_gradient_phase(round_id=0, depth=0);

    // Submit gradients until frozen
    sh.receive_gradients(0, 0, [1u8; 32], make_share(1, &[1.0]), 0);
    sh.receive_gradients(0, 0, [2u8; 32], make_share(1, &[2.0]), 0);
    assert_eq!(sh.phase(), Phase::FrozenGradients);

    // Aggregator submits splits consensus
    let step = StepId::gradients(0, 0);
    sh.submit_vote(step, 0, vec![1u8; 32], "splits".into());
    // Advances to next depth
    assert_eq!(sh.phase(), Phase::CollectingGradients);
    assert_eq!(sh.current_depth(), 1);
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement state machine**

Add to `ShareHolder`:
- `phase: Phase` enum field
- `current_round: i32`
- `current_depth: i32`
- `max_depth: usize` and `n_trees: usize` (from run config)

Phase transitions:
- `CollectingStats` → `FrozenStats` (auto, when stats freeze)
- `FrozenStats` → `CollectingGradients(0, 0)` (when bins consensus reached)
- `CollectingGradients` → `FrozenGradients` (auto, when gradients freeze)
- `FrozenGradients` → `CollectingGradients(round, depth+1)` (when splits consensus, if depth < max_depth and splits found)
- `FrozenGradients` → `CollectingGradients(round+1, 0)` (when tree consensus, round complete)
- Last round complete → `TrainingComplete`

Update `submit_vote` to trigger phase transitions after consensus.

**Step 4: Run tests**

Run: `cd rust && cargo test shareholder 2>&1`
Expected: all pass

**Step 5: Commit**

```bash
git add rust/src/domain/shareholder.rs
git commit -m "feat(domain): add state machine to ShareHolder for round progression"
```

---

### Task 7: ShareholderService Rewrite

Rewrite the gRPC service to support the new protocol: run lifecycle from coordinator, consensus RPCs, and client-facing result endpoints.

**Files:**
- Modify: `rust/src/grpc/shareholder_service.rs`

**Step 1: Restructure ShareholderServiceImpl**

Replace the current session/cancelled management with run-based management:

```rust
pub struct ShareholderServiceImpl {
    min_clients: usize,
    coordinator_url: String,
    runs: Arc<Mutex<HashMap<String, Arc<RwLock<ShareHolder>>>>>,
}
```

**Step 2: Add coordinator polling background task**

On startup, spawn a task that polls the coordinator every 5 seconds via `ListRuns`. For each run:
- If new and active → create ShareHolder, configure from `GetRunConfig`
- If cancelled/complete → remove from local `runs` map

This replaces the `cancelled` HashSet entirely.

**Step 3: Implement new RPCs**

Keep existing share submission RPCs (`SubmitStats`, `SubmitGradients`) but change them to look up runs instead of sessions. Remove `cancel_session` and `reset` RPCs.

Add new RPCs:
- `SubmitResult` — aggregator submits computed result, delegates to `sh.submit_vote()`
- `GetConsensusBins` — client fetches consensus bins, returns `ready=false` if not yet available
- `GetConsensusSplits` — client fetches consensus splits for round/depth
- `GetConsensusModel` — client fetches final model
- `GetRunState` — returns current phase, round, depth, vote progress

Keep existing query RPCs for aggregators (`GetStatsCommitments`, `GetGradientCommitments`, `GetGradientNodeIds`, `GetStatsSum`, `GetGradientsSum`).

**Step 4: Update proto conversion helpers**

Add helpers for converting between proto and domain types for consensus results (BinsResult, SplitsResult, TreeResult).

**Step 5: Build and fix compilation errors**

Run: `cd rust && cargo build 2>&1`
Fix any remaining compilation issues.

**Step 6: Commit**

```bash
git add rust/src/grpc/shareholder_service.rs
git commit -m "feat(grpc): rewrite ShareholderService with consensus and coordinator polling"
```

---

### Task 8: Aggregator Stateless Compute

Refactor the aggregator domain to be a stateless compute engine. Remove the training loop — the aggregator now responds to shareholder state changes.

**Files:**
- Modify: `rust/src/domain/aggregator.rs`

**Step 1: Extract pure compute functions**

The current `Aggregator` struct holds shareholders, config, AND accumulated state (model, splits, node_totals). Split this:

Keep `Aggregator` but make it a simpler struct that just holds config and shareholders. The accumulated state (model, splits) moves to the shareholders (via consensus).

Key methods become:
```rust
impl Aggregator {
    /// Given stats shares from shareholders, compute bin configurations.
    pub async fn compute_bins(&self) -> Result<(Vec<BinConfiguration>, f64)>

    /// Given gradient shares, compute split decisions for a depth.
    /// Takes existing node_totals as input (from previous depths).
    pub async fn compute_splits(
        &self,
        depth: Depth,
        min_samples: usize,
        node_totals: &mut HashMap<NodeId, NodeTotals>,
        existing_splits: &HashMap<NodeId, SplitDecision>,
    ) -> Result<(HashMap<NodeId, SplitDecision>, bool)>

    /// Given splits and node totals, build the tree.
    pub fn build_tree(
        &self,
        splits: &HashMap<NodeId, SplitDecision>,
        node_totals: &HashMap<NodeId, NodeTotals>,
    ) -> Tree
}
```

`define_bins` → `compute_bins` (returns result instead of storing it)
`compute_splits` → takes/returns node_totals instead of mutating self
`finish_round` → `build_tree` (pure function)

**Step 2: Update tests**

Existing Aggregator tests (if any) need updating. The shamir/shareholder tests should still pass unchanged.

**Step 3: Run tests**

Run: `cd rust && cargo test 2>&1`
Expected: all pass

**Step 4: Commit**

```bash
git add rust/src/domain/aggregator.rs
git commit -m "refactor(domain): make Aggregator stateless compute engine"
```

---

### Task 9: AggregatorService Rewrite

Replace the old aggregator gRPC service with a polling loop that watches shareholder state and submits computed results.

**Files:**
- Modify: `rust/src/grpc/aggregator_service.rs`
- Modify: `rust/src/grpc/remote_shareholder.rs` (add new RPC clients)

**Step 1: Update RemoteShareHolder**

Add client methods for the new shareholder RPCs:
```rust
impl RemoteShareHolder {
    pub async fn submit_result(&self, run_id: &str, step: StepId, agg_id: i32, hash: Vec<u8>, result: ...) -> Result<bool>
    pub async fn get_run_state(&self, run_id: &str) -> Result<RunState>
}
```

**Step 2: Rewrite AggregatorService as a polling loop**

The aggregator no longer has a gRPC service. Instead, it's a main loop:

```rust
pub async fn run_aggregator(
    aggregator_id: i32,
    coordinator_url: String,
    shareholder_addresses: Vec<String>,
    threshold: usize,
) -> Result<()> {
    // 1. Poll coordinator for active runs
    // 2. For each run, connect to shareholders
    // 3. Poll shareholder run state
    // 4. When phase is FrozenStats:
    //    - Fetch commitments, reconstruct, compute bins
    //    - Submit bins result to all shareholders
    // 5. When phase is FrozenGradients:
    //    - Fetch commitments, reconstruct, compute splits
    //    - Submit splits result to all shareholders
    // 6. Repeat until TrainingComplete
}
```

**Step 3: Remove old AggregatorServiceImpl, SessionState, AggregatorConfig**

These are no longer needed — the aggregator is not a gRPC server.

**Step 4: Build**

Run: `cd rust && cargo build 2>&1`

**Step 5: Commit**

```bash
git add rust/src/grpc/aggregator_service.rs rust/src/grpc/remote_shareholder.rs
git commit -m "feat(grpc): rewrite aggregator as stateless polling loop"
```

---

### Task 10: Binaries

Update all three binaries for the new topology.

**Files:**
- Modify: `rust/src/bin/coordinator.rs`
- Modify: `rust/src/bin/shareholder.rs`
- Modify: `rust/src/bin/aggregator.rs`

**Step 1: Coordinator binary**

Already created in Task 3. Reads `PORT` from env, starts gRPC server.

**Step 2: Shareholder binary**

Update to accept:
- `PORT` (default: 50051)
- `MIN_CLIENTS` (default: 10)
- `COORDINATOR` (required: coordinator gRPC address)

Start shareholder gRPC server and coordinator polling task.

**Step 3: Aggregator binary**

Rewrite to be a client, not a server:
- `AGGREGATOR_ID` (required: unique integer)
- `COORDINATOR` (required: coordinator gRPC address)
- `SHAREHOLDERS` (required: comma-separated addresses)
- `THRESHOLD` (default: 2)

Calls `run_aggregator()` from Task 9.

**Step 4: Build and verify**

Run: `cd rust && cargo build --release 2>&1`
Expected: all three binaries compile

**Step 5: Commit**

```bash
git add rust/src/bin/
git commit -m "feat: update binaries for multi-aggregator topology"
```

---

### Task 11: Docker Compose

Update Docker configuration for the new topology: 1 coordinator, 3 shareholders, 3 aggregators.

**Files:**
- Modify: `Dockerfile.rust`
- Modify: `docker-compose.rust.yml`

**Step 1: Update Dockerfile**

Add coordinator binary to the build:

```dockerfile
COPY --from=builder /app/rust/target/release/coordinator /usr/local/bin/
```

**Step 2: Write docker-compose.rust.yml**

```yaml
services:
  coordinator:
    build:
      context: .
      dockerfile: Dockerfile.rust
    entrypoint: ["coordinator"]
    environment:
      PORT: "50053"
    ports:
      - "50053:50053"

  shareholder-1:
    build:
      context: .
      dockerfile: Dockerfile.rust
    entrypoint: ["shareholder"]
    environment:
      PORT: "50051"
      MIN_CLIENTS: "10"
      COORDINATOR: "coordinator:50053"

  shareholder-2:
    # same as shareholder-1

  shareholder-3:
    # same as shareholder-1

  aggregator-1:
    build:
      context: .
      dockerfile: Dockerfile.rust
    entrypoint: ["aggregator"]
    environment:
      AGGREGATOR_ID: "1"
      COORDINATOR: "coordinator:50053"
      SHAREHOLDERS: "shareholder-1:50051,shareholder-2:50051,shareholder-3:50051"
      THRESHOLD: "2"

  aggregator-2:
    # same with AGGREGATOR_ID=2

  aggregator-3:
    # same with AGGREGATOR_ID=3
```

**Step 3: Build Docker images**

Run: `docker compose -f docker-compose.rust.yml build`

**Step 4: Commit**

```bash
git add Dockerfile.rust docker-compose.rust.yml
git commit -m "feat(docker): multi-aggregator topology with coordinator"
```

---

### Task 12: Integration Test

End-to-end test: create a run via coordinator, run Python clients against Rust shareholders, verify consensus training completes.

**Files:**
- Modify: `src/privateboost/grpc/simulate.py` (update for new protocol)

**Step 1: Update simulation script**

The Python simulation client needs to:
1. Call coordinator to create a run (new)
2. Submit shares to shareholders (unchanged)
3. Poll shareholders for run state (was: poll aggregator)
4. Fetch consensus bins from shareholders (was: from aggregator)
5. Fetch consensus splits from shareholders (was: from aggregator)
6. Fetch final model from shareholders (was: from aggregator)

**Step 2: Run integration test**

```bash
docker compose -f docker-compose.rust.yml up -d
# Wait for services
python -m privateboost.grpc.simulate \
  --coordinator localhost:50053 \
  --shareholders localhost:50051,localhost:50052,localhost:50053 \
  --dataset heart_disease
```

Expected: training completes, >75% accuracy, all 3 aggregators agree on every step.

**Step 3: Commit**

```bash
git add src/privateboost/grpc/simulate.py
git commit -m "feat: update simulation for multi-aggregator protocol"
```

---

### Task 13: Remove Python Aggregator and Shareholder Servers

The Python gRPC servers are replaced by Rust. Remove them.

**Files:**
- Delete: `src/privateboost/grpc/aggregator_server.py`
- Delete: `src/privateboost/grpc/shareholder_server.py`
- Modify: `src/privateboost/grpc/__init__.py` (remove exports)
- Delete: `docker-compose.yml` (the Python-based compose file)

**Step 1: Remove files**

```bash
rm src/privateboost/grpc/aggregator_server.py
rm src/privateboost/grpc/shareholder_server.py
```

**Step 2: Update imports**

Remove references to deleted modules from `__init__.py` and any other importers.

**Step 3: Verify Python package still works**

Run: `cd /home/bold/workspace/privateboost && make lint`
Expected: clean (ruff passes)

**Step 4: Commit**

```bash
git add -u
git commit -m "chore: remove Python aggregator and shareholder servers (replaced by Rust)"
```
