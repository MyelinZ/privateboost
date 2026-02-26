# gRPC Implementation Design for Privateboost

Revised design based on review of the [original gRPC interface design](2026-01-29-grpc-interface-design.md). This document incorporates all agreed changes and serves as the implementation spec.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| RPC style | Unary (request-response) | Mobile clients can't hold persistent streams. |
| Topology | Shareholders + Aggregator as servers | Clients connect outbound only (mobile-friendly). Aggregator is also a gRPC client of shareholders. |
| Client sync | Polling via `GetRoundState` | Stateless — clients connect, act, disconnect. |
| Coordination | Aggregator polls shareholders | Aggregator checks commitment counts on shareholders to know when enough clients have submitted. |
| Array serialization | Raw bytes + dtype + shape | Industry standard (TensorFlow, ONNX, Triton). Cross-language compatible. |
| IDs | UUIDs (string) | Session, client, and shareholder IDs are UUIDs. Node/round IDs remain integers. |
| Auth | Firebase ID tokens via gRPC metadata | Out of scope for proto definitions. Validated by server interceptors. |
| Session scoping | `session_id` on all shareholder RPCs | Shareholders can serve multiple concurrent sessions. |
| Dropout | Target count/fraction, no timeout | Aggregator waits for target participation. Operator intervention if stuck. |
| Cancellation | `CancelSession` RPC + `NOT_FOUND` status | No special `CANCELLED` phase. Clients discover cancellation via gRPC `NOT_FOUND` on next call. |
| min_clients | Deploy-time config per shareholder | Shareholders don't trust the aggregator to set their security threshold. |

---

## Shared Messages

```protobuf
syntax = "proto3";
package privateboost;

enum DType {
  FLOAT64 = 0;
  FLOAT32 = 1;
  INT32 = 2;
  INT64 = 3;
}

message NdArray {
  DType dtype = 1;
  repeated int64 shape = 2;
  bytes data = 3;              // raw little-endian bytes
}

message Share {
  int32 x = 1;                // evaluation point (1, 2, ..., n_shareholders)
  NdArray y = 2;              // polynomial evaluated at x
}
```

---

## ShareholderService

The shareholder is a passive secure accumulator. It stores Shamir shares indexed by commitment hash and returns sums on request. All RPCs include `session_id` to scope storage. Shareholders reject RPCs for cancelled sessions with `NOT_FOUND`.

```protobuf
service ShareholderService {
  // Client-facing: receive shares
  rpc SubmitStats(SubmitStatsRequest)         returns (SubmitStatsResponse);
  rpc SubmitGradients(SubmitGradientsRequest) returns (SubmitGradientsResponse);

  // Aggregator-facing: query commitments and sums
  rpc GetStatsCommitments(GetStatsCommitmentsRequest)       returns (GetStatsCommitmentsResponse);
  rpc GetGradientCommitments(GetGradientCommitmentsRequest) returns (GetGradientCommitmentsResponse);
  rpc GetGradientNodeIds(GetGradientNodeIdsRequest)         returns (GetGradientNodeIdsResponse);
  rpc GetStatsSum(GetStatsSumRequest)                       returns (GetStatsSumResponse);
  rpc GetGradientsSum(GetGradientsSumRequest)               returns (GetGradientsSumResponse);

  // Session lifecycle
  rpc CancelSession(CancelSessionRequest)                   returns (CancelSessionResponse);
  rpc Reset(ResetRequest)                                   returns (ResetResponse);
}
```

### Client-facing RPCs

**SubmitStats** — Client sends Shamir-shared feature statistics. Uses a deterministic commitment (`SHA256(session_id || client_id)`) for natural deduplication — same client always produces the same commitment hash, so shareholders overwrite rather than duplicate.

```protobuf
message SubmitStatsRequest {
  string session_id = 1;
  bytes commitment = 2;       // SHA256(session_id || client_id) — deterministic
  Share share = 3;
}
message SubmitStatsResponse {}
```

**SubmitGradients** — Client sends Shamir-shared gradient histogram. Uses a fresh nonce per round to prevent cross-round linkability.

```protobuf
message SubmitGradientsRequest {
  string session_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
  bytes commitment = 4;       // SHA256(round_id || client_id || fresh_nonce)
  Share share = 5;
  int32 node_id = 6;
}
message SubmitGradientsResponse {}
```

### Aggregator-facing RPCs

```protobuf
message GetStatsCommitmentsRequest {
  string session_id = 1;
}
message GetStatsCommitmentsResponse {
  repeated bytes commitments = 1;
}

message GetGradientCommitmentsRequest {
  string session_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
}
message GetGradientCommitmentsResponse {
  repeated bytes commitments = 1;
}

message GetGradientNodeIdsRequest {
  string session_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
}
message GetGradientNodeIdsResponse {
  repeated int32 node_ids = 1;
}

message GetStatsSumRequest {
  string session_id = 1;
  repeated bytes commitments = 2;
}
message GetStatsSumResponse {
  int32 x_coord = 1;
  NdArray sum = 2;
}

message GetGradientsSumRequest {
  string session_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
  repeated bytes commitments = 4;
  int32 node_id = 5;
}
message GetGradientsSumResponse {
  int32 x_coord = 1;
  NdArray sum = 2;
}

message CancelSessionRequest {
  string session_id = 1;
}
message CancelSessionResponse {}

message ResetRequest {
  string session_id = 1;
}
message ResetResponse {}
```

---

## AggregatorService

The aggregator orchestrates the protocol and serves as the coordination/discovery point for clients.

```protobuf
service AggregatorService {
  // Session management
  rpc JoinSession(JoinSessionRequest)             returns (JoinSessionResponse);
  rpc CancelSession(CancelSessionRequest)         returns (CancelSessionResponse);

  // Round state (lightweight poll)
  rpc GetRoundState(GetRoundStateRequest)         returns (GetRoundStateResponse);

  // Training data (full state for gradient computation)
  rpc GetTrainingState(GetTrainingStateRequest)   returns (GetTrainingStateResponse);

  // Final model (lightweight, inference only)
  rpc GetModel(GetModelRequest)                   returns (GetModelResponse);
}
```

### Session Management

```protobuf
message JoinSessionRequest {
  string session_id = 1;
}

message JoinSessionResponse {
  string session_id = 1;
  repeated ShareholderInfo shareholders = 2;
  int32 threshold = 3;
  TrainingConfig config = 4;
}

message ShareholderInfo {
  string id = 1;
  string address = 2;         // e.g. "https://sh1.example.com:443"
  int32 x_coord = 3;
}

message TrainingConfig {
  repeated FeatureSpec features = 1;
  string target_column = 2;
  string loss = 3;            // "squared" or "logistic"
  int32 n_bins = 4;
  int32 n_trees = 5;
  int32 max_depth = 6;
  double learning_rate = 7;
  double lambda_reg = 8;
  int32 min_clients = 9;
  oneof target {
    int32 target_count = 10;
    float target_fraction = 11;
  }
}

message FeatureSpec {
  int32 index = 1;
  string name = 2;
}
```

### Round State

Lightweight poll endpoint. Clients call this frequently to know what action to take.

```protobuf
enum Phase {
  WAITING_FOR_CLIENTS = 0;
  COLLECTING_GRADIENTS = 1;
  TRAINING_COMPLETE = 2;
}

message GetRoundStateRequest {
  string session_id = 1;
}
message GetRoundStateResponse {
  Phase phase = 1;
  int32 round_id = 2;
  int32 depth = 3;
}
```

### Training State

Returns everything a client needs to compute and submit gradients. Called once per depth iteration.

```protobuf
message GetTrainingStateRequest {
  string session_id = 1;
}
message GetTrainingStateResponse {
  Model model = 1;
  map<int32, SplitDecision> current_splits = 2;
  repeated BinConfiguration bins = 3;
  int32 round_id = 4;
  int32 depth = 5;
}

message BinConfiguration {
  int32 feature_idx = 1;
  NdArray edges = 2;
  NdArray inner_edges = 3;
  int32 n_bins = 4;
}

message SplitDecision {
  int32 node_id = 1;
  int32 feature_idx = 2;
  double threshold = 3;
  double gain = 4;
  int32 left_child_id = 5;
  int32 right_child_id = 6;
}
```

### Final Model

Lightweight endpoint for fetching the trained ensemble after training completes.

```protobuf
message GetModelRequest {
  string session_id = 1;
}
message GetModelResponse {
  Model model = 1;
}

message Model {
  double initial_prediction = 1;
  double learning_rate = 2;
  repeated Tree trees = 3;
}

message Tree {
  TreeNode root = 1;
}

message TreeNode {
  oneof node {
    SplitNode split = 1;
    LeafNode leaf = 2;
  }
}

message SplitNode {
  int32 feature_idx = 1;
  double threshold = 2;
  TreeNode left = 3;
  TreeNode right = 4;
}

message LeafNode {
  double value = 1;
}
```

---

## Client Protocol Flow

```
1. JOIN
   -> Aggregator.JoinSession(session_id)
   <- shareholders[], threshold, TrainingConfig
   Open gRPC channels to each shareholder.

2. SUBMIT STATS
   Compute stats vector [x0, x0^2, x1, x1^2, ..., target, target^2].
   Create Shamir shares with deterministic commitment: SHA256(session_id || client_id).
   -> Shareholder[i].SubmitStats(session_id, commitment, share)  for each shareholder

3. POLL
   Poll Aggregator.GetRoundState() until phase = COLLECTING_GRADIENTS.

4. GRADIENT ROUNDS (repeat for each round_id / depth)
   -> Aggregator.GetRoundState()
   <- phase = COLLECTING_GRADIENTS, round_id = R, depth = D

   -> Aggregator.GetTrainingState()
   <- model, current_splits, bins, round_id, depth

   Compute node_id from current_splits.
   Compute prediction from model, then gradient.
   Bin gradient into histogram, create Shamir shares with fresh nonce.
   -> Shareholder[i].SubmitGradients(session_id, R, D, commitment, share, node_id)

   Poll GetRoundState() until depth/round advances, then repeat.

5. COMPLETE
   <- phase = TRAINING_COMPLETE
   -> Aggregator.GetModel()
   <- final trained model
```

---

## Aggregator Internal Flow

Server-side loop (not exposed as RPCs):

```
1. Wait for enough clients to submit stats.
   Poll shareholders: GetStatsCommitments(session_id) to count.

2. INITIAL BINS
   When target reached:
     Select best threshold shareholders by commitment overlap.
     Call GetStatsSum(session_id, commitments) on each.
     Reconstruct via Lagrange -> means, variances.
     Compute bin configurations.
     Set phase = COLLECTING_GRADIENTS, round_id = 0, depth = 0.

3. GRADIENT ROUNDS
   For round_id = 0 to n_trees - 1:
     For depth = 0 to max_depth - 1:
       Poll shareholders: GetGradientCommitments(session_id, round_id, depth).
       When target reached:
         For each active node_id (from GetGradientNodeIds):
           Call GetGradientsSum(session_id, round_id, depth, commitments, node_id).
           Reconstruct gradient/hessian histograms.
           Find best split.
         Advance depth.
     Build tree from splits, add to model.

4. Set phase = TRAINING_COMPLETE.
```

---

## Dropout Handling

- **min_clients**: Deploy-time config on each shareholder. Hard security floor — shareholders refuse to return sums below it.
- **target_count / target_fraction**: Aggregator's practical participation threshold in TrainingConfig.
- **No timeout**: If the target is never reached, the system waits. Operator intervention required (cancel the session).
- **Client reconnection**: Poll `GetRoundState`, call `GetTrainingState`, participate from current round/depth onward.
- **Partial shareholder submission**: Commitment-overlap selection handles it — aggregator only uses commitments present across all selected shareholders.

---

## Cancellation

1. Admin calls `AggregatorService.CancelSession(session_id)`.
2. Aggregator stops its training loop, propagates `ShareholderService.CancelSession(session_id)` to each shareholder with idempotent retries.
3. Shareholders reject future RPCs for that session with `NOT_FOUND` and free stored shares.
4. Clients discover cancellation via `NOT_FOUND` on their next RPC call (GetRoundState, SubmitStats, etc.) and stop participating.

---

## Security Properties

| Property | How it's preserved |
|---|---|
| **Threshold security** | Shareholders only release sums, never individual shares. Reconstruction requires `threshold` shareholders. |
| **Aggregate-only** | Aggregator sees only reconstructed sums. Clients send shares directly to shareholders — aggregator never proxies share data. |
| **Anonymous** | Stats: deterministic commitment `SHA256(session_id \|\| client_id)`. Gradients: fresh nonce per round prevents cross-round linkability. |
| **Minimum N** | Shareholders enforce deploy-time `min_clients`. Refuse to return sums below threshold. |
| **Auth separation** | Firebase ID tokens in gRPC metadata identify who can participate. Commitment scheme ensures aggregator can't link participation to data. |
| **Transport security** | TLS required on all gRPC channels. |

---

## Communication Map

```
Client ----------------------------------------- Aggregator (gRPC server)
  |  JoinSession, GetRoundState,                      |
  |  GetTrainingState, GetModel                       |
  |                                                    |
  |                                                    | (gRPC client)
  |                                                    |
  +---- SubmitStats ----------------------> Shareholder 1 (gRPC server)
  +---- SubmitGradients ------------------> Shareholder 2 (gRPC server)
  +---- SubmitStats/Gradients ------------> Shareholder N (gRPC server)
                                                 ^
                                                 |
                                 Aggregator calls:
                                 GetStatsCommitments, GetGradientCommitments,
                                 GetGradientNodeIds, GetStatsSum,
                                 GetGradientsSum, CancelSession, Reset
```

---

## Changes from Original Design (2026-01-29)

| Change | Rationale |
|---|---|
| Removed `CANCELLED` phase | Clients discover cancellation via `NOT_FOUND` gRPC status. Simpler. |
| Removed `GetSplits` RPC | Splits folded into `GetTrainingState`. |
| Removed `GetBinConfigs` RPC | Bins folded into `GetTrainingState`. |
| Added `GetTrainingState` RPC | Single call returns model + splits + bins + round_id + depth. One RPC per depth iteration instead of three. |
| `GetModel` returns latest, no round_id in request | Client always wants latest state. Round context comes from `GetTrainingState`. |
| `min_clients` is deploy-time shareholder config | Shareholders don't trust the aggregator to set their security threshold. |
| Explicit TLS requirement | Noted in security properties. |
