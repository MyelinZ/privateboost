# gRPC Interface Design for Privateboost

## Overview

This document defines gRPC services for the three parties in the privateboost protocol: **shareholders**, **aggregator**, and **clients**. The design targets thousands of mobile clients communicating over the internet.

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| RPC style | Unary (request-response) | Mobile clients can't hold persistent streams. Flower migrated from bidi streaming to unary for the same reason. |
| Topology | Shareholders + Aggregator as servers | Clients connect outbound only (mobile-friendly). Aggregator is also a gRPC client of shareholders. |
| Client sync | Polling via `GetRoundState` | Stateless — clients connect, act, disconnect. No persistent connections. |
| Coordination | Aggregator polls shareholders | Aggregator checks commitment counts on shareholders to know when enough clients have submitted. |
| Array serialization | Raw bytes + dtype + shape | Industry standard (TensorFlow, ONNX, Triton). Cross-language compatible (Python, Rust). |
| IDs | UUIDs (string) | Session, client, and shareholder IDs are UUIDs. Node IDs and round IDs remain integers (structural indices). |
| Auth | Firebase ID tokens via gRPC metadata | Out of scope for proto definitions. Token passed as `authorization` metadata header, validated by server interceptors. |
| Session scoping | `session_id` on all shareholder RPCs | Shareholders can serve multiple concurrent sessions. Fully stateless — no implicit "current session". |
| Dropout | Target count/fraction, no timeout | Aggregator waits for target participation. Operator intervention if stuck. |
| Cancellation | `CANCELLED` phase + `CancelSession` RPC | Admin cancels via aggregator, which propagates to shareholders. Clients discover via polling. Eventually consistent with idempotent retries. |

---

## Shared Messages

```protobuf
syntax = "proto3";
package privateboost;

// -- Tensor representation (follows TensorFlow/ONNX convention) --

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

// -- Shamir share --

message Share {
  int32 x = 1;                // evaluation point (1, 2, ..., n_shareholders)
  NdArray y = 2;              // polynomial evaluated at x
}
```

---

## Shareholder Service

The shareholder is a passive secure accumulator. It stores Shamir shares indexed by commitment hash and returns sums on request.

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

All shareholder RPCs include `session_id` to scope storage. Shareholders reject RPCs for cancelled sessions with `FAILED_PRECONDITION`.

### Client-facing RPCs

**SubmitStats** — Client sends Shamir-shared feature statistics. Can be called at any point after joining. Uses a deterministic commitment for natural deduplication — same client always produces the same commitment hash, so shareholders overwrite rather than duplicate.

```protobuf
message SubmitStatsRequest {
  string session_id = 1;      // UUID of the training session
  bytes commitment = 2;       // SHA256(session_id || client_id) — deterministic, no random nonce
  Share share = 3;            // share of [x₀, x₀², x₁, x₁², ..., target, target²]
}
message SubmitStatsResponse {}
```

The `client_id` used here is a secret known only to the client, separate from the Firebase UID. The aggregator cannot reverse the hash to de-anonymize stats.

**SubmitGradients** — Client sends Shamir-shared gradient histogram. Called once per client per tree depth.

```protobuf
message SubmitGradientsRequest {
  string session_id = 1;      // UUID of the training session
  int32 round_id = 2;         // tree index (0..n_trees)
  int32 depth = 3;            // depth level within the tree
  bytes commitment = 4;       // SHA256(round_id || client_id || fresh_nonce)
  Share share = 5;            // share of binned [g, h] histogram
  int32 node_id = 6;          // tree node this client belongs to
}
message SubmitGradientsResponse {}
```

### Aggregator-facing RPCs

**GetStatsCommitments** — Returns the set of commitment hashes that have stats shares. Used by the aggregator to discover how many clients have submitted.

```protobuf
message GetStatsCommitmentsRequest {
  string session_id = 1;
}
message GetStatsCommitmentsResponse {
  repeated bytes commitments = 1;
}
```

**GetGradientCommitments** — Returns commitment hashes with gradient shares at a specific round and depth.

```protobuf
message GetGradientCommitmentsRequest {
  string session_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
}
message GetGradientCommitmentsResponse {
  repeated bytes commitments = 1;
}
```

**GetGradientNodeIds** — Returns all node IDs that have gradient data at a round and depth. Used to discover active nodes to split.

```protobuf
message GetGradientNodeIdsRequest {
  string session_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
}
message GetGradientNodeIdsResponse {
  repeated int32 node_ids = 1;
}
```

**GetStatsSum** — Sums the shares for the requested commitments and returns (x_coord, sum). Enforces min_clients threshold — returns an error if `len(commitments) < min_clients`.

```protobuf
message GetStatsSumRequest {
  string session_id = 1;
  repeated bytes commitments = 2;
}
message GetStatsSumResponse {
  int32 x_coord = 1;          // shareholder's Shamir x-coordinate
  NdArray sum = 2;            // sum of share values for requested commitments
}
```

**GetGradientsSum** — Same as GetStatsSum but for gradient histograms, scoped to a round, depth, and node.

```protobuf
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
```

**CancelSession** — Marks a session as cancelled. Shareholder rejects all future RPCs for this session with `FAILED_PRECONDITION` and frees stored shares. Idempotent — safe to call multiple times.

```protobuf
message CancelSessionRequest {
  string session_id = 1;
}
message CancelSessionResponse {}
```

**Reset** — Clears all stored shares for a session. Called between training rounds or at session end.

```protobuf
message ResetRequest {
  string session_id = 1;
}
message ResetResponse {}
```

---

## Aggregator Service

The aggregator orchestrates the protocol and serves as the coordination/discovery point for clients.

```protobuf
service AggregatorService {
  // Session management
  rpc JoinSession(JoinSessionRequest)       returns (JoinSessionResponse);
  rpc CancelSession(CancelSessionRequest)   returns (CancelSessionResponse);

  // Round state (clients poll this)
  rpc GetRoundState(GetRoundStateRequest)   returns (GetRoundStateResponse);

  // Data distribution
  rpc GetBinConfigs(GetBinConfigsRequest)   returns (GetBinConfigsResponse);
  rpc GetSplits(GetSplitsRequest)           returns (GetSplitsResponse);
  rpc GetModel(GetModelRequest)             returns (GetModelResponse);
}
```

### Session Management

**JoinSession** — Client registers for a training session. Returns shareholder addresses, Shamir parameters, and the full training configuration. Returns `FAILED_PRECONDITION` if the session is cancelled.

```protobuf
message JoinSessionRequest {
  string session_id = 1;      // UUID v4 of the training session
  // Client identity derived from Firebase auth token in gRPC metadata.
  // The cryptographic client_id used in commitments stays private to the client.
}

message JoinSessionResponse {
  string session_id = 1;
  repeated ShareholderInfo shareholders = 2;
  int32 threshold = 3;        // m in m-of-n Shamir
  TrainingConfig config = 4;
}

message ShareholderInfo {
  string id = 1;               // UUID
  string address = 2;          // URL, e.g. "https://sh1.example.com:443"
  int32 x_coord = 3;           // Shamir evaluation point
}

message TrainingConfig {
  repeated FeatureSpec features = 1;
  string target_column = 2;
  string loss = 3;             // "squared" or "logistic"
  int32 n_bins = 4;
  int32 n_trees = 5;
  int32 max_depth = 6;
  double learning_rate = 7;
  double lambda_reg = 8;
  int32 min_clients = 9;      // hard security minimum
  oneof target {
    int32 target_count = 10;   // proceed when this many clients submit
    float target_fraction = 11; // proceed when this fraction of registered clients submit
  }
}

message FeatureSpec {
  int32 index = 1;             // feature index in the protocol (0, 1, 2, ...)
  string name = 2;             // column name in the client's local data
}
```

**CancelSession** — Admin cancels a training session. The aggregator marks the session as cancelled, stops its training loop, and propagates the cancellation to all shareholders (with idempotent retries for unreachable shareholders).

```protobuf
// Reuses the same CancelSessionRequest/Response as the shareholder service
```

### Round State

**GetRoundState** — Client polls to learn the current protocol phase and what action to take.

```protobuf
enum Phase {
  WAITING_FOR_CLIENTS = 0;     // not enough clients have submitted stats yet
  COLLECTING_GRADIENTS = 1;    // clients should submit gradients for current round/depth
  TRAINING_COMPLETE = 2;       // training finished, fetch final model
  CANCELLED = 3;               // session cancelled by admin — stop participating
}

message GetRoundStateRequest {
  string session_id = 1;       // UUID v4
}
message GetRoundStateResponse {
  Phase phase = 1;
  int32 round_id = 2;         // current tree index
  int32 depth = 3;            // current depth level
}
```

### Data Distribution

**GetBinConfigs** — Client fetches histogram bin definitions for a specific round. Bins may be recomputed between rounds as new clients contribute stats.

```protobuf
message GetBinConfigsRequest {
  string session_id = 1;       // UUID v4
  int32 round_id = 2;
}
message GetBinConfigsResponse {
  repeated BinConfiguration bins = 1;
}

message BinConfiguration {
  int32 feature_idx = 1;
  NdArray edges = 2;           // bin edges including ±∞
  NdArray inner_edges = 3;     // finite bin edges only
  int32 n_bins = 4;
}
```

**GetSplits** — Client fetches split decisions for the current round to compute its tree node assignment.

```protobuf
message GetSplitsRequest {
  string session_id = 1;       // UUID v4
  int32 round_id = 2;
}
message GetSplitsResponse {
  map<int32, SplitDecision> splits = 1;  // node_id → split
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

**GetModel** — Client fetches the current ensemble model to compute predictions and gradients.

```protobuf
message GetModelRequest {
  string session_id = 1;       // UUID v4
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

A client executes this sequence of RPCs during a training session:

```
1. JOIN
   → Aggregator.JoinSession(session_id)
   ← shareholders[], threshold, TrainingConfig
   Open gRPC channels to each shareholder.

2. SUBMIT STATS (can happen at any time after joining)
   Compute stats vector [x₀, x₀², x₁, x₁², ..., target, target²].
   Create Shamir shares with deterministic commitment: SHA256(session_id || client_id).
   → Shareholder[i].SubmitStats(session_id, commitment, share)  for each shareholder
   Shareholders deduplicate by commitment hash (same client → same hash → overwrite).

3. WAIT FOR TRAINING TO START
   Poll Aggregator.GetRoundState() until phase = COLLECTING_GRADIENTS.

4. GRADIENT ROUNDS (repeat for each round_id / depth)
   → Aggregator.GetRoundState()
   ← phase = COLLECTING_GRADIENTS, round_id = R, depth = D

   → Aggregator.GetBinConfigs(R)    (fetch bins for this round — may change between rounds)
   → Aggregator.GetModel()          (compute current prediction)
   → Aggregator.GetSplits(R)        (find node assignment)

   Compute gradients, bin into histogram, create Shamir shares with random nonce.
   → Shareholder[i].SubmitGradients(session_id, round_id, depth, commitment, share, node_id)

   Poll GetRoundState() until depth/round advances, then repeat.

5. COMPLETE
   → Aggregator.GetRoundState()
   ← phase = TRAINING_COMPLETE
   → Aggregator.GetModel()          (fetch final trained model)
```

Clients that join mid-training submit stats immediately and start participating
from the current round/depth. Their stats are incorporated when bins are
recomputed at the start of the next round.

---

## Aggregator Internal Flow

The aggregator runs this loop server-side (not exposed as RPCs):

```
1. Wait for enough clients to join and submit stats (target_count or target_fraction).
   Poll shareholders: GetStatsCommitments(session_id) to count stats submissions.

2. INITIAL BINS
   When target reached:
     Select best `threshold` shareholders by commitment overlap.
     Call GetStatsSum(session_id, commitments) on each selected shareholder.
     Reconstruct via Lagrange interpolation → means, variances.
     Compute bin configurations.
     Set phase = COLLECTING_GRADIENTS, round_id = 0, depth = 0.

3. GRADIENT ROUNDS
   For round_id = 0 to n_trees - 1:
     (Optional) Recompute bins if new stats commitments have appeared since last round.
     For depth = 0 to max_depth - 1:
       Poll shareholders: GetGradientCommitments(session_id, round_id, depth).
       When target reached:
         For each active node_id (from GetGradientNodeIds):
           Call GetGradientsSum(session_id, round_id, depth, commitments, node_id).
           Reconstruct gradient/hessian histograms.
           Find best split (maximize gain = G²/(H+λ)).
         Update splits, advance depth.
     Build tree from splits, add to model.

4. Set phase = TRAINING_COMPLETE.
```

---

## Dropout Handling

- **min_clients**: Hard security minimum enforced by shareholders. They refuse to return sums if `len(commitments) < min_clients`.
- **target_count / target_fraction**: Practical threshold. Aggregator waits until this many clients have submitted before proceeding.
- **No timeout**: If the target is never reached, the system waits. Operator intervention required (cancel the session).
- **Client reconnection**: A client that disconnects and reconnects polls `GetRoundState`, fetches current model/splits/bins, and participates from the current round/depth onward. Missed rounds are simply missed.
- **Partial shareholder submission**: If a client crashes mid-submission (sent shares to 2 of 3 shareholders), the commitment-overlap selection handles this — the aggregator only uses commitments present across all selected shareholders.

---

## Cancellation

An admin cancels a training session via `AggregatorService.CancelSession(session_id)`. The flow:

1. Aggregator marks session as cancelled, stops its training loop.
2. Aggregator calls `ShareholderService.CancelSession(session_id)` on each shareholder (best effort).
3. If a shareholder is unreachable, the aggregator retries with idempotent calls until it succeeds. Stale shares on an unreachable shareholder are harmless — no one will ever read them.
4. Shareholders that receive the cancel immediately reject future RPCs for that session with `FAILED_PRECONDITION` and free stored shares.
5. Clients discover cancellation through two paths:
   - **Polling**: `GetRoundState()` returns `phase = CANCELLED` — client stops.
   - **Submission rejected**: `SubmitStats`/`SubmitGradients` returns `FAILED_PRECONDITION` — client checks `GetRoundState` to confirm cancellation.

Both paths lead to the client stopping. The eventual consistency window is small — between the aggregator cancelling and the next client poll/submission.

---

## Communication Map

```
Client ──────────────────────────────────────── Aggregator (gRPC server)
  │  JoinSession, GetRoundState,                    │
  │  GetBinConfigs, GetSplits, GetModel              │
  │                                                   │
  │                                                   │ (gRPC client)
  │                                                   │
  ├──── SubmitStats ────────────────────> Shareholder 1 (gRPC server)
  ├──── SubmitGradients ────────────────> Shareholder 2 (gRPC server)
  └──── SubmitStats/Gradients ──────────> Shareholder N (gRPC server)
                                               ▲
                                               │
                               Aggregator calls:
                               GetStatsCommitments, GetGradientCommitments,
                               GetGradientNodeIds, GetStatsSum,
                               GetGradientsSum, Reset
```

## Security Properties Preserved

- **Threshold security**: Shareholders only release sums, never individual shares. Reconstruction requires `threshold` shareholders.
- **Aggregate-only**: Aggregator sees only reconstructed sums (means, variances, gradient histograms), never individual client data.
- **Anonymous**: Aggregator sees commitment hashes, never client IDs. Fresh nonce per round prevents linkability. The `JoinSession` RPC knows the client_id, but cannot link it to any commitment.
- **Minimum N**: Shareholders enforce `min_clients` before releasing any aggregate.
- **Auth separation**: Firebase ID tokens in gRPC metadata, validated by interceptors. Orthogonal to the commitment-based anonymity scheme.
