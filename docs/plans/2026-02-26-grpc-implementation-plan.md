# gRPC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement gRPC servers and client for the privateboost federated XGBoost protocol, turning the in-process simulation into a networked system.

**Architecture:** The ShareholderService and AggregatorService are gRPC servers wrapping the existing `ShareHolder` and `Aggregator` classes. A new `NetworkClient` replaces direct method calls with gRPC RPCs. Serialization helpers convert between Python dataclasses/numpy arrays and protobuf messages. An integration test spins up all servers in-process and runs the full protocol over gRPC.

**Tech Stack:** grpcio, grpcio-tools, protobuf (Python), pytest, existing privateboost core

---

### Task 1: Add gRPC dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add grpcio and grpcio-tools to dependencies**

In `pyproject.toml`, add to `dependencies`:
```toml
dependencies = [
    "numpy>=2.0.0",
    "pandas>=2.0.0",
    "pandas-stubs~=2.3.3",
    "scikit-learn>=1.8.0",
    "xgboost>=3.1.3",
    "grpcio>=1.70.0",
    "grpcio-tools>=1.70.0",
    "protobuf>=5.29.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync --all-extras`
Expected: Dependencies install successfully.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add gRPC and protobuf dependencies"
```

---

### Task 2: Define protobuf schema

**Files:**
- Create: `proto/privateboost.proto`

**Step 1: Create the proto file**

```protobuf
syntax = "proto3";
package privateboost;

// -- Shared messages --

enum DType {
  FLOAT64 = 0;
  FLOAT32 = 1;
  INT32 = 2;
  INT64 = 3;
}

message NdArray {
  DType dtype = 1;
  repeated int64 shape = 2;
  bytes data = 3;
}

message Share {
  int32 x = 1;
  NdArray y = 2;
}

// -- ShareholderService --

service ShareholderService {
  rpc SubmitStats(SubmitStatsRequest) returns (SubmitStatsResponse);
  rpc SubmitGradients(SubmitGradientsRequest) returns (SubmitGradientsResponse);

  rpc GetStatsCommitments(GetStatsCommitmentsRequest) returns (GetStatsCommitmentsResponse);
  rpc GetGradientCommitments(GetGradientCommitmentsRequest) returns (GetGradientCommitmentsResponse);
  rpc GetGradientNodeIds(GetGradientNodeIdsRequest) returns (GetGradientNodeIdsResponse);
  rpc GetStatsSum(GetStatsSumRequest) returns (GetStatsSumResponse);
  rpc GetGradientsSum(GetGradientsSumRequest) returns (GetGradientsSumResponse);

  rpc CancelSession(CancelSessionRequest) returns (CancelSessionResponse);
  rpc Reset(ResetRequest) returns (ResetResponse);
}

message SubmitStatsRequest {
  string session_id = 1;
  bytes commitment = 2;
  Share share = 3;
}
message SubmitStatsResponse {}

message SubmitGradientsRequest {
  string session_id = 1;
  int32 round_id = 2;
  int32 depth = 3;
  bytes commitment = 4;
  Share share = 5;
  int32 node_id = 6;
}
message SubmitGradientsResponse {}

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

// -- AggregatorService --

service AggregatorService {
  rpc JoinSession(JoinSessionRequest) returns (JoinSessionResponse);
  rpc CancelSession(CancelSessionRequest) returns (CancelSessionResponse);
  rpc GetRoundState(GetRoundStateRequest) returns (GetRoundStateResponse);
  rpc GetTrainingState(GetTrainingStateRequest) returns (GetTrainingStateResponse);
  rpc GetModel(GetModelRequest) returns (GetModelResponse);
}

enum Phase {
  WAITING_FOR_CLIENTS = 0;
  COLLECTING_GRADIENTS = 1;
  TRAINING_COMPLETE = 2;
}

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
  string address = 2;
  int32 x_coord = 3;
}

message TrainingConfig {
  repeated FeatureSpec features = 1;
  string target_column = 2;
  string loss = 3;
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

message GetRoundStateRequest {
  string session_id = 1;
}
message GetRoundStateResponse {
  Phase phase = 1;
  int32 round_id = 2;
  int32 depth = 3;
}

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

message GetModelRequest {
  string session_id = 1;
}
message GetModelResponse {
  Model model = 1;
}

// -- Shared model messages --

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

**Step 2: Generate Python stubs**

Run:
```bash
uv run python -m grpc_tools.protoc \
  -I proto \
  --python_out=src/privateboost/grpc \
  --pyi_out=src/privateboost/grpc \
  --grpc_python_out=src/privateboost/grpc \
  proto/privateboost.proto
```

Create `src/privateboost/grpc/__init__.py` as an empty file.

Expected: `privateboost_pb2.py`, `privateboost_pb2.pyi`, `privateboost_pb2_grpc.py` are generated in `src/privateboost/grpc/`.

**Step 3: Verify the generated code imports cleanly**

Run: `uv run python -c "from privateboost.grpc import privateboost_pb2, privateboost_pb2_grpc; print('OK')"`
Expected: `OK`

**Step 4: Add a Makefile target for proto generation**

Add to `Makefile`:
```makefile
proto:
	uv run python -m grpc_tools.protoc \
		-I proto \
		--python_out=src/privateboost/grpc \
		--pyi_out=src/privateboost/grpc \
		--grpc_python_out=src/privateboost/grpc \
		proto/privateboost.proto
```

**Step 5: Commit**

```bash
git add proto/ src/privateboost/grpc/ Makefile
git commit -m "Add protobuf schema and generate Python stubs"
```

---

### Task 3: Serialization helpers (NdArray, Share, BinConfiguration, SplitDecision, Model)

**Files:**
- Create: `src/privateboost/grpc/converters.py`
- Create: `tests/test_grpc_converters.py`

These convert between Python dataclasses/numpy and protobuf messages. Every other gRPC task depends on these, so we test them thoroughly first.

**Step 1: Write the failing tests**

```python
"""Tests for protobuf <-> Python converters."""

import numpy as np
import pytest

from privateboost.crypto import Share
from privateboost.messages import BinConfiguration, SplitDecision
from privateboost.tree import Leaf, Model, SplitNode, Tree

from privateboost.grpc.converters import (
    ndarray_to_pb,
    pb_to_ndarray,
    share_to_pb,
    pb_to_share,
    bin_config_to_pb,
    pb_to_bin_config,
    split_decision_to_pb,
    pb_to_split_decision,
    model_to_pb,
    pb_to_model,
)


def test_ndarray_roundtrip_float64():
    arr = np.array([1.5, -2.3, 0.0, 42.0])
    pb = ndarray_to_pb(arr)
    result = pb_to_ndarray(pb)
    np.testing.assert_array_equal(result, arr)
    assert result.dtype == np.float64


def test_ndarray_roundtrip_2d():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    pb = ndarray_to_pb(arr)
    result = pb_to_ndarray(pb)
    np.testing.assert_array_equal(result, arr)
    assert result.shape == (2, 2)


def test_share_roundtrip():
    s = Share(x=3, y=np.array([10.0, 20.0, 30.0]))
    pb = share_to_pb(s)
    result = pb_to_share(pb)
    assert result.x == 3
    np.testing.assert_array_equal(result.y, s.y)


def test_bin_config_roundtrip():
    edges = np.array([-np.inf, 0.0, 1.0, 2.0, np.inf])
    inner_edges = np.array([0.0, 1.0, 2.0])
    bc = BinConfiguration(feature_idx=2, edges=edges, inner_edges=inner_edges, n_bins=2)
    pb = bin_config_to_pb(bc)
    result = pb_to_bin_config(pb)
    assert result.feature_idx == 2
    assert result.n_bins == 2
    np.testing.assert_array_equal(result.edges, edges)
    np.testing.assert_array_equal(result.inner_edges, inner_edges)


def test_split_decision_roundtrip():
    sd = SplitDecision(
        node_id=0, feature_idx=3, threshold=1.5, gain=0.8,
        left_child_id=1, right_child_id=2,
        g_left=-0.5, h_left=10.0, g_right=0.5, h_right=10.0,
    )
    pb = split_decision_to_pb(sd)
    result = pb_to_split_decision(pb)
    assert result.node_id == 0
    assert result.feature_idx == 3
    assert result.threshold == 1.5
    assert result.gain == 0.8
    assert result.left_child_id == 1
    assert result.right_child_id == 2


def test_model_roundtrip_empty():
    m = Model(initial_prediction=0.46, learning_rate=0.1, trees=[])
    pb = model_to_pb(m)
    result = pb_to_model(pb)
    assert result.initial_prediction == pytest.approx(0.46)
    assert result.learning_rate == pytest.approx(0.1)
    assert result.trees == []


def test_model_roundtrip_with_trees():
    tree = Tree(root=SplitNode(
        feature_idx=0,
        threshold=1.5,
        gain=0.8,
        left=Leaf(value=-0.1, n_samples=50),
        right=Leaf(value=0.2, n_samples=30),
    ))
    m = Model(initial_prediction=0.5, learning_rate=0.15, trees=[tree])
    pb = model_to_pb(m)
    result = pb_to_model(pb)
    assert len(result.trees) == 1
    root = result.trees[0].root
    assert isinstance(root, SplitNode)
    assert root.feature_idx == 0
    assert root.threshold == 1.5
    assert isinstance(root.left, Leaf)
    assert root.left.value == pytest.approx(-0.1)
    assert isinstance(root.right, Leaf)
    assert root.right.value == pytest.approx(0.2)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grpc_converters.py -v`
Expected: FAIL — `ImportError: cannot import name 'ndarray_to_pb' from 'privateboost.grpc.converters'`

**Step 3: Implement converters**

Create `src/privateboost/grpc/converters.py`:

```python
"""Convert between Python dataclasses/numpy and protobuf messages."""

import numpy as np

from privateboost.crypto import Share
from privateboost.messages import BinConfiguration, SplitDecision
from privateboost.tree import Leaf, Model, SplitNode, Tree, TreeNode

from . import privateboost_pb2 as pb

_DTYPE_MAP = {
    pb.FLOAT64: np.float64,
    pb.FLOAT32: np.float32,
    pb.INT32: np.int32,
    pb.INT64: np.int64,
}

_NP_TO_DTYPE = {v: k for k, v in _DTYPE_MAP.items()}


def ndarray_to_pb(arr: np.ndarray) -> pb.NdArray:
    dtype = _NP_TO_DTYPE.get(arr.dtype.type, pb.FLOAT64)
    return pb.NdArray(dtype=dtype, shape=list(arr.shape), data=arr.tobytes())


def pb_to_ndarray(msg: pb.NdArray) -> np.ndarray:
    dtype = _DTYPE_MAP.get(msg.dtype, np.float64)
    return np.frombuffer(msg.data, dtype=dtype).reshape(msg.shape).copy()


def share_to_pb(s: Share) -> pb.Share:
    return pb.Share(x=s.x, y=ndarray_to_pb(s.y))


def pb_to_share(msg: pb.Share) -> Share:
    return Share(x=msg.x, y=pb_to_ndarray(msg.y))


def bin_config_to_pb(bc: BinConfiguration) -> pb.BinConfiguration:
    return pb.BinConfiguration(
        feature_idx=bc.feature_idx,
        edges=ndarray_to_pb(bc.edges),
        inner_edges=ndarray_to_pb(bc.inner_edges),
        n_bins=bc.n_bins,
    )


def pb_to_bin_config(msg: pb.BinConfiguration) -> BinConfiguration:
    return BinConfiguration(
        feature_idx=msg.feature_idx,
        edges=pb_to_ndarray(msg.edges),
        inner_edges=pb_to_ndarray(msg.inner_edges),
        n_bins=msg.n_bins,
    )


def split_decision_to_pb(sd: SplitDecision) -> pb.SplitDecision:
    return pb.SplitDecision(
        node_id=sd.node_id,
        feature_idx=sd.feature_idx,
        threshold=sd.threshold,
        gain=sd.gain,
        left_child_id=sd.left_child_id,
        right_child_id=sd.right_child_id,
    )


def pb_to_split_decision(msg: pb.SplitDecision) -> SplitDecision:
    return SplitDecision(
        node_id=msg.node_id,
        feature_idx=msg.feature_idx,
        threshold=msg.threshold,
        gain=msg.gain,
        left_child_id=msg.left_child_id,
        right_child_id=msg.right_child_id,
        g_left=0.0,
        h_left=0.0,
        g_right=0.0,
        h_right=0.0,
    )


def _tree_node_to_pb(node: TreeNode) -> pb.TreeNode:
    match node:
        case SplitNode(feature_idx=f, threshold=t, left=l, right=r):
            return pb.TreeNode(split=pb.SplitNode(
                feature_idx=f,
                threshold=t,
                left=_tree_node_to_pb(l),
                right=_tree_node_to_pb(r),
            ))
        case Leaf(value=v):
            return pb.TreeNode(leaf=pb.LeafNode(value=v))


def _pb_to_tree_node(msg: pb.TreeNode) -> TreeNode:
    if msg.HasField("split"):
        s = msg.split
        return SplitNode(
            feature_idx=s.feature_idx,
            threshold=s.threshold,
            gain=0.0,
            left=_pb_to_tree_node(s.left),
            right=_pb_to_tree_node(s.right),
        )
    else:
        return Leaf(value=msg.leaf.value, n_samples=0)


def model_to_pb(m: Model) -> pb.Model:
    trees = [pb.Tree(root=_tree_node_to_pb(t.root)) for t in m.trees]
    return pb.Model(
        initial_prediction=m.initial_prediction,
        learning_rate=m.learning_rate,
        trees=trees,
    )


def pb_to_model(msg: pb.Model) -> Model:
    trees = [Tree(root=_pb_to_tree_node(t.root)) for t in msg.trees]
    return Model(
        initial_prediction=msg.initial_prediction,
        learning_rate=msg.learning_rate,
        trees=trees,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grpc_converters.py -v`
Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/privateboost/grpc/converters.py tests/test_grpc_converters.py
git commit -m "Add protobuf <-> Python serialization converters with tests"
```

---

### Task 4: ShareholderService gRPC server

**Files:**
- Create: `src/privateboost/grpc/shareholder_server.py`
- Create: `tests/test_grpc_shareholder.py`

The server wraps the existing `ShareHolder` class. It manages multiple sessions, each backed by its own `ShareHolder` instance. Sessions are implicitly created on first `SubmitStats`.

**Step 1: Write the failing test**

```python
"""Tests for ShareholderService gRPC server."""

import grpc
import numpy as np
import pytest
from concurrent import futures

from privateboost.crypto import share, Share
from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.converters import share_to_pb, pb_to_ndarray
from privateboost.grpc.shareholder_server import ShareholderServicer


@pytest.fixture()
def shareholder_channel():
    """Start a ShareholderService gRPC server and return a channel to it."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ShareholderServicer(x_coord=1, min_clients=2)
    pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel
    channel.close()
    server.stop(grace=0)


def _make_shares(values: np.ndarray, n_parties: int = 3) -> list[Share]:
    return share(values, n_parties=n_parties, threshold=2)


def test_submit_and_get_stats_commitments(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-1"

    values = np.array([1.0, 2.0, 3.0, 4.0])
    shares = _make_shares(values)
    commitment = b"\x01" * 32

    stub.SubmitStats(pb.SubmitStatsRequest(
        session_id=session_id,
        commitment=commitment,
        share=share_to_pb(shares[0]),
    ))

    resp = stub.GetStatsCommitments(pb.GetStatsCommitmentsRequest(session_id=session_id))
    assert commitment in resp.commitments


def test_stats_sum(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-2"

    np.random.seed(42)
    values_a = np.array([10.0, 20.0])
    values_b = np.array([30.0, 40.0])
    shares_a = _make_shares(values_a)
    shares_b = _make_shares(values_b)
    commitment_a = b"\x01" * 32
    commitment_b = b"\x02" * 32

    stub.SubmitStats(pb.SubmitStatsRequest(
        session_id=session_id, commitment=commitment_a, share=share_to_pb(shares_a[0]),
    ))
    stub.SubmitStats(pb.SubmitStatsRequest(
        session_id=session_id, commitment=commitment_b, share=share_to_pb(shares_b[0]),
    ))

    resp = stub.GetStatsSum(pb.GetStatsSumRequest(
        session_id=session_id, commitments=[commitment_a, commitment_b],
    ))
    assert resp.x_coord == 1
    result = pb_to_ndarray(resp.sum)
    expected = shares_a[0].y + shares_b[0].y
    np.testing.assert_array_almost_equal(result, expected)


def test_min_clients_enforcement(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-3"

    values = np.array([1.0, 2.0])
    shares = _make_shares(values)
    commitment = b"\x03" * 32

    stub.SubmitStats(pb.SubmitStatsRequest(
        session_id=session_id, commitment=commitment, share=share_to_pb(shares[0]),
    ))

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.GetStatsSum(pb.GetStatsSumRequest(
            session_id=session_id, commitments=[commitment],
        ))
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_cancel_session(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-4"

    values = np.array([1.0, 2.0])
    shares = _make_shares(values)
    stub.SubmitStats(pb.SubmitStatsRequest(
        session_id=session_id, commitment=b"\x04" * 32, share=share_to_pb(shares[0]),
    ))

    stub.CancelSession(pb.CancelSessionRequest(session_id=session_id))

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.GetStatsCommitments(pb.GetStatsCommitmentsRequest(session_id=session_id))
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


def test_session_isolation(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)

    values = np.array([1.0, 2.0])
    shares = _make_shares(values)

    stub.SubmitStats(pb.SubmitStatsRequest(
        session_id="session-A", commitment=b"\x0a" * 32, share=share_to_pb(shares[0]),
    ))
    stub.SubmitStats(pb.SubmitStatsRequest(
        session_id="session-B", commitment=b"\x0b" * 32, share=share_to_pb(shares[0]),
    ))

    resp_a = stub.GetStatsCommitments(pb.GetStatsCommitmentsRequest(session_id="session-A"))
    resp_b = stub.GetStatsCommitments(pb.GetStatsCommitmentsRequest(session_id="session-B"))
    assert len(resp_a.commitments) == 1
    assert len(resp_b.commitments) == 1
    assert resp_a.commitments[0] != resp_b.commitments[0]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grpc_shareholder.py -v`
Expected: FAIL — `ImportError: cannot import name 'ShareholderServicer' from 'privateboost.grpc.shareholder_server'`

**Step 3: Implement the ShareholderServicer**

Create `src/privateboost/grpc/shareholder_server.py`:

```python
"""gRPC server implementation for ShareholderService."""

import threading
from typing import Dict

import grpc
import numpy as np

from privateboost.crypto import Share
from privateboost.messages import CommittedGradientShare, CommittedStatsShare
from privateboost.shareholder import ShareHolder

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .converters import ndarray_to_pb, pb_to_share


class ShareholderServicer(pb_grpc.ShareholderServiceServicer):
    """gRPC servicer wrapping ShareHolder instances, one per session."""

    def __init__(self, x_coord: int, min_clients: int = 10):
        self._x_coord = x_coord
        self._min_clients = min_clients
        self._sessions: Dict[str, ShareHolder] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def _get_or_create_session(self, session_id: str, context: grpc.ServicerContext) -> ShareHolder | None:
        with self._lock:
            if session_id in self._cancelled:
                context.abort(grpc.StatusCode.NOT_FOUND, f"Session {session_id} cancelled")
                return None
            if session_id not in self._sessions:
                self._sessions[session_id] = ShareHolder(
                    party_id=0,
                    x_coord=self._x_coord,
                    min_clients=self._min_clients,
                )
            return self._sessions[session_id]

    def _get_session(self, session_id: str, context: grpc.ServicerContext) -> ShareHolder | None:
        with self._lock:
            if session_id in self._cancelled:
                context.abort(grpc.StatusCode.NOT_FOUND, f"Session {session_id} cancelled")
                return None
            sh = self._sessions.get(session_id)
            if sh is None:
                context.abort(grpc.StatusCode.NOT_FOUND, f"Session {session_id} not found")
                return None
            return sh

    def SubmitStats(self, request: pb.SubmitStatsRequest, context: grpc.ServicerContext) -> pb.SubmitStatsResponse:
        sh = self._get_or_create_session(request.session_id, context)
        if sh is None:
            return pb.SubmitStatsResponse()
        share = pb_to_share(request.share)
        sh.receive_stats(CommittedStatsShare(commitment=request.commitment, share=share))
        return pb.SubmitStatsResponse()

    def SubmitGradients(self, request: pb.SubmitGradientsRequest, context: grpc.ServicerContext) -> pb.SubmitGradientsResponse:
        sh = self._get_or_create_session(request.session_id, context)
        if sh is None:
            return pb.SubmitGradientsResponse()
        share = pb_to_share(request.share)
        sh.receive_gradients(CommittedGradientShare(
            round_id=request.round_id,
            depth=request.depth,
            commitment=request.commitment,
            share=share,
            node_id=request.node_id,
        ))
        return pb.SubmitGradientsResponse()

    def GetStatsCommitments(self, request: pb.GetStatsCommitmentsRequest, context: grpc.ServicerContext) -> pb.GetStatsCommitmentsResponse:
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetStatsCommitmentsResponse()
        commitments = sh.get_stats_commitments()
        return pb.GetStatsCommitmentsResponse(commitments=list(commitments))

    def GetGradientCommitments(self, request: pb.GetGradientCommitmentsRequest, context: grpc.ServicerContext) -> pb.GetGradientCommitmentsResponse:
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetGradientCommitmentsResponse()
        commitments = sh.get_gradient_commitments(request.depth)
        return pb.GetGradientCommitmentsResponse(commitments=list(commitments))

    def GetGradientNodeIds(self, request: pb.GetGradientNodeIdsRequest, context: grpc.ServicerContext) -> pb.GetGradientNodeIdsResponse:
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetGradientNodeIdsResponse()
        node_ids = sh.get_gradient_node_ids(request.depth)
        return pb.GetGradientNodeIdsResponse(node_ids=list(node_ids))

    def GetStatsSum(self, request: pb.GetStatsSumRequest, context: grpc.ServicerContext) -> pb.GetStatsSumResponse:
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetStatsSumResponse()
        try:
            x_coord, total = sh.get_stats_sum(list(request.commitments))
        except ValueError as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            return pb.GetStatsSumResponse()
        return pb.GetStatsSumResponse(x_coord=x_coord, sum=ndarray_to_pb(total))

    def GetGradientsSum(self, request: pb.GetGradientsSumRequest, context: grpc.ServicerContext) -> pb.GetGradientsSumResponse:
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetGradientsSumResponse()
        try:
            x_coord, total = sh.get_gradients_sum(
                request.depth, list(request.commitments), request.node_id,
            )
        except ValueError as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            return pb.GetGradientsSumResponse()
        return pb.GetGradientsSumResponse(x_coord=x_coord, sum=ndarray_to_pb(total))

    def CancelSession(self, request: pb.CancelSessionRequest, context: grpc.ServicerContext) -> pb.CancelSessionResponse:
        with self._lock:
            self._cancelled.add(request.session_id)
            self._sessions.pop(request.session_id, None)
        return pb.CancelSessionResponse()

    def Reset(self, request: pb.ResetRequest, context: grpc.ServicerContext) -> pb.ResetResponse:
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.ResetResponse()
        sh.reset()
        return pb.ResetResponse()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grpc_shareholder.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/privateboost/grpc/shareholder_server.py tests/test_grpc_shareholder.py
git commit -m "Add ShareholderService gRPC server with session management"
```

---

### Task 5: AggregatorService gRPC server

**Files:**
- Create: `src/privateboost/grpc/aggregator_server.py`
- Create: `tests/test_grpc_aggregator.py`

The aggregator server wraps the existing `Aggregator` class. It maintains session state (phase, round_id, depth) and runs the training loop in a background thread. The `Aggregator` currently takes `ShareHolder` instances directly — the gRPC server instead uses shareholder stubs to call remote shareholders.

This task has two parts:
1. A `RemoteShareHolder` adapter that implements the same interface as `ShareHolder` but calls gRPC
2. The `AggregatorServicer` itself

**Step 1: Write the failing test**

```python
"""Tests for AggregatorService gRPC server."""

import grpc
import numpy as np
import pytest
from concurrent import futures

from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.shareholder_server import ShareholderServicer
from privateboost.grpc.aggregator_server import AggregatorServicer
from privateboost.grpc.converters import share_to_pb
from privateboost.crypto import share


def _start_shareholder(x_coord: int, min_clients: int = 2) -> tuple[grpc.Server, str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ShareholderServicer(x_coord=x_coord, min_clients=min_clients)
    pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, f"localhost:{port}"


@pytest.fixture()
def grpc_cluster():
    """Start 3 shareholder servers and 1 aggregator server."""
    sh_servers = []
    sh_addresses = []
    for i in range(3):
        server, addr = _start_shareholder(x_coord=i + 1, min_clients=2)
        sh_servers.append(server)
        sh_addresses.append(addr)

    agg_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = AggregatorServicer(
        shareholder_addresses=sh_addresses,
        shareholder_x_coords=[1, 2, 3],
        n_bins=5,
        threshold=2,
        min_clients=2,
        learning_rate=0.15,
        lambda_reg=2.0,
        n_trees=2,
        max_depth=2,
        loss="squared",
        target_count=3,
        features=[pb.FeatureSpec(index=0, name="f0"), pb.FeatureSpec(index=1, name="f1")],
        target_column="target",
    )
    pb_grpc.add_AggregatorServiceServicer_to_server(servicer, agg_server)
    agg_port = agg_server.add_insecure_port("[::]:0")
    agg_server.start()

    agg_channel = grpc.insecure_channel(f"localhost:{agg_port}")

    yield agg_channel, sh_addresses

    agg_channel.close()
    agg_server.stop(grace=0)
    for s in sh_servers:
        s.stop(grace=0)


def test_join_session(grpc_cluster):
    agg_channel, sh_addresses = grpc_cluster
    stub = pb_grpc.AggregatorServiceStub(agg_channel)
    session_id = "test-session"

    resp = stub.JoinSession(pb.JoinSessionRequest(session_id=session_id))
    assert resp.session_id == session_id
    assert len(resp.shareholders) == 3
    assert resp.threshold == 2
    assert resp.config.n_bins == 5
    assert resp.config.n_trees == 2
    assert resp.config.loss == "squared"


def test_initial_round_state(grpc_cluster):
    agg_channel, _ = grpc_cluster
    stub = pb_grpc.AggregatorServiceStub(agg_channel)
    session_id = "test-session-state"

    stub.JoinSession(pb.JoinSessionRequest(session_id=session_id))
    resp = stub.GetRoundState(pb.GetRoundStateRequest(session_id=session_id))
    assert resp.phase == pb.WAITING_FOR_CLIENTS
    assert resp.round_id == 0
    assert resp.depth == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grpc_aggregator.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement the AggregatorServicer**

Create `src/privateboost/grpc/aggregator_server.py`. This is the largest piece — it wraps `Aggregator` and manages session state.

The `RemoteShareHolder` adapter converts `ShareHolder` method calls into gRPC RPCs. It implements enough of the `ShareHolder` interface that `Aggregator` can use it directly (the aggregator-facing methods: `get_stats_commitments`, `get_gradient_commitments`, `get_gradient_node_ids`, `get_stats_sum`, `get_gradients_sum`).

```python
"""gRPC server implementation for AggregatorService."""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import grpc
import numpy as np

from privateboost.aggregator import Aggregator
from privateboost.crypto import Share
from privateboost.messages import BinConfiguration, SplitDecision

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .converters import (
    bin_config_to_pb,
    model_to_pb,
    pb_to_ndarray,
    split_decision_to_pb,
)


class RemoteShareHolder:
    """Adapter: calls a remote ShareholderService via gRPC.

    Implements the aggregator-facing subset of the ShareHolder interface
    so that the existing Aggregator class can use it unchanged.
    """

    def __init__(self, address: str, x_coord: int, session_id: str):
        self.x_coord = x_coord
        self._session_id = session_id
        self._channel = grpc.insecure_channel(address)
        self._stub = pb_grpc.ShareholderServiceStub(self._channel)

    def get_stats_commitments(self) -> Set[bytes]:
        resp = self._stub.GetStatsCommitments(
            pb.GetStatsCommitmentsRequest(session_id=self._session_id)
        )
        return set(resp.commitments)

    def get_gradient_commitments(self, depth: int) -> Set[bytes]:
        resp = self._stub.GetGradientCommitments(
            pb.GetGradientCommitmentsRequest(
                session_id=self._session_id, round_id=0, depth=depth,
            )
        )
        return set(resp.commitments)

    def get_gradient_node_ids(self, depth: int) -> Set[int]:
        resp = self._stub.GetGradientNodeIds(
            pb.GetGradientNodeIdsRequest(
                session_id=self._session_id, round_id=0, depth=depth,
            )
        )
        return set(resp.node_ids)

    def get_stats_sum(self, commitments: List[bytes]) -> Tuple[int, np.ndarray]:
        resp = self._stub.GetStatsSum(
            pb.GetStatsSumRequest(
                session_id=self._session_id, commitments=commitments,
            )
        )
        return (resp.x_coord, pb_to_ndarray(resp.sum))

    def get_gradients_sum(
        self, depth: int, commitments: List[bytes], node_id: int,
    ) -> Tuple[int, np.ndarray]:
        resp = self._stub.GetGradientsSum(
            pb.GetGradientsSumRequest(
                session_id=self._session_id,
                round_id=0,
                depth=depth,
                commitments=commitments,
                node_id=node_id,
            )
        )
        return (resp.x_coord, pb_to_ndarray(resp.sum))

    def close(self):
        self._channel.close()


@dataclass
class SessionState:
    phase: pb.Phase = pb.WAITING_FOR_CLIENTS
    round_id: int = 0
    depth: int = 0
    aggregator: Aggregator | None = None
    remote_shareholders: list[RemoteShareHolder] = field(default_factory=list)
    training_thread: threading.Thread | None = None


class AggregatorServicer(pb_grpc.AggregatorServiceServicer):
    """gRPC servicer for the Aggregator."""

    def __init__(
        self,
        shareholder_addresses: list[str],
        shareholder_x_coords: list[int],
        n_bins: int = 10,
        threshold: int = 2,
        min_clients: int = 10,
        learning_rate: float = 0.1,
        lambda_reg: float = 1.0,
        n_trees: int = 10,
        max_depth: int = 3,
        loss: str = "squared",
        target_count: int | None = None,
        target_fraction: float | None = None,
        features: list[pb.FeatureSpec] | None = None,
        target_column: str = "target",
    ):
        self._sh_addresses = shareholder_addresses
        self._sh_x_coords = shareholder_x_coords
        self._n_bins = n_bins
        self._threshold = threshold
        self._min_clients = min_clients
        self._learning_rate = learning_rate
        self._lambda_reg = lambda_reg
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._loss = loss
        self._target_count = target_count
        self._target_fraction = target_fraction
        self._features = features or []
        self._target_column = target_column
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def _get_or_create_session(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                remote_shs = [
                    RemoteShareHolder(addr, x_coord, session_id)
                    for addr, x_coord in zip(self._sh_addresses, self._sh_x_coords)
                ]
                aggregator = Aggregator(
                    shareholders=remote_shs,
                    n_bins=self._n_bins,
                    threshold=self._threshold,
                    min_clients=self._min_clients,
                    learning_rate=self._learning_rate,
                    lambda_reg=self._lambda_reg,
                )
                self._sessions[session_id] = SessionState(
                    aggregator=aggregator,
                    remote_shareholders=remote_shs,
                )
            return self._sessions[session_id]

    def _get_session(self, session_id: str, context: grpc.ServicerContext) -> SessionState | None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                context.abort(grpc.StatusCode.NOT_FOUND, f"Session {session_id} not found")
                return None
            return state

    def _run_training(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None or state.aggregator is None:
            return

        agg = state.aggregator
        target = self._target_count or self._min_clients

        # Wait for enough stats submissions
        while True:
            try:
                commitments = agg._shareholders[0].get_stats_commitments()
                if len(commitments) >= target:
                    break
            except Exception:
                pass
            import time
            time.sleep(0.1)

        # Define bins
        agg.define_bins()

        # Training loop
        for round_id in range(self._n_trees):
            with self._lock:
                state.phase = pb.COLLECTING_GRADIENTS
                state.round_id = round_id
                state.depth = 0

            for depth in range(self._max_depth):
                with self._lock:
                    state.depth = depth

                # Wait for enough gradient submissions
                while True:
                    try:
                        commitments = agg._shareholders[0].get_gradient_commitments(depth)
                        if len(commitments) >= target:
                            break
                    except Exception:
                        pass
                    import time
                    time.sleep(0.1)

                if not agg.compute_splits(depth=depth, min_samples=1):
                    break

            agg.finish_round()

        with self._lock:
            state.phase = pb.TRAINING_COMPLETE

    def JoinSession(self, request: pb.JoinSessionRequest, context: grpc.ServicerContext) -> pb.JoinSessionResponse:
        state = self._get_or_create_session(request.session_id)

        shareholders = [
            pb.ShareholderInfo(id=str(i), address=addr, x_coord=x)
            for i, (addr, x) in enumerate(zip(self._sh_addresses, self._sh_x_coords))
        ]

        config = pb.TrainingConfig(
            features=self._features,
            target_column=self._target_column,
            loss=self._loss,
            n_bins=self._n_bins,
            n_trees=self._n_trees,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            lambda_reg=self._lambda_reg,
            min_clients=self._min_clients,
        )
        if self._target_count is not None:
            config.target_count = self._target_count
        elif self._target_fraction is not None:
            config.target_fraction = self._target_fraction

        # Start training loop on first join
        if state.training_thread is None:
            state.training_thread = threading.Thread(
                target=self._run_training, args=(request.session_id,), daemon=True,
            )
            state.training_thread.start()

        return pb.JoinSessionResponse(
            session_id=request.session_id,
            shareholders=shareholders,
            threshold=self._threshold,
            config=config,
        )

    def GetRoundState(self, request: pb.GetRoundStateRequest, context: grpc.ServicerContext) -> pb.GetRoundStateResponse:
        state = self._get_session(request.session_id, context)
        if state is None:
            return pb.GetRoundStateResponse()
        with self._lock:
            return pb.GetRoundStateResponse(
                phase=state.phase,
                round_id=state.round_id,
                depth=state.depth,
            )

    def GetTrainingState(self, request: pb.GetTrainingStateRequest, context: grpc.ServicerContext) -> pb.GetTrainingStateResponse:
        state = self._get_session(request.session_id, context)
        if state is None:
            return pb.GetTrainingStateResponse()
        agg = state.aggregator
        with self._lock:
            return pb.GetTrainingStateResponse(
                model=model_to_pb(agg.model),
                current_splits={
                    nid: split_decision_to_pb(sd)
                    for nid, sd in agg.splits.items()
                },
                bins=[bin_config_to_pb(bc) for bc in agg._bin_configs],
                round_id=state.round_id,
                depth=state.depth,
            )

    def GetModel(self, request: pb.GetModelRequest, context: grpc.ServicerContext) -> pb.GetModelResponse:
        state = self._get_session(request.session_id, context)
        if state is None:
            return pb.GetModelResponse()
        return pb.GetModelResponse(model=model_to_pb(state.aggregator.model))

    def CancelSession(self, request: pb.CancelSessionRequest, context: grpc.ServicerContext) -> pb.CancelSessionResponse:
        with self._lock:
            state = self._sessions.pop(request.session_id, None)
        if state is not None:
            for rsh in state.remote_shareholders:
                try:
                    rsh._stub.CancelSession(
                        pb.CancelSessionRequest(session_id=request.session_id)
                    )
                except Exception:
                    pass
                rsh.close()
        return pb.CancelSessionResponse()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grpc_aggregator.py -v`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/privateboost/grpc/aggregator_server.py tests/test_grpc_aggregator.py
git commit -m "Add AggregatorService gRPC server with RemoteShareHolder adapter"
```

---

### Task 6: NetworkClient

**Files:**
- Create: `src/privateboost/grpc/network_client.py`
- Create: `tests/test_grpc_network_client.py`

The `NetworkClient` replaces the in-process `Client`. It talks to shareholders via gRPC for share submission and to the aggregator for session management / training state. It reuses the existing crypto and gradient computation logic.

**Step 1: Write the failing test**

```python
"""Tests for NetworkClient gRPC client."""

import grpc
import numpy as np
import pytest
from concurrent import futures

from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.shareholder_server import ShareholderServicer
from privateboost.grpc.network_client import NetworkClient


def _start_shareholder(x_coord: int, min_clients: int = 2) -> tuple[grpc.Server, str, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ShareholderServicer(x_coord=x_coord, min_clients=min_clients)
    pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, f"localhost:{port}", x_coord


@pytest.fixture()
def shareholder_cluster():
    servers = []
    infos = []
    for i in range(3):
        server, addr, x_coord = _start_shareholder(x_coord=i + 1, min_clients=2)
        servers.append(server)
        infos.append((addr, x_coord))
    yield infos
    for s in servers:
        s.stop(grace=0)


def test_network_client_submit_stats(shareholder_cluster):
    session_id = "test-nc-stats"
    client = NetworkClient(
        client_id="client_0",
        features=np.array([1.0, 2.0]),
        target=1.0,
        session_id=session_id,
        shareholder_addresses=[addr for addr, _ in shareholder_cluster],
        shareholder_x_coords=[x for _, x in shareholder_cluster],
        threshold=2,
    )

    client.submit_stats()

    # Verify shares were received by querying a shareholder directly
    channel = grpc.insecure_channel(shareholder_cluster[0][0])
    stub = pb_grpc.ShareholderServiceStub(channel)
    resp = stub.GetStatsCommitments(pb.GetStatsCommitmentsRequest(session_id=session_id))
    assert len(resp.commitments) == 1
    channel.close()
    client.close()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grpc_network_client.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement NetworkClient**

Create `src/privateboost/grpc/network_client.py`:

```python
"""gRPC-based client for the privateboost protocol."""

import hashlib
from typing import Dict, List

import grpc
import numpy as np

from privateboost.client import _find_bin_index
from privateboost.crypto import generate_nonce, compute_commitment, share
from privateboost.messages import BinConfiguration, Loss, SplitDecision
from privateboost.tree import Model

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .converters import share_to_pb


class NetworkClient:
    """A data owner that secret-shares its features with remote shareholders via gRPC."""

    def __init__(
        self,
        client_id: str,
        features: np.ndarray,
        target: float,
        session_id: str,
        shareholder_addresses: list[str],
        shareholder_x_coords: list[int],
        threshold: int = 2,
    ):
        self.client_id = client_id
        self.features = np.asarray(features)
        self.target = target
        self.session_id = session_id
        self.threshold = threshold
        self._n_parties = len(shareholder_addresses)

        self._channels = [
            grpc.insecure_channel(addr) for addr in shareholder_addresses
        ]
        self._stubs = [
            pb_grpc.ShareholderServiceStub(ch) for ch in self._channels
        ]

    def _stats_commitment(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.session_id.encode("utf-8"))
        h.update(self.client_id.encode("utf-8"))
        return h.digest()

    def submit_stats(self) -> None:
        commitment = self._stats_commitment()
        feature_stats = np.column_stack([self.features, self.features ** 2]).ravel()
        values = np.concatenate([feature_stats, [self.target, self.target ** 2]])
        shares = share(values, n_parties=self._n_parties, threshold=self.threshold)

        for stub, s in zip(self._stubs, shares):
            stub.SubmitStats(pb.SubmitStatsRequest(
                session_id=self.session_id,
                commitment=commitment,
                share=share_to_pb(s),
            ))

    def submit_gradients(
        self,
        bins: List[BinConfiguration],
        model: Model,
        splits: Dict[int, SplitDecision],
        round_id: int,
        depth: int,
        loss: Loss = "squared",
    ) -> None:
        node_id = self._get_node_id(splits)
        nonce = generate_nonce()
        commitment = compute_commitment(round_id, self.client_id, nonce)
        prediction = model.predict_one(self.features)

        if loss == "squared":
            gradient = prediction - self.target
            hessian = 1.0
        else:
            p = 1.0 / (1.0 + np.exp(-prediction))
            gradient = p - self.target
            hessian = p * (1.0 - p)

        all_gradients = []
        all_hessians = []

        for config in bins:
            value = self.features[config.feature_idx]
            n_total_bins = config.n_bins + 2
            bin_idx = _find_bin_index(value, config.edges, n_total_bins)

            g_vec = np.zeros(n_total_bins)
            h_vec = np.zeros(n_total_bins)
            g_vec[bin_idx] = gradient
            h_vec[bin_idx] = hessian

            all_gradients.append(g_vec)
            all_hessians.append(h_vec)

        values = np.concatenate(
            [np.concatenate(all_gradients), np.concatenate(all_hessians)]
        )
        shares = share(values, n_parties=self._n_parties, threshold=self.threshold)

        for stub, s in zip(self._stubs, shares):
            stub.SubmitGradients(pb.SubmitGradientsRequest(
                session_id=self.session_id,
                round_id=round_id,
                depth=depth,
                commitment=commitment,
                share=share_to_pb(s),
                node_id=node_id,
            ))

    def _get_node_id(self, splits: Dict[int, SplitDecision]) -> int:
        node_id = 0
        while node_id in splits:
            split = splits[node_id]
            if self.features[split.feature_idx] <= split.threshold:
                node_id = split.left_child_id
            else:
                node_id = split.right_child_id
        return node_id

    def close(self):
        for ch in self._channels:
            ch.close()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grpc_network_client.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/privateboost/grpc/network_client.py tests/test_grpc_network_client.py
git commit -m "Add NetworkClient for gRPC-based share submission"
```

---

### Task 7: End-to-end integration test over gRPC

**Files:**
- Create: `tests/test_grpc_integration.py`

This is the key test. It mirrors `test_xgboost_heart_disease_shamir` from the existing integration tests, but runs the full protocol over gRPC: 3 shareholder servers, 1 aggregator server, many NetworkClients. Verifies the federated model achieves >75% accuracy, same as the in-process test.

**Step 1: Write the test**

```python
"""End-to-end integration test: full protocol over gRPC."""

import time
from concurrent import futures
from typing import List

import grpc
import numpy as np
import pandas as pd
import pytest

from privateboost.messages import BinConfiguration, SplitDecision

from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.shareholder_server import ShareholderServicer
from privateboost.grpc.aggregator_server import AggregatorServicer
from privateboost.grpc.network_client import NetworkClient
from privateboost.grpc.converters import pb_to_model, pb_to_bin_config, pb_to_split_decision

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
COLUMNS = [*FEATURES, "target"]


@pytest.fixture(scope="session")
def heart_disease_df(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    cache_dir = tmp_path_factory.mktemp("data")
    cache_file = cache_dir / "heart_disease.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        df = pd.read_csv(url, names=COLUMNS, na_values="?")
        df = df.dropna()
        df["target"] = (df["target"] > 0).astype(int)
        df.to_csv(cache_file, index=False)
    return df


def test_grpc_xgboost_heart_disease(heart_disease_df: pd.DataFrame):
    """End-to-end: federated XGBoost over gRPC achieves >75% accuracy."""
    np.random.seed(42)
    df = heart_disease_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    threshold = 2
    min_clients = 10
    learning_rate = 0.15
    lambda_reg = 2.0
    n_trees = 15
    max_depth = 3
    session_id = "integration-test"

    # Start 3 shareholder servers
    sh_servers: list[grpc.Server] = []
    sh_addresses: list[str] = []
    for i in range(3):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        servicer = ShareholderServicer(x_coord=i + 1, min_clients=min_clients)
        pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("[::]:0")
        server.start()
        sh_servers.append(server)
        sh_addresses.append(f"localhost:{port}")

    # Start aggregator server
    agg_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    feature_specs = [pb.FeatureSpec(index=i, name=name) for i, name in enumerate(FEATURES)]
    agg_servicer = AggregatorServicer(
        shareholder_addresses=sh_addresses,
        shareholder_x_coords=[1, 2, 3],
        n_bins=10,
        threshold=threshold,
        min_clients=min_clients,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        n_trees=n_trees,
        max_depth=max_depth,
        loss="squared",
        target_count=len(df_train),
        features=feature_specs,
        target_column="target",
    )
    pb_grpc.add_AggregatorServiceServicer_to_server(agg_servicer, agg_server)
    agg_port = agg_server.add_insecure_port("[::]:0")
    agg_server.start()
    agg_channel = grpc.insecure_channel(f"localhost:{agg_port}")
    agg_stub = pb_grpc.AggregatorServiceStub(agg_channel)

    try:
        # Clients join and submit stats
        join_resp = agg_stub.JoinSession(pb.JoinSessionRequest(session_id=session_id))

        clients: list[NetworkClient] = []
        for idx, row in df_train.iterrows():
            client = NetworkClient(
                client_id=f"client_{idx}",
                features=row[FEATURES].values.astype(float),
                target=float(row["target"]),
                session_id=session_id,
                shareholder_addresses=sh_addresses,
                shareholder_x_coords=[1, 2, 3],
                threshold=threshold,
            )
            clients.append(client)

        for client in clients:
            client.submit_stats()

        # Wait for bins to be computed
        while True:
            state = agg_stub.GetRoundState(pb.GetRoundStateRequest(session_id=session_id))
            if state.phase == pb.COLLECTING_GRADIENTS:
                break
            time.sleep(0.2)

        # Training loop: clients submit gradients for each round/depth
        current_round = -1
        current_depth = -1
        while True:
            state = agg_stub.GetRoundState(pb.GetRoundStateRequest(session_id=session_id))
            if state.phase == pb.TRAINING_COMPLETE:
                break

            if state.round_id == current_round and state.depth == current_depth:
                time.sleep(0.1)
                continue

            current_round = state.round_id
            current_depth = state.depth

            # Get training state
            ts = agg_stub.GetTrainingState(pb.GetTrainingStateRequest(session_id=session_id))
            model = pb_to_model(ts.model)
            bins = [pb_to_bin_config(b) for b in ts.bins]
            splits = {nid: pb_to_split_decision(sd) for nid, sd in ts.current_splits.items()}

            for client in clients:
                client.submit_gradients(
                    bins=bins,
                    model=model,
                    splits=splits,
                    round_id=current_round,
                    depth=current_depth,
                    loss="squared",
                )

        # Fetch final model
        model_resp = agg_stub.GetModel(pb.GetModelRequest(session_id=session_id))
        final_model = pb_to_model(model_resp.model)

        # Evaluate
        test_features = df_test[FEATURES].values.astype(float)
        test_targets = df_test["target"].values
        test_preds = final_model.predict(test_features)
        test_classes = (test_preds >= 0.5).astype(int)
        accuracy = np.mean(test_classes == test_targets)

        print(f"gRPC test accuracy: {accuracy:.2%}")
        assert accuracy > 0.75, f"Expected >75% accuracy, got {accuracy:.2%}"

    finally:
        for client in clients:
            client.close()
        agg_channel.close()
        agg_server.stop(grace=0)
        for s in sh_servers:
            s.stop(grace=0)
```

**Step 2: Run the test**

Run: `uv run pytest tests/test_grpc_integration.py -v -s`
Expected: PASS with accuracy >75%.

This test validates the entire gRPC stack end-to-end: serialization, shareholder server, aggregator server with background training loop, network client, and the full training protocol.

**Step 3: Commit**

```bash
git add tests/test_grpc_integration.py
git commit -m "Add end-to-end gRPC integration test"
```

---

### Task 8: Package exports and cleanup

**Files:**
- Modify: `src/privateboost/grpc/__init__.py`
- Run: `make lint` and fix any issues

**Step 1: Set up the grpc package exports**

Update `src/privateboost/grpc/__init__.py`:

```python
"""gRPC servers and client for the privateboost protocol."""

from .aggregator_server import AggregatorServicer, RemoteShareHolder
from .converters import (
    bin_config_to_pb,
    model_to_pb,
    ndarray_to_pb,
    pb_to_bin_config,
    pb_to_model,
    pb_to_ndarray,
    pb_to_share,
    pb_to_split_decision,
    share_to_pb,
    split_decision_to_pb,
)
from .network_client import NetworkClient
from .shareholder_server import ShareholderServicer

__all__ = [
    "AggregatorServicer",
    "NetworkClient",
    "RemoteShareHolder",
    "ShareholderServicer",
    "bin_config_to_pb",
    "model_to_pb",
    "ndarray_to_pb",
    "pb_to_bin_config",
    "pb_to_model",
    "pb_to_ndarray",
    "pb_to_share",
    "pb_to_split_decision",
    "share_to_pb",
    "split_decision_to_pb",
]
```

**Step 2: Lint and fix**

Run: `make fix && make lint`
Expected: No warnings or errors.

**Step 3: Run full test suite**

Run: `make test`
Expected: All tests pass (existing + new gRPC tests).

**Step 4: Commit**

```bash
git add src/privateboost/grpc/__init__.py
git commit -m "Add gRPC package exports"
```

---

### Task 9: Final verification

**Step 1: Run full build/test/lint cycle**

```bash
make test && make lint
```

Expected: All tests pass, zero lint warnings.

**Step 2: Review the diff**

```bash
git diff main --stat
```

Verify the changeset includes:
- `pyproject.toml` — grpcio/protobuf deps
- `proto/privateboost.proto` — protobuf schema
- `src/privateboost/grpc/` — generated stubs + converters + servers + client
- `tests/test_grpc_*.py` — converter, shareholder, aggregator, network client, integration tests
- `Makefile` — proto target
- `docs/plans/` — design + plan docs

No changes to existing core files (`client.py`, `shareholder.py`, `aggregator.py`, `tree.py`, `messages.py`, `crypto/`).
