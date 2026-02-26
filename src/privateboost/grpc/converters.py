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
            return pb.TreeNode(
                split=pb.SplitNode(
                    feature_idx=f,
                    threshold=t,
                    left=_tree_node_to_pb(l),
                    right=_tree_node_to_pb(r),
                )
            )
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
