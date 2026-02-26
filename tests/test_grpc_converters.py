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
        node_id=0,
        feature_idx=3,
        threshold=1.5,
        gain=0.8,
        left_child_id=1,
        right_child_id=2,
        g_left=-0.5,
        h_left=10.0,
        g_right=0.5,
        h_right=10.0,
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
    tree = Tree(
        root=SplitNode(
            feature_idx=0,
            threshold=1.5,
            gain=0.8,
            left=Leaf(value=-0.1, n_samples=50),
            right=Leaf(value=0.2, n_samples=30),
        )
    )
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
