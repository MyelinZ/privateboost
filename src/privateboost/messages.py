"""Public dataclasses for protocol results."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .crypto import Share

Loss = Literal["squared", "logistic"]


@dataclass
class CommittedShare:
    """Base class for shares with commitment."""

    round_id: int
    commitment: bytes
    share: Share


@dataclass
class CommittedStatsShare(CommittedShare):
    """Stats share: y contains [x0, x0², x1, x1², ..., target, target²]."""

    pass


@dataclass
class CommittedGradientShare(CommittedShare):
    """Gradient share for a specific tree node."""

    node_id: int


@dataclass
class NodeTotals:
    """Gradient and hessian totals for a tree node."""

    gradient_sum: float
    hessian_sum: float


@dataclass
class GradientHistogram:
    """Gradient and hessian histograms for a feature."""

    gradient: np.ndarray
    hessian: np.ndarray


@dataclass
class BinConfiguration:
    """Histogram bin configuration for a feature."""

    feature_idx: int
    edges: np.ndarray
    inner_edges: np.ndarray
    n_bins: int


@dataclass
class MedianResult:
    """Result of median estimation for a feature."""

    feature_idx: int
    estimated_median: float
    histogram: np.ndarray
    bin_edges: np.ndarray


@dataclass
class SplitDecision:
    """A split decision for a tree node."""

    node_id: int
    feature_idx: int
    threshold: float
    gain: float
    left_child_id: int
    right_child_id: int
    g_left: float
    h_left: float
    g_right: float
    h_right: float


@dataclass
class LeafNode:
    """A leaf node with its prediction value."""

    node_id: int
    value: float
    n_samples: int
