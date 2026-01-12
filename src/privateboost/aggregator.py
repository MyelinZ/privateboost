"""Aggregator class for privacy-preserving federated learning."""

from itertools import combinations
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .crypto import Share, reconstruct
from .messages import (
    BinConfiguration,
    GradientHistogram,
    NodeTotals,
    SplitDecision,
)
from .shareholder import ShareHolder
from .tree import Leaf, Model, SplitNode, Tree, TreeNode

BIN_RANGE_STDS = 3
MIN_HESSIAN_THRESHOLD = 0.1


class Aggregator:
    """Reconstructs aggregate statistics from shareholder submissions."""

    def __init__(
        self,
        shareholders: List[ShareHolder],
        n_bins: int = 10,
        threshold: int = 2,
        min_clients: int = 10,
        learning_rate: float = 0.1,
        lambda_reg: float = 1.0,
    ):
        if len(shareholders) < threshold:
            raise ValueError(
                f"Need at least {threshold} shareholders, got {len(shareholders)}"
            )

        self._shareholders = shareholders
        self.n_bins = n_bins
        self.threshold = threshold
        self.min_clients = min_clients
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg

        self._n_clients: int = 0
        self._n_features: int = 0
        self._means: Optional[np.ndarray] = None
        self._variances: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._bin_configs: List[BinConfiguration] = []
        self._model = Model(initial_prediction=0.0, learning_rate=learning_rate)

        self._next_node_id: int = 1
        self._node_totals: Dict[int, NodeTotals] = {}
        self._splits: Dict[int, SplitDecision] = {}

    def _select_shareholders(
        self, get_commitments: Callable[[ShareHolder], Set[bytes]]
    ) -> Tuple[List[ShareHolder], List[bytes]]:
        """Select threshold shareholders with largest commitment overlap."""
        sh_commitments = [
            (sh, get_commitments(sh)) for sh in self._shareholders
        ]

        best_overlap: Set[bytes] = set()
        best_group: List[ShareHolder] = []

        for combo in combinations(sh_commitments, self.threshold):
            shareholders = [sh for sh, _ in combo]
            commit_sets = [commits for _, commits in combo]
            overlap = set.intersection(*commit_sets)

            if len(overlap) > len(best_overlap):
                best_overlap = overlap
                best_group = shareholders

        if len(best_overlap) < self.min_clients:
            raise ValueError(
                f"Best overlap has {len(best_overlap)} clients, need {self.min_clients}"
            )

        return best_group, list(best_overlap)

    def _collect_stats_shares(
        self,
        selected_shs: List[ShareHolder],
        commitments: List[bytes],
    ) -> List[Share]:
        """Collect summed stats shares from shareholders."""
        shares = []
        for sh in selected_shs:
            try:
                x, y = sh.get_stats_sum(commitments)
                shares.append(Share(x=x, y=y))
            except ValueError:
                continue
        return shares

    def _collect_gradient_shares(
        self,
        selected_shs: List[ShareHolder],
        depth: int,
        commitments: List[bytes],
        node_id: int,
    ) -> List[Share]:
        """Collect summed gradient shares from shareholders."""
        shares = []
        for sh in selected_shs:
            try:
                x, y = sh.get_gradients_sum(depth, commitments, node_id)
                shares.append(Share(x=x, y=y))
            except ValueError:
                continue
        return shares

    def define_bins(self) -> List[BinConfiguration]:
        """Reconstruct statistics and define histogram bins."""
        selected_shs, commitments = self._select_shareholders(
            lambda sh: sh.get_stats_commitments()
        )
        self._n_clients = len(commitments)

        shares = self._collect_stats_shares(selected_shs, commitments)
        totals = reconstruct(shares, threshold=self.threshold)

        n_values = len(totals)
        n_total = n_values // 2
        self._n_features = n_total - 1

        means = np.zeros(n_total)
        variances = np.zeros(n_total)

        for idx in range(n_total):
            total_x = totals[idx * 2]
            total_x2 = totals[idx * 2 + 1]
            means[idx] = total_x / self._n_clients
            variances[idx] = (total_x2 / self._n_clients) - (means[idx] ** 2)

        self._means = means
        self._variances = variances
        self._stds = np.sqrt(np.maximum(variances, 0))

        self._bin_configs = []
        for idx in range(self._n_features):
            mean = self._means[idx]
            std = self._stds[idx]

            range_min = mean - BIN_RANGE_STDS * std
            range_max = mean + BIN_RANGE_STDS * std
            inner_edges = np.linspace(range_min, range_max, self.n_bins + 1)
            edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])

            self._bin_configs.append(BinConfiguration(
                feature_idx=idx,
                edges=edges,
                inner_edges=inner_edges,
                n_bins=self.n_bins,
            ))

        self._model.initial_prediction = float(self._means[-1])

        return self._bin_configs

    def compute_splits(
        self,
        depth: int,
        min_gain: float = 0.0,
        min_samples: int = 1,
    ) -> bool:
        """Compute and store the best split for each active node.

        Returns True if any new splits were found at this depth.
        """
        if depth == 0:
            self._next_node_id = 1
            self._node_totals.clear()
            self._splits.clear()

        n_splits_before = len(self._splits)
        selected_shs, commitments = self._select_shareholders(
            lambda sh: sh.get_gradient_commitments(depth)
        )

        active_nodes: Set[int] = set()
        for sh in selected_shs:
            active_nodes.update(sh.get_gradient_node_ids(depth))

        for node_id in active_nodes:
            shares = self._collect_gradient_shares(
                selected_shs, depth, commitments, node_id
            )

            if len(shares) < self.threshold:
                continue

            totals = reconstruct(shares, threshold=self.threshold)

            n_bins_total = self.n_bins + 2
            n_features = self._n_features
            grad_size = n_features * n_bins_total

            gradient_flat = totals[:grad_size]
            hessian_flat = totals[grad_size:]

            histograms: List[GradientHistogram] = []
            for f in range(n_features):
                start = f * n_bins_total
                end = start + n_bins_total
                histograms.append(GradientHistogram(
                    gradient=gradient_flat[start:end],
                    hessian=hessian_flat[start:end],
                ))

            best_gain = min_gain
            best_split = None

            # Each feature's histogram sums to the same total (one bin per client per feature)
            total_g = histograms[0].gradient.sum()
            total_h = histograms[0].hessian.sum()

            self._node_totals[node_id] = NodeTotals(
                gradient_sum=total_g, hessian_sum=total_h
            )

            n_samples = int(round(total_h))
            if n_samples < min_samples:
                continue

            base_score = (total_g**2) / (total_h + self.lambda_reg)

            for feature_idx, hist in enumerate(histograms):
                if feature_idx >= len(self._bin_configs):
                    continue
                config = self._bin_configs[feature_idx]

                g_cumsum = np.cumsum(hist.gradient)
                h_cumsum = np.cumsum(hist.hessian)

                for i in range(len(hist.gradient) - 1):
                    g_left = g_cumsum[i]
                    h_left = h_cumsum[i]
                    g_right = total_g - g_left
                    h_right = total_h - h_left

                    if h_left < MIN_HESSIAN_THRESHOLD or h_right < MIN_HESSIAN_THRESHOLD:
                        continue

                    left_score = (g_left**2) / (h_left + self.lambda_reg)
                    right_score = (g_right**2) / (h_right + self.lambda_reg)
                    gain = left_score + right_score - base_score

                    if gain > best_gain:
                        best_gain = gain
                        threshold = config.edges[i + 1]

                        best_split = SplitDecision(
                            node_id=node_id,
                            feature_idx=feature_idx,
                            threshold=threshold,
                            gain=gain,
                            left_child_id=self._next_node_id,
                            right_child_id=self._next_node_id + 1,
                            g_left=g_left,
                            h_left=h_left,
                            g_right=g_right,
                            h_right=h_right,
                        )

            if best_split is not None:
                self._splits[node_id] = best_split
                self._node_totals[best_split.left_child_id] = NodeTotals(
                    gradient_sum=best_split.g_left, hessian_sum=best_split.h_left
                )
                self._node_totals[best_split.right_child_id] = NodeTotals(
                    gradient_sum=best_split.g_right, hessian_sum=best_split.h_right
                )
                self._next_node_id += 2

        return len(self._splits) > n_splits_before

    @property
    def splits(self) -> Dict[int, SplitDecision]:
        """Return the current split decisions."""
        return self._splits

    def finish_round(self) -> None:
        """Finalize the current round by building a tree from splits and adding to model."""

        def compute_leaf_value(node_id: int) -> Leaf:
            if node_id in self._node_totals:
                totals = self._node_totals[node_id]
                total_g, total_h = totals.gradient_sum, totals.hessian_sum
            else:
                total_g, total_h = 0.0, 0.0

            if total_h + self.lambda_reg > 0:
                value = -total_g / (total_h + self.lambda_reg)
            else:
                value = 0.0

            return Leaf(value=value, n_samples=int(round(total_h)))

        def build_node(node_id: int) -> TreeNode:
            if node_id in self._splits:
                split = self._splits[node_id]
                return SplitNode(
                    feature_idx=split.feature_idx,
                    threshold=split.threshold,
                    gain=split.gain,
                    left=build_node(split.left_child_id),
                    right=build_node(split.right_child_id),
                )
            else:
                return compute_leaf_value(node_id)

        tree = Tree(root=build_node(0))
        self._model.add_tree(tree)

    @property
    def model(self) -> Model:
        """Return the current model with all built trees."""
        return self._model

    @property
    def means(self) -> Optional[np.ndarray]:
        return self._means

    @property
    def variances(self) -> Optional[np.ndarray]:
        return self._variances

    @property
    def stds(self) -> Optional[np.ndarray]:
        return self._stds

    @property
    def n_clients(self) -> int:
        return self._n_clients

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._means = None
        self._variances = None
        self._stds = None
        self._bin_configs = []
        self._model = Model(initial_prediction=0.0, learning_rate=self.learning_rate)
        self._n_clients = 0
        self._n_features = 0
        self._next_node_id = 1
        self._node_totals.clear()
        self._splits.clear()

