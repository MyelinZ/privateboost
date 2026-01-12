"""Aggregator class for privacy-preserving federated learning."""

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .crypto import Share, reconstruct
from .messages import (
    BinConfiguration,
    GradientHistogram,
    LeafNode,
    NodeTotals,
    RoundType,
    SplitDecision,
)
from .shareholder import ShareHolder

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
    ):
        if len(shareholders) < threshold:
            raise ValueError(
                f"Need at least {threshold} shareholders, got {len(shareholders)}"
            )

        self._shareholders = shareholders
        self.n_bins = n_bins
        self.threshold = threshold
        self.min_clients = min_clients

        self._n_clients: int = 0
        self._n_features: int = 0
        self._means: Optional[np.ndarray] = None
        self._variances: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._bin_configs: List[BinConfiguration] = []

        self._next_node_id: int = 1
        self._node_totals: Dict[int, NodeTotals] = {}

    def _select_shareholders(
        self, round_id: int, round_type: RoundType = "stats"
    ) -> Tuple[List[ShareHolder], List[bytes]]:
        """Select shareholders with largest commitment overlap (for threshold=2)."""
        sh_commitments = [
            (sh, sh.get_commitments(round_id, round_type)) for sh in self._shareholders
        ]

        best_overlap: Set[bytes] = set()
        best_pair: List[ShareHolder] = []

        for (sh_a, commits_a), (sh_b, commits_b) in combinations(sh_commitments, 2):
            overlap = commits_a & commits_b
            if len(overlap) > len(best_overlap):
                best_overlap = overlap
                best_pair = [sh_a, sh_b]

        if len(best_overlap) < self.min_clients:
            raise ValueError(
                f"Best overlap has {len(best_overlap)} clients, need {self.min_clients}"
            )

        return best_pair, list(best_overlap)

    def _select_shareholders_for_threshold(
        self, round_id: int, round_type: RoundType = "stats"
    ) -> Tuple[List[ShareHolder], List[bytes]]:
        """Select `threshold` shareholders with largest commitment overlap."""
        if self.threshold == 2:
            return self._select_shareholders(round_id, round_type)

        sh_commitments = [
            (sh, sh.get_commitments(round_id, round_type)) for sh in self._shareholders
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

    def _collect_shares(
        self,
        selected_shs: List[ShareHolder],
        round_id: int,
        commitments: List[bytes],
        node_id: Optional[int] = None,
    ) -> List[Share]:
        """Collect summed shares from shareholders."""
        shares = []
        for sh in selected_shs:
            try:
                if node_id is None:
                    x, y = sh.get_stats_sum(round_id, commitments)
                else:
                    x, y = sh.get_gradients_sum(round_id, commitments, node_id)
                shares.append(Share(x=x, y=y))
            except ValueError:
                continue
        return shares

    def define_bins(self, round_id: int = 0) -> List[BinConfiguration]:
        """Reconstruct statistics and define histogram bins."""
        selected_shs, commitments = self._select_shareholders_for_threshold(
            round_id, "stats"
        )
        self._n_clients = len(commitments)

        shares = self._collect_shares(selected_shs, round_id, commitments)
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

        return self._bin_configs

    def find_best_splits(
        self,
        round_id: int,
        active_nodes: List[int],
        lambda_reg: float = 1.0,
        min_gain: float = 0.0,
        min_samples: int = 1,
    ) -> Dict[int, SplitDecision]:
        """Find the best split for each active node."""
        selected_shs, commitments = self._select_shareholders_for_threshold(
            round_id, "gradients"
        )

        splits: Dict[int, SplitDecision] = {}

        for node_id in active_nodes:
            shares = self._collect_shares(selected_shs, round_id, commitments, node_id)

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

            total_g = histograms[0].gradient.sum()
            total_h = histograms[0].hessian.sum()

            self._node_totals[node_id] = NodeTotals(
                gradient_sum=total_g, hessian_sum=total_h
            )

            n_samples = int(round(total_h))
            if n_samples < min_samples:
                continue

            base_score = (total_g**2) / (total_h + lambda_reg)

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

                    left_score = (g_left**2) / (h_left + lambda_reg)
                    right_score = (g_right**2) / (h_right + lambda_reg)
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
                splits[node_id] = best_split
                self._next_node_id += 2

        return splits

    def compute_leaf_values(
        self,
        round_id: int,
        leaf_nodes: List[int],
        lambda_reg: float = 1.0,
    ) -> Dict[int, LeafNode]:
        """Compute prediction values for leaf nodes."""
        selected_shs, commitments = self._select_shareholders_for_threshold(
            round_id, "gradients"
        )

        leaves: Dict[int, LeafNode] = {}

        for node_id in leaf_nodes:
            if node_id in self._node_totals:
                totals = self._node_totals[node_id]
                total_g, total_h = totals.gradient_sum, totals.hessian_sum
            else:
                shares = self._collect_shares(selected_shs, round_id, commitments, node_id)

                if len(shares) < self.threshold:
                    total_g, total_h = 0.0, 0.0
                else:
                    totals_arr = reconstruct(shares, threshold=self.threshold)
                    n_bins_total = self.n_bins + 2
                    n_features = self._n_features
                    grad_size = n_features * n_bins_total

                    total_g = totals_arr[:grad_size].sum() / n_features
                    total_h = totals_arr[grad_size:].sum() / n_features

            if total_h + lambda_reg > 0:
                value = -total_g / (total_h + lambda_reg)
            else:
                value = 0.0

            leaves[node_id] = LeafNode(
                node_id=node_id,
                value=value,
                n_samples=int(round(total_h)),
            )

        return leaves

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
        self._n_clients = 0
        self._n_features = 0
        self._next_node_id = 1
        self._node_totals.clear()

    def reset_tree(self) -> None:
        """Reset state for a new tree."""
        self._next_node_id = 1
        self._node_totals.clear()
