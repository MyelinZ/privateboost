"""Client class for privacy-preserving federated learning."""

from typing import Dict, List

import numpy as np

from .crypto import compute_commitment, generate_nonce, share
from .messages import (
    BinConfiguration,
    CommittedGradientShare,
    CommittedStatsShare,
    Loss,
    SplitDecision,
)
from .tree import Model
from .shareholder import ShareHolder


def _find_bin_index(value: float, edges: np.ndarray, n_total_bins: int) -> int:
    """Find the bin index for a value given bin edges."""
    bin_idx = int(np.searchsorted(edges, value, side="right")) - 1
    return max(0, min(bin_idx, n_total_bins - 1))


class Client:
    """A data owner that secret-shares its features with shareholders."""

    def __init__(
        self,
        client_id: str,
        features: np.ndarray,
        target: float,
        shareholders: List[ShareHolder],
        threshold: int = 2,
        seed: int | None = None,
    ):
        self.client_id = client_id
        self.features = np.asarray(features)
        self.target = target
        self.shareholders = shareholders
        self._n_parties = len(shareholders)
        self.threshold = threshold
        self._rng = np.random.default_rng(seed)

        if threshold > self._n_parties:
            raise ValueError(f"Threshold {threshold} cannot exceed n_parties {self._n_parties}")

    def _get_node_id(self, splits: Dict[int, SplitDecision]) -> int:
        """Compute current node by traversing splits from root."""
        node_id = 0
        while node_id in splits:
            split = splits[node_id]
            if self.features[split.feature_idx] <= split.threshold:
                node_id = split.left_child_id
            else:
                node_id = split.right_child_id
        return node_id

    @staticmethod
    def _get_active_node_ids(splits: Dict[int, SplitDecision], depth: int) -> List[int]:
        """Get all active (leaf) node IDs at the current tree depth.

        At depth 0, only node 0 is active. At deeper depths, the active nodes
        are the leaf nodes of the partially-built tree — nodes that appear as
        children in splits but are not themselves split yet.
        """
        if depth == 0:
            return [0]
        # Collect all child node IDs that are not themselves split
        active = []
        for split in splits.values():
            if split.left_child_id not in splits:
                active.append(split.left_child_id)
            if split.right_child_id not in splits:
                active.append(split.right_child_id)
        return sorted(active) if active else [0]

    def _new_commitment(self, round_id: int) -> bytes:
        """Generate fresh nonce and compute commitment for a round."""
        nonce = generate_nonce()
        return compute_commitment(round_id, self.client_id, nonce)

    def submit_stats(self) -> None:
        """Secret-share feature values using Shamir with commitment."""
        nonce = generate_nonce()
        commitment = compute_commitment(0, self.client_id, nonce)

        # Interleave x and x² for each feature, then append target stats
        feature_stats = np.column_stack([self.features, self.features**2]).ravel()
        values = np.concatenate([feature_stats, [self.target, self.target**2]])

        shares = share(values, n_parties=self._n_parties, threshold=self.threshold, rng=self._rng)

        for shareholder, s in zip(self.shareholders, shares):
            shareholder.receive_stats(
                CommittedStatsShare(
                    commitment=commitment,
                    share=s,
                )
            )

    def submit_gradients(
        self,
        bins: List[BinConfiguration],
        model: Model,
        splits: Dict[int, SplitDecision],
        round_id: int,
        depth: int,
        loss: Loss = "squared",
        hide_path: bool = False,
    ) -> None:
        """Compute gradients and submit Shamir shares with commitment.

        If hide_path=True, submit shares for ALL active nodes (zeros for
        non-matching nodes), so shareholders cannot learn the client's
        tree path. This costs O(2^d) communication per client per round.
        """
        true_node_id = self._get_node_id(splits)
        commitment = self._new_commitment(round_id)
        prediction = model.predict_one(self.features)

        if loss == "squared":
            gradient = prediction - self.target
            hessian = 1.0
        else:
            p = 1.0 / (1.0 + np.exp(-prediction))
            gradient = p - self.target
            hessian = p * (1.0 - p)

        if hide_path:
            active_nodes = self._get_active_node_ids(splits, depth)
        else:
            active_nodes = [true_node_id]

        for node_id in active_nodes:
            is_true_node = node_id == true_node_id

            all_gradients = []
            all_hessians = []

            for config in bins:
                n_total_bins = config.n_bins + 2
                g_vec = np.zeros(n_total_bins)
                h_vec = np.zeros(n_total_bins)

                if is_true_node:
                    value = self.features[config.feature_idx]
                    bin_idx = _find_bin_index(value, config.edges, n_total_bins)
                    g_vec[bin_idx] = gradient
                    h_vec[bin_idx] = hessian

                all_gradients.append(g_vec)
                all_hessians.append(h_vec)

            values = np.concatenate([np.concatenate(all_gradients), np.concatenate(all_hessians)])
            shares = share(
                values, n_parties=self._n_parties, threshold=self.threshold, rng=self._rng
            )

            for shareholder, s in zip(self.shareholders, shares):
                shareholder.receive_gradients(
                    CommittedGradientShare(
                        round_id=round_id,
                        depth=depth,
                        commitment=commitment,
                        share=s,
                        node_id=node_id,
                    )
                )
