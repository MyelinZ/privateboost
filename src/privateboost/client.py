"""Client class for privacy-preserving federated learning."""

from typing import TYPE_CHECKING, Dict, List

import numpy as np

from .crypto import compute_commitment, generate_nonce, share
from .messages import (
    BinConfiguration,
    CommittedGradientShare,
    CommittedStatsShare,
    LeafNode,
    Loss,
    SplitDecision,
)

if TYPE_CHECKING:
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
        features: List[float],
        target: float,
        shareholders: List["ShareHolder"],
        threshold: int = 2,
    ):
        self.client_id = client_id
        self.features = features
        self.target = target
        self.shareholders = shareholders
        self._n_parties = len(shareholders)
        self.threshold = threshold

        if threshold > self._n_parties:
            raise ValueError(
                f"Threshold {threshold} cannot exceed n_parties {self._n_parties}"
            )

        self.prediction: float = 0.0
        self.node_id: int = 0

    def _new_commitment(self, round_id: int) -> bytes:
        """Generate fresh nonce and compute commitment for a round."""
        nonce = generate_nonce()
        return compute_commitment(round_id, self.client_id, nonce)

    def submit_stats(self, round_id: int = 0) -> None:
        """Secret-share feature values using Shamir with commitment."""
        commitment = self._new_commitment(round_id)

        values = []
        for x in self.features:
            values.extend([x, x * x])
        values.extend([self.target, self.target * self.target])

        shares = share(np.array(values), n_parties=self._n_parties, threshold=self.threshold)

        for shareholder, s in zip(self.shareholders, shares):
            shareholder.receive_stats(CommittedStatsShare(
                round_id=round_id,
                commitment=commitment,
                share=s,
            ))

    def submit_gradients(
        self,
        bins: List[BinConfiguration],
        active_nodes: List[int],
        round_id: int,
        loss: Loss = "squared",
    ) -> None:
        """Compute gradients and submit Shamir shares with commitment."""
        if self.node_id not in active_nodes:
            return

        commitment = self._new_commitment(round_id)

        if loss == "squared":
            gradient = self.prediction - self.target
            hessian = 1.0
        else:
            p = 1.0 / (1.0 + np.exp(-self.prediction))
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

        for shareholder, s in zip(self.shareholders, shares):
            shareholder.receive_gradients(CommittedGradientShare(
                round_id=round_id,
                commitment=commitment,
                share=s,
                node_id=self.node_id,
            ))

    def apply_splits(self, splits: Dict[int, SplitDecision]) -> None:
        """Apply split decisions to update node assignment."""
        if self.node_id not in splits:
            return

        split = splits[self.node_id]
        feature_value = self.features[split.feature_idx]

        if feature_value <= split.threshold:
            self.node_id = split.left_child_id
        else:
            self.node_id = split.right_child_id

    def update_prediction(
        self, leaves: Dict[int, LeafNode], learning_rate: float
    ) -> None:
        """Update prediction based on leaf values."""
        if self.node_id in leaves:
            self.prediction += learning_rate * leaves[self.node_id].value

    def reset_node(self) -> None:
        """Reset node assignment to root for a new tree."""
        self.node_id = 0
