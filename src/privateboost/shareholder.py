"""ShareHolder class for privacy-preserving federated learning."""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from .crypto import Share
from .messages import CommittedGradientShare, CommittedStatsShare

# Type alias for gradient storage: depth -> commitment -> node_id -> Share
GradientStore = Dict[int, Dict[bytes, Dict[int, Share]]]


def _new_gradient_store() -> GradientStore:
    """Create a new nested defaultdict for gradient storage."""
    return defaultdict(lambda: defaultdict(dict))


class ShareHolder:
    """Aggregates secret shares from clients and forwards to the aggregator.

    ShareHolders are intermediate parties that collect and sum secret shares
    without seeing raw client values. They store shares indexed by commitment
    and enforce a minimum client threshold before releasing aggregates.
    """

    def __init__(self, party_id: int, x_coord: int, min_clients: int = 10):
        self.party_id = party_id
        self.x_coord = x_coord
        self.min_clients = min_clients

        self._stats: Dict[bytes, Share] = {}
        self._gradients: GradientStore = _new_gradient_store()
        self._current_round_id: int = -1

    @property
    def current_round_id(self) -> int:
        """The round_id of the most recently received gradient data."""
        return self._current_round_id

    def receive_stats(self, msg: CommittedStatsShare) -> None:
        """Receive and store stats share from a client."""
        self._stats[msg.commitment] = msg.share

    def receive_gradients(self, msg: CommittedGradientShare) -> None:
        """Receive and store gradient share from a client."""
        if msg.round_id > self._current_round_id:
            self._gradients = _new_gradient_store()
            self._current_round_id = msg.round_id

        self._gradients[msg.depth][msg.commitment][msg.node_id] = msg.share

    def get_stats_commitments(self) -> Set[bytes]:
        """Get set of commitments received for stats."""
        return set(self._stats.keys())

    def get_gradient_commitments(self, depth: int) -> Set[bytes]:
        """Get set of commitments received for a depth."""
        return set(self._gradients.get(depth, {}).keys())

    def get_gradient_node_ids(self, depth: int) -> Set[int]:
        """Get set of node_ids that have gradient data for a depth."""
        node_ids: Set[int] = set()
        for client_nodes in self._gradients.get(depth, {}).values():
            node_ids.update(client_nodes.keys())
        return node_ids

    def get_stats_sum(self, commitments: List[bytes]) -> Tuple[int, np.ndarray]:
        """Sum stats shares for requested commitments."""
        if len(commitments) < self.min_clients:
            raise ValueError(
                f"Requested {len(commitments)} clients, minimum is {self.min_clients}"
            )

        total_y = None

        for commitment in commitments:
            s = self._stats.get(commitment)
            if s is None:
                raise ValueError(f"Unknown commitment: {commitment.hex()[:16]}...")

            if total_y is None:
                total_y = s.y.copy()
            else:
                total_y += s.y

        if total_y is None:
            raise ValueError("No shares to sum")

        return (self.x_coord, total_y)

    def get_gradients_sum(
        self, depth: int, commitments: List[bytes], node_id: int
    ) -> Tuple[int, np.ndarray]:
        """Sum gradient shares for requested commitments and node."""
        if len(commitments) < self.min_clients:
            raise ValueError(
                f"Requested {len(commitments)} clients, minimum is {self.min_clients}"
            )

        depth_data = self._gradients.get(depth, {})
        total_y = None

        for commitment in commitments:
            client_nodes = depth_data.get(commitment, {})
            s = client_nodes.get(node_id)

            if s is not None:
                if total_y is None:
                    total_y = s.y.copy()
                else:
                    total_y += s.y

        if total_y is None:
            raise ValueError(f"No shares for node {node_id}")

        return (self.x_coord, total_y)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._stats.clear()
        self._gradients = _new_gradient_store()
