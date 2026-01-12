"""ShareHolder class for privacy-preserving federated learning."""

from typing import Dict, List, Set, Tuple

import numpy as np

from .crypto import Share
from .messages import CommittedGradientShare, CommittedStatsShare


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

        self._stats: Dict[int, Dict[bytes, Share]] = {}
        self._gradients: Dict[int, Dict[bytes, Dict[int, Share]]] = {}

    def receive_stats(self, msg: CommittedStatsShare) -> None:
        """Receive and store stats share from a client."""
        if msg.round_id not in self._stats:
            self._stats[msg.round_id] = {}
        self._stats[msg.round_id][msg.commitment] = msg.share

    def receive_gradients(self, msg: CommittedGradientShare) -> None:
        """Receive and store gradient share from a client."""
        if msg.round_id not in self._gradients:
            self._gradients[msg.round_id] = {}
        if msg.commitment not in self._gradients[msg.round_id]:
            self._gradients[msg.round_id][msg.commitment] = {}
        self._gradients[msg.round_id][msg.commitment][msg.node_id] = msg.share

    def get_commitments(self, round_id: int, round_type: str = "stats") -> Set[bytes]:
        """Get set of commitments received for a round."""
        if round_type == "stats":
            return set(self._stats.get(round_id, {}).keys())
        else:
            return set(self._gradients.get(round_id, {}).keys())

    def get_stats_sum(
        self, round_id: int, commitments: List[bytes]
    ) -> Tuple[int, np.ndarray]:
        """Sum stats shares for requested commitments."""
        if len(commitments) < self.min_clients:
            raise ValueError(
                f"Requested {len(commitments)} clients, minimum is {self.min_clients}"
            )

        round_data = self._stats.get(round_id, {})
        total_y = None

        for commitment in commitments:
            s = round_data.get(commitment)
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
        self, round_id: int, commitments: List[bytes], node_id: int
    ) -> Tuple[int, np.ndarray]:
        """Sum gradient shares for requested commitments and node."""
        if len(commitments) < self.min_clients:
            raise ValueError(
                f"Requested {len(commitments)} clients, minimum is {self.min_clients}"
            )

        round_data = self._gradients.get(round_id, {})
        total_y = None

        for commitment in commitments:
            client_nodes = round_data.get(commitment, {})
            s = client_nodes.get(node_id)

            if s is not None:
                if total_y is None:
                    total_y = s.y.copy()
                else:
                    total_y += s.y

        if total_y is None:
            raise ValueError(f"No shares for node {node_id}")

        return (self.x_coord, total_y)

    def clear_round(self, round_id: int) -> None:
        """Clear data for a completed round."""
        self._stats.pop(round_id, None)
        self._gradients.pop(round_id, None)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._stats.clear()
        self._gradients.clear()
