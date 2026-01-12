"""privateboost: Privacy-preserving federated XGBoost via Shamir secret sharing.

Example:
    >>> from privateboost import Client, ShareHolder, Aggregator
    >>>
    >>> # Setup with 2-of-3 threshold
    >>> shareholders = [ShareHolder(i, x_coord=i+1, min_clients=10) for i in range(3)]
    >>> aggregator = Aggregator(shareholders, n_bins=10, threshold=2, min_clients=10)
    >>>
    >>> # Create clients
    >>> clients = [Client(f"c{i}", features, target, shareholders, threshold=2) for ...]
    >>>
    >>> # Compute statistics
    >>> for client in clients:
    ...     client.submit_stats(round_id=0)
    >>> bins = aggregator.define_bins(round_id=0)
"""

from .aggregator import Aggregator
from .client import Client
from .crypto import Share
from .messages import BinConfiguration, LeafNode, Loss, MedianResult, SplitDecision
from .shareholder import ShareHolder

__version__ = "0.1.0"

__all__ = [
    "Aggregator",
    "BinConfiguration",
    "Client",
    "LeafNode",
    "Loss",
    "MedianResult",
    "Share",
    "ShareHolder",
    "SplitDecision",
]
