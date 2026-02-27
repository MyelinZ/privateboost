"""gRPC client for the privateboost protocol."""

from .converters import (
    bin_config_to_pb,
    model_to_pb,
    ndarray_to_pb,
    pb_to_bin_config,
    pb_to_model,
    pb_to_ndarray,
    pb_to_share,
    pb_to_split_decision,
    share_to_pb,
    split_decision_to_pb,
)
from .network_client import NetworkClient

__all__ = [
    "NetworkClient",
    "bin_config_to_pb",
    "model_to_pb",
    "ndarray_to_pb",
    "pb_to_bin_config",
    "pb_to_model",
    "pb_to_ndarray",
    "pb_to_share",
    "pb_to_split_decision",
    "share_to_pb",
    "split_decision_to_pb",
]
