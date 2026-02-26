"""gRPC servers and client for the privateboost protocol."""

from .aggregator_server import AggregatorServicer, RemoteShareHolder
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
from .shareholder_server import ShareholderServicer

__all__ = [
    "AggregatorServicer",
    "NetworkClient",
    "RemoteShareHolder",
    "ShareholderServicer",
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
