"""CLI entrypoint for privateboost gRPC servers."""

import os
import sys
from concurrent import futures

import grpc

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .aggregator_server import AggregatorServicer
from .shareholder_server import ShareholderServicer


def _run_shareholder():
    port = os.environ.get("PORT", "50051")
    min_clients = int(os.environ.get("MIN_CLIENTS", "10"))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = ShareholderServicer(min_clients=min_clients)
    pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Shareholder server listening on port {port} (min_clients={min_clients})")
    server.wait_for_termination()


def _run_aggregator():
    port = os.environ.get("PORT", "50052")
    shareholders_str = os.environ.get("SHAREHOLDERS")
    if not shareholders_str:
        print(
            "ERROR: SHAREHOLDERS env var is required (comma-separated host:port list)"
        )
        sys.exit(1)
    shareholder_addresses = [s.strip() for s in shareholders_str.split(",")]

    features_str = os.environ.get("FEATURES", "")
    feature_specs = [
        pb.FeatureSpec(index=i, name=name.strip())
        for i, name in enumerate(features_str.split(","))
        if name.strip()
    ]

    target_count_str = os.environ.get("TARGET_COUNT")
    target_count = int(target_count_str) if target_count_str else None
    target_fraction_str = os.environ.get("TARGET_FRACTION")
    target_fraction = float(target_fraction_str) if target_fraction_str else None

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = AggregatorServicer(
        shareholder_addresses=shareholder_addresses,
        n_bins=int(os.environ.get("N_BINS", "10")),
        threshold=int(os.environ.get("THRESHOLD", "2")),
        min_clients=int(os.environ.get("MIN_CLIENTS", "10")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "0.15")),
        lambda_reg=float(os.environ.get("LAMBDA_REG", "2.0")),
        n_trees=int(os.environ.get("N_TREES", "15")),
        max_depth=int(os.environ.get("MAX_DEPTH", "3")),
        loss=os.environ.get("LOSS", "squared"),
        target_count=target_count,
        target_fraction=target_fraction,
        features=feature_specs,
        target_column=os.environ.get("TARGET_COLUMN", "target"),
    )
    pb_grpc.add_AggregatorServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Aggregator server listening on port {port}")
    print(f"  Shareholders: {shareholder_addresses}")
    print(
        f"  Threshold: {servicer._threshold}, Trees: {servicer._n_trees}, Depth: {servicer._max_depth}"
    )
    server.wait_for_termination()


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m privateboost.grpc.serve <shareholder|aggregator>")
        sys.exit(1)

    role = sys.argv[1]
    if role == "shareholder":
        _run_shareholder()
    elif role == "aggregator":
        _run_aggregator()
    else:
        print(f"Unknown role: {role}. Use 'shareholder' or 'aggregator'.")
        sys.exit(1)
