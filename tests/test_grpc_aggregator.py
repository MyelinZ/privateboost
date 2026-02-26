"""Tests for AggregatorService gRPC server."""

import grpc
import pytest
from concurrent import futures

from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.shareholder_server import ShareholderServicer
from privateboost.grpc.aggregator_server import AggregatorServicer


def _start_shareholder(min_clients: int = 2) -> tuple[grpc.Server, str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ShareholderServicer(min_clients=min_clients)
    pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, f"localhost:{port}"


@pytest.fixture()
def grpc_cluster():
    sh_servers = []
    sh_addresses = []
    for _ in range(3):
        server, addr = _start_shareholder(min_clients=2)
        sh_servers.append(server)
        sh_addresses.append(addr)

    agg_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = AggregatorServicer(
        shareholder_addresses=sh_addresses,
        n_bins=5,
        threshold=2,
        min_clients=2,
        learning_rate=0.15,
        lambda_reg=2.0,
        n_trees=2,
        max_depth=2,
        loss="squared",
        target_count=3,
        features=[
            pb.FeatureSpec(index=0, name="f0"),
            pb.FeatureSpec(index=1, name="f1"),
        ],
        target_column="target",
    )
    pb_grpc.add_AggregatorServiceServicer_to_server(servicer, agg_server)
    agg_port = agg_server.add_insecure_port("[::]:0")
    agg_server.start()

    agg_channel = grpc.insecure_channel(f"localhost:{agg_port}")

    yield agg_channel, sh_addresses

    agg_channel.close()
    agg_server.stop(grace=0)
    for s in sh_servers:
        s.stop(grace=0)


def test_join_session(grpc_cluster):
    agg_channel, sh_addresses = grpc_cluster
    stub = pb_grpc.AggregatorServiceStub(agg_channel)

    resp = stub.JoinSession(pb.JoinSessionRequest(session_id="test-session"))
    assert resp.session_id == "test-session"
    assert len(resp.shareholders) == 3
    assert resp.threshold == 2
    assert resp.config.n_bins == 5
    assert resp.config.n_trees == 2
    assert resp.config.loss == "squared"
    # x_coords assigned by aggregator: 1, 2, 3
    assert [sh.x_coord for sh in resp.shareholders] == [1, 2, 3]


def test_initial_round_state(grpc_cluster):
    agg_channel, _ = grpc_cluster
    stub = pb_grpc.AggregatorServiceStub(agg_channel)

    stub.JoinSession(pb.JoinSessionRequest(session_id="test-state"))
    resp = stub.GetRoundState(pb.GetRoundStateRequest(session_id="test-state"))
    assert resp.phase == pb.WAITING_FOR_CLIENTS
    assert resp.round_id == 0
    assert resp.depth == 0
