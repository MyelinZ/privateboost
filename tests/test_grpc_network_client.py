"""Tests for NetworkClient gRPC client."""

import grpc
import numpy as np
import pytest
from concurrent import futures

from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.shareholder_server import ShareholderServicer
from privateboost.grpc.network_client import NetworkClient


def _start_shareholder(min_clients: int = 2) -> tuple[grpc.Server, str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ShareholderServicer(min_clients=min_clients)
    pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, f"localhost:{port}"


@pytest.fixture()
def shareholder_cluster():
    servers = []
    addresses = []
    for _ in range(3):
        server, addr = _start_shareholder(min_clients=2)
        servers.append(server)
        addresses.append(addr)
    yield addresses
    for s in servers:
        s.stop(grace=0)


def test_network_client_submit_stats(shareholder_cluster):
    session_id = "test-nc-stats"
    client = NetworkClient(
        client_id="client_0",
        features=np.array([1.0, 2.0]),
        target=1.0,
        session_id=session_id,
        shareholder_addresses=shareholder_cluster,
        threshold=2,
    )

    client.submit_stats()

    channel = grpc.insecure_channel(shareholder_cluster[0])
    stub = pb_grpc.ShareholderServiceStub(channel)
    resp = stub.GetStatsCommitments(
        pb.GetStatsCommitmentsRequest(session_id=session_id)
    )
    assert len(resp.commitments) == 1
    channel.close()
    client.close()
