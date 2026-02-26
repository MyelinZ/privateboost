"""Tests for ShareholderService gRPC server."""

import grpc
import numpy as np
import pytest
from concurrent import futures

from privateboost.crypto import share, Share
from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.converters import share_to_pb, pb_to_ndarray
from privateboost.grpc.shareholder_server import ShareholderServicer


@pytest.fixture()
def shareholder_channel():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = ShareholderServicer(min_clients=2)
    pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel
    channel.close()
    server.stop(grace=0)


def _make_shares(values: np.ndarray, n_parties: int = 3) -> list[Share]:
    return share(values, n_parties=n_parties, threshold=2)


def test_submit_and_get_stats_commitments(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-1"

    values = np.array([1.0, 2.0, 3.0, 4.0])
    shares = _make_shares(values)
    commitment = b"\x01" * 32

    stub.SubmitStats(
        pb.SubmitStatsRequest(
            session_id=session_id,
            commitment=commitment,
            share=share_to_pb(shares[0]),
        )
    )

    resp = stub.GetStatsCommitments(
        pb.GetStatsCommitmentsRequest(session_id=session_id)
    )
    assert commitment in resp.commitments


def test_stats_sum(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-2"

    np.random.seed(42)
    values_a = np.array([10.0, 20.0])
    values_b = np.array([30.0, 40.0])
    shares_a = _make_shares(values_a)
    shares_b = _make_shares(values_b)
    commitment_a = b"\x01" * 32
    commitment_b = b"\x02" * 32

    stub.SubmitStats(
        pb.SubmitStatsRequest(
            session_id=session_id,
            commitment=commitment_a,
            share=share_to_pb(shares_a[0]),
        )
    )
    stub.SubmitStats(
        pb.SubmitStatsRequest(
            session_id=session_id,
            commitment=commitment_b,
            share=share_to_pb(shares_b[0]),
        )
    )

    resp = stub.GetStatsSum(
        pb.GetStatsSumRequest(
            session_id=session_id,
            commitments=[commitment_a, commitment_b],
        )
    )
    result = pb_to_ndarray(resp.sum)
    expected = shares_a[0].y + shares_b[0].y
    np.testing.assert_array_almost_equal(result, expected)


def test_min_clients_enforcement(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-3"

    values = np.array([1.0, 2.0])
    shares = _make_shares(values)
    commitment = b"\x03" * 32

    stub.SubmitStats(
        pb.SubmitStatsRequest(
            session_id=session_id,
            commitment=commitment,
            share=share_to_pb(shares[0]),
        )
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.GetStatsSum(
            pb.GetStatsSumRequest(
                session_id=session_id,
                commitments=[commitment],
            )
        )
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_cancel_session(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)
    session_id = "test-session-4"

    values = np.array([1.0, 2.0])
    shares = _make_shares(values)
    stub.SubmitStats(
        pb.SubmitStatsRequest(
            session_id=session_id,
            commitment=b"\x04" * 32,
            share=share_to_pb(shares[0]),
        )
    )

    stub.CancelSession(pb.CancelSessionRequest(session_id=session_id))

    with pytest.raises(grpc.RpcError) as exc_info:
        stub.GetStatsCommitments(pb.GetStatsCommitmentsRequest(session_id=session_id))
    assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND


def test_session_isolation(shareholder_channel):
    stub = pb_grpc.ShareholderServiceStub(shareholder_channel)

    values = np.array([1.0, 2.0])
    shares = _make_shares(values)

    stub.SubmitStats(
        pb.SubmitStatsRequest(
            session_id="session-A",
            commitment=b"\x0a" * 32,
            share=share_to_pb(shares[0]),
        )
    )
    stub.SubmitStats(
        pb.SubmitStatsRequest(
            session_id="session-B",
            commitment=b"\x0b" * 32,
            share=share_to_pb(shares[0]),
        )
    )

    resp_a = stub.GetStatsCommitments(
        pb.GetStatsCommitmentsRequest(session_id="session-A")
    )
    resp_b = stub.GetStatsCommitments(
        pb.GetStatsCommitmentsRequest(session_id="session-B")
    )
    assert len(resp_a.commitments) == 1
    assert len(resp_b.commitments) == 1
    assert resp_a.commitments[0] != resp_b.commitments[0]
