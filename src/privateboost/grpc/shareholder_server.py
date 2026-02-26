"""gRPC server implementation for ShareholderService."""

import threading
from typing import Dict

import grpc

from privateboost.messages import CommittedGradientShare, CommittedStatsShare
from privateboost.shareholder import ShareHolder

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .converters import ndarray_to_pb, pb_to_share


class ShareholderServicer(pb_grpc.ShareholderServiceServicer):
    """gRPC servicer wrapping ShareHolder instances, one per session.

    Shareholders have no identity (no x_coord). They are pure storage
    engines. The aggregator assigns x_coords from its own config.
    """

    def __init__(self, min_clients: int = 10):
        self._min_clients = min_clients
        self._sessions: Dict[str, ShareHolder] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def _get_or_create_session(
        self, session_id: str, context: grpc.ServicerContext
    ) -> ShareHolder | None:
        with self._lock:
            if session_id in self._cancelled:
                context.abort(
                    grpc.StatusCode.NOT_FOUND, f"Session {session_id} cancelled"
                )
                return None
            if session_id not in self._sessions:
                self._sessions[session_id] = ShareHolder(
                    party_id=0, x_coord=0, min_clients=self._min_clients
                )
            return self._sessions[session_id]

    def _get_session(
        self, session_id: str, context: grpc.ServicerContext
    ) -> ShareHolder | None:
        with self._lock:
            if session_id in self._cancelled:
                context.abort(
                    grpc.StatusCode.NOT_FOUND, f"Session {session_id} cancelled"
                )
                return None
            sh = self._sessions.get(session_id)
            if sh is None:
                context.abort(
                    grpc.StatusCode.NOT_FOUND, f"Session {session_id} not found"
                )
                return None
            return sh

    def SubmitStats(self, request, context):
        sh = self._get_or_create_session(request.session_id, context)
        if sh is None:
            return pb.SubmitStatsResponse()
        share = pb_to_share(request.share)
        sh.receive_stats(
            CommittedStatsShare(commitment=request.commitment, share=share)
        )
        return pb.SubmitStatsResponse()

    def SubmitGradients(self, request, context):
        sh = self._get_or_create_session(request.session_id, context)
        if sh is None:
            return pb.SubmitGradientsResponse()
        share = pb_to_share(request.share)
        sh.receive_gradients(
            CommittedGradientShare(
                round_id=request.round_id,
                depth=request.depth,
                commitment=request.commitment,
                share=share,
                node_id=request.node_id,
            )
        )
        return pb.SubmitGradientsResponse()

    def GetStatsCommitments(self, request, context):
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetStatsCommitmentsResponse()
        return pb.GetStatsCommitmentsResponse(
            commitments=list(sh.get_stats_commitments())
        )

    def GetGradientCommitments(self, request, context):
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetGradientCommitmentsResponse()
        return pb.GetGradientCommitmentsResponse(
            commitments=list(sh.get_gradient_commitments(request.depth))
        )

    def GetGradientNodeIds(self, request, context):
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetGradientNodeIdsResponse()
        return pb.GetGradientNodeIdsResponse(
            node_ids=list(sh.get_gradient_node_ids(request.depth))
        )

    def GetStatsSum(self, request, context):
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetStatsSumResponse()
        try:
            _x_coord, total = sh.get_stats_sum(list(request.commitments))
        except ValueError as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            return pb.GetStatsSumResponse()
        return pb.GetStatsSumResponse(sum=ndarray_to_pb(total))

    def GetGradientsSum(self, request, context):
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.GetGradientsSumResponse()
        try:
            _x_coord, total = sh.get_gradients_sum(
                request.depth, list(request.commitments), request.node_id
            )
        except ValueError as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            return pb.GetGradientsSumResponse()
        return pb.GetGradientsSumResponse(sum=ndarray_to_pb(total))

    def CancelSession(self, request, context):
        with self._lock:
            self._cancelled.add(request.session_id)
            self._sessions.pop(request.session_id, None)
        return pb.CancelSessionResponse()

    def Reset(self, request, context):
        sh = self._get_session(request.session_id, context)
        if sh is None:
            return pb.ResetResponse()
        sh.reset()
        return pb.ResetResponse()
