"""gRPC server implementation for AggregatorService."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import grpc
import numpy as np

from privateboost.aggregator import Aggregator

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .converters import (
    bin_config_to_pb,
    model_to_pb,
    pb_to_ndarray,
    split_decision_to_pb,
)


class RemoteShareHolder:
    """Adapter: calls a remote ShareholderService via gRPC.

    Implements the aggregator-facing subset of the ShareHolder interface.
    The aggregator assigns x_coord -- the remote shareholder doesn't know it.
    """

    def __init__(self, address: str, x_coord: int, session_id: str):
        self.x_coord = x_coord
        self._session_id = session_id
        self._round_id = 0
        self._channel = grpc.insecure_channel(address)
        self._stub = pb_grpc.ShareholderServiceStub(self._channel)

    def get_stats_commitments(self) -> Set[bytes]:
        resp = self._stub.GetStatsCommitments(
            pb.GetStatsCommitmentsRequest(session_id=self._session_id)
        )
        return set(resp.commitments)

    def get_gradient_commitments(self, depth: int) -> Set[bytes]:
        resp = self._stub.GetGradientCommitments(
            pb.GetGradientCommitmentsRequest(
                session_id=self._session_id,
                round_id=self._round_id,
                depth=depth,
            )
        )
        return set(resp.commitments)

    def get_gradient_node_ids(self, depth: int) -> Set[int]:
        resp = self._stub.GetGradientNodeIds(
            pb.GetGradientNodeIdsRequest(
                session_id=self._session_id,
                round_id=self._round_id,
                depth=depth,
            )
        )
        return set(resp.node_ids)

    def get_stats_sum(self, commitments: List[bytes]) -> Tuple[int, np.ndarray]:
        resp = self._stub.GetStatsSum(
            pb.GetStatsSumRequest(
                session_id=self._session_id,
                commitments=commitments,
            )
        )
        return (self.x_coord, pb_to_ndarray(resp.sum))

    def get_gradients_sum(
        self,
        depth: int,
        commitments: List[bytes],
        node_id: int,
    ) -> Tuple[int, np.ndarray]:
        resp = self._stub.GetGradientsSum(
            pb.GetGradientsSumRequest(
                session_id=self._session_id,
                round_id=self._round_id,
                depth=depth,
                commitments=commitments,
                node_id=node_id,
            )
        )
        return (self.x_coord, pb_to_ndarray(resp.sum))

    def close(self):
        self._channel.close()


@dataclass
class SessionState:
    phase: int = 0  # pb.WAITING_FOR_CLIENTS
    round_id: int = 0
    depth: int = 0
    aggregator: Aggregator | None = None
    remote_shareholders: list[RemoteShareHolder] = field(default_factory=list)
    training_thread: threading.Thread | None = None


class AggregatorServicer(pb_grpc.AggregatorServiceServicer):
    """gRPC servicer for the Aggregator.

    Assigns x_coords (1, 2, 3, ...) to shareholders based on their
    position in the shareholder_addresses list.
    """

    def __init__(
        self,
        shareholder_addresses: list[str],
        n_bins: int = 10,
        threshold: int = 2,
        min_clients: int = 10,
        learning_rate: float = 0.1,
        lambda_reg: float = 1.0,
        n_trees: int = 10,
        max_depth: int = 3,
        loss: str = "squared",
        target_count: int | None = None,
        target_fraction: float | None = None,
        features: list[pb.FeatureSpec] | None = None,
        target_column: str = "target",
    ):
        self._sh_addresses = shareholder_addresses
        self._n_bins = n_bins
        self._threshold = threshold
        self._min_clients = min_clients
        self._learning_rate = learning_rate
        self._lambda_reg = lambda_reg
        self._n_trees = n_trees
        self._max_depth = max_depth
        self._loss = loss
        self._target_count = target_count
        self._target_fraction = target_fraction
        self._features = features or []
        self._target_column = target_column
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def _get_or_create_session(self, session_id: str) -> SessionState:
        with self._lock:
            if session_id not in self._sessions:
                remote_shs = [
                    RemoteShareHolder(addr, x_coord=i + 1, session_id=session_id)
                    for i, addr in enumerate(self._sh_addresses)
                ]
                aggregator = Aggregator(
                    shareholders=remote_shs,  # type: ignore[arg-type]
                    n_bins=self._n_bins,
                    threshold=self._threshold,
                    min_clients=self._min_clients,
                    learning_rate=self._learning_rate,
                    lambda_reg=self._lambda_reg,
                )
                self._sessions[session_id] = SessionState(
                    aggregator=aggregator,
                    remote_shareholders=remote_shs,
                )
            return self._sessions[session_id]

    def _get_session(
        self, session_id: str, context: grpc.ServicerContext
    ) -> SessionState | None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Session {session_id} not found",
                )
                return None
            return state

    def _run_training(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None or state.aggregator is None:
            return

        agg = state.aggregator
        target = self._target_count or self._min_clients

        while True:
            try:
                commitments = agg._shareholders[0].get_stats_commitments()
                if len(commitments) >= target:
                    break
            except Exception:
                pass
            time.sleep(0.1)

        agg.define_bins()

        for round_id in range(self._n_trees):
            with self._lock:
                state.phase = pb.COLLECTING_GRADIENTS
                state.round_id = round_id
                state.depth = 0

            for rsh in state.remote_shareholders:
                rsh._round_id = round_id

            for depth in range(self._max_depth):
                with self._lock:
                    state.depth = depth

                while True:
                    try:
                        commitments = agg._shareholders[0].get_gradient_commitments(
                            depth
                        )
                        if len(commitments) >= target:
                            break
                    except Exception:
                        pass
                    time.sleep(0.1)

                if not agg.compute_splits(depth=depth, min_samples=1):
                    break

            agg.finish_round()

        with self._lock:
            state.phase = pb.TRAINING_COMPLETE

    def JoinSession(self, request, context):
        state = self._get_or_create_session(request.session_id)

        shareholders = [
            pb.ShareholderInfo(id=str(i), address=addr, x_coord=i + 1)
            for i, addr in enumerate(self._sh_addresses)
        ]

        config = pb.TrainingConfig(
            features=self._features,
            target_column=self._target_column,
            loss=self._loss,
            n_bins=self._n_bins,
            n_trees=self._n_trees,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            lambda_reg=self._lambda_reg,
            min_clients=self._min_clients,
        )
        if self._target_count is not None:
            config.target_count = self._target_count
        elif self._target_fraction is not None:
            config.target_fraction = self._target_fraction

        if state.training_thread is None:
            state.training_thread = threading.Thread(
                target=self._run_training,
                args=(request.session_id,),
                daemon=True,
            )
            state.training_thread.start()

        return pb.JoinSessionResponse(
            session_id=request.session_id,
            shareholders=shareholders,
            threshold=self._threshold,
            config=config,
        )

    def GetRoundState(self, request, context):
        state = self._get_session(request.session_id, context)
        if state is None:
            return pb.GetRoundStateResponse()
        with self._lock:
            return pb.GetRoundStateResponse(
                phase=state.phase,
                round_id=state.round_id,
                depth=state.depth,
            )

    def GetTrainingState(self, request, context):
        state = self._get_session(request.session_id, context)
        if state is None:
            return pb.GetTrainingStateResponse()
        agg = state.aggregator
        with self._lock:
            return pb.GetTrainingStateResponse(
                model=model_to_pb(agg.model),
                current_splits={
                    nid: split_decision_to_pb(sd) for nid, sd in agg.splits.items()
                },
                bins=[bin_config_to_pb(bc) for bc in agg._bin_configs],
                round_id=state.round_id,
                depth=state.depth,
            )

    def GetModel(self, request, context):
        state = self._get_session(request.session_id, context)
        if state is None:
            return pb.GetModelResponse()
        return pb.GetModelResponse(model=model_to_pb(state.aggregator.model))

    def CancelSession(self, request, context):
        with self._lock:
            state = self._sessions.pop(request.session_id, None)
        if state is not None:
            for rsh in state.remote_shareholders:
                try:
                    rsh._stub.CancelSession(
                        pb.CancelSessionRequest(session_id=request.session_id)
                    )
                except Exception:
                    pass
                rsh.close()
        return pb.CancelSessionResponse()
