"""gRPC-based client for the privateboost protocol."""

import hashlib
from typing import Dict, List

import grpc
import numpy as np

from privateboost.client import _find_bin_index
from privateboost.crypto import generate_nonce, compute_commitment, share
from privateboost.messages import BinConfiguration, Loss, SplitDecision
from privateboost.tree import Model

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .converters import share_to_pb


class NetworkClient:
    """A data owner that secret-shares its features with remote shareholders via gRPC."""

    def __init__(
        self,
        client_id: str,
        features: np.ndarray,
        target: float,
        session_id: str,
        shareholder_addresses: list[str],
        threshold: int = 2,
    ):
        self.client_id = client_id
        self.features = np.asarray(features)
        self.target = target
        self.session_id = session_id
        self.threshold = threshold
        self._n_parties = len(shareholder_addresses)

        self._channels = [grpc.insecure_channel(addr) for addr in shareholder_addresses]
        self._stubs = [pb_grpc.ShareholderServiceStub(ch) for ch in self._channels]

    def _stats_commitment(self) -> bytes:
        h = hashlib.sha256()
        h.update(self.session_id.encode("utf-8"))
        h.update(self.client_id.encode("utf-8"))
        return h.digest()

    def submit_stats(self) -> None:
        commitment = self._stats_commitment()
        feature_stats = np.column_stack([self.features, self.features**2]).ravel()
        values = np.concatenate([feature_stats, [self.target, self.target**2]])
        shares = share(values, n_parties=self._n_parties, threshold=self.threshold)

        for stub, s in zip(self._stubs, shares):
            stub.SubmitStats(
                pb.SubmitStatsRequest(
                    session_id=self.session_id,
                    commitment=commitment,
                    share=share_to_pb(s),
                )
            )

    def submit_gradients(
        self,
        bins: List[BinConfiguration],
        model: Model,
        splits: Dict[int, SplitDecision],
        round_id: int,
        depth: int,
        loss: Loss = "squared",
    ) -> None:
        node_id = self._get_node_id(splits)
        nonce = generate_nonce()
        commitment = compute_commitment(round_id, self.client_id, nonce)
        prediction = model.predict_one(self.features)

        if loss == "squared":
            gradient = prediction - self.target
            hessian = 1.0
        else:
            p = 1.0 / (1.0 + np.exp(-prediction))
            gradient = p - self.target
            hessian = p * (1.0 - p)

        all_gradients = []
        all_hessians = []
        for config in bins:
            value = self.features[config.feature_idx]
            n_total_bins = config.n_bins + 2
            bin_idx = _find_bin_index(value, config.edges, n_total_bins)
            g_vec = np.zeros(n_total_bins)
            h_vec = np.zeros(n_total_bins)
            g_vec[bin_idx] = gradient
            h_vec[bin_idx] = hessian
            all_gradients.append(g_vec)
            all_hessians.append(h_vec)

        values = np.concatenate(
            [np.concatenate(all_gradients), np.concatenate(all_hessians)]
        )
        shares = share(values, n_parties=self._n_parties, threshold=self.threshold)

        for stub, s in zip(self._stubs, shares):
            stub.SubmitGradients(
                pb.SubmitGradientsRequest(
                    session_id=self.session_id,
                    round_id=round_id,
                    depth=depth,
                    commitment=commitment,
                    share=share_to_pb(s),
                    node_id=node_id,
                )
            )

    def _get_node_id(self, splits: Dict[int, SplitDecision]) -> int:
        node_id = 0
        while node_id in splits:
            split = splits[node_id]
            if self.features[split.feature_idx] <= split.threshold:
                node_id = split.left_child_id
            else:
                node_id = split.right_child_id
        return node_id

    def close(self):
        for ch in self._channels:
            ch.close()
