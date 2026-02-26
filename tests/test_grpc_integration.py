"""End-to-end integration test: full protocol over gRPC."""

import time
from concurrent import futures

import grpc
import numpy as np
import pandas as pd
import pytest

from privateboost.grpc import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from privateboost.grpc.aggregator_server import AggregatorServicer
from privateboost.grpc.converters import (
    pb_to_bin_config,
    pb_to_model,
    pb_to_split_decision,
)
from privateboost.grpc.network_client import NetworkClient
from privateboost.grpc.shareholder_server import ShareholderServicer

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
COLUMNS = [*FEATURES, "target"]


@pytest.fixture(scope="session")
def heart_disease_df(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    cache_dir = tmp_path_factory.mktemp("data")
    cache_file = cache_dir / "heart_disease.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        df = pd.read_csv(url, names=COLUMNS, na_values="?")
        df = df.dropna()
        df["target"] = (df["target"] > 0).astype(int)
        df.to_csv(cache_file, index=False)
    return df


def test_grpc_xgboost_heart_disease(heart_disease_df: pd.DataFrame):
    """End-to-end: federated XGBoost over gRPC achieves >75% accuracy."""
    np.random.seed(42)
    df = heart_disease_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    threshold = 2
    min_clients = 10
    n_trees = 15
    max_depth = 3
    session_id = "integration-test"

    # Start 3 shareholder servers on random ports
    sh_servers = []
    sh_addresses = []
    for _ in range(3):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        servicer = ShareholderServicer(min_clients=min_clients)
        pb_grpc.add_ShareholderServiceServicer_to_server(servicer, server)
        port = server.add_insecure_port("[::]:0")
        server.start()
        sh_servers.append(server)
        sh_addresses.append(f"localhost:{port}")

    # Start aggregator server
    agg_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    feature_specs = [pb.FeatureSpec(index=i, name=n) for i, n in enumerate(FEATURES)]
    agg_servicer = AggregatorServicer(
        shareholder_addresses=sh_addresses,
        n_bins=10,
        threshold=threshold,
        min_clients=min_clients,
        learning_rate=0.15,
        lambda_reg=2.0,
        n_trees=n_trees,
        max_depth=max_depth,
        loss="squared",
        target_count=len(df_train),
        features=feature_specs,
        target_column="target",
    )
    pb_grpc.add_AggregatorServiceServicer_to_server(agg_servicer, agg_server)
    agg_port = agg_server.add_insecure_port("[::]:0")
    agg_server.start()
    agg_channel = grpc.insecure_channel(f"localhost:{agg_port}")
    agg_stub = pb_grpc.AggregatorServiceStub(agg_channel)

    clients = []
    try:
        # Join session (starts background training thread)
        agg_stub.JoinSession(pb.JoinSessionRequest(session_id=session_id))

        # Create network clients
        for idx, row in df_train.iterrows():
            clients.append(
                NetworkClient(
                    client_id=f"client_{idx}",
                    features=row[FEATURES].values.astype(float),
                    target=float(row["target"]),
                    session_id=session_id,
                    shareholder_addresses=sh_addresses,
                    threshold=threshold,
                )
            )

        # Submit statistics to shareholders
        for client in clients:
            client.submit_stats()

        # Wait for aggregator to compute bins
        while True:
            state = agg_stub.GetRoundState(
                pb.GetRoundStateRequest(session_id=session_id)
            )
            if state.phase == pb.COLLECTING_GRADIENTS:
                break
            time.sleep(0.2)

        # Training loop: submit gradients when round/depth advances
        current_round = -1
        current_depth = -1
        while True:
            state = agg_stub.GetRoundState(
                pb.GetRoundStateRequest(session_id=session_id)
            )
            if state.phase == pb.TRAINING_COMPLETE:
                break

            if state.round_id == current_round and state.depth == current_depth:
                time.sleep(0.1)
                continue

            current_round = state.round_id
            current_depth = state.depth

            ts = agg_stub.GetTrainingState(
                pb.GetTrainingStateRequest(session_id=session_id)
            )
            model = pb_to_model(ts.model)
            bins = [pb_to_bin_config(b) for b in ts.bins]
            splits = {
                nid: pb_to_split_decision(sd) for nid, sd in ts.current_splits.items()
            }

            for client in clients:
                client.submit_gradients(
                    bins=bins,
                    model=model,
                    splits=splits,
                    round_id=current_round,
                    depth=current_depth,
                    loss="squared",
                )

        # Evaluate final model
        model_resp = agg_stub.GetModel(pb.GetModelRequest(session_id=session_id))
        final_model = pb_to_model(model_resp.model)

        test_features = df_test[FEATURES].values.astype(float)
        test_targets = df_test["target"].values
        test_preds = final_model.predict(test_features)
        test_classes = (test_preds >= 0.5).astype(int)
        accuracy = np.mean(test_classes == test_targets)

        print(f"gRPC test accuracy: {accuracy:.2%}")
        assert accuracy > 0.75, f"Expected >75% accuracy, got {accuracy:.2%}"

    finally:
        for client in clients:
            client.close()
        agg_channel.close()
        agg_server.stop(grace=0)
        for s in sh_servers:
            s.stop(grace=0)
