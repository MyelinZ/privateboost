"""Simulation script: runs the full federated XGBoost protocol over gRPC."""

import os
import time
import uuid

import grpc
import numpy as np
import pandas as pd

from . import privateboost_pb2 as pb, privateboost_pb2_grpc as pb_grpc
from .converters import pb_to_bin_config, pb_to_model, pb_to_split_decision
from .network_client import NetworkClient

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


def _load_heart_disease() -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, names=COLUMNS, na_values="?")
    df = df.dropna()
    df["target"] = (df["target"] > 0).astype(int)
    return df


def main():
    aggregator_addr = os.environ.get("AGGREGATOR", "localhost:50052")
    session_id = os.environ.get("SESSION_ID", str(uuid.uuid4()))

    print(f"Connecting to aggregator at {aggregator_addr}")
    print(f"Session ID: {session_id}")

    channel = grpc.insecure_channel(aggregator_addr)
    stub = pb_grpc.AggregatorServiceStub(channel)

    # Join session
    join_resp = stub.JoinSession(pb.JoinSessionRequest(session_id=session_id))
    sh_addresses = [sh.address for sh in join_resp.shareholders]
    threshold = join_resp.threshold
    config = join_resp.config
    print(
        f"Joined session with {len(sh_addresses)} shareholders, threshold={threshold}"
    )
    print(
        f"Config: {config.n_trees} trees, depth {config.max_depth}, lr={config.learning_rate}"
    )

    # Load dataset
    print("Loading Heart Disease dataset...")
    df = _load_heart_disease()
    np.random.seed(42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")

    # Create clients
    clients = []
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

    # Submit stats
    print("Submitting statistics...")
    for i, client in enumerate(clients):
        client.submit_stats()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(clients)} clients submitted stats")
    print(f"All {len(clients)} clients submitted stats")

    # Wait for training to start
    print("Waiting for aggregator to compute bins...")
    while True:
        state = stub.GetRoundState(pb.GetRoundStateRequest(session_id=session_id))
        if state.phase == pb.COLLECTING_GRADIENTS:
            break
        time.sleep(0.5)
    print("Bins computed, starting gradient rounds")

    # Training loop
    current_round = -1
    current_depth = -1
    while True:
        state = stub.GetRoundState(pb.GetRoundStateRequest(session_id=session_id))
        if state.phase == pb.TRAINING_COMPLETE:
            break

        if state.round_id == current_round and state.depth == current_depth:
            time.sleep(0.1)
            continue

        current_round = state.round_id
        current_depth = state.depth
        print(f"  Round {current_round}, depth {current_depth}")

        ts = stub.GetTrainingState(pb.GetTrainingStateRequest(session_id=session_id))
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
                loss=config.loss,
            )

    # Fetch final model
    print("Training complete, fetching model...")
    model_resp = stub.GetModel(pb.GetModelRequest(session_id=session_id))
    final_model = pb_to_model(model_resp.model)
    print(f"Model has {len(final_model.trees)} trees")

    # Evaluate
    test_features = df_test[FEATURES].values.astype(float)
    test_targets = df_test["target"].values
    test_preds = final_model.predict(test_features)
    test_classes = (test_preds >= 0.5).astype(int)
    accuracy = np.mean(test_classes == test_targets)

    print(f"\nTest accuracy: {accuracy:.2%}")

    # Cleanup
    for client in clients:
        client.close()
    channel.close()


if __name__ == "__main__":
    main()
