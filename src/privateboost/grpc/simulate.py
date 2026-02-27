"""Simulation script: runs the full federated XGBoost protocol over gRPC.

New multi-aggregator protocol:
  1. Coordinator creates a run
  2. Clients submit stats/gradients to shareholders
  3. Aggregators independently compute results and submit to shareholders
  4. Shareholders vote on consensus
  5. Clients poll shareholders for consensus results
"""

import os
import time

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
    coordinator_addr = os.environ.get("COORDINATOR", "localhost:50053")
    shareholder_addrs = os.environ.get(
        "SHAREHOLDERS", "localhost:50051,localhost:50052,localhost:50053"
    ).split(",")
    n_trees = int(os.environ.get("N_TREES", "3"))
    max_depth = int(os.environ.get("MAX_DEPTH", "3"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.15"))
    lambda_reg = float(os.environ.get("LAMBDA_REG", "2.0"))
    n_bins = int(os.environ.get("N_BINS", "10"))
    min_clients = int(os.environ.get("MIN_CLIENTS", "0"))
    loss = os.environ.get("LOSS", "logistic")
    threshold = int(os.environ.get("THRESHOLD", "2"))

    print(f"Connecting to coordinator at {coordinator_addr}")
    print(f"Shareholders: {shareholder_addrs}")

    # Create run via coordinator
    coord_channel = grpc.insecure_channel(coordinator_addr)
    coord_stub = pb_grpc.CoordinatorServiceStub(coord_channel)

    # Load dataset first to know how many clients we have
    print("Loading Heart Disease dataset...")
    df = _load_heart_disease()
    np.random.seed(42)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")

    target_count = min_clients if min_clients > 0 else len(df_train)

    config = pb.TrainingConfig(
        features=[pb.FeatureSpec(index=i, name=f) for i, f in enumerate(FEATURES)],
        target_column="target",
        loss=loss,
        n_bins=n_bins,
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        min_clients=target_count,
        target_count=target_count,
    )

    resp = coord_stub.CreateRun(pb.CreateRunRequest(config=config))
    run_id = resp.run_id
    print(f"Created run: {run_id}")
    print(
        f"Config: {n_trees} trees, depth {max_depth}, lr={learning_rate}, loss={loss}"
    )

    # Connect to a shareholder for polling
    sh_channel = grpc.insecure_channel(shareholder_addrs[0])
    sh_stub = pb_grpc.ShareholderServiceStub(sh_channel)

    # Create clients
    clients = []
    for idx, row in df_train.iterrows():
        clients.append(
            NetworkClient(
                client_id=f"client_{idx}",
                features=row[FEATURES].values.astype(float),
                target=float(row["target"]),
                run_id=run_id,
                shareholder_addresses=shareholder_addrs,
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

    # Wait for stats to freeze and bins consensus
    print("Waiting for bins consensus...")
    while True:
        state = sh_stub.GetRunState(pb.GetRunStateRequest(run_id=run_id))
        if state.phase == pb.COLLECTING_GRADIENTS:
            break
        if state.phase == pb.TRAINING_COMPLETE:
            print("Training completed early (during stats phase)")
            return
        time.sleep(0.5)

    # Fetch consensus bins
    bins_resp = sh_stub.GetConsensusBins(pb.GetConsensusBinsRequest(run_id=run_id))
    bins = [pb_to_bin_config(b) for b in bins_resp.bins]
    initial_prediction = bins_resp.initial_prediction
    print(
        f"Bins computed ({len(bins)} features), initial_prediction={initial_prediction:.4f}"
    )

    # Build initial model (no trees yet)
    from privateboost.tree import Model

    model = Model(
        initial_prediction=initial_prediction,
        learning_rate=learning_rate,
        trees=[],
    )

    # Training loop
    current_round = -1
    current_depth = -1
    splits: dict[int, object] = {}

    while True:
        state = sh_stub.GetRunState(pb.GetRunStateRequest(run_id=run_id))
        if state.phase == pb.TRAINING_COMPLETE:
            break

        # Waiting for gradients to be needed
        if state.phase in (pb.COLLECTING_GRADIENTS, pb.FROZEN_STATS):
            if state.round_id == current_round and state.depth == current_depth:
                time.sleep(0.1)
                continue

            current_round = state.round_id
            current_depth = state.depth

            # Fetch latest splits for navigation
            if current_depth > 0:
                splits_resp = sh_stub.GetConsensusSplits(
                    pb.GetConsensusSplitsRequest(
                        run_id=run_id,
                        round_id=current_round,
                        depth=current_depth - 1,
                    )
                )
                if splits_resp.ready:
                    for nid, sd in splits_resp.splits.items():
                        splits[nid] = pb_to_split_decision(sd)
            elif current_round > 0:
                # New round: fetch model with previous trees, reset splits
                model_resp = sh_stub.GetConsensusModel(
                    pb.GetConsensusModelRequest(run_id=run_id)
                )
                if model_resp.model and model_resp.model.trees:
                    model = pb_to_model(model_resp.model)
                splits = {}

            print(f"  Round {current_round}, depth {current_depth}")

            for client in clients:
                client.submit_gradients(
                    bins=bins,
                    model=model,
                    splits=splits,
                    round_id=current_round,
                    depth=current_depth,
                    loss=loss,
                )
        else:
            time.sleep(0.2)

    # Fetch final model
    print("Training complete, fetching model...")
    model_resp = sh_stub.GetConsensusModel(pb.GetConsensusModelRequest(run_id=run_id))
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
    sh_channel.close()
    coord_channel.close()


if __name__ == "__main__":
    main()
