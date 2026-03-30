"""Integration tests: Federated XGBoost with Shamir secret sharing."""

from typing import List

import numpy as np
import pandas as pd
import pytest

from privateboost import (
    Aggregator,
    BinConfiguration,
    Client,
    ShareHolder,
)

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
    """Load UCI Heart Disease dataset with caching."""
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


def create_clients(
    df: pd.DataFrame,
    features: List[str],
    shareholders: List[ShareHolder],
    threshold: int = 2,
) -> List[Client]:
    """Create Client instances from DataFrame rows."""
    return [
        Client(
            client_id=f"client_{idx}",
            features=row[features].values,
            target=float(row["target"]),
            shareholders=shareholders,
            threshold=threshold,
        )
        for idx, row in df.iterrows()
    ]


def compute_initial_statistics(
    clients: List[Client],
    aggregator: Aggregator,
) -> tuple[List[BinConfiguration], float]:
    """Compute bins and initial prediction from client statistics."""
    for client in clients:
        client.submit_stats()

    bins = aggregator.define_bins()
    initial_pred = float(aggregator.means[-1])

    return bins, initial_pred


def train_tree(
    clients: List[Client],
    aggregator: Aggregator,
    bins: List[BinConfiguration],
    max_depth: int,
    round_id: int,
) -> None:
    """Train a single tree."""
    for depth in range(max_depth):
        for client in clients:
            client.submit_gradients(
                bins,
                aggregator.model,
                aggregator.splits,
                round_id=round_id,
                depth=depth,
                loss="logistic",
            )

        if not aggregator.compute_splits(depth=depth, min_child_weight=1.0):
            break

    aggregator.finish_round()


def test_xgboost_heart_disease_shamir(heart_disease_df: pd.DataFrame):
    """End-to-end test: federated XGBoost with Shamir 2-of-3 achieves >75% accuracy."""
    np.random.seed(42)

    df = heart_disease_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    threshold = 2
    min_clients = 10
    learning_rate = 0.15
    lambda_reg = 2.0

    shareholders = [
        ShareHolder(party_id=i, x_coord=i + 1, min_clients=min_clients) for i in range(3)
    ]
    aggregator = Aggregator(
        shareholders,
        n_bins=10,
        threshold=threshold,
        min_clients=min_clients,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
    )

    train_clients = create_clients(df_train, FEATURES, shareholders, threshold=threshold)

    bins, initial_pred = compute_initial_statistics(train_clients, aggregator)

    print(f"Number of training clients: {aggregator.n_clients}")
    print(f"Initial prediction (target mean): {initial_pred:.4f}")

    n_trees = 15
    max_depth = 3

    for round_id in range(n_trees):
        train_tree(
            train_clients,
            aggregator,
            bins,
            max_depth,
            round_id,
        )

    test_features_arr = df_test[FEATURES].values.astype(float)
    test_targets = df_test["target"].values

    test_preds = aggregator.model.predict(test_features_arr)
    test_classes = (test_preds >= 0.5).astype(int)
    test_accuracy = np.mean(test_classes == test_targets)

    print(f"Test accuracy: {test_accuracy:.2%}")
    assert test_accuracy > 0.75, f"Expected >75% accuracy, got {test_accuracy:.2%}"


def test_xgboost_heart_disease_path_hiding(heart_disease_df: pd.DataFrame):
    """End-to-end test: path hiding produces same accuracy as without."""
    np.random.seed(42)

    df = heart_disease_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    threshold = 2
    min_clients = 10
    learning_rate = 0.15
    lambda_reg = 2.0

    shareholders = [
        ShareHolder(party_id=i, x_coord=i + 1, min_clients=min_clients) for i in range(3)
    ]
    aggregator = Aggregator(
        shareholders,
        n_bins=10,
        threshold=threshold,
        min_clients=min_clients,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
    )

    train_clients = create_clients(df_train, FEATURES, shareholders, threshold=threshold)

    bins, _ = compute_initial_statistics(train_clients, aggregator)

    n_trees = 15
    max_depth = 3

    for round_id in range(n_trees):
        for depth in range(max_depth):
            for client in train_clients:
                client.submit_gradients(
                    bins,
                    aggregator.model,
                    aggregator.splits,
                    round_id=round_id,
                    depth=depth,
                    loss="logistic",
                    hide_path=True,
                )

            if not aggregator.compute_splits(depth=depth, min_child_weight=1.0):
                break

        aggregator.finish_round()

    test_features_arr = df_test[FEATURES].values.astype(float)
    test_targets = df_test["target"].values

    test_preds = aggregator.model.predict(test_features_arr)
    test_classes = (test_preds >= 0.5).astype(int)
    test_accuracy = np.mean(test_classes == test_targets)

    print(f"Test accuracy (path hiding): {test_accuracy:.2%}")
    assert test_accuracy > 0.75, f"Expected >75% accuracy with path hiding, got {test_accuracy:.2%}"


def test_min_clients_enforcement():
    """Test that shareholders enforce minimum client threshold."""
    min_clients = 10
    shareholders = [
        ShareHolder(party_id=i, x_coord=i + 1, min_clients=min_clients) for i in range(3)
    ]

    clients = [Client(f"c{i}", [1.0, 2.0], 0.0, shareholders, threshold=2) for i in range(5)]

    for client in clients:
        client.submit_stats()

    commitments = list(shareholders[0].get_stats_commitments())

    with pytest.raises(ValueError, match="minimum"):
        shareholders[0].get_stats_sum(commitments=commitments)
