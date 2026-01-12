"""Integration tests: Federated XGBoost with Shamir secret sharing."""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from privateboost import (
    Aggregator,
    BinConfiguration,
    Client,
    LeafNode,
    ShareHolder,
    SplitDecision,
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
            features=[float(row[name]) for name in features],
            target=float(row["target"]),
            shareholders=shareholders,
            threshold=threshold,
        )
        for idx, row in df.iterrows()
    ]


def compute_initial_statistics(
    clients: List[Client],
    aggregator: Aggregator,
    round_id: int = 0,
) -> tuple[List[BinConfiguration], float]:
    """Run statistics round to compute bins and initial prediction."""
    for client in clients:
        client.submit_stats(round_id=round_id)

    bins = aggregator.define_bins(round_id=round_id)
    initial_pred = float(aggregator.means[-1])

    return bins, initial_pred


def train_tree(
    clients: List[Client],
    aggregator: Aggregator,
    bins: List[BinConfiguration],
    max_depth: int,
    lambda_reg: float,
    learning_rate: float,
    base_round_id: int,
) -> tuple[Dict[int, SplitDecision], Dict[int, LeafNode]]:
    """Train a single tree and update client predictions."""
    aggregator.reset_tree()
    for client in clients:
        client.reset_node()

    all_splits: Dict[int, SplitDecision] = {}
    active_nodes = [0]

    for depth in range(max_depth):
        round_id = base_round_id + depth

        for client in clients:
            client.submit_gradients(
                bins, active_nodes, round_id=round_id, loss="squared"
            )

        splits = aggregator.find_best_splits(
            round_id=round_id,
            active_nodes=active_nodes,
            lambda_reg=lambda_reg,
            min_samples=5,
        )
        if not splits:
            break

        all_splits.update(splits)

        for client in clients:
            client.apply_splits(splits)

        active_nodes = []
        for split in splits.values():
            active_nodes.extend([split.left_child_id, split.right_child_id])

    split_node_ids = set(all_splits.keys())
    all_child_ids = set()
    for split in all_splits.values():
        all_child_ids.add(split.left_child_id)
        all_child_ids.add(split.right_child_id)

    leaf_node_ids = list(all_child_ids - split_node_ids)
    if 0 not in split_node_ids:
        leaf_node_ids.append(0)

    leaf_round_id = base_round_id + max_depth
    for client in clients:
        client.submit_gradients(
            bins, leaf_node_ids, round_id=leaf_round_id, loss="squared"
        )

    leaves = aggregator.compute_leaf_values(
        round_id=leaf_round_id,
        leaf_nodes=leaf_node_ids,
        lambda_reg=lambda_reg,
    )

    for client in clients:
        client.update_prediction(leaves, learning_rate)

    return all_splits, leaves


def predict_sample(
    features: List[float],
    trees: List[tuple[Dict[int, SplitDecision], Dict[int, LeafNode]]],
    initial_pred: float,
    learning_rate: float,
) -> float:
    """Predict a single sample using trained trees."""
    pred = initial_pred
    for splits, leaves in trees:
        node_id = 0
        while node_id in splits:
            split = splits[node_id]
            if features[split.feature_idx] <= split.threshold:
                node_id = split.left_child_id
            else:
                node_id = split.right_child_id
        if node_id in leaves:
            pred += learning_rate * leaves[node_id].value
    return pred


def test_xgboost_heart_disease_shamir(heart_disease_df: pd.DataFrame):
    """End-to-end test: federated XGBoost with Shamir 2-of-3 achieves >75% accuracy."""
    np.random.seed(42)

    df = heart_disease_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    threshold = 2
    min_clients = 10

    shareholders = [
        ShareHolder(party_id=i, x_coord=i + 1, min_clients=min_clients)
        for i in range(3)
    ]
    aggregator = Aggregator(shareholders, n_bins=10, threshold=threshold, min_clients=min_clients)

    train_clients = create_clients(
        df_train, FEATURES, shareholders, threshold=threshold
    )

    bins, initial_pred = compute_initial_statistics(
        train_clients, aggregator, round_id=0
    )

    print(f"Number of training clients: {aggregator.n_clients}")
    print(f"Initial prediction (target mean): {initial_pred:.4f}")

    for client in train_clients:
        client.prediction = initial_pred

    n_trees = 15
    max_depth = 3
    learning_rate = 0.15
    lambda_reg = 2.0

    all_trees = []
    for tree_idx in range(n_trees):
        base_round_id = 1000 + tree_idx * 100

        splits, leaves = train_tree(
            train_clients,
            aggregator,
            bins,
            max_depth,
            lambda_reg,
            learning_rate,
            base_round_id,
        )
        all_trees.append((splits, leaves))

        for sh in shareholders:
            for rid in range(base_round_id, base_round_id + max_depth + 2):
                sh.clear_round(rid)

    test_features_arr = df_test[FEATURES].values.astype(float)
    test_targets = df_test["target"].values

    test_preds = []
    for i in range(len(test_targets)):
        features = list(test_features_arr[i])
        pred = predict_sample(features, all_trees, initial_pred, learning_rate)
        test_preds.append(pred)

    test_preds = np.array(test_preds)
    test_classes = (test_preds >= 0.5).astype(int)
    test_accuracy = np.mean(test_classes == test_targets)

    print(f"Test accuracy: {test_accuracy:.2%}")
    assert test_accuracy > 0.75, f"Expected >75% accuracy, got {test_accuracy:.2%}"


def test_min_clients_enforcement():
    """Test that shareholders enforce minimum client threshold."""
    min_clients = 10
    shareholders = [
        ShareHolder(party_id=i, x_coord=i + 1, min_clients=min_clients)
        for i in range(3)
    ]

    clients = [
        Client(f"c{i}", [1.0, 2.0], 0.0, shareholders, threshold=2) for i in range(5)
    ]

    for client in clients:
        client.submit_stats(round_id=0)

    commitments = list(shareholders[0].get_commitments(round_id=0, round_type="stats"))

    with pytest.raises(ValueError, match="minimum"):
        shareholders[0].get_stats_sum(round_id=0, commitments=commitments)
