#!/usr/bin/env python3
"""Generate figures and results for the privateboost paper."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.metrics import accuracy_score
from pathlib import Path

from privateboost import Client, ShareHolder, Aggregator

# Hyperparameters
N_TREES = 15
MAX_DEPTH = 3
LEARNING_RATE = 0.15
LAMBDA_REG = 2.0
N_BINS = 10
N_SHAREHOLDERS = 3
THRESHOLD = 2
MIN_CLIENTS = 10

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_datasets():
    """Load all three datasets."""
    np.random.seed(SEED)
    datasets = {}

    # Heart Disease
    print("Loading Heart Disease...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(url, names=columns, na_values="?").dropna()
    df["target"] = (df["target"] > 0).astype(int)
    features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    datasets["Heart Disease"] = {
        "X_train": df.iloc[:split_idx][features].values.astype(float),
        "y_train": df.iloc[:split_idx]["target"].values,
        "X_test": df.iloc[split_idx:][features].values.astype(float),
        "y_test": df.iloc[split_idx:]["target"].values,
        "n_train": split_idx,
        "n_total": len(df),
    }

    # Breast Cancer
    print("Loading Breast Cancer...")
    data = load_breast_cancer()
    indices = np.random.permutation(len(data.data))
    split = int(len(data.data) * 0.8)
    datasets["Breast Cancer"] = {
        "X_train": data.data[indices[:split]],
        "y_train": data.target[indices[:split]],
        "X_test": data.data[indices[split:]],
        "y_test": data.target[indices[split:]],
        "n_train": split,
        "n_total": len(data.data),
    }

    # Pima Diabetes
    print("Loading Pima Diabetes...")
    diabetes = fetch_openml(name='diabetes', version=1, as_frame=False)
    X, y = diabetes.data, (diabetes.target == 'tested_positive').astype(int)
    indices = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    datasets["Pima Diabetes"] = {
        "X_train": X[indices[:split]],
        "y_train": y[indices[:split]],
        "X_test": X[indices[split:]],
        "y_test": y[indices[split:]],
        "n_train": split,
        "n_total": len(X),
    }

    return datasets


def train_privateboost(X_train, y_train, X_test, y_test, return_history=True):
    """Train privateboost and return test accuracy (and optionally history)."""
    shs = [ShareHolder(party_id=i, x_coord=i + 1, min_clients=MIN_CLIENTS) for i in range(N_SHAREHOLDERS)]
    agg = Aggregator(shs, n_bins=N_BINS, threshold=THRESHOLD, min_clients=MIN_CLIENTS,
                     learning_rate=LEARNING_RATE, lambda_reg=LAMBDA_REG)
    clients = [Client(f"c_{i}", X_train[i], float(y_train[i]), shs, threshold=THRESHOLD)
               for i in range(len(X_train))]

    for c in clients:
        c.submit_stats()
    bins = agg.define_bins()

    history = []
    for round_id in range(N_TREES):
        for depth in range(MAX_DEPTH):
            for c in clients:
                c.submit_gradients(bins, agg.model, agg.splits, round_id=round_id, depth=depth, loss="squared")
            if not agg.compute_splits(depth=depth, min_samples=5):
                break
        agg.finish_round()
        te_p = agg.model.predict(X_test)
        history.append(accuracy_score(y_test, (te_p >= 0.5).astype(int)))

    if return_history:
        return history
    return history[-1]


def train_xgboost(X_train, y_train, X_test, y_test, matched=True, return_history=True):
    """Train XGBoost and return test accuracy (and optionally history)."""
    history = []
    for n in range(1, N_TREES + 1):
        if matched:
            model = xgb.XGBClassifier(
                n_estimators=n, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                reg_lambda=LAMBDA_REG, tree_method='hist', max_bin=N_BINS, random_state=SEED
            )
        else:
            model = xgb.XGBClassifier(n_estimators=n, random_state=SEED)
        model.fit(X_train, y_train, verbose=False)
        history.append(accuracy_score(y_test, model.predict(X_test)))

    if return_history:
        return history
    return history[-1]


def train_with_dropout(X_train, y_train, X_test, y_test, dropout_rate, seed=42):
    """Train privateboost with client dropout. Returns final test accuracy or None if failed."""
    rng = np.random.default_rng(seed)

    shs = [ShareHolder(party_id=i, x_coord=i + 1, min_clients=MIN_CLIENTS) for i in range(N_SHAREHOLDERS)]
    agg = Aggregator(shs, n_bins=N_BINS, threshold=THRESHOLD, min_clients=MIN_CLIENTS,
                     learning_rate=LEARNING_RATE, lambda_reg=LAMBDA_REG)
    clients = [Client(f"c_{i}", X_train[i], float(y_train[i]), shs, threshold=THRESHOLD)
               for i in range(len(X_train))]

    for c in clients:
        c.submit_stats()

    try:
        bins = agg.define_bins()
    except ValueError:
        return None

    try:
        for round_id in range(N_TREES):
            # Per-round dropout
            round_mask = rng.random(len(clients)) >= dropout_rate
            round_clients = [c for c, active in zip(clients, round_mask) if active]

            for depth in range(MAX_DEPTH):
                for c in round_clients:
                    c.submit_gradients(bins, agg.model, agg.splits, round_id=round_id, depth=depth, loss="squared")
                if not agg.compute_splits(depth=depth, min_samples=5):
                    break
            agg.finish_round()
    except ValueError:
        return None

    test_preds = agg.model.predict(X_test)
    return accuracy_score(y_test, (test_preds >= 0.5).astype(int))


def run_learning_curves(datasets):
    """Run learning curve experiments and return results."""
    results = {}
    for name, data in datasets.items():
        print(f"  Training on {name}...")
        results[name] = {
            "privateboost": train_privateboost(data["X_train"], data["y_train"], data["X_test"], data["y_test"]),
            "xgboost_matched": train_xgboost(data["X_train"], data["y_train"], data["X_test"], data["y_test"], matched=True),
            "xgboost_default": train_xgboost(data["X_train"], data["y_train"], data["X_test"], data["y_test"], matched=False),
        }
    return results


def run_dropout_experiments(datasets, dropout_rates, n_trials=5):
    """Run dropout experiments and return results."""
    results = {}
    for name, data in datasets.items():
        print(f"  Dropout experiments on {name}...")
        results[name] = {rate: [] for rate in dropout_rates}
        for rate in dropout_rates:
            for trial in range(n_trials):
                acc = train_with_dropout(
                    data["X_train"], data["y_train"], data["X_test"], data["y_test"],
                    rate, seed=SEED + trial
                )
                results[name][rate].append(acc)
    return results


def plot_learning_curves(results, datasets):
    """Generate learning curves figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    trees = range(1, N_TREES + 1)

    for ax, name in zip(axes, results.keys()):
        ax.plot(trees, results[name]["privateboost"], 'o-', color='#2ecc71', label='privateboost', markersize=5)
        ax.plot(trees, results[name]["xgboost_matched"], 's--', color='#3498db', label='XGBoost (matched)', markersize=5)
        ax.plot(trees, results[name]["xgboost_default"], '^:', color='#e74c3c', label='XGBoost (default)', markersize=5)
        ax.set_xlabel('Number of Trees')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(name)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150, bbox_inches='tight')
    print(f"Saved learning_curves.png")
    plt.close()


def plot_dropout_resilience(results, datasets, dropout_rates, n_trials):
    """Generate dropout resilience figure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.array(dropout_rates) * 100

    colors = {'Heart Disease': '#2ecc71', 'Breast Cancer': '#3498db', 'Pima Diabetes': '#e74c3c'}
    markers = {'Heart Disease': 'o', 'Breast Cancer': 's', 'Pima Diabetes': '^'}

    for name in results.keys():
        means, stds = [], []
        for rate in dropout_rates:
            successes = [a for a in results[name][rate] if a is not None]
            if successes:
                means.append(np.mean(successes))
                stds.append(np.std(successes))
            else:
                means.append(np.nan)
                stds.append(0)

        means, stds = np.array(means), np.array(stds)
        ax.errorbar(x, means * 100, yerr=stds * 100, fmt=f'{markers[name]}-', color=colors[name],
                    capsize=4, markersize=6, label=name)

    ax.set_xlabel('Dropout Rate (%)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-5, 85)
    ax.set_ylim(50, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dropout_resilience.png", dpi=150, bbox_inches='tight')
    print(f"Saved dropout_resilience.png")
    plt.close()


def export_results(learning_results, dropout_results, dropout_rates):
    """Export results to CSV."""
    # Learning curves - final accuracies
    rows = []
    for name, res in learning_results.items():
        rows.append({
            "dataset": name,
            "privateboost": res["privateboost"][-1],
            "xgboost_matched": res["xgboost_matched"][-1],
            "xgboost_default": res["xgboost_default"][-1],
        })
    df_acc = pd.DataFrame(rows)
    df_acc.to_csv(OUTPUT_DIR / "accuracy_results.csv", index=False)
    print(f"Saved accuracy_results.csv")

    # Dropout results
    rows = []
    for name, res in dropout_results.items():
        for rate in dropout_rates:
            successes = [a for a in res[rate] if a is not None]
            rows.append({
                "dataset": name,
                "dropout_rate": rate,
                "mean_accuracy": np.mean(successes) if successes else None,
                "std_accuracy": np.std(successes) if successes else None,
                "success_rate": len(successes) / len(res[rate]),
            })
    df_dropout = pd.DataFrame(rows)
    df_dropout.to_csv(OUTPUT_DIR / "dropout_results.csv", index=False)
    print(f"Saved dropout_results.csv")

    return df_acc, df_dropout


def main():
    print("=" * 60)
    print("Generating figures for privateboost paper")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading datasets...")
    datasets = load_datasets()

    # Learning curves
    print("\n[2/4] Running learning curve experiments...")
    learning_results = run_learning_curves(datasets)
    plot_learning_curves(learning_results, datasets)

    # Dropout
    print("\n[3/4] Running dropout experiments...")
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_trials = 5
    dropout_results = run_dropout_experiments(datasets, dropout_rates, n_trials)
    plot_dropout_resilience(dropout_results, datasets, dropout_rates, n_trials)

    # Export
    print("\n[4/4] Exporting results...")
    df_acc, df_dropout = export_results(learning_results, dropout_results, dropout_rates)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\nFinal Test Accuracies:")
    print(df_acc.to_string(index=False))

    print("\nDropout Resilience (0% vs 30%):")
    for name in datasets:
        acc_0 = df_dropout[(df_dropout["dataset"] == name) & (df_dropout["dropout_rate"] == 0.0)]["mean_accuracy"].values[0]
        acc_30 = df_dropout[(df_dropout["dataset"] == name) & (df_dropout["dropout_rate"] == 0.3)]["mean_accuracy"].values[0]
        if acc_0 and acc_30:
            print(f"  {name}: {acc_0:.1%} -> {acc_30:.1%} (delta: {(acc_30-acc_0)*100:+.1f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()
