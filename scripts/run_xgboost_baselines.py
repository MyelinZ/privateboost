"""Run centralized XGBoost baselines on all datasets using same splits as Rust benchmark."""
import csv
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DATASETS = {
    "heart_disease": ("heart_disease.csv", "target"),
    "breast_cancer": ("breast_cancer.csv", "target"),
    "pima_diabetes": ("pima_diabetes.csv", "target"),
    "cdc_diabetes": ("cdc_diabetes.csv", "target"),
}


def load_csv(path, target_col):
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        target_idx = headers.index(target_col)
        data = [[float(x) for x in row] for row in reader]
    data = np.array(data)
    X = np.delete(data, target_idx, axis=1)
    y = data[:, target_idx]
    return X, y


def load_split_indices(splits_dir, split_id):
    """Load train/test indices exported by the Rust benchmark."""
    path = os.path.join(splits_dir, f"split_{split_id}.csv")
    train_idx, test_idx = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            if row["set"] == "train":
                train_idx.append(idx)
            else:
                test_idx.append(idx)
    return np.array(train_idx), np.array(test_idx)


def run_baseline(name, csv_file, target_col, n_splits=5):
    print(f"\n=== {name} ===")
    X, y = load_csv(os.path.join(DATA_DIR, csv_file), target_col)

    # Look for split indices from the Rust benchmark
    # Check common result directory patterns
    splits_dir = None
    for depth in [3]:
        candidate = os.path.join(RESULTS_DIR, f"{name}_depth_{depth}", "splits")
        if os.path.isdir(candidate):
            splits_dir = candidate
            break

    if splits_dir is None:
        print(f"  ERROR: No split indices found for {name}. Run Rust benchmark first.")
        return

    print(f"  Using splits from: {splits_dir}")

    out_dir = os.path.join(RESULTS_DIR, f"{name}_xgboost")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.csv")
    curve_path = os.path.join(out_dir, "learning_curve.csv")

    with open(metrics_path, "w", newline="") as mf, open(curve_path, "w", newline="") as cf:
        mw = csv.writer(mf)
        mw.writerow(["split_id", "method", "accuracy", "auc_roc", "f1", "n_train", "n_test", "n_features"])

        cw = csv.writer(cf)
        cw.writerow(["split_id", "method", "n_trees", "accuracy", "auc_roc", "f1"])

        for split_id in range(n_splits):
            train_idx, test_idx = load_split_indices(splits_dir, split_id)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            n_features = X_train.shape[1]

            for method, params in [
                ("xgb_matched", dict(
                    n_estimators=15, max_depth=3, learning_rate=0.15,
                    reg_lambda=2.0, max_bin=10, tree_method="hist",
                )),
                ("xgb_default", dict(n_estimators=15)),
            ]:
                n_trees = params.get("n_estimators", 100)
                clf = XGBClassifier(
                    random_state=split_id, eval_metric="logloss", verbosity=0, **params
                )
                clf.fit(X_train, y_train)

                probs = clf.predict_proba(X_test)[:, 1]
                preds = (probs >= 0.5).astype(int)
                acc = accuracy_score(y_test, preds)
                auc = roc_auc_score(y_test, probs)
                f1 = f1_score(y_test, preds)

                mw.writerow([split_id, method, f"{acc:.6f}", f"{auc:.6f}", f"{f1:.6f}",
                             len(train_idx), len(test_idx), n_features])

                print(f"  Split {split_id} {method}: acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}")

                # Learning curve: train incrementally
                for n_t in range(1, n_trees + 1):
                    clf_partial = XGBClassifier(
                        random_state=split_id, eval_metric="logloss", verbosity=0,
                        n_estimators=n_t,
                        **(
                            {k: v for k, v in params.items() if k != "n_estimators"}
                            if "n_estimators" in params
                            else params
                        ),
                    )
                    clf_partial.fit(X_train, y_train)
                    p = clf_partial.predict_proba(X_test)[:, 1]
                    a = accuracy_score(y_test, (p >= 0.5).astype(int))
                    au = roc_auc_score(y_test, p)
                    f = f1_score(y_test, (p >= 0.5).astype(int))
                    cw.writerow([split_id, method, n_t, f"{a:.6f}", f"{au:.6f}", f"{f:.6f}"])

    print(f"  Results written to {out_dir}")


def main():
    for name, (csv_file, target_col) in DATASETS.items():
        run_baseline(name, csv_file, target_col)


if __name__ == "__main__":
    main()
