"""Reproduce the manuscript's gain-retention and bin-uniformity tables.

Gain retention (Table tab:gain_retention): average node-level retention
within each training run and feature-type group, then mean and sample SD
across five matched splits, separately by dataset. Paired differences are
Gaussian minus equal-width run means. These are descriptive summaries,
not independent-node hypothesis tests or confidence intervals.

Bin uniformity (Table tab:bin_uniformity): coefficient of variation of bin
occupancy per feature under both binning methods, same pooling and
stratification, with a two-sided Wilcoxon signed-rank test paired by
feature (zero differences dropped, so discrete features contribute
nothing and the pooled test equals the continuous one).

A feature counts as continuous with more than 20 unique values, discrete
otherwise.

Inputs: data/<dataset>.csv and
results/gain_retention/<dataset>_<method>/gain_retention.csv, produced by
scripts/run_experiments.sh.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from plot_binning_analysis import compute_gaussian_bins, compute_uniform_bins

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "gain_retention")

DATASETS = ["heart_disease", "breast_cancer", "pima_diabetes", "cdc_diabetes"]
METHODS = ["gaussian", "uniform"]
DISCRETE_MAX_UNIQUE = 20
N_BINS = 10


def load_features():
    """One row per (dataset, feature): values and continuity classification."""
    rows = []
    for ds in DATASETS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ds}.csv"))
        features = [c for c in df.columns if c != "target"]
        for idx, name in enumerate(features):
            values = df[name].to_numpy(dtype=float)
            rows.append(
                {
                    "dataset": ds,
                    "feature_idx": idx,
                    "feature": name,
                    "continuous": np.unique(values).size > DISCRETE_MAX_UNIQUE,
                    "values": values,
                }
            )
    return rows


def gain_retention_table(features):
    """Summarise existing runs; nodes and overlapping splits are not replicates
    for an independence-based hypothesis test. Features classify selected
    splits, not independent observations. Exact gain is for the chosen feature.
    """
    key = {(f["dataset"], f["feature_idx"]): f["continuous"] for f in features}
    runs = []
    for ds in DATASETS:
        for method in METHODS:
            path = os.path.join(RESULTS_DIR, f"{ds}_{method}", "gain_retention.csv")
            df = pd.read_csv(path)
            assert set(df["split_id"]) == set(range(5)), path
            assert not df.duplicated(["split_id", "tree", "node_id"]).any(), path
            df["continuous"] = [key[(ds, i)] for i in df["feature_idx"]]
            for group, mask in [("Continuous", df["continuous"]),
                                ("Discrete", ~df["continuous"]),
                                ("All", pd.Series(True, index=df.index))]:
                for split_id, part in df[mask].groupby("split_id"):
                    runs.append(dict(dataset=ds, method=method, group=group,
                                     split_id=split_id, nodes=len(part),
                                     retention=part["retention"].mean()))
    runs = pd.DataFrame(runs)
    runs.to_csv(os.path.join(RESULTS_DIR, "run_summary.csv"), index=False,
                float_format="%.10f")
    rows = []
    for (ds, group), part in runs.groupby(["dataset", "group"], sort=False):
        paired = part.pivot(index="split_id", columns="method", values="retention")
        assert len(paired) == 5 and not paired.isna().any().any(), (ds, group)
        diff = paired["gaussian"] - paired["uniform"]
        rows.append(dict(dataset=ds, group=group, n_runs=len(paired),
                         gaussian_mean=paired["gaussian"].mean(),
                         gaussian_sd=paired["gaussian"].std(ddof=1),
                         uniform_mean=paired["uniform"].mean(),
                         uniform_sd=paired["uniform"].std(ddof=1),
                         difference_mean=diff.mean(), difference_sd=diff.std(ddof=1)))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "descriptive_summary.csv"), index=False,
                   float_format="%.10f")
    print("Gain retention: mean ± sample SD across matched runs (no hypothesis test)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()


def occupancy_cv(values, method):
    fn = compute_gaussian_bins if method == "gaussian" else compute_uniform_bins
    edges, _ = fn(values, N_BINS)
    counts, _ = np.histogram(values, bins=edges)
    pcts = 100.0 * counts / counts.sum()
    return np.std(pcts) / np.mean(pcts)


def bin_uniformity_table(features):
    cv = {
        m: np.array([occupancy_cv(f["values"], m) for f in features]) for m in METHODS
    }
    cont = np.array([f["continuous"] for f in features])

    print("Bin occupancy uniformity (CV of bin counts, lower is better)")
    print(f"{'Features':<12} {'Gaussian':>9} {'Equal-width':>12} {'p':>10}")
    for label, mask in [("Cont.", cont), ("Disc.", ~cont), ("All", np.ones_like(cont))]:
        g, u = cv["gaussian"][mask.astype(bool)], cv["uniform"][mask.astype(bool)]
        # CV depends only on the multiset of counts, so a discrete feature
        # whose values land in different bin positions under the two methods
        # still ties exactly; drop those pairs (as Wilcoxon's zero-dropping
        # would) rather than let float summation noise count them as ties
        # broken one way or the other.
        nonzero = np.abs(g - u) > 1e-12
        if not nonzero.any():
            p_str = "---"
        else:
            p_str = f"{stats.wilcoxon(g[nonzero], u[nonzero]).pvalue:.1e}"
        print(f"{label} ({mask.sum()}){'':<3} {g.mean():>9.2f} {u.mean():>12.2f} {p_str:>10}")
    print()


def main():
    features = load_features()
    gain_retention_table(features)
    bin_uniformity_table(features)


if __name__ == "__main__":
    main()
