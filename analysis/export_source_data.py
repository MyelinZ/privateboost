"""Export the numerical values underlying each data figure as Supplementary
Data workbooks (one .xlsx per figure, one sheet per panel or data series).

Reuses the loaders and computations of generate_figures.py and
plot_binning_analysis.py so the exported values are exactly what the figures
plot. Figure-to-file mapping (must match the Data Availability statement in
manuscript/main.tex):

  Fig 4  learning_curves      -> Supplementary_Data_1.xlsx
  Fig 5  binning_analysis     -> Supplementary_Data_2.xlsx
  Fig 6  dropout_resilience   -> Supplementary_Data_3.xlsx
  Fig 7  depth_tradeoff       -> Supplementary_Data_4.xlsx
  Fig 8  fleet_timeline       -> Supplementary_Data_5.xlsx
  Fig 9  fleet_auc            -> Supplementary_Data_6.xlsx
  Fig 10 fleet_participation  -> Supplementary_Data_7.xlsx
"""
import os

import numpy as np
import pandas as pd

import generate_figures as gf
import plot_binning_analysis as pba
from paths import SUPPLEMENTARY_DIR as OUT_DIR


def export_learning_curves(path):
    with pd.ExcelWriter(path) as xl:
        for ds in gf.DATASETS:
            frames = []
            pb = os.path.join(gf.RESULTS_DIR, ds, "learning_curve.csv")
            xgb = os.path.join(gf.RESULTS_DIR, f"{ds}_xgboost", "learning_curve.csv")
            if os.path.exists(pb):
                df = pd.read_csv(pb)[["split_id", "n_trees", "auc_roc"]]
                df.insert(0, "method", "privateboost")
                frames.append(df)
            if os.path.exists(xgb):
                df = pd.read_csv(xgb)[["method", "split_id", "n_trees", "auc_roc"]]
                frames.append(df)
            pd.concat(frames).to_excel(xl, sheet_name=ds, index=False)


def export_binning(path):
    df = pd.read_csv(pba.DATA_PATH)
    methods = [
        ("equal_width", pba.compute_uniform_bins),
        ("gaussian", pba.compute_gaussian_bins),
    ]
    with pd.ExcelWriter(path) as xl:
        for feat, _label in pba.FEATURES:
            vals = df[feat].values
            pd.DataFrame({feat: vals}).to_excel(
                xl, sheet_name=f"{feat}_values", index=False)
            rows = []
            for name, fn in methods:
                edges, _ = fn(vals, pba.N_BINS)
                counts, _ = np.histogram(vals, bins=edges)
                pcts = counts / counts.sum() * 100
                for i, (lo, hi, c, p) in enumerate(
                        zip(edges[:-1], edges[1:], counts, pcts)):
                    rows.append({"method": name, "bin_index": i,
                                 "lower_edge": lo, "upper_edge": hi,
                                 "count": int(c), "percent": p})
            pd.DataFrame(rows).to_excel(
                xl, sheet_name=f"{feat}_occupancy", index=False)


def export_dropout(path):
    frames = []
    for ds in gf.DATASETS:
        p = os.path.join(gf.RESULTS_DIR, ds, "dropout.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)[["dropout_rate", "split_id", "auc_roc"]]
        df.insert(0, "dataset", ds)
        frames.append(df)
    with pd.ExcelWriter(path) as xl:
        pd.concat(frames).to_excel(xl, sheet_name="dropout", index=False)


def export_depth_tradeoff(path):
    quality_rows, cost_rows = [], []
    for ds in gf.DATASETS:
        depths, _means, _stds, auc_values = gf._load_depth_data(ds)
        for d, vals in zip(depths, auc_values):
            for split_id, auc in enumerate(vals):
                quality_rows.append({"dataset": ds, "max_depth": d,
                                     "split_id": split_id, "auc_roc": auc})
        m_depths, m_base_mb, m_hiding_mb = gf._load_measured_wire(ds)
        for d, base, hiding in zip(m_depths, m_base_mb, m_hiding_mb):
            cost_rows.append({"dataset": ds, "max_depth": d,
                              "configuration": "base", "per_client_mb": base})
            cost_rows.append({"dataset": ds, "max_depth": d,
                              "configuration": "path_hiding",
                              "per_client_mb": hiding})
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame(quality_rows).to_excel(
            xl, sheet_name="model_quality", index=False)
        pd.DataFrame(cost_rows).to_excel(
            xl, sheet_name="communication_cost", index=False)


def _fleet_submitted():
    df = pd.read_csv(os.path.join(gf.RESULTS_DIR, "fleet_round_metrics.csv"))
    return df[df["roundId"].notna()].copy()


def export_fleet_timeline(path):
    submitted = _fleet_submitted()
    cols = ["ts", "deviceModel", "datasetId", "triggerSource"]
    with pd.ExcelWriter(path) as xl:
        submitted[cols].sort_values("ts").to_excel(
            xl, sheet_name="submitted_rounds", index=False)


def export_fleet_auc(path):
    tree_df = pd.read_csv(os.path.join(gf.RESULTS_DIR, "fleet_tree_metrics.csv"))
    submitted = _fleet_submitted()
    submitted["totalBytes"] = submitted["txBytes"] + submitted["rxBytes"]
    per_device = (
        submitted.groupby(["deviceModel", "datasetId"])["totalBytes"].sum() / 1e6
    ).reset_index().rename(columns={"totalBytes": "total_mb"})
    with pd.ExcelWriter(path) as xl:
        tree_df.sort_values(["datasetId", "treeIdx"])[
            ["datasetId", "treeIdx", "auc"]
        ].to_excel(xl, sheet_name="per_tree_auc", index=False)
        per_device.to_excel(xl, sheet_name="per_client_communication", index=False)


def export_fleet_participation(path):
    submitted = _fleet_submitted()
    counts = (
        submitted.groupby(["deviceModel", "triggerSource"]).size()
        .reset_index(name="n_submitted_rounds")
    )
    gaps = gf._device_inter_round_gaps_min(submitted)
    gap_rows = [{"deviceModel": dev, "gap_minutes": g}
                for dev, vals in gaps.items() for g in vals]
    with pd.ExcelWriter(path) as xl:
        counts.to_excel(xl, sheet_name="submissions_by_trigger", index=False)
        pd.DataFrame(gap_rows).to_excel(
            xl, sheet_name="inter_round_gaps", index=False)


EXPORTS = [
    ("Supplementary_Data_1.xlsx", export_learning_curves),
    ("Supplementary_Data_2.xlsx", export_binning),
    ("Supplementary_Data_3.xlsx", export_dropout),
    ("Supplementary_Data_4.xlsx", export_depth_tradeoff),
    ("Supplementary_Data_5.xlsx", export_fleet_timeline),
    ("Supplementary_Data_6.xlsx", export_fleet_auc),
    ("Supplementary_Data_7.xlsx", export_fleet_participation),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, fn in EXPORTS:
        path = os.path.join(OUT_DIR, name)
        fn(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
