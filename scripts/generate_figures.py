"""Generate all paper figures from benchmark CSV results."""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import style
style.apply()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")

DATASETS = ["heart_disease", "breast_cancer", "pima_diabetes", "cdc_diabetes"]
DATASET_LABELS = {
    "heart_disease": "Heart Disease (297)",
    "breast_cancer": "Breast Cancer (569)",
    "pima_diabetes": "Pima Diabetes (768)",
    "cdc_diabetes": "CDC BRFSS (70,692)",
}

COLORS = style.COLORS

DATASET_COLORS = {
    "heart_disease": "#0173B2",
    "breast_cancer": "#D55E00",
    "pima_diabetes": "#E69F00",
    "cdc_diabetes": "#009E73",
}
DATASET_MARKERS = {
    "heart_disease": "o",
    "breast_cancer": "s",
    "pima_diabetes": "^",
    "cdc_diabetes": "D",
}
DATASET_SHORT = {
    "heart_disease": "Heart Disease",
    "breast_cancer": "Breast Cancer",
    "pima_diabetes": "Pima Diabetes",
    "cdc_diabetes": "CDC BRFSS",
}


def load_learning_curves():
    curves = {}
    for ds in DATASETS:
        pb_path = os.path.join(RESULTS_DIR, ds, "learning_curve.csv")
        xgb_path = os.path.join(RESULTS_DIR, f"{ds}_xgboost", "learning_curve.csv")
        if os.path.exists(pb_path):
            curves[f"{ds}_pb"] = pd.read_csv(pb_path)
        if os.path.exists(xgb_path):
            curves[f"{ds}_xgb"] = pd.read_csv(xgb_path)
    return curves


def generate_learning_curves(curves):
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        label = DATASET_LABELS[ds]
        metric = "auc_roc"

        pb_key = f"{ds}_pb"
        if pb_key in curves:
            df = curves[pb_key]
            grouped = df.groupby("n_trees")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(grouped["n_trees"], grouped["mean"], "-",
                    color=COLORS["pb"], linewidth=1.0)
            ax.fill_between(grouped["n_trees"],
                            grouped["mean"] - grouped["std"],
                            grouped["mean"] + grouped["std"],
                            alpha=0.10, color=COLORS["pb"], linewidth=0)

        xgb_key = f"{ds}_xgb"
        if xgb_key in curves:
            df = curves[xgb_key]
            for method, style, color in [
                ("xgb_matched", "--", COLORS["xgb_matched"]),
                ("xgb_default", ":", COLORS["xgb_default"]),
            ]:
                sub = df[df["method"] == method]
                if not sub.empty:
                    grouped = sub.groupby("n_trees")[metric].agg(["mean", "std"]).reset_index()
                    ax.plot(grouped["n_trees"], grouped["mean"], style,
                            color=color, linewidth=0.9)
                    ax.fill_between(grouped["n_trees"],
                                    grouped["mean"] - grouped["std"],
                                    grouped["mean"] + grouped["std"],
                                    alpha=0.10, color=color, linewidth=0)

        ax.set_title(label, fontsize=8, pad=4)
        ax.set_xlabel("Number of trees")
        ax.set_ylabel("AUC-ROC")
        ax.set_xlim(0.5, 15.5)
        ax.set_xticks([1, 5, 10, 15])
        ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color=COLORS["pb"], linewidth=1.0, linestyle="-",
               label="PrivateBoost"),
        Line2D([0], [0], color=COLORS["xgb_matched"], linewidth=0.9, linestyle="--",
               label="XGBoost (matched)"),
        Line2D([0], [0], color=COLORS["xgb_default"], linewidth=0.9, linestyle=":",
               label="XGBoost (defaults)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIGURES_DIR, "learning_curves.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def generate_metrics_table():
    print("\n=== Metrics Table (LaTeX) ===\n")
    for ds in DATASETS:
        label = DATASET_LABELS[ds]
        pb_path = os.path.join(RESULTS_DIR, ds, "metrics.csv")
        xgb_path = os.path.join(RESULTS_DIR, f"{ds}_xgboost", "metrics.csv")

        if os.path.exists(pb_path):
            pb = pd.read_csv(pb_path)
            print(f"    {label:<28s} & PB     & ${pb['accuracy'].mean()*100:.1f} \\pm {pb['accuracy'].std()*100:.1f}$ "
                  f"& ${pb['auc_roc'].mean():.3f} \\pm {pb['auc_roc'].std():.3f}$ "
                  f"& ${pb['f1'].mean():.3f} \\pm {pb['f1'].std():.3f}$ \\\\")

        if os.path.exists(xgb_path):
            xgb = pd.read_csv(xgb_path)
            for method, ml in [("xgb_matched", "XGB-M"), ("xgb_default", "XGB-D")]:
                sub = xgb[xgb["method"] == method]
                if not sub.empty:
                    print(f"    {'':<28s} & {ml:<6s} & ${sub['accuracy'].astype(float).mean()*100:.1f} \\pm {sub['accuracy'].astype(float).std()*100:.1f}$ "
                          f"& ${sub['auc_roc'].astype(float).mean():.3f} \\pm {sub['auc_roc'].astype(float).std():.3f}$ "
                          f"& ${sub['f1'].astype(float).mean():.3f} \\pm {sub['f1'].astype(float).std():.3f}$ \\\\")


def generate_dropout_figure():
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for ds in DATASETS:
        path = os.path.join(RESULTS_DIR, ds, "dropout.csv")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        grouped = df.groupby("dropout_rate")["auc_roc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            grouped["dropout_rate"] * 100, grouped["mean"],
            yerr=grouped["std"],
            fmt=f"-{DATASET_MARKERS[ds]}", color=DATASET_COLORS[ds], markersize=3,
            linewidth=0.8, capsize=1.5, capthick=0.5, label=DATASET_SHORT[ds],
        )

    ax.set_xlabel("Client dropout rate (%)")
    ax.set_ylabel("AUC-ROC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(FIGURES_DIR, "dropout_resilience.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def generate_traffic_per_round(ds="cdc_diabetes"):
    path = os.path.join(RESULTS_DIR, ds, "traffic_per_round.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    # Filter to C2S traffic for the bar chart
    c2s = df[df["direction"] == "ClientToShareholder"]
    grouped = c2s.groupby(["tree", "depth"])["bytes"].sum().reset_index()
    grouped["round"] = range(len(grouped))

    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    ax.bar(grouped["round"], grouped["bytes"] / 1e6,
           color=COLORS["pb"], alpha=0.8, width=0.8, edgecolor="none")
    ax.set_xlabel("Round (tree x depth)")
    ax.set_ylabel("Client -> shareholder (MB)")
    ax.grid(axis="y", alpha=0.3)

    for t in range(1, 15):
        ax.axvline(x=t * 3 - 0.5, color="#cbd5e1", linewidth=0.3, linestyle="-")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "traffic_per_round.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def _load_depth_data(ds):
    """Load depth sweep data for one dataset."""
    depths = []
    auc_means, auc_stds = [], []
    base_kb_means, hiding_kb_means = [], []

    for d in range(1, 7):
        base_prefix = None
        for prefix in [f"{ds}_depth_{d}", f"cdc_depth_{d}"]:
            if os.path.exists(os.path.join(RESULTS_DIR, prefix, "metrics.csv")):
                base_prefix = prefix
                break
        if base_prefix is None:
            continue

        m = pd.read_csv(os.path.join(RESULTS_DIR, base_prefix, "metrics.csv"))
        t = pd.read_csv(os.path.join(RESULTS_DIR, base_prefix, "traffic_totals.csv"))

        depths.append(d)
        auc_means.append(m["auc_roc"].mean())
        auc_stds.append(m["auc_roc"].std())

        n_train = int(m["n_train"].iloc[0])
        per_client_base = t["total_bytes"].mean() / n_train / 1024
        base_kb_means.append(per_client_base)

        hiding_prefix = None
        for prefix in [f"{ds}_depth_{d}_hiding", f"cdc_depth_{d}_hiding"]:
            if os.path.exists(os.path.join(RESULTS_DIR, prefix, "traffic_totals.csv")):
                hiding_prefix = prefix
                break

        if hiding_prefix is not None:
            t_h = pd.read_csv(os.path.join(RESULTS_DIR, hiding_prefix, "traffic_totals.csv"))
            hiding_kb_means.append(t_h["total_bytes"].mean() / n_train / 1024)
        else:
            avg_nodes = sum(2**i for i in range(d)) / d if d > 0 else 1
            hiding_kb_means.append(per_client_base * avg_nodes)

    return depths, auc_means, auc_stds, base_kb_means, hiding_kb_means


def generate_depth_tradeoff():
    """4-row x 2-col figure: AUC-ROC and comm cost vs depth for all datasets."""
    fig, axes = plt.subplots(4, 2, figsize=(5.5, 7.5),
                             gridspec_kw={"width_ratios": [1, 1]})

    for row, ds in enumerate(DATASETS):
        data = _load_depth_data(ds)
        depths, auc_means, auc_stds, base_kb, hiding_kb = data

        if not depths:
            print(f"  No depth data for {ds}, skipping")
            continue

        ax_auc = axes[row, 0]
        ax_cost = axes[row, 1]
        label = DATASET_LABELS[ds]

        # Left: AUC-ROC vs depth
        ax_auc.errorbar(depths, auc_means, yerr=auc_stds, fmt="-o",
                        color=COLORS["pb"], markersize=3, linewidth=0.9,
                        capsize=2, capthick=0.4)
        ax_auc.set_xticks(depths)
        ax_auc.grid(True, alpha=0.3)
        ax_auc.set_ylabel("AUC-ROC", fontsize=7)

        # Right: Communication cost vs depth
        width = 0.35
        x = np.array(depths)
        ax_cost.bar(x - width/2, base_kb, width, color=COLORS["pb"],
                    alpha=0.8, label="Base", edgecolor="none")
        ax_cost.bar(x + width/2, hiding_kb, width, color=COLORS["accent"],
                    alpha=0.8, label="+ Path hiding", edgecolor="none")
        ax_cost.set_xticks(depths)
        ax_cost.grid(axis="y", alpha=0.3)
        ax_cost.set_ylabel("Per-client (KB)", fontsize=7)

        # Dataset name as row title above the left subplot
        if row == 0:
            ax_auc.set_title(f"Model quality\n{label}", fontsize=8, pad=4,
                             fontweight="normal")
            ax_cost.set_title("Communication cost", fontsize=8, pad=4)
        else:
            ax_auc.set_title(label, fontsize=8, pad=4, fontweight="normal")

    # X labels on all rows
    for col in range(2):
        for row in range(4):
            axes[row, col].set_xlabel("Maximum tree depth")

    # Shared legend at bottom
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["pb"], alpha=0.8, ec="none",
                       label="Base protocol"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["accent"], alpha=0.8, ec="none",
                       label="+ Path hiding"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.65, -0.01), frameon=False, fontsize=7)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.subplots_adjust(hspace=0.35)
    out = os.path.join(FIGURES_DIR, "depth_tradeoff.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    curves = load_learning_curves()
    generate_learning_curves(curves)
    generate_metrics_table()
    generate_dropout_figure()
    generate_traffic_per_round()
    generate_depth_tradeoff()


if __name__ == "__main__":
    main()
