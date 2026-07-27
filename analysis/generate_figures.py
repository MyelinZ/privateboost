"""Generate all paper figures from benchmark CSV results."""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from paths import FIGURES_DIR, RESULTS_DIR
import style
style.apply()

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

# deviceModel (as reported by the platform) -> display name, in the fixed
# order the fleet figures present devices. Must match fleet_table.py.
DEVICE_DISPLAY = {
    "Google Pixel 9 Pro": "Google Pixel 9 Pro",
    "samsung SM-X200": "Samsung Galaxy Tab A8",
    "Lenovo TB328FU": "Lenovo Tab M8",
    "iPhone iPhone": "Apple iPhone 13 Pro",
}
TRIGGER_DISPLAY = {"push": "Remote wakeup", "workmanager": "Periodic"}
TRIGGER_MARKERS = {"push": "o", "workmanager": "s"}


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
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0))
    axes = axes.flatten()
    rng = np.random.default_rng(0)

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        path = os.path.join(RESULTS_DIR, ds, "dropout.csv")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        jitter = rng.uniform(-1.5, 1.5, size=len(df))
        ax.scatter(df["dropout_rate"] * 100 + jitter, df["auc_roc"], s=4,
                   color=DATASET_COLORS[ds], alpha=0.35, linewidths=0, zorder=2)
        grouped = df.groupby("dropout_rate")["auc_roc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            grouped["dropout_rate"] * 100, grouped["mean"],
            yerr=grouped["std"],
            fmt=f"-{DATASET_MARKERS[ds]}", color=DATASET_COLORS[ds], markersize=3,
            linewidth=0.8, capsize=1.5, capthick=0.5, zorder=3,
        )
        ax.set_title(DATASET_LABELS[ds], fontsize=8, pad=4)
        ax.set_xlabel("Client dropout rate (%)")
        ax.set_ylabel("AUC-ROC")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "dropout_resilience.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def _load_depth_data(ds):
    """Per-depth AUC-ROC across splits for one dataset. Communication cost for
    the same sweep comes from `_load_measured_wire`, on its own depth axis:
    the two need not cover the same depths."""
    depths = []
    auc_means, auc_stds, auc_values = [], [], []

    for d in range(1, 7):
        base_prefix = None
        for prefix in [f"{ds}_depth_{d}", f"cdc_depth_{d}"]:
            if os.path.exists(os.path.join(RESULTS_DIR, prefix, "metrics.csv")):
                base_prefix = prefix
                break
        if base_prefix is None:
            continue

        m = pd.read_csv(os.path.join(RESULTS_DIR, base_prefix, "metrics.csv"))

        depths.append(d)
        auc_means.append(m["auc_roc"].mean())
        auc_stds.append(m["auc_roc"].std())
        auc_values.append(m["auc_roc"].to_numpy())

    return depths, auc_means, auc_stds, auc_values


WIRE_MEASURED_PATH = os.path.join(RESULTS_DIR, "wire_measured.csv")


def _load_measured_wire(ds):
    """Measured per-client wire total (MB) vs depth for one dataset, 2-of-3
    threshold, from `results/wire_measured.csv` (see `wire.py` and
    `crates/pbr-e2e/tests/wire_grid.rs`): the real transport stack (HTTP/2
    framing and connection handshakes) over a loopback cluster, client-side
    only (`total_tx + total_rx`).
    Returns (depths, base_mb, hiding_mb); empty lists if the CSV or a
    depth/arm combination is missing."""
    if not os.path.exists(WIRE_MEASURED_PATH):
        return [], [], []
    df = pd.read_csv(WIRE_MEASURED_PATH)
    df = df[(df["dataset"] == ds) & (df["threshold"] == 2)].copy()
    df["hide_path"] = df["hide_path"].astype(str).str.lower() == "true"
    df["total_mb"] = (df["total_tx"] + df["total_rx"]) / (1024 * 1024)

    depths, base_mb, hiding_mb = [], [], []
    for d in sorted(df["depth"].unique()):
        base_row = df[(df["depth"] == d) & (~df["hide_path"])]
        hiding_row = df[(df["depth"] == d) & (df["hide_path"])]
        if base_row.empty or hiding_row.empty:
            continue
        depths.append(int(d))
        base_mb.append(base_row["total_mb"].iloc[0])
        hiding_mb.append(hiding_row["total_mb"].iloc[0])
    return depths, base_mb, hiding_mb


def generate_depth_tradeoff():
    """4-row x 2-col figure: AUC-ROC and comm cost vs depth for all datasets."""
    fig, axes = plt.subplots(4, 2, figsize=(5.5, 7.5),
                             gridspec_kw={"width_ratios": [1, 1]})

    rng = np.random.default_rng(0)
    for row, ds in enumerate(DATASETS):
        depths, auc_means, auc_stds, auc_values = _load_depth_data(ds)

        if not depths:
            print(f"  No depth data for {ds}, skipping")
            continue

        ax_auc = axes[row, 0]
        ax_cost = axes[row, 1]
        label = DATASET_LABELS[ds]

        # Left: AUC-ROC vs depth
        for d, vals in zip(depths, auc_values):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax_auc.scatter(np.full(len(vals), d) + jitter, vals, s=4,
                           color=COLORS["pb"], alpha=0.35, linewidths=0,
                           zorder=2)
        ax_auc.errorbar(depths, auc_means, yerr=auc_stds, fmt="-o",
                        color=COLORS["pb"], markersize=3, linewidth=0.9,
                        capsize=2, capthick=0.4, zorder=3)
        ax_auc.set_xticks(depths)
        ax_auc.grid(True, alpha=0.3)
        ax_auc.set_ylabel("AUC-ROC", fontsize=7)

        # Right: measured per-client communication cost vs depth
        width = 0.35
        m_depths, m_base_mb, m_hiding_mb = _load_measured_wire(ds)
        mx = np.array(m_depths)
        ax_cost.bar(mx - width/2, m_base_mb, width, color=COLORS["pb"],
                    alpha=0.8, label="Base", edgecolor="none")
        ax_cost.bar(mx + width/2, m_hiding_mb, width, color=COLORS["accent"],
                    alpha=0.8, label="+ Path hiding", edgecolor="none")
        ax_cost.set_xticks(m_depths)
        ax_cost.grid(axis="y", alpha=0.3)
        ax_cost.set_ylabel("Per-client (MB)", fontsize=7)

        # Dataset name as row title above the left subplot
        if row == 0:
            ax_auc.set_title(f"Model quality\n{label}", fontsize=8, pad=4,
                             fontweight="normal")
            ax_cost.set_title("Communication cost", fontsize=8, pad=4)
        else:
            ax_auc.set_title(label, fontsize=8, pad=4, fontweight="normal")

    # X label on the bottom row only, so a row's label cannot collide with the
    # title of the row beneath it (the depth axis is shared down each column).
    for col in range(2):
        axes[3, col].set_xlabel("Maximum tree depth")

    # Shared legend at bottom
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["pb"], alpha=0.8, ec="none",
                       label="Base protocol"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["accent"], alpha=0.8, ec="none",
                       label="+ Path hiding"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=6.5,
               columnspacing=1.0)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.subplots_adjust(hspace=0.35)
    out = os.path.join(FIGURES_DIR, "depth_tradeoff.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def generate_fleet_auc():
    """Fleet capture, two panels: (a) per-tree test AUC of the three
    concurrently trained sessions (results/fleet_tree_metrics.csv); (b)
    measured per-client communication (results/fleet_round_metrics.csv),
    summed over submitted rounds per (device, dataset) and averaged across
    the four devices, with each device's own total plotted as a point over
    its dataset's bar."""
    tree_df = pd.read_csv(os.path.join(RESULTS_DIR, "fleet_tree_metrics.csv"))
    round_df = pd.read_csv(os.path.join(RESULTS_DIR, "fleet_round_metrics.csv"))
    submitted = round_df[round_df["roundId"].notna()].copy()

    fig, (ax_auc, ax_comm) = plt.subplots(1, 2, figsize=(6.0, 2.6))

    for ds, group in tree_df.groupby("datasetId"):
        group = group.sort_values("treeIdx")
        ax_auc.plot(
            group["treeIdx"] + 1,
            group["auc"],
            marker=DATASET_MARKERS.get(ds, "o"),
            markersize=3.5,
            linewidth=1.2,
            color=DATASET_COLORS.get(ds, COLORS["pb"]),
            label=DATASET_SHORT.get(ds, ds),
        )
    ax_auc.set_xlabel("Tree")
    ax_auc.set_ylabel("Test AUC-ROC")
    ax_auc.set_xticks(sorted((tree_df["treeIdx"] + 1).unique()))
    ax_auc.set_ylim(0.5, 1.02)
    ax_auc.legend(frameon=False, loc="lower right", fontsize=6)

    submitted["totalBytes"] = submitted["txBytes"] + submitted["rxBytes"]
    per_device = (
        submitted.groupby(["deviceModel", "datasetId"])["totalBytes"].sum() / 1e6
    ).reset_index()
    datasets_present = [d for d in DATASETS if d in per_device["datasetId"].unique()]
    means = per_device.groupby("datasetId")["totalBytes"].mean()
    xpos = np.arange(len(datasets_present))
    ax_comm.bar(
        xpos, [means[ds] for ds in datasets_present],
        color=[DATASET_COLORS[ds] for ds in datasets_present],
        alpha=0.8, width=0.6, edgecolor="none",
    )
    rng = np.random.default_rng(0)
    for i, ds in enumerate(datasets_present):
        vals = per_device.loc[per_device["datasetId"] == ds, "totalBytes"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax_comm.scatter(np.full(len(vals), i) + jitter, vals, color="black",
                         s=6, zorder=3, linewidths=0)
    ax_comm.set_xticks(xpos)
    ax_comm.set_xticklabels([DATASET_SHORT[ds] for ds in datasets_present],
                             rotation=20, fontsize=6)
    ax_comm.set_ylabel("Per-client comm. (MB)")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fleet_auc.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def _device_inter_round_gaps_min(submitted):
    """Minutes between a device's successive submitted rounds within the same
    session, pooled across its sessions. Grouping by session first avoids
    treating the sub-second gap between two concurrently trained sessions'
    near-simultaneous submissions in one wake as an inter-round interval."""
    gaps = {}
    for dev in DEVICE_DISPLAY:
        dsub = submitted[submitted["deviceModel"] == dev]
        dev_gaps = []
        for _, g in dsub.groupby("datasetId"):
            ts = g.sort_values("clientTsMs")["clientTsMs"].to_numpy()
            dev_gaps.extend((np.diff(ts) / 60000.0).tolist())
        gaps[dev] = np.array(dev_gaps)
    return gaps


def generate_fleet_participation():
    """Fleet capture, two panels: (a) submitted rounds per device by wake
    trigger, plus the fleet average; (b) time between a device's successive
    submitted rounds (results/fleet_round_metrics.csv)."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "fleet_round_metrics.csv"))
    submitted = df[df["roundId"].notna()].copy()

    fig, (ax_trig, ax_gap) = plt.subplots(1, 2, figsize=(6.0, 2.6))

    devices = list(DEVICE_DISPLAY)
    counts = (
        submitted.groupby(["deviceModel", "triggerSource"]).size()
        .unstack(fill_value=0)
        .reindex(devices, fill_value=0)
    )
    counts.loc["Average"] = counts.mean(axis=0)
    labels = [DEVICE_DISPLAY[d] for d in devices] + ["Average"]
    trigger_colors = {"push": COLORS["pb"], "workmanager": COLORS["accent"]}

    xpos = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for trig in ("push", "workmanager"):
        if trig not in counts.columns:
            continue
        vals = counts[trig].to_numpy()
        ax_trig.bar(xpos, vals, bottom=bottom, color=trigger_colors[trig],
                    label=TRIGGER_DISPLAY[trig], width=0.65, edgecolor="none")
        bottom += vals
    ax_trig.set_xticks(xpos)
    ax_trig.set_xticklabels(labels, rotation=25, ha="right", fontsize=6)
    ax_trig.set_ylabel("Submissions")
    ax_trig.set_ylim(0, bottom.max() * 1.15)
    ax_trig.legend(frameon=False, ncol=2, loc="lower center",
                    bbox_to_anchor=(0.5, 1.02), fontsize=6)

    gaps = _device_inter_round_gaps_min(submitted)
    ax_gap.boxplot([gaps[d] for d in devices], widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, dev in enumerate(devices, start=1):
        vals = gaps[dev]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax_gap.scatter(np.full(len(vals), i) + jitter, vals, color="black",
                        s=5, alpha=0.5, zorder=3, linewidths=0)
    ax_gap.set_xticks(range(1, len(devices) + 1))
    ax_gap.set_xticklabels([DEVICE_DISPLAY[d] for d in devices],
                            rotation=25, ha="right", fontsize=6)
    ax_gap.set_ylabel("Inter-round interval (min)")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fleet_participation.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def generate_fleet_timeline():
    """Timeline of the fleet capture: one marker per submitted round,
    positioned by device (row) and elapsed hours since the capture's first
    submitted round (results/fleet_round_metrics.csv). Colour identifies the
    dataset/session, marker shape the wake trigger; a small per-dataset
    vertical offset separates the three concurrently trained sessions within
    a device's row."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "fleet_round_metrics.csv"))
    submitted = df[df["roundId"].notna()].copy()
    t0 = submitted["clientTsMs"].min()
    submitted["hoursSinceStart"] = (submitted["clientTsMs"] - t0) / 3_600_000.0

    fig, ax = plt.subplots(figsize=(7.0, 2.5))

    devices = list(DEVICE_DISPLAY)
    device_y = {d: i for i, d in enumerate(devices)}
    datasets_present = [d for d in DATASETS if d in submitted["datasetId"].unique()]
    offset_step = 0.18
    dataset_offset = {
        ds: (i - (len(datasets_present) - 1) / 2) * offset_step
        for i, ds in enumerate(datasets_present)
    }

    for (dev, ds, trig), g in submitted.groupby(
        ["deviceModel", "datasetId", "triggerSource"]
    ):
        y = device_y[dev] + dataset_offset[ds]
        ax.scatter(
            g["hoursSinceStart"], np.full(len(g), y),
            marker=TRIGGER_MARKERS.get(trig, "o"),
            color=DATASET_COLORS.get(ds, COLORS["pb"]),
            s=14, linewidths=0, zorder=3,
        )

    ax.set_yticks([device_y[d] for d in devices])
    ax.set_yticklabels([DEVICE_DISPLAY[d] for d in devices])
    ax.set_ylim(-0.6, len(devices) - 1 + 0.6)
    ax.set_xlabel("Time since start (h)")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0f}h")
    ax.grid(axis="x", alpha=0.3)

    dataset_legend = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4,
               color=DATASET_COLORS[ds], label=DATASET_SHORT[ds])
        for ds in datasets_present
    ]
    trigger_legend = [
        Line2D([0], [0], marker=m, linestyle="none", markersize=4,
               color="black", label=TRIGGER_DISPLAY[t])
        for t, m in TRIGGER_MARKERS.items()
    ]
    fig.legend(handles=dataset_legend, title="Dataset", loc="center left",
               bbox_to_anchor=(0.82, 0.68), frameon=False, fontsize=6)
    fig.legend(handles=trigger_legend, title="Trigger", loc="center left",
               bbox_to_anchor=(0.82, 0.28), frameon=False, fontsize=6)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    out = os.path.join(FIGURES_DIR, "fleet_timeline.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    curves = load_learning_curves()
    generate_learning_curves(curves)
    generate_metrics_table()
    generate_dropout_figure()
    generate_depth_tradeoff()
    generate_fleet_auc()
    generate_fleet_participation()
    generate_fleet_timeline()


if __name__ == "__main__":
    main()
