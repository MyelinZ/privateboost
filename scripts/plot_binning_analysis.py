"""Visualize how PrivateBoost's Gaussian quantile binning compares to equal-width."""
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

import style
style.apply()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pima_diabetes.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")

N_BINS = 10

COLOR_PRIMARY = style.COLORS["pb"]
COLOR_ACCENT = style.COLORS["accent"]
COLOR_HIGHLIGHT = style.COLORS["highlight"]
COLOR_HIST = "#3a3a3a"
COLOR_MUTED = style.COLORS["muted"]

FEATURES = [
    ("glucose", "Glucose (mg/dL)"),
    ("insulin", "Insulin (\u03bcU/mL)"),
]


def compute_uniform_bins(values, n_bins):
    mean = np.mean(values)
    std = np.std(values)
    k = stats.norm.ppf(1.0 - 1.0 / (n_bins + 2))
    range_min = mean - k * std
    range_max = mean + k * std
    inner_edges = np.linspace(range_min, range_max, n_bins + 1)
    edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    return edges, inner_edges


def compute_gaussian_bins(values, n_bins):
    mean = np.mean(values)
    std = np.std(values)
    n_inner = n_bins + 1
    quantiles_inner = np.array([(i + 1.0) / (n_inner + 1.0) for i in range(n_inner)])
    inner_edges = stats.norm.ppf(quantiles_inner, loc=mean, scale=std)
    edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    return edges, inner_edges


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.8),
                             gridspec_kw={"height_ratios": [1.0, 1.2]})

    methods = [
        ("Equal-width", COLOR_ACCENT, compute_uniform_bins),
        ("Gaussian", COLOR_PRIMARY, compute_gaussian_bins),
    ]

    for col, (feat, label) in enumerate(FEATURES):
        vals = df[feat].values
        mean = np.mean(vals)
        std = np.std(vals)
        skewness = stats.skew(vals)

        ax_dist = axes[0, col]
        ax_bars = axes[1, col]

        # --- Row 1: Distribution + bin edges ---
        # Clip x range to 99th percentile for better visibility
        p99 = np.percentile(vals, 99)
        xlim = (vals.min() - (p99 - vals.min()) * 0.05, p99 * 1.05)

        ax_dist.hist(vals, bins=50, range=xlim, color=COLOR_HIST, edgecolor="#2c2c2c",
                     linewidth=0.3, alpha=0.85, density=True)

        ax_dist.axvline(mean, color=COLOR_HIGHLIGHT, linewidth=1.0, zorder=5)
        ax_dist.axvspan(mean - std, mean + std, color=COLOR_HIGHLIGHT,
                        alpha=0.08, zorder=0)

        # Equal-width edges
        _, u_inner = compute_uniform_bins(vals, N_BINS)
        for e in u_inner:
            if xlim[0] <= e <= xlim[1]:
                ax_dist.axvline(e, color=COLOR_ACCENT, linewidth=0.5, alpha=0.6,
                                zorder=3)

        # Gaussian edges
        _, g_inner = compute_gaussian_bins(vals, N_BINS)
        for e in g_inner:
            if xlim[0] <= e <= xlim[1]:
                ax_dist.axvline(e, color=COLOR_PRIMARY, linewidth=0.5, alpha=0.6,
                                linestyle="--", zorder=3)

        ax_dist.set_xlim(xlim)

        ax_dist.text(0.97, 0.88, f"$\\gamma_1 = {skewness:.2f}$",
                     transform=ax_dist.transAxes, fontsize=7, ha="right", va="top",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor=COLOR_MUTED, linewidth=0.4, alpha=0.95),
                     zorder=6)

        ax_dist.set_title(label, fontsize=9, pad=4)
        ax_dist.set_xticklabels([])
        ax_dist.set_yticks([])

        # --- Row 2: Grouped bar chart ---
        all_pcts = {}
        for name, color, fn in methods:
            edges, _ = fn(vals, N_BINS)
            counts, _ = np.histogram(vals, bins=edges)
            pcts = counts / counts.sum() * 100
            all_pcts[name] = pcts

        n_total = len(all_pcts["Equal-width"])
        x_base = np.arange(n_total)
        bar_width = 0.35

        for i, (name, color, _) in enumerate(methods):
            pcts = all_pcts[name]
            offset = (i - 0.5) * bar_width
            ax_bars.bar(x_base + offset, pcts, bar_width, color=color, alpha=0.8,
                        edgecolor="none", label=name)

        # Ideal uniform line
        ideal = 100.0 / n_total
        ax_bars.axhline(ideal, color=COLOR_MUTED, linewidth=0.6, linestyle="--",
                        alpha=0.7)

        # CV annotation per method
        for i, (name, color, _) in enumerate(methods):
            pcts = all_pcts[name]
            cv = np.std(pcts) / np.mean(pcts)
            ax_bars.text(0.97, 0.92 - i * 0.10, f"CV = {cv:.2f}",
                         transform=ax_bars.transAxes, fontsize=6.5,
                         ha="right", va="top", color=color, fontweight="bold")

        ax_bars.grid(axis="y", alpha=0.4)

        ax_bars.set_xticks(x_base)
        tick_labels = ["$-\\infty$"] + [str(i+1) for i in range(n_total - 2)] + ["$+\\infty$"]
        ax_bars.set_xticklabels(tick_labels, fontsize=6)
        ax_bars.set_xlabel("Bin")

        ax_bars.grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel("Density", fontsize=8)
    axes[1, 0].set_ylabel("Samples (%)", fontsize=8)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=COLOR_HIGHLIGHT, linewidth=1.0, label="\u03bc"),
        Patch(facecolor=COLOR_HIGHLIGHT, alpha=0.08, edgecolor="none",
              label=r"$\pm 1\sigma$"),
        Patch(facecolor=COLOR_ACCENT, alpha=0.8, edgecolor="none",
              label="Equal-width"),
        Patch(facecolor=COLOR_PRIMARY, alpha=0.8, edgecolor="none",
              label="Gaussian quantile"),
        Line2D([0], [0], color=COLOR_MUTED, linewidth=0.6, linestyle="--",
               alpha=0.7, label="Ideal uniform"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=7)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(FIGURES_DIR, "binning_analysis.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
