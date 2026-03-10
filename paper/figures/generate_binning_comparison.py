#!/usr/bin/env python3
"""Generate binning comparison figure: μ±3σ equal-width vs quantile bins."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

plt.rcParams.update({"font.size": 14})

N_BINS = 10
BIN_RANGE_STDS = 3
N_SAMPLES = 100_000

sigma_values = [0.2, 0.4, 0.6, 0.8]
titles = [
    "Near-Symmetric (skewness ≈ 0.6)",
    "Mild Skew (skewness ≈ 1.3)",
    "Moderate Skew (skewness ≈ 2.3)",
    "Heavy Skew (skewness ≈ 3.7)",
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()

for ax, sigma, title in zip(axes, sigma_values, titles):
    np.random.seed(42)
    data = np.random.lognormal(mean=0, sigma=sigma, size=N_SAMPLES)

    mu = np.mean(data)
    std = np.std(data)
    range_min = mu - BIN_RANGE_STDS * std
    range_max = mu + BIN_RANGE_STDS * std
    pb_edges = np.linspace(range_min, range_max, N_BINS + 1)

    quantile_edges = np.percentile(data, np.linspace(0, 100, N_BINS + 1))

    x = np.linspace(max(0, data.min() - 0.2), np.percentile(data, 99.5), 500)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(0))
    ax.fill_between(x, pdf, alpha=0.12, color="gray")
    ax.plot(x, pdf, color="gray", linewidth=1.5)

    for i, edge in enumerate(pb_edges):
        if data.min() - 0.5 < edge < np.percentile(data, 99.9):
            ax.axvline(
                edge,
                color="#e74c3c",
                linewidth=1.3,
                linestyle="--",
                alpha=0.75,
                label="μ±3σ Equal-Width" if i == 0 else None,
            )

    for i, edge in enumerate(quantile_edges[1:-1]):
        ax.axvline(
            edge,
            color="#3498db",
            linewidth=1.3,
            linestyle="-",
            alpha=0.75,
            label="Quantile (XGBoost)" if i == 0 else None,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlim(max(0, data.min() - 0.1), np.percentile(data, 99.5))
    ax.set_ylabel("Density")
    ax.set_xlabel("Feature Value")
    ax.grid(alpha=0.2)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("binning_comparison.png", dpi=150, bbox_inches="tight")
print("Saved binning_comparison.png")
