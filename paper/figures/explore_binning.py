#!/usr/bin/env python3
"""Explore different skewness levels for the binning comparison figure."""

import numpy as np
import matplotlib.pyplot as plt

N_BINS = 10
N_SAMPLES = 10_000
BIN_RANGE_STDS = 3

sigma_values = [0.2, 0.4, 0.6, 0.8]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, sigma in zip(axes, sigma_values):
    np.random.seed(42)
    data = np.random.lognormal(mean=0, sigma=sigma, size=N_SAMPLES)

    # privateboost binning: equal-width from mu +/- 3*std
    mu = np.mean(data)
    std = np.std(data)
    range_min = mu - BIN_RANGE_STDS * std
    range_max = mu + BIN_RANGE_STDS * std
    pb_edges = np.linspace(range_min, range_max, N_BINS + 1)

    # Quantile binning (XGBoost default)
    quantile_edges = np.percentile(data, np.linspace(0, 100, N_BINS + 1))

    # Plot density
    from scipy.stats import lognorm
    x = np.linspace(max(0, data.min() - 0.5), np.percentile(data, 99.5), 500)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(0))
    ax.fill_between(x, pdf, alpha=0.15, color='gray')
    ax.plot(x, pdf, color='gray', linewidth=1.5)

    # Plot bin edges
    for i, edge in enumerate(pb_edges):
        if data.min() - 1 < edge < np.percentile(data, 99.9):
            ax.axvline(edge, color='#e74c3c', linewidth=1.2, linestyle='--', alpha=0.7,
                       label='μ±3σ bins' if i == 0 else None)

    for i, edge in enumerate(quantile_edges[1:-1]):  # skip 0th and 100th percentile
        ax.axvline(edge, color='#3498db', linewidth=1.2, linestyle='-', alpha=0.7,
                   label='Quantile bins' if i == 0 else None)

    skewness = lognorm.stats(s=sigma, scale=np.exp(0), moments='s')
    ax.set_title(f'σ={sigma} (skewness≈{float(skewness):.1f})', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(max(0, data.min() - 0.3), np.percentile(data, 99.5))
    ax.set_ylabel('Density')

plt.suptitle('Log-normal distributions: μ±3σ bins (red) vs Quantile bins (blue)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('explore_binning.png', dpi=150, bbox_inches='tight')
print("Saved explore_binning.png")
