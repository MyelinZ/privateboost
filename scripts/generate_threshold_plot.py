"""Generate threshold comparison plot for design document."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from privateboost import Client, ShareHolder, Aggregator

np.random.seed(42)

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

df = pd.read_csv(url, names=columns, na_values="?")
df = df.dropna()
df["target"] = (df["target"] > 0).astype(int)

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

X = df[FEATURES].values.astype(float)
y = df["target"].values.astype(float)

# Setup privateboost
N_SHAREHOLDERS = 3
N_BINS = 10
LAMBDA_REG = 1.0

aggregator = Aggregator(n_bins=N_BINS)
shareholders = [ShareHolder(i, aggregator) for i in range(N_SHAREHOLDERS)]

clients = [
    Client(
        client_id=f"patient_{idx}",
        features=[float(row[name]) for name in FEATURES],
        target=float(row["target"]),
        shareholders=shareholders,
    )
    for idx, row in df.iterrows()
]

for client in clients:
    client.submit_feature_shares_for_stats()
for sh in shareholders:
    sh.submit_stats()
bins = aggregator.define_bins()

initial_pred = float(aggregator.means[-1])
predictions = np.full(len(y), initial_pred)


def compute_optimal_split_for_feature(X, y, predictions, feature_idx, lambda_reg=1.0):
    """Find optimal split for a specific feature."""
    gradients = predictions - y
    hessians = np.ones_like(gradients)

    total_g = gradients.sum()
    total_h = hessians.sum()
    base_score = (total_g ** 2) / (total_h + lambda_reg)

    feature_values = X[:, feature_idx]
    sorted_indices = np.argsort(feature_values)
    sorted_values = feature_values[sorted_indices]
    sorted_g = gradients[sorted_indices]
    sorted_h = hessians[sorted_indices]

    best_gain = 0
    best_threshold = None

    g_left = 0
    h_left = 0

    for i in range(len(sorted_values) - 1):
        g_left += sorted_g[i]
        h_left += sorted_h[i]
        g_right = total_g - g_left
        h_right = total_h - h_left

        if sorted_values[i] == sorted_values[i + 1]:
            continue

        if h_left < 0.1 or h_right < 0.1:
            continue

        left_score = (g_left ** 2) / (h_left + lambda_reg)
        right_score = (g_right ** 2) / (h_right + lambda_reg)
        gain = left_score + right_score - base_score

        if gain > best_gain:
            best_gain = gain
            best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

    return best_threshold, best_gain


def compute_pb_split_for_feature(X, y, predictions, feature_idx, bins, lambda_reg=1.0):
    """Simulate privateboost split using histogram binning."""
    gradients = predictions - y
    hessians = np.ones_like(gradients)

    total_g = gradients.sum()
    total_h = hessians.sum()
    base_score = (total_g ** 2) / (total_h + lambda_reg)

    bin_config = bins[feature_idx]
    n_total_bins = bin_config.n_bins + 2

    grad_hist = np.zeros(n_total_bins)
    hess_hist = np.zeros(n_total_bins)

    for i in range(len(X)):
        value = X[i, feature_idx]
        bin_idx = int(np.searchsorted(bin_config.edges, value, side="right")) - 1
        bin_idx = max(0, min(bin_idx, n_total_bins - 1))
        grad_hist[bin_idx] += gradients[i]
        hess_hist[bin_idx] += hessians[i]

    g_cumsum = np.cumsum(grad_hist)
    h_cumsum = np.cumsum(hess_hist)

    best_gain = 0
    best_threshold = None

    for i in range(len(grad_hist) - 1):
        g_left = g_cumsum[i]
        h_left = h_cumsum[i]
        g_right = total_g - g_left
        h_right = total_h - h_left

        if h_left < 0.1 or h_right < 0.1:
            continue

        left_score = (g_left ** 2) / (h_left + lambda_reg)
        right_score = (g_right ** 2) / (h_right + lambda_reg)
        gain = left_score + right_score - base_score

        if gain > best_gain:
            best_gain = gain
            best_threshold = bin_config.edges[i + 1]

    return best_threshold, best_gain


# Compute for all features
all_features_data = []

for feat_idx, feat_name in enumerate(FEATURES):
    opt_thresh, opt_g = compute_optimal_split_for_feature(X, y, predictions, feat_idx, LAMBDA_REG)
    pb_thresh, pb_g = compute_pb_split_for_feature(X, y, predictions, feat_idx, bins, LAMBDA_REG)

    feat_min = X[:, feat_idx].min()
    feat_max = X[:, feat_idx].max()
    feat_range = feat_max - feat_min

    all_features_data.append({
        'feature': feat_name,
        'opt_threshold': opt_thresh,
        'pb_threshold': pb_thresh,
        'feat_min': feat_min,
        'feat_range': feat_range,
        'gain_retention': (pb_g / opt_g * 100) if opt_g > 0 else 100,
    })

# Generate plot
fig, ax = plt.subplots(figsize=(10, 7))

for i, row in enumerate(all_features_data):
    ax.plot([0, 1], [i, i], 'k-', alpha=0.2, linewidth=8)

    if row['feat_range'] > 0:
        if row['opt_threshold'] is not None:
            opt_n = (row['opt_threshold'] - row['feat_min']) / row['feat_range']
            ax.scatter(opt_n, i, s=120, c='#e74c3c', marker='o', edgecolor='black',
                      label='Optimal (exhaustive search)' if i == 0 else '', zorder=5)

        if row['pb_threshold'] is not None:
            pb_n = (row['pb_threshold'] - row['feat_min']) / row['feat_range']
            ax.scatter(pb_n, i, s=120, c='#2ecc71', marker='s', edgecolor='black',
                      label='privateboost (histogram-based)' if i == 0 else '', zorder=5)

        if row['opt_threshold'] is not None and row['pb_threshold'] is not None:
            ax.plot([opt_n, pb_n], [i, i], 'b-', alpha=0.4, linewidth=2)

ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels(FEATURES)
ax.set_xlabel('Normalized position in feature range (0 = min, 1 = max)')
ax.set_ylabel('Feature')
ax.set_title('Split Threshold Comparison: privateboost vs Optimal\n(UCI Heart Disease Dataset, n=297)')
ax.set_xlim(-0.05, 1.05)
ax.legend(loc='upper right')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('docs/plans/threshold_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to docs/plans/threshold_comparison.png")

# Print summary
mean_retention = np.mean([r['gain_retention'] for r in all_features_data])
print(f"Mean gain retention: {mean_retention:.1f}%")
