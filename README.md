# privateboost

Privacy-preserving federated XGBoost via secret sharing.

Train gradient boosted trees across distributed clients where each client holds a single sample. Individual data is never revealed - only aggregate statistics are computed through secure multi-party computation.

## Installation

```bash
pip install privateboost
```

Or with uv:

```bash
uv add privateboost
```

## Quick Start

```python
from privateboost import Client, ShareHolder, Aggregator
import pandas as pd

# Load your data
df = pd.read_csv("data.csv")

# Setup infrastructure
aggregator = Aggregator(n_bins=10)
shareholders = [ShareHolder(i, aggregator) for i in range(3)]

# Create clients (one per row)
clients = Client.from_dataframe(
    df,
    features=["age", "income", "score"],
    target="label",
    shareholders=shareholders
)

# Round 1: Compute statistics for binning
for client in clients:
    client.submit_feature_shares_for_stats()
for sh in shareholders:
    sh.submit_stats()
bins = aggregator.define_bins()

# Access computed statistics
print(aggregator.means)      # {"age": 45.2, "income": 52000, ...}
print(aggregator.variances)  # {"age": 156.3, ...}

# Round 2: Estimate medians (optional)
for client in clients:
    client.submit_histogram_shares(bins)
for sh in shareholders:
    sh.submit_histogram()
medians = aggregator.compute_medians()
```

## XGBoost Training

```python
# Initialize predictions
initial_pred = aggregator.means["_target"]  # Include target in stats
for client in clients:
    client.prediction = initial_pred

# Train trees
bins_dict = {b.feature_name: b for b in bins}

for tree in range(n_trees):
    aggregator.reset_tree()
    for client in clients:
        client.reset_node()
        client.compute_gradients()

    # Build tree level by level
    active_nodes = [0]
    all_splits = {}

    for depth in range(max_depth):
        for client in clients:
            client.submit_gradients(bins_dict, active_nodes)
        for sh in shareholders:
            sh.submit_gradients()
            sh.reset_gradients()

        splits = aggregator.find_best_splits(lambda_reg=1.0)
        if not splits:
            break

        all_splits.update(splits)
        for client in clients:
            client.apply_splits(splits, bins_dict)

        active_nodes = []
        for s in splits.values():
            active_nodes.extend([s.left_child_id, s.right_child_id])

    # Compute leaf values and update predictions
    leaf_ids = ...  # Nodes that weren't split
    leaves = aggregator.compute_leaf_values(leaf_ids)
    for client in clients:
        client.update_prediction(leaves, learning_rate=0.1)
```

## How It Works

1. **Secret Sharing**: Each client splits their values into random shares distributed across 3+ ShareHolders. No single ShareHolder can reconstruct the original value.

2. **Aggregation**: ShareHolders sum their received shares and forward totals to the Aggregator. The Aggregator combines shares from all ShareHolders to reconstruct only the aggregate (sum, not individual values).

3. **XGBoost**: Gradient histograms are aggregated the same way. The Aggregator finds optimal splits from histogram totals without seeing individual gradients.

```
Clients          ShareHolders       Aggregator
┌─────┐          ┌─────┐
│  C1 │─shares──>│ SH1 │─sum─┐
└─────┘          └─────┘     │      ┌─────────┐
┌─────┐          ┌─────┐     ├─────>│   AGG   │──> Statistics
│  C2 │─shares──>│ SH2 │─sum─┤      └─────────┘
└─────┘          └─────┘     │
┌─────┐          ┌─────┐     │
│  C3 │─shares──>│ SH3 │─sum─┘
└─────┘          └─────┘
```

## API Reference

### Client

```python
Client(client_id, features, target, shareholders)
Client.from_dataframe(df, features, target, shareholders)

client.submit_feature_shares_for_stats()  # Share x and x² for mean/variance
client.submit_histogram_shares(bins)       # Share bin votes for median
client.compute_gradients(loss="squared")   # Compute local gradient/hessian
client.submit_gradients(bins, active_nodes)# Share gradient histograms
client.apply_splits(splits, bins)          # Update node assignment
client.update_prediction(leaves, lr)       # Add leaf value to prediction
client.reset_node()                        # Reset to root for new tree
```

### ShareHolder

```python
ShareHolder(party_id, aggregator)

sh.submit_stats()      # Forward aggregated statistics
sh.submit_histogram()  # Forward aggregated histogram
sh.submit_gradients()  # Forward aggregated gradients
sh.reset()             # Clear all accumulators
sh.reset_gradients()   # Clear only gradient accumulators
```

### Aggregator

```python
Aggregator(n_bins=10)

aggregator.define_bins()                    # -> List[BinConfiguration]
aggregator.compute_medians()                # -> Dict[str, MedianResult]
aggregator.find_best_splits(lambda_reg)     # -> Dict[int, SplitDecision]
aggregator.compute_leaf_values(leaf_ids)    # -> Dict[int, LeafNode]
aggregator.reset()                          # Clear everything
aggregator.reset_tree()                     # Clear tree state only

aggregator.means       # Dict[str, float]
aggregator.variances   # Dict[str, float]
aggregator.stds        # Dict[str, float]
```

## Notebooks

- `notebooks/01_secure_aggregation.ipynb` - Mean, variance, and median computation
- `notebooks/02_federated_xgboost.ipynb` - Full XGBoost training with train/test split

## Security Model

- Requires 3+ ShareHolders (no single point of compromise)
- Assumes ShareHolders don't collude
- Reveals only aggregate statistics, never individual values
- Split thresholds are revealed (necessary for prediction)

## Citation

If you use this library in research, please cite:

```bibtex
@software{privateboost,
  title = {privateboost: Privacy-preserving federated XGBoost via secret sharing},
  author = {Specht, Bernhard},
  year = {2026},
  url = {https://github.com/username/privateboost}
}
```

## License

MIT
