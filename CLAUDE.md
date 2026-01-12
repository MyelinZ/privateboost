# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (use uv)
uv sync --all-extras

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_integration.py::test_xgboost_heart_disease_shamir -v

# Type checking
uv run mypy src/

# Run Jupyter notebooks
uv run jupyter notebook
```

## Architecture

privateboost implements privacy-preserving federated XGBoost via **Shamir secret sharing** with commitments. The protocol uses m-of-n threshold sharing (default 2-of-3).

```
Clients          ShareHolders       Aggregator
┌─────┐          ┌─────┐
│  C1 │─shares──>│ SH1 │─sum─┐
└─────┘          └─────┘     │      ┌─────────┐
┌─────┐          ┌─────┐     ├─────>│   AGG   │──> Statistics/Splits
│  C2 │─shares──>│ SH2 │─sum─┤      └─────────┘
└─────┘          └─────┘     │
                 ┌─────┐     │
                 │ SH3 │─────┘
                 └─────┘
```

### Core Modules

- **`client.py`**: Holds one data sample. Creates Shamir shares with commitments and distributes to shareholders. Maintains XGBoost state (prediction, node assignment).

- **`shareholder.py`**: Stores shares by commitment. Sums shares for requested commitments and returns (x_coord, sum) for Shamir reconstruction. Enforces minimum N clients.

- **`aggregator.py`**: Selects shareholders with largest commitment overlap, reconstructs aggregates using Lagrange interpolation. Computes mean/variance, defines histogram bins, finds optimal splits.

- **`crypto.py`**: Shamir sharing (`shamir_share_vector`, `shamir_reconstruct_vector`) and commitment scheme (`compute_commitment`).

- **`messages.py`**: Data classes (`ShamirShare`, `CommittedStatsShare`, `CommittedGradientShare`, `BinConfiguration`, `SplitDecision`, `LeafNode`).

### Protocol Flow

1. **Statistics round**: Clients share x and x² with commitments → aggregator reconstructs mean/variance
2. **XGBoost training**: Per tree depth level:
   - Clients compute gradients, create Shamir shares with commitments
   - Aggregator selects shareholders, requests sums for valid commitments
   - Shareholders sum shares, aggregator reconstructs via Lagrange interpolation
   - Aggregator finds best splits, broadcasts to clients
   - Leaf values computed from gradient sums

### Security Properties

- **Threshold security**: Any m-1 colluding shareholders learn nothing
- **Aggregate-only**: Aggregator sees only sums, never individual values
- **Anonymous**: Aggregator sees commitment hashes, never client IDs
- **Minimum N**: Shareholders reject requests for < N clients
