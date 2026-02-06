# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (use uv, requires Python >=3.12)
uv sync --all-extras

# Run tests (integration tests download UCI datasets, needs network on first run)
make test                # or: uv run pytest
uv run pytest tests/test_integration.py::test_xgboost_heart_disease_shamir -v  # single test

# Lint and format
make lint                # check only (ruff check + ruff format --check)
make fix                 # auto-fix (ruff check --fix + ruff format)

# Type checking
uv run mypy src/

# No build step — pure Python package. Verification is: make test && make lint
```

## Architecture

privateboost implements privacy-preserving federated XGBoost via **Shamir secret sharing** with commitments. The protocol uses m-of-n threshold sharing (default 2-of-3).

The key insight: Shamir shares are **additively homomorphic** — shareholders can sum shares from many clients and the aggregator reconstructs the sum of the original values without ever seeing individual data.

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

### Core Modules (`src/privateboost/`)

- **`client.py`**: One instance per data sample. Creates Shamir shares with commitments and distributes to shareholders. Maintains XGBoost state (prediction, node assignment). Computes gradients and bins them into histograms.

- **`shareholder.py`**: Stores shares indexed by commitment hash. Sums shares for requested commitments and returns `(x_coord, sum)` for Lagrange reconstruction. Enforces minimum N clients before releasing any aggregate.

- **`aggregator.py`**: Orchestrates the protocol. Selects shareholders with largest commitment overlap, reconstructs aggregates via Lagrange interpolation, computes mean/variance, defines histogram bins, finds optimal XGBoost splits (gain = `G²/(H+λ)`).

- **`tree.py`**: `Tree` and `Model` dataclasses for prediction. `Model` is the ensemble: `prediction = initial + lr × Σ tree.predict()`. Uses `match` statements for tree traversal.

- **`crypto/`**: `shamir.py` — `share()`, `reconstruct()`, `Share` dataclass. `commitment.py` — SHA256-based `compute_commitment(round_id, client_id, nonce)` for client anonymity.

- **`messages.py`**: Protocol data classes (`CommittedStatsShare`, `CommittedGradientShare`, `BinConfiguration`, `SplitDecision`, `LeafNode`, etc.).

### Protocol Flow

1. **Statistics round**: Clients share `[x, x²]` per feature with commitments → aggregator reconstructs mean/variance → defines equal-width bins from `μ ± 3σ`
2. **XGBoost training**: For each tree, for each depth level:
   - Clients compute gradients (`g = ŷ − y`, `h = 1` for squared; `g = p − y`, `h = p(1−p)` for logistic)
   - Clients bin gradients into histograms, create Shamir shares with fresh commitments
   - Shareholders sum shares; aggregator selects shareholders, reconstructs histogram sums
   - Aggregator scans bins to find best split per node (maximizing XGBoost gain)
   - Clients update node assignments; leaf values = `−G/(H+λ)`

### Security Properties

- **Threshold security**: Any m-1 colluding shareholders learn nothing
- **Aggregate-only**: Aggregator sees only sums, never individual values
- **Anonymous**: Aggregator sees commitment hashes, never client IDs (fresh nonce per round)
- **Minimum N**: Shareholders reject requests for < N clients
