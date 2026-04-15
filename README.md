# PrivateBoost

Privacy-preserving federated XGBoost via Shamir secret sharing.

Companion code for the paper: *PrivateBoost: Privacy-Preserving Federated Gradient Boosting for Cross-Device Medical Data*.

## Overview

PrivateBoost implements a federated gradient boosting protocol where clients
collaboratively train an XGBoost model without revealing their raw data. Gradient
statistics are split into Shamir secret shares over a Mersenne prime field
(2^61 - 1), distributed to shareholders, and reconstructed only when a
configurable threshold of shareholders cooperate.

The core protocol is implemented in Rust (`rust/privateboost/`). Python is used
only for dataset preparation, figure generation, and XGBoost baselines
(`scripts/`).

## Prerequisites

- Rust >= 1.85 (edition 2024)
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup

```bash
# Python dependencies (for scripts and figures)
uv sync --all-extras --all-groups

# Rust build
cd rust/privateboost && cargo build --release && cd ../..
```

## Download datasets

The benchmark datasets are not checked in. Fetch and prepare them with:

```bash
uv run python scripts/prepare_datasets.py
```

This downloads UCI Heart Disease, Pima Diabetes, and CDC BRFSS Diabetes, and
generates the sklearn Breast Cancer CSV, writing everything to `data/`.

## Run tests

```bash
cd rust/privateboost && cargo test
```

## Reproduce paper figures

```bash
make figures
```

## Project structure

```
rust/privateboost/         Rust implementation
  src/crypto/              Mersenne field, Shamir sharing, encoding
  src/protocol/            Client, aggregator, shareholder
  src/model/               Tree structures
  src/bin/                 Benchmark binaries
  tests/                   Integration and crypto tests

scripts/                   Python helpers
  prepare_datasets.py      Download and preprocess datasets
  generate_figures.py      Generate paper figures from results/
  run_xgboost_baselines.py Centralized XGBoost baselines
  plot_binning_analysis.py Binning analysis plot

results/                   Benchmark outputs (CSV)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
