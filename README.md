# PrivateBoost

Privacy-preserving federated XGBoost via Shamir secret sharing.

Companion code for the paper: *PrivateBoost: Privacy-Preserving Federated Gradient Boosting for Cross-Device Medical Data*.

## Overview

PrivateBoost implements a federated gradient boosting protocol where clients
collaboratively train an XGBoost model without revealing their raw data. Gradient
statistics are split into Shamir secret shares over a Mersenne prime field
(2^61 - 1), distributed to shareholders, and reconstructed only when a
configurable threshold of shareholders cooperate.

The repository contains two implementations:

- **Python** (`src/privateboost/`) — reference implementation for clarity and experimentation
- **Rust** (`rust/privateboost/`) — performance-oriented implementation of the same protocol

## Prerequisites

- Python >= 3.12
- Rust >= 1.85 (for edition 2024)
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup

```bash
# Python dependencies
uv sync --all-extras --all-groups

# Rust build
cd rust/privateboost && cargo build && cd ../..
```

## Run tests

```bash
# Python tests
make test

# Rust tests
cd rust/privateboost && cargo test && cd ../..
```

## Lint

```bash
make lint
```

## Reproduce paper figures

```bash
make figures
```

## Project structure

```
src/privateboost/          Python implementation
  aggregator.py            Coordinates training rounds
  client.py                Data holder, computes and shares gradients
  shareholder.py           Holds and reconstructs secret shares
  messages.py              Protocol message types
  tree.py                  Tree model structures
  crypto/                  Shamir sharing, commitments, field arithmetic

rust/privateboost/         Rust implementation
  src/crypto/              Mersenne field, Shamir sharing, encoding
  src/protocol/            Client, aggregator, shareholder
  src/model/               Tree structures
  tests/                   Integration and crypto tests

paper/                     LaTeX source and figures
tests/                     Python integration tests
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
