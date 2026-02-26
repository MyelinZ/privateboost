# Docker & Simulation Design for Privateboost

## Overview

Docker containers for shareholder and aggregator gRPC servers, plus a host-side simulation script that runs the full federated XGBoost protocol against them.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Container scope | Shareholder + Aggregator only | Clients run on host (or mobile in production). No client container needed. |
| Dockerfile | Single image, role via command | One package to install, compose `command` selects shareholder or aggregator. |
| Shareholder identity | No x_coord config | Shareholders are pure storage engines. Aggregator assigns x_coords from its ordered address list. |
| Configuration | Environment variables | Works for local compose and production deployment. Sensible defaults for all training params. |
| Simulation | Host-side Python script | `python -m privateboost.grpc.simulate` against Dockerized servers. Built-in Heart Disease dataset. |
| Scaling | Named services in compose | Explicit shareholder-1, shareholder-2, shareholder-3. Add more by adding service blocks and updating aggregator SHAREHOLDERS env. |

---

## x_coord Assignment Change

The original gRPC design had shareholders self-reporting their x_coord. This is a security and configuration risk — a shareholder should not control its own Shamir evaluation point.

**New design:** Shareholders don't know their x_coord. The aggregator assigns x_coords (1, 2, 3, ...) based on the order of addresses in its `SHAREHOLDERS` config. When the aggregator calls a shareholder and gets a sum back, it attaches the x_coord from its own mapping.

**Proto changes:**
- `GetStatsSumResponse` and `GetGradientsSumResponse` drop the `x_coord` field — they return only `NdArray sum`
- `ShareholderServicer` constructor drops `x_coord` parameter

**JoinSession is unchanged** — the aggregator still returns `ShareholderInfo` with addresses and x_coords to clients. Clients use this to create the right shares.

---

## Container Configuration

### Shareholder

Entrypoint: `python -m privateboost.grpc.serve shareholder`

| Env var | Default | Description |
|---|---|---|
| `PORT` | `50051` | gRPC listen port |
| `MIN_CLIENTS` | `10` | Security minimum for releasing sums |

### Aggregator

Entrypoint: `python -m privateboost.grpc.serve aggregator`

| Env var | Default | Description |
|---|---|---|
| `PORT` | `50052` | gRPC listen port |
| `SHAREHOLDERS` | (required) | Comma-separated `host:port` list |
| `THRESHOLD` | `2` | m in m-of-n Shamir |
| `N_BINS` | `10` | Histogram bins |
| `N_TREES` | `15` | Number of boosting rounds |
| `MAX_DEPTH` | `3` | Max tree depth |
| `LEARNING_RATE` | `0.15` | Boosting learning rate |
| `LAMBDA_REG` | `2.0` | L2 regularization |
| `MIN_CLIENTS` | `10` | Aggregator participation target |
| `LOSS` | `squared` | Loss function |
| `FEATURES` | (required) | Comma-separated feature names |
| `TARGET_COLUMN` | `target` | Target column name |

### Simulation Script

Run on host: `uv run python -m privateboost.grpc.simulate`

| Env var | Default | Description |
|---|---|---|
| `AGGREGATOR` | `localhost:50052` | Aggregator address |
| `SESSION_ID` | random UUID | Session identifier |

---

## Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install .

ENTRYPOINT ["python", "-m", "privateboost.grpc.serve"]
```

---

## docker-compose.yml

```yaml
services:
  shareholder-1:
    build: .
    command: shareholder
    environment:
      PORT: "50051"
      MIN_CLIENTS: "10"

  shareholder-2:
    build: .
    command: shareholder
    environment:
      PORT: "50051"
      MIN_CLIENTS: "10"

  shareholder-3:
    build: .
    command: shareholder
    environment:
      PORT: "50051"
      MIN_CLIENTS: "10"

  aggregator:
    build: .
    command: aggregator
    ports:
      - "50052:50052"
    environment:
      PORT: "50052"
      SHAREHOLDERS: "shareholder-1:50051,shareholder-2:50051,shareholder-3:50051"
      THRESHOLD: "2"
      N_BINS: "10"
      N_TREES: "15"
      MAX_DEPTH: "3"
      LEARNING_RATE: "0.15"
      LAMBDA_REG: "2.0"
      MIN_CLIENTS: "10"
      LOSS: "squared"
      FEATURES: "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal"
      TARGET_COLUMN: "target"
    depends_on:
      - shareholder-1
      - shareholder-2
      - shareholder-3
```

The aggregator port is published for the host-side simulation script to connect to.

---

## Simulation Flow

```
1. Connect to aggregator at AGGREGATOR address
2. Call JoinSession(session_id) -> shareholders[], threshold, config
3. Download Heart Disease dataset (cached)
4. 80/20 train/test split, seed 42
5. Create one NetworkClient per training row
6. All clients submit_stats() to shareholders
7. Poll GetRoundState() until COLLECTING_GRADIENTS
8. Loop:
     GetTrainingState() -> model, splits, bins
     All clients submit_gradients()
     Poll until round/depth advances
9. On TRAINING_COMPLETE: GetModel() -> final model
10. Evaluate on test set, print accuracy
```

---

## Security Properties

- Shareholders have zero identity configuration — no x_coord to misconfigure or tamper with
- Aggregator controls x_coord assignment, consistent with its role as coordinator
- min_clients enforced independently per shareholder at deploy time
- TLS can be enabled via additional cert env vars (future work)
- Same trust model works in Docker and production deployment
