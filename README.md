# PrivateBoost

Privacy-preserving federated gradient boosting for cross-device medical data.

Each device holds a slice of the training set and never sends a record
anywhere. It computes gradient and Hessian histograms locally, splits
them into Shamir secret shares over the Mersenne prime field (2^61 - 1), and
sends one share to each of three shareholders. The shareholders sum the shares
they receive and forward partial sums to an aggregator, which reconstructs only
the aggregate by Lagrange interpolation and decides the next split. The
aggregator holds no shares and any single shareholder holds one point of a
threshold scheme, so no party sees an individual contribution.

This repository is the deployable implementation: one aggregator and three
shareholders as separate gRPC processes, plus a Flutter app that participates
from real Android and iOS devices under real background-scheduling constraints.

## Prerequisites

`devenv.nix` pins the environment (Rust, Python + uv, protoc, TeX Live,
gitleaks). With [devenv](https://devenv.sh) and direnv:

```bash
direnv allow          # or: devenv shell
```

Android builds additionally need the rustup stable toolchain ahead of the nix
one on `PATH`, because the nix cargo carries no Android std. iOS builds need a
Mac carrying its own toolchain: Xcode, CocoaPods, the `aarch64-apple-ios` Rust
target, and `protoc` on `PATH`, since the Xcode build compiles the Rust library
through cargokit outside any devenv shell.

## Quickstart

```bash
just build            # cargo build --workspace
just test             # every workspace test
just figures          # regenerate the figures and tables from results/
```

## Reproducing the results

**Deterministic offline.** The protocol simulations, the XGBoost baselines, the
figures and the tables. Anyone can regenerate these from `data/` and get the
committed bytes back.

```bash
just reproduce            # everything derivable from data/, hours
just reproduce-verify     # regenerate into a scratch tree and diff the CSVs
just figures              # figures and tables from committed results/, minutes
```

`just reproduce-verify` reruns the experiments into a temporary directory and
compares every git-tracked CSV byte-for-byte, skipping only `timing.csv`, which
records wall-clock. Pass dataset names to widen it beyond the default
`heart_disease`.

`data/` holds the four prepared datasets and is the canonical input: every result
in `results/` derives from exactly these bytes. The raw sources they are derived
from are committed beside them (`heart_disease_raw.csv`, `pima_raw.csv`,
`cdc_raw.csv`; Breast Cancer needs none, since it ships inside scikit-learn), so
`analysis/prepare_datasets.py` rebuilds all four **offline** and all four come back
byte-identical. The inputs are verifiable rather than merely archived:

```bash
rm data/{heart_disease,breast_cancer,pima_diabetes,cdc_diabetes}.csv
uv run python analysis/prepare_datasets.py
git status --short data      # clean means the rebuild matched
```

Keeping the raw files is what makes that check durable: the download URLs are a
UCI archive path, one GitHub repository, and the UCI repository API, and an
artifact that needs three third-party endpoints alive to rebuild its own inputs
stops working eventually. Delete a raw file and the script fetches it again.

Row order matters as much as row content, because it fixes the stratified
train/test splits every benchmark derives. The CDC balancing and shuffle therefore
index with numpy's `RandomState` directly, whose stream numpy policy freezes,
instead of passing one instance through two pandas calls.

**Local cluster.** `just e2e` brings up a real four-process TLS cluster and
gates on AUC. `just e2e-wire-cost` measures per-client bytes below rustls
across a 56-config grid into `results/wire_measured.csv` (~36 min, a fresh
cluster per configuration).

`wire_measured.csv` is the one committed artifact that does not reproduce
byte-for-byte. Re-measuring reproduces the grid shape exactly, `n_rounds`
included, but the byte counters move by up to 0.9% per cell: they sit below
rustls, so TLS record boundaries and HTTP/2 window updates land differently
between runs. Over the totals the paper draws on, the spread is 0.1% on average
and 0.2% at worst. `just reproduce-verify` does not compare it, because
`scripts/run_experiments.sh` does not produce it.

**Field measurements.** `results/fleet_round_metrics.csv` and
`results/fleet_tree_metrics.csv` come from four consumer devices (Pixel 9 Pro,
Galaxy Tab A8, Lenovo Tab M8, iPhone 13 Pro), each holding a quarter of the
training split, training three sessions against a single cloud VM on 2026-07-24.
Wakes came from silent FCM data messages at the 15-minute per-account floor plus
platform background work, with screens off and no foreground polling.

Reproducing these needs the device fleet, so the committed CSVs are the primary
record. `analysis/export_fleet.py` derives them from the fetched telemetry and
is where the per-person and per-device identifiers (`uid`, `deviceId`,
`appVersion`) are dropped, so it also documents what the published data
contains.

## Layout

- `crates/pbr-core` — the protocol: Shamir sharing, the Mersenne field, the
  dense-histogram client and aggregator, model and metrics
- `crates/pbr-proto` — tonic/prost codegen and conversions
- `crates/pbr-server` — one binary, two roles (`--role shareholder | aggregator`)
- `crates/pbr-client` — client protocol library and CLI driver
- `crates/pbr-admin` — admin-plane CLI, including session creation
- `crates/pbr-e2e` — the four-process end-to-end and the wire-cost sweep
- `proto/` — the wire contract
- `app/` — the Flutter app; `app/rust/` bridges it to `pbr-client`
- `deploy/` — loopback, emulator and single-VM cluster configurations; its
  `README.md` covers how to bring a cluster up
- `scripts/` — the experiment and publish shell drivers
- `analysis/` — the Python project: dataset preparation, XGBoost baselines,
  figure and table generation, fleet telemetry export
- `data/`, `results/`, `figures/` — inputs, outputs, figures

## License

Apache License 2.0. See `LICENSE`.
