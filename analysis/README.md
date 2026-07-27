# The Python project

Every Python module in the repository lives here, driven by the root
`pyproject.toml`. Two groups of work share it:

- **The paper pipeline** — `prepare_datasets.py` builds `data/`,
  `run_xgboost_baselines.py` produces the centralised baselines, and
  `generate_figures.py`, `plot_binning_analysis.py`, `binning_tables.py`,
  `fleet_table.py` and `export_source_data.py` turn the committed `results/`
  into the manuscript's figures, tables and supplementary workbooks. All of it
  runs from `just figures` and needs no Firestore.
- **Fleet telemetry** — `fetch.py`, `export_fleet.py` and `wire.py`, described
  below. Only `fetch.py` and `export_fleet.py` need the `fleet` dependency
  group; they are what turned the live deployment telemetry into the committed
  `results/fleet_*.csv` the paper's Figures 8-10 are built from.

Run everything from the repository root, not from this directory.

## Fleet telemetry

Fetch the two per-simulation metric collections from Firestore:

- **Systems** — `paperSimRoundMetrics`: one document per training round a device
  performs (on-device compute in µs, wire bytes below TLS, poll/submit latency,
  FCM wake delay). Written by the app under Firebase Auth.
- **Quality** — `paperSimTreeMetrics`: one document per tree, scored by the
  aggregator at each tree boundary against a public held-out benchmark split
  (AUC and friends over training time). Written with the aggregator's admin
  service account, which bypasses the security rules; clients are locked out of
  the collection entirely.

## Setup

    uv sync --group fleet

The `fleet` group adds google-cloud-firestore and pyarrow, which only `fetch.py`
and `export_fleet.py` need. A plain `uv sync` is enough for the paper pipeline
and keeps `just figures` from resolving grpcio.

Firestore access uses Application Default Credentials. ADC is rule-exempt, so it
reads `paperSimTreeMetrics` despite the client lock:

    gcloud auth application-default login

## Fetch data

    uv run python analysis/fetch.py

Streams both collections to `data/paperSimRoundMetrics.parquet` and
`data/paperSimTreeMetrics.parquet`. It is empty-collection safe (an unpopulated
collection yields an empty typed parquet) and prints the row count for each.

## Derive the committed CSVs

    uv run python analysis/export_fleet.py

Turns the parquet into `results/fleet_round_metrics.csv` and
`results/fleet_tree_metrics.csv`, dropping every column that could name a person
or device. Those two CSVs are the repository's durable record of the capture and
the only fleet input the manuscript's figures read — a reader can regenerate
Figures 8-10 without any Firestore access.

`data/` is git-ignored; `fetch.py` regenerates it.

## Figures

There is no separate plotting module. The manuscript's figures are the only
figures: `generate_figures.py` builds Figures 4 and 7-10 plus the metrics table,
and `plot_binning_analysis.py` builds Figure 5. A figure with no place in the
paper has no code here.

## Caveats

- **Wake latency** is device-perceived wake delay and includes server/device
  clock skew; it is not a pure transport measurement.
- **Wire bytes** are counted below rustls, so they include TLS handshake and
  HTTP/2 framing on top of the ciphertext payload.
- The aggregator holding the labelled test set discloses nothing: the split
  is public benchmark data committed to this repo, contributed by no client
  and never trained on. The privacy property — clients submit only secret
  shares; the aggregator never sees client records or labels — is untouched.
- **`breast_cancer`'s held-out split is class-skewed — do not use its AUC in a
  paper quality figure as-is.** Every dataset's train/held-out split is
  positional: the app's bundled train asset is the first 80% of
  `data/<name>.csv` and the aggregator's held-out slice is the remaining 20% of
  the same file (**except `heart_disease`**, whose held-out slice is the last
  20% of `crates/pbr-core/tests/data/heart_disease.csv` instead — the same
  rows in a different order; `data/heart_disease.csv`'s last 20% overlaps the app's
  train split and must not be used as its held-out source). That rule only
  yields a representative held-out set if the
  source rows are already shuffled. `heart_disease`, `pima_diabetes`, and
  `cdc_diabetes` satisfy this (`cdc_diabetes.csv` is shuffled by
  `prepare_datasets.py`; the other two happen to be balanced enough
  in native order — measured train-vs-held-out positive-rate deltas: 0.069,
  0.010, 0.008). `breast_cancer.csv` is in its native download order and is
  not: measured positive-rate delta is **0.18** (59.1% positive in train vs.
  77.2% in the held-out fifth). A `breast_cancer` held-out AUC is therefore a
  biased estimate, not comparable to the other three datasets', and must not
  be used for a paper quality figure until the split is fixed. Fixing it
  means shuffling consistently on both sides — the app's bundled train asset
  and the aggregator's held-out slice — since they must stay complementary
  halves of the same shuffle; it is a data-generation fix, not something a
  figure script can correct after the fact. Do not reshuffle `data/*.csv` to fix
  this without also regenerating the committed `results/`, which
  `scripts/run_experiments.sh` produced against the current (unshuffled) file
  order.

## Layout

Paper pipeline:

- `paths.py` — `RESULTS_DIR`, `FIGURES_DIR` and `SUPPLEMENTARY_DIR`, all
  resolved relative to the repository root. The latter two follow the tree:
  `manuscript/figures/` and `manuscript/supplementary/` here, `figures/` and
  `supplementary/` in the published artifact, which carries no `manuscript/`.
- `style.py` — the manuscript figure style (7pt journal column) and the
  Okabe-Ito palette, applied at import by every figure script.
- `prepare_datasets.py` — rebuilds `data/` from source, byte-identically.
- `run_xgboost_baselines.py` — the centralised baselines over the split indices
  the protocol runs export.
- `generate_figures.py` — the manuscript's plotted figures and the metrics table.
- `plot_binning_analysis.py`, `binning_tables.py` — the binning figure and its
  two tables.
- `fleet_table.py` — the LaTeX device table, printed to stdout.
- `export_source_data.py` — the per-figure supplementary workbooks.

Fleet telemetry:

- `fetch.py` — Firestore → parquet; the `*_to_frame` parsers pin each schema.
- `export_fleet.py` — parquet → the committed `results/fleet_*.csv`, dropping
  every column that could name a person or device.
- `wire.py` — reconciles measured against analytic wire cost.
