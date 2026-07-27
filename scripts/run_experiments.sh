#!/usr/bin/env bash
# Regenerates every results/ directory from the data/ CSVs using the release
# binaries in crates/pbr-core. The protocol runs are deterministic: a
# rerun reproduces the committed CSVs byte-for-byte (timing.csv, which
# measures wall-clock, is the exception).
#
# Full regeneration takes a few hours, dominated by the CDC BRFSS sweeps.
# Pass one or more dataset names to restrict the run, e.g.:
#   scripts/run_experiments.sh heart_disease pima_diabetes
set -euo pipefail
cd "$(dirname "$0")/.."

# Where the run writes. `just reproduce-verify` points this at a scratch tree so a
# rerun can be diffed against the committed CSVs.
RESULTS_ROOT=${RESULTS_ROOT:-results}

cargo build --release -p pbr-core --bins
BENCH=target/release/benchmark
GAIN=target/release/gain_retention

DATASETS=("${@:-heart_disease breast_cancer pima_diabetes cdc_diabetes}")
# shellcheck disable=SC2128,SC2086
DATASETS=($DATASETS)

for ds in "${DATASETS[@]}"; do
  data=data/$ds.csv

  # Main configuration (15 trees, depth 3, 2-of-3) plus the gradient-round
  # dropout sweep (Fig. dropout_resilience, Table metrics).
  $BENCH --dataset "$data" --dropout-sweep --output-dir "$RESULTS_ROOT/$ds"

  # Depth sweep with and without path hiding (Fig. depth_tradeoff). The
  # depth-3 run also exports the split indices the XGBoost baselines reuse.
  for d in 1 2 3 4 5 6; do
    $BENCH --dataset "$data" --max-depth "$d" --output-dir "$RESULTS_ROOT/${ds}_depth_$d"
    $BENCH --dataset "$data" --max-depth "$d" --hide-path --output-dir "$RESULTS_ROOT/${ds}_depth_${d}_hiding"
  done

  # Statistics-phase dropout and per-shareholder message loss (Table robustness).
  $BENCH --dataset "$data" --stats-dropout-sweep --share-loss-sweep --output-dir "$RESULTS_ROOT/robustness/$ds"

  # Timing runs (per-role computation cost; timing.csv is wall-clock and
  # varies run to run — every other CSV these runs produce is deterministic).
  $BENCH --dataset "$data" --output-dir "$RESULTS_ROOT/timing/$ds"
  $BENCH --dataset "$data" --hide-path --output-dir "$RESULTS_ROOT/timing/${ds}_hiding"

  # Gain retention of binned vs exact split finding under both binning
  # methods (Tables gain_retention/bin_uniformity via analysis/binning_tables.py).
  $GAIN --dataset "$data" --bin-method gaussian --output-dir "$RESULTS_ROOT/gain_retention/${ds}_gaussian"
  $GAIN --dataset "$data" --bin-method uniform --output-dir "$RESULTS_ROOT/gain_retention/${ds}_uniform"
done

case " ${DATASETS[*]} " in
  *" cdc_diabetes "*)
    $BENCH --dataset data/cdc_diabetes.csv --n-shareholders 5 --threshold 3 \
      --output-dir "$RESULTS_ROOT/timing/cdc_diabetes_3of5"
    ;;
esac

# Centralized XGBoost baselines over the exact split indices exported above.
# Restricted to the datasets this run swept, so a narrowed run does not fail
# looking for splits it never produced.
uv run python analysis/run_xgboost_baselines.py "${DATASETS[@]}"
