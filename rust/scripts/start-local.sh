#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
BIN="$ROOT/target/release"
LOG_DIR="$ROOT/logs"
PID_FILE="$LOG_DIR/pids"

# Accept optional YAML config to read training params from
CONFIG="${1:-}"

if [ -n "$CONFIG" ] && [ -f "$CONFIG" ]; then
    # Extract training params from YAML (simple grep, no yq dependency)
    yaml_val() { grep "^  $1:" "$CONFIG" | head -1 | awk '{print $2}'; }
    N_TREES="${N_TREES:-$(yaml_val n_trees)}"
    MAX_DEPTH="${MAX_DEPTH:-$(yaml_val max_depth)}"
    LEARNING_RATE="${LEARNING_RATE:-$(yaml_val learning_rate)}"
    LAMBDA_REG="${LAMBDA_REG:-$(yaml_val lambda_reg)}"
    N_BINS="${N_BINS:-$(yaml_val n_bins)}"
    THRESHOLD="${THRESHOLD:-$(yaml_val threshold)}"
    LOSS="${LOSS:-$(yaml_val loss)}"
    SCALE="${SCALE:-$(yaml_val scale)}"
fi

# Defaults if not set via config or env
N_TREES="${N_TREES:-15}"
MAX_DEPTH="${MAX_DEPTH:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.15}"
LAMBDA_REG="${LAMBDA_REG:-2.0}"
N_BINS="${N_BINS:-10}"
THRESHOLD="${THRESHOLD:-2}"

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
    echo "Services may already be running (found $PID_FILE)."
    echo "Run scripts/stop-local.sh first."
    exit 1
fi

# Build if needed
if [ ! -f "$BIN/coordinator" ] || [ ! -f "$BIN/shareholder" ] || [ ! -f "$BIN/aggregator" ]; then
    echo "Building release binaries..."
    (cd "$ROOT" && cargo build --release)
fi

wait_for_port() {
    local port=$1 tries=0
    while ! ss -tlnp 2>/dev/null | grep -q ":${port} " && [ $tries -lt 30 ]; do
        sleep 0.1
        tries=$((tries + 1))
    done
    if [ $tries -ge 30 ]; then
        echo "  WARNING: port $port not ready after 3s — check logs/$2.log"
    fi
}

echo "Starting services (n_trees=$N_TREES, max_depth=$MAX_DEPTH, lr=$LEARNING_RATE)..."
> "$PID_FILE"

# Coordinator
PORT=50053 RUST_LOG=info "$BIN/coordinator" \
    > "$LOG_DIR/coordinator.log" 2>&1 &
echo $! >> "$PID_FILE"
echo "  coordinator    pid=$!  port=50053"
wait_for_port 50053 coordinator

# Shareholders
for i in 1 2 3; do
    port=$((50060 + i))
    PORT=$port MIN_CLIENTS=10 EXPECTED_AGGREGATORS=3 RUST_LOG=info "$BIN/shareholder" \
        > "$LOG_DIR/shareholder-$i.log" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  shareholder-$i  pid=$!  port=$port"
    wait_for_port $port "shareholder-$i"
done

# Aggregators
for i in 1 2 3; do
    AGGREGATOR_ID=$i \
    SHAREHOLDERS=localhost:50061,localhost:50062,localhost:50063 \
    THRESHOLD="$THRESHOLD" N_BINS="$N_BINS" N_TREES="$N_TREES" MAX_DEPTH="$MAX_DEPTH" \
    LEARNING_RATE="$LEARNING_RATE" LAMBDA_REG="$LAMBDA_REG" MIN_CLIENTS=10 \
    RUN_ID=${RUN_ID:-default} RUST_LOG=info \
    "$BIN/aggregator" \
        > "$LOG_DIR/aggregator-$i.log" 2>&1 &
    echo $! >> "$PID_FILE"
    echo "  aggregator-$i   pid=$!"
done

echo ""
echo "All services running. Logs in $LOG_DIR/"
echo "Stop with: scripts/stop-local.sh"
echo ""
echo "Test with:"
echo "  target/release/pbctl list"
echo "  target/release/pbctl submit --config examples/heart-disease.yaml --run-id default"
