#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
LOG_DIR="$ROOT/logs"
PID_FILE="$LOG_DIR/pids"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Services may not be running."
    exit 0
fi

echo "Stopping services..."
while read -r pid; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && echo "  killed pid $pid" || true
    fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "All services stopped."
