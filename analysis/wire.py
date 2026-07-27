"""Typed loader for `results/wire_measured.csv`: the measured per-client wire
bytes, counted below rustls over the real TLS/HTTP2 transport (see
`crates/pbr-e2e/tests/wire_grid.rs`).

The counters are client-side and never observe shareholder->aggregator
traffic. They are the session totals a run's `WireRun`/`run_collecting`
snapshot at start and end, so unlike a per-round byte count they are a
complete, summable per-session payload cost.
"""

import pandas as pd

# Column -> coercion kind, mirroring fetch.py's typed-frame idiom.
WIRE_SCHEMA: dict[str, str] = {
    "dataset": "str",
    "depth": "Int64",
    "threshold": "Int64",
    "hide_path": "bool",
    "total_tx": "Int64",
    "total_rx": "Int64",
    "submit_tx": "Int64",
    "submit_rx": "Int64",
    "n_rounds": "Int64",
}


def load_wire_measured(path) -> pd.DataFrame:
    """Typed `results/wire_measured.csv` frame: `hide_path` as a real bool
    and every byte/round count as nullable `Int64`."""
    df = pd.read_csv(path, dtype=str)
    for col, kind in WIRE_SCHEMA.items():
        s = df[col]
        if kind == "Int64":
            df[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
        elif kind == "bool":
            df[col] = s.str.strip().str.lower().map({"true": True, "false": False})
        else:  # str
            df[col] = s
    return df
