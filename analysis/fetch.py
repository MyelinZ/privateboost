"""Stream both metric collections from Firestore into local parquet files.

Access is via Application Default Credentials
(`gcloud auth application-default login`); ADC bypasses the security rules, so
it can read `paperSimTreeMetrics` even though every client is locked out of it.
Re-run any time to refresh; the plots only read the parquet.

The two `*_to_frame(list_of_dicts)` parsers pin each collection's schema and
dtypes independently of Firestore, so they are unit-testable and the plots can
run against synthetic rows before any live data exists.
"""

from pathlib import Path

import pandas as pd

PROJECT = "pboost-test-12345"

ROUND_COLLECTION = "paperSimRoundMetrics"
TREE_COLLECTION = "paperSimTreeMetrics"

DATA = Path(__file__).resolve().parent / "data"
ROUND_OUT = DATA / "paperSimRoundMetrics.parquet"
TREE_OUT = DATA / "paperSimTreeMetrics.parquet"

# Column -> coercion kind. The order also pins the schema: a document missing a
# field lands as NA/NaN in that typed column rather than dropping out. `Int64`
# is the nullable integer dtype (treeIdx/depth/wakeLatencyMs are absent on
# whole classes of rows: stats rounds carry no tree, only push rounds carry a
# wake latency).
ROUND_SCHEMA: dict[str, str] = {
    "ts": "datetime",
    "uid": "str",
    "deviceId": "str",
    "deviceModel": "str",
    "appVersion": "str",
    "sessionId": "str",
    "triggerSource": "str",
    "batchId": "Int64",
    "batchCount": "Int64",
    "datasetId": "str",
    "roundId": "Int64",
    "nRecords": "Int64",
    "treeIdx": "Int64",
    "depth": "Int64",
    "roundKind": "str",
    "outcome": "str",
    "lastError": "str",
    "nPeersAttempted": "Int64",
    "nPeersAccepted": "Int64",
    "networkType": "str",
    "batteryState": "str",
    "batteryLevel": "Int64",
    "wallMs": "Int64",
    "clientTsMs": "Int64",
    "pollUs": "Int64",
    "computeUs": "Int64",
    "submitUs": "Int64",
    "txBytes": "Int64",
    "rxBytes": "Int64",
    "rssBytes": "Int64",
    "wakeLatencyMs": "Int64",
}

TREE_SCHEMA: dict[str, str] = {
    "ts": "datetime",
    "sessionId": "str",
    "treeIdx": "Int64",
    "auc": "float",
    "accuracy": "float",
    "precision": "float",
    "recall": "float",
    "f1": "float",
    "logloss": "float",
    "nTest": "Int64",
    "thresholdUsed": "float",
}


def _frame(rows: list[dict], schema: dict[str, str]) -> pd.DataFrame:
    """Build a typed frame with exactly ``schema``'s columns, coercing each to
    its declared dtype. Safe on an empty ``rows`` list (yields an empty typed
    frame), which keeps the fetch valid before either collection has data."""
    fields = list(schema)
    df = pd.DataFrame([{f: r.get(f) for f in fields} for r in rows], columns=fields)
    for col, kind in schema.items():
        s = df[col]
        if kind == "datetime":
            df[col] = pd.to_datetime(s, utc=True, errors="coerce")
        elif kind == "Int64":
            df[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
        elif kind == "float":
            df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:  # str
            df[col] = s.where(s.notna(), None).astype("object")
    return df


def round_metrics_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Parse `paperSimRoundMetrics` documents (the app-written systems axis) into a typed frame."""
    return _frame(rows, ROUND_SCHEMA)


def tree_metrics_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Parse `paperSimTreeMetrics` documents (the aggregator-written quality axis) into a typed frame."""
    return _frame(rows, TREE_SCHEMA)


def _stream(client, collection: str) -> list[dict]:
    return [d.to_dict() for d in client.collection(collection).stream()]


def _fetch(collection: str, to_frame, out: Path) -> pd.DataFrame:
    from google.cloud import firestore

    client = firestore.Client(project=PROJECT)
    df = to_frame(_stream(client, collection))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"fetched {len(df)} docs from {collection} -> {out}")
    return df


def fetch_round_metrics(out: Path = ROUND_OUT) -> pd.DataFrame:
    return _fetch(ROUND_COLLECTION, round_metrics_to_frame, out)


def fetch_tree_metrics(out: Path = TREE_OUT) -> pd.DataFrame:
    return _fetch(TREE_COLLECTION, tree_metrics_to_frame, out)


def fetch_all() -> None:
    fetch_round_metrics()
    fetch_tree_metrics()


if __name__ == "__main__":
    fetch_all()
