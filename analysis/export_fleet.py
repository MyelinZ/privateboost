"""Derive the committed fleet CSVs in `results/` from fetch.py's parquet.

The parquet under `analysis/data/` is git-ignored and re-fetchable; the two
CSVs written here are the repository's durable record of the fleet capture
and the direct inputs of `generate_figures.py`'s fleet figures and
`fleet_table.py`. Tree metrics carry no dataset id of their own
(the aggregator writes only `sessionId`), so it is joined from the round
frame's observed `(sessionId, datasetId)` pairs. Identifier columns that
could name a person or device (`uid`, `deviceId`, `appVersion`) never reach
the CSVs; `deviceModel` is the analysis key.
"""

from pathlib import Path

import pandas as pd

from fetch import ROUND_OUT, TREE_OUT

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

DROP_COLUMNS = ["uid", "deviceId", "appVersion"]

# The overnight 2026-07-24/25 capture's three session ids (heart_disease,
# breast_cancer, pima_diabetes). Both Firestore collections also hold an
# older capture; this pins which sessionIds the committed CSVs cover.
FLEET_SESSIONS = frozenset(
    {
        "b379008f-f450-4987-bba4-51517776290a",
        "8290562d-2ac2-4701-a08f-14f98f73dfa7",
        "5293ac2e-1915-4c3b-8f9d-5d7a5ccf62d1",
    }
)


def export_fleet(
    round_parquet: Path, tree_parquet: Path, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounds = pd.read_parquet(round_parquet)
    trees = pd.read_parquet(tree_parquet)
    rounds = rounds[rounds["sessionId"].isin(FLEET_SESSIONS)]
    trees = trees[trees["sessionId"].isin(FLEET_SESSIONS)]

    session_dataset = (
        rounds.loc[rounds["datasetId"].notna(), ["sessionId", "datasetId"]]
        .drop_duplicates()
    )
    tree_df = trees.merge(session_dataset, on="sessionId", how="left").sort_values("ts")
    round_df = rounds.drop(columns=DROP_COLUMNS).sort_values("ts")

    out_dir.mkdir(parents=True, exist_ok=True)
    tree_df.to_csv(out_dir / "fleet_tree_metrics.csv", index=False)
    round_df.to_csv(out_dir / "fleet_round_metrics.csv", index=False)
    return tree_df, round_df


if __name__ == "__main__":
    tree_df, round_df = export_fleet(ROUND_OUT, TREE_OUT, RESULTS)
    print(
        f"wrote {len(tree_df)} tree rows, {len(round_df)} round rows to {RESULTS}"
    )
