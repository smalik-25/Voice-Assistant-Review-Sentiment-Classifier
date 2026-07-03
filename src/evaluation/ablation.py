"""Build the ablation table from MLflow, not by hand.

Reads every run in the ablation experiment, keeps the most recent run per variant, and
returns a table sorted by held-out PR-AUC with the majority baseline pinned on top. PR-AUC
is the ranking metric: F1 at the default 0.5 threshold understates models that operate at a
tuned threshold, so it is a column but not the sort key.

Usage:
    python -m src.evaluation.ablation
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

EXPERIMENT = "sentiment-ablation"
TRACKING_URI = "file:./mlruns"
METRICS = [
    "holdout_neg_recall",
    "holdout_neg_precision",
    "holdout_neg_f1",
    "holdout_specificity",
    "holdout_pr_auc",
    "cv_neg_recall",
]


def get_ablation_table(experiment: str = EXPERIMENT, tracking_uri: str = TRACKING_URI) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"])
    if runs.empty:
        return pd.DataFrame(columns=["variant", *METRICS])

    runs = runs.drop_duplicates(subset="params.variant", keep="first")  # newest per variant
    cols = {"params.variant": "variant", "params.framework": "framework"}
    for m in METRICS:
        cols[f"metrics.{m}"] = m
    table = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)

    # Baseline pinned on top, everything else by held-out PR-AUC (ranking quality).
    is_baseline = table["variant"].eq("baseline_dummy")
    ranked = pd.concat([
        table[is_baseline],
        table[~is_baseline].sort_values("holdout_pr_auc", ascending=False),
    ], ignore_index=True)
    return ranked


def to_markdown(table: pd.DataFrame) -> str:
    show = table.copy()
    for m in METRICS:
        if m in show.columns:
            show[m] = show[m].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    headers = list(show.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    table = get_ablation_table()
    if table.empty:
        print(f"no runs found in experiment {EXPERIMENT!r}; run some variants first")
        return
    print(to_markdown(table))


if __name__ == "__main__":
    main()
