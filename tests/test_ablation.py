"""The ablation table is built from MLflow and pins the baseline on top."""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402

from src.evaluation.ablation import get_ablation_table  # noqa: E402


def _log(variant, framework, pr_auc):
    with mlflow.start_run():
        mlflow.log_param("variant", variant)
        mlflow.log_param("framework", framework)
        mlflow.log_metric("holdout_pr_auc", pr_auc)
        mlflow.log_metric("holdout_neg_f1", pr_auc)


def test_baseline_on_top_and_sorted(tmp_path):
    uri = f"file:{tmp_path}/mlruns"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("t-abl")
    _log("sklearn_linear", "sklearn", 0.58)
    _log("pytorch_mlp", "pytorch", 0.52)
    _log("baseline_dummy", "sklearn", 0.09)

    table = get_ablation_table("t-abl", uri)
    order = list(table["variant"])
    assert order[0] == "baseline_dummy"          # baseline pinned on top
    rest = order[1:]
    assert rest.index("sklearn_linear") < rest.index("pytorch_mlp")  # sorted by PR-AUC desc
