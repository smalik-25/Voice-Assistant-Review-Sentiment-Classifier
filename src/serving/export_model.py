"""Export the serving model bundle.

Fits the selected pipeline on the full curated dataset, computes the operating threshold on
out-of-fold predictions, and saves a self-contained bundle to ``models/model.joblib``. The
bundle is a plain scikit-learn pipeline plus the threshold and some metadata, so the serving
container loads it with scikit-learn alone (no MLflow, no project code).

Usage:
    python -m src.serving.export_model
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.evaluation.threshold import choose_threshold
from src.features.text import build_vectorizer

CONFIG_PATH = Path("configs/serve_model.yaml")
OUT = Path("models/model.joblib")
NEG = 0


def build_pipeline(config: dict) -> Pipeline:
    rep, m, seed = config["representation"], config["model"], int(config["seed"])
    return Pipeline([
        ("tfidf", build_vectorizer(rep["kind"], ngram_range=rep["ngram_range"],
                                   min_df=rep["min_df"], max_df=rep["max_df"])),
        ("clf", LogisticRegression(
            penalty=m["penalty"], C=m["C"], class_weight=m["class_weight"],
            solver=m["solver"], max_iter=m["max_iter"], random_state=seed)),
    ])


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seed = int(config["seed"])
    d = config["data"]
    df = pd.read_parquet(d["curated_path"])
    X, y = df[d["text_col"]], df[d["label_col"]]

    pipeline = build_pipeline(config)
    # Operating threshold on out-of-fold predictions, then fit on all data for deployment.
    oof_neg = cross_val_predict(
        pipeline, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, NEG]
    threshold = choose_threshold(y, oof_neg, target_recall=config["selection"]["target_recall"])
    pipeline.fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "target_recall": config["selection"]["target_recall"],
        "classes": pipeline.classes_.tolist(),
        "config_name": config["name"],
        "seed": seed,
        "trained_rows": int(len(df)),
        "exported_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]
    print(f"wrote {OUT} threshold={threshold:.4f} rows={len(df)} sha256[:12]={digest}")


if __name__ == "__main__":
    main()
