"""Build the drift reference from the training data and the serving model.

Computes the monitored features on the curated dataset with the fitted serving pipeline and
stores them as the reference distribution. In a real deployment the reference is the
training population; incoming request batches are then compared against it.

Usage:
    python -m src.monitoring.build_reference
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.monitoring.features import extract_features

BUNDLE = Path("models/model.joblib")
CURATED = Path("data/processed/curated.parquet")
OUT = Path("models/drift_reference.json")
MAX_SAMPLES = 2000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-col", default="verified_reviews")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundle = joblib.load(BUNDLE)
    df = pd.read_parquet(CURATED)
    texts = df[args.text_col]
    if len(texts) > MAX_SAMPLES:
        texts = texts.sample(MAX_SAMPLES, random_state=args.seed)

    features = extract_features(bundle, texts)
    reference = {
        "features": {k: np.asarray(v).tolist() for k, v in features.items()},
        "n": int(len(texts)),
        "built_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reference))
    print(f"wrote {OUT} features={list(features)} n={reference['n']}")


if __name__ == "__main__":
    main()
