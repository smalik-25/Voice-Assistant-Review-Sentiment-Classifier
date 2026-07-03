"""Run a drift check of a batch against the stored training reference.

Designed-and-implemented monitoring: there is no live production traffic here, so a batch is
supplied as a file. In a deployment this batch would be a window of logged request texts.
The check prints a JSON drift report and exits non-zero when drift is detected, so it can run
as a scheduled job with alerting wired to the exit code.

Usage:
    python -m src.monitoring.check --batch path/to/reviews.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.monitoring.drift import DEFAULT_PSI_THRESHOLD, drift_report
from src.monitoring.features import extract_features

BUNDLE = Path("models/model.joblib")
REFERENCE = Path("models/drift_reference.json")


def _load_batch(path: str, text_col: str) -> pd.Series:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in {".csv", ".tsv"}:
        df = pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")
    else:  # plain text, one review per line
        return pd.Series(p.read_text().splitlines())
    return df[text_col]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--text-col", default="verified_reviews")
    parser.add_argument("--psi-threshold", type=float, default=DEFAULT_PSI_THRESHOLD)
    parser.add_argument("--out", default=None, help="optional path to write the JSON report")
    args = parser.parse_args()

    bundle = joblib.load(BUNDLE)
    reference = json.loads(REFERENCE.read_text())["features"]
    current = extract_features(bundle, _load_batch(args.batch, args.text_col))
    current = {k: v.tolist() for k, v in current.items()}

    report = drift_report(reference, current, psi_threshold=args.psi_threshold)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    sys.exit(1 if report["drifted"] else 0)


if __name__ == "__main__":
    main()
