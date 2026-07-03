"""Bootstrap confidence intervals for held-out PR-AUC (negative class).

The test set has only about 40 negative reviews, so a single PR-AUC number is noisy and a
small gap between two models might be nothing. This resamples the test rows with replacement
to put an interval on each model's PR-AUC, and does a paired resample (the same rows for both
models) to compare two models without the comparison being thrown off by which test draw you
happened to get.

Every variant saves its held-out predictions through the harness, so the comparison runs on
identical test rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

NEG = 0
PRED_DIR = Path("models/predictions")


def _ap(y_bin: np.ndarray, score: np.ndarray) -> float:
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return np.nan  # a resample with only one class has no defined PR-AUC
    return average_precision_score(y_bin, score)


def pr_auc_ci(y_true, neg_score, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    y = (np.asarray(y_true) == NEG).astype(int)
    s = np.asarray(neg_score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = _ap(y[idx], s[idx])
        if not np.isnan(v):
            boot.append(v)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"pr_auc": float(_ap(y, s)), "lo": float(lo), "hi": float(hi), "n": int(n), "n_boot": len(boot)}


def paired_diff(y_true, score_a, score_b, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Paired bootstrap of PR-AUC(a) - PR-AUC(b) on the same resampled rows."""
    y = (np.asarray(y_true) == NEG).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        va, vb = _ap(y[idx], a[idx]), _ap(y[idx], b[idx])
        if not (np.isnan(va) or np.isnan(vb)):
            diffs.append(va - vb)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": float(_ap(y, a) - _ap(y, b)),
        "lo": float(lo),
        "hi": float(hi),
        "prob_a_gt_b": float((diffs > 0).mean()),
        "n_boot": len(diffs),
    }


def save_predictions(variant: str, y_true, neg_score, out_dir: Path = PRED_DIR) -> str:
    """Persist a variant's held-out predictions so the bootstrap can compare on shared rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant}.parquet"
    pd.DataFrame({"y_true": np.asarray(y_true), "neg_score": np.asarray(neg_score)}).to_parquet(path, index=False)
    return str(path)
