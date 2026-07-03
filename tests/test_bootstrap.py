"""Bootstrap PR-AUC intervals: point sits inside the interval, paired diff behaves."""
from __future__ import annotations

import numpy as np

from src.evaluation.bootstrap import paired_diff, pr_auc_ci


def _labelled(seed=0, n_neg=50, n_pos=150):
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_neg + [1] * n_pos)
    separating = np.concatenate([rng.uniform(0.6, 1.0, n_neg), rng.uniform(0.0, 0.5, n_pos)])
    return y, separating, rng


def test_ci_brackets_the_point():
    y, s, _ = _labelled()
    r = pr_auc_ci(y, s, n_boot=500, seed=1)
    assert r["lo"] <= r["pr_auc"] <= r["hi"]
    assert r["pr_auc"] > 0.5


def test_paired_diff_zero_for_identical_scores():
    y, s, _ = _labelled()
    d = paired_diff(y, s, s, n_boot=500, seed=1)
    assert abs(d["diff"]) < 1e-9
    assert d["lo"] <= 0 <= d["hi"]


def test_paired_diff_detects_the_better_model():
    y, good, rng = _labelled(seed=2)
    bad = rng.random(len(y))
    d = paired_diff(y, good, bad, n_boot=800, seed=3)
    assert d["diff"] > 0
    assert d["prob_a_gt_b"] > 0.9
