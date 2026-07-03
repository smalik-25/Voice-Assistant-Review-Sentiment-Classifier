"""Operating-point selection: meet the recall floor at the best available precision."""
from __future__ import annotations

import numpy as np

from src.evaluation.threshold import choose_threshold, evaluate_at_threshold

# Negatives (label 0) score higher than positives, with a clean separation.
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
NEG_SCORE = np.array([0.90, 0.80, 0.70, 0.60, 0.55, 0.50, 0.40, 0.30, 0.20, 0.10])


def test_threshold_meets_recall_floor():
    t = choose_threshold(Y, NEG_SCORE, target_recall=0.80)
    m = evaluate_at_threshold(Y, NEG_SCORE, t)
    assert m["neg_recall"] >= 0.80


def test_lower_target_allows_higher_threshold():
    t_high_recall = choose_threshold(Y, NEG_SCORE, target_recall=0.80)
    t_low_recall = choose_threshold(Y, NEG_SCORE, target_recall=0.40)
    # Requiring less recall lets us keep a stricter (higher) threshold.
    assert t_low_recall >= t_high_recall


def test_unreachable_target_falls_back_to_max_recall():
    # Target above what any threshold yields on this data returns the loosest threshold.
    t = choose_threshold(Y, NEG_SCORE, target_recall=1.01)
    m = evaluate_at_threshold(Y, NEG_SCORE, t)
    assert m["neg_recall"] == 1.0
