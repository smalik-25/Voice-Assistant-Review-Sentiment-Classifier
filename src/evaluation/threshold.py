"""Pick the operating point (decision threshold) for the negative class.

Model selection is done on PR-AUC. This is the separate second decision: given the chosen
model's scores, find the threshold that meets a negative-class recall target while giving
up as little precision as possible. Concretely, the highest threshold whose negative recall
still clears the target, and the precision and specificity that point costs.

Scores here are the predicted probability of the NEGATIVE class; predict negative when that
probability is at or above the threshold.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.evaluation.metrics import evaluate

NEG = 0


def predict_negative(neg_score, threshold: float):
    return np.where(np.asarray(neg_score) >= threshold, NEG, 1)


def choose_threshold(y_true, neg_score, target_recall: float = 0.80) -> float:
    """Highest threshold whose negative-class recall is at least ``target_recall``.

    Higher thresholds mean fewer negative flags, so more precision but less recall. Among
    the thresholds that still meet the recall floor, the largest is the best-precision one.
    If the target is unreachable, returns the lowest threshold (maximum recall).
    """
    y = (np.asarray(y_true) == NEG).astype(int)
    precision, recall, thresholds = precision_recall_curve(y, np.asarray(neg_score))
    # precision/recall have one extra trailing point with no threshold; align on thresholds.
    recall = recall[:-1]
    feasible = np.where(recall >= target_recall)[0]
    if len(feasible) == 0:
        return float(thresholds.min())
    return float(thresholds[feasible].max())


def evaluate_at_threshold(y_true, neg_score, threshold: float) -> dict:
    """Full negative-class metrics when predicting at the given threshold."""
    return evaluate(y_true, predict_negative(neg_score, threshold), neg_score=neg_score)


def sweep(y_true, neg_score, thresholds=None) -> list[dict]:
    """A table of negative-class metrics across thresholds, for plotting the trade-off."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    out = []
    for t in thresholds:
        m = evaluate_at_threshold(y_true, neg_score, float(t))
        out.append({"threshold": float(t), "neg_recall": m["neg_recall"],
                    "neg_precision": m["neg_precision"], "specificity": m["specificity"]})
    return out
