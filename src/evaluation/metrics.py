"""Shared evaluation. Every variant is scored through this one function so results are
comparable.

The negative class (``feedback == 0``) is the class of interest throughout: it is the
minority, and flagging genuine negatives is the point of the task. Two metrics that get
confused are kept distinct here:

- ``neg_precision``: of the reviews flagged negative, how many really are. It answers
  "how trustworthy is a negative flag."
- ``specificity``: of the genuinely positive reviews, how many we leave unflagged. It
  answers "how often are positive reviews wrongly flagged as negative." This is the
  metric the success criterion is really about, not precision.

``pr_auc`` is average precision for the negative class, which is more informative than
ROC-AUC under heavy imbalance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

NEG = 0  # negative class: the minority and the class of interest


def evaluate(y_true, y_pred, neg_score=None) -> dict:
    """Return negative-class metrics plus a confusion matrix.

    ``neg_score`` is the predicted probability (or score) of the NEGATIVE class. When
    given, PR-AUC for the negative class is included. The confusion matrix has rows as
    true [0, 1] and columns as predicted [0, 1].
    """
    y_true = np.asarray(y_true)
    out = {
        "neg_precision": precision_score(y_true, y_pred, pos_label=NEG, zero_division=0),
        "neg_recall": recall_score(y_true, y_pred, pos_label=NEG, zero_division=0),
        "neg_f1": f1_score(y_true, y_pred, pos_label=NEG, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if neg_score is not None:
        out["pr_auc"] = average_precision_score((y_true == NEG).astype(int), neg_score)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def negative_scores(estimator, X) -> np.ndarray:
    """Predicted probability of the negative class from a fitted estimator.

    Uses ``classes_`` to find the column for label 0, so it does not assume ordering.
    """
    proba = estimator.predict_proba(X)
    neg_col = list(estimator.classes_).index(NEG)
    return proba[:, neg_col]
