"""Tests for the shared evaluation and CV protocol.

Self-contained: they synthesize a small imbalanced text dataset rather than depend on
the gitignored real data, so they run in CI. The point is to prove that different
estimators are scored through the same function on the same folds, and that the metric
definitions (specificity vs precision, negative class as class of interest) are correct.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.cv import cross_validate_negative, make_holdout
from src.evaluation.metrics import evaluate
from src.features.text import build_vectorizer

POS_WORDS = ["love", "great", "perfect", "excellent", "amazing", "works", "wonderful"]
NEG_WORDS = ["broke", "terrible", "useless", "refund", "disappointed", "awful", "stopped"]


def make_synthetic(n=400, neg_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []
    for _ in range(n):
        is_neg = rng.random() < neg_rate
        vocab = NEG_WORDS if is_neg else POS_WORDS
        words = rng.choice(vocab, size=rng.integers(3, 7))
        texts.append(" ".join(words))
        labels.append(0 if is_neg else 1)
    return np.array(texts, dtype=object), np.array(labels)


def logreg_pipeline():
    return Pipeline([
        ("tfidf", build_vectorizer("tfidf", min_df=1)),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", class_weight="balanced", max_iter=1000,
        )),
    ])


def test_evaluate_metric_definitions():
    # Hand-checked example: negative class (0) is the class of interest.
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    m = evaluate(y_true, y_pred, neg_score=[0.9, 0.2, 0.1, 0.1, 0.6])
    assert m["neg_recall"] == 0.5          # caught 1 of 2 true negatives
    assert m["neg_precision"] == 0.5       # 1 of 2 negative flags correct
    assert abs(m["specificity"] - 2 / 3) < 1e-9  # 2 of 3 positives left unflagged
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert 0.0 <= m["pr_auc"] <= 1.0


def test_same_function_scores_both_estimators():
    X, y = make_synthetic()
    keys = {"neg_precision", "neg_recall", "neg_f1", "specificity", "pr_auc"}

    logreg = cross_validate_negative(logreg_pipeline(), X, y, n_splits=3, n_repeats=2, seed=42)
    dummy = cross_validate_negative(
        DummyClassifier(strategy="most_frequent"), X, y, n_splits=3, n_repeats=2, seed=42
    )

    # Both estimators return the same metric surface: comparable by construction.
    assert keys.issubset(logreg["mean"])
    assert keys.issubset(dummy["mean"])
    assert logreg["n_folds"] == dummy["n_folds"] == 6

    # Baseline-relative sanity: the majority baseline never catches a negative.
    assert dummy["mean"]["neg_recall"] == 0.0
    assert logreg["mean"]["neg_recall"] > dummy["mean"]["neg_recall"]
    for v in logreg["mean"].values():
        assert 0.0 <= v <= 1.0


def test_make_holdout_preserves_prevalence():
    X, y = make_synthetic(n=600, neg_rate=0.1, seed=1)
    X_tr, X_te, y_tr, y_te = make_holdout(X, y, test_size=0.2, seed=42)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    assert len(X_te) == 120
