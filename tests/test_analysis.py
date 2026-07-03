"""Tests for the Phase 4 analysis helpers: interpretation and fairness separation."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.fairness import separation_by_group
from src.evaluation.interpret import top_coefficients
from src.features.text import build_vectorizer


def test_top_coefficients_surface_negative_words():
    texts = ["broke terrible useless"] * 12 + ["love great perfect"] * 28
    y = [0] * 12 + [1] * 28
    pipe = Pipeline([
        ("vec", build_vectorizer("tfidf", min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(texts, y)

    result = top_coefficients(pipe, n=3)
    assert 0.0 <= result["sparsity"] <= 1.0
    assert set(result["negative_drivers"]["word"]) & {"broke", "terrible", "useless"}


def test_separation_reports_group_spread():
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    pred = [0, 0, 1, 1, 1, 1, 1, 1]           # group A perfect on negatives, group B misses them
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    table, spread = separation_by_group(y, pred, groups, min_negatives=2)

    by = table.set_index("variation")
    assert by.loc["A", "neg_recall"] == 1.0
    assert by.loc["B", "neg_recall"] == 0.0
    assert spread["neg_recall_spread"] == 1.0
