"""Drift detection: PSI behaves, the report flags a shifted feature, features extract."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.text import build_vectorizer
from src.monitoring.drift import drift_report, psi
from src.monitoring.features import extract_features


def test_psi_low_for_same_high_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    same = rng.normal(0, 1, 2000)
    shifted = rng.normal(1.5, 1, 2000)
    assert psi(ref, same) < 0.1
    assert psi(ref, shifted) >= 0.2


def test_drift_report_flags_shifted_feature():
    rng = np.random.default_rng(1)
    ref = {"a": rng.normal(0, 1, 1500), "b": rng.normal(0, 1, 1500)}
    similar = {"a": rng.normal(0, 1, 1500), "b": rng.normal(0, 1, 1500)}
    shifted = {"a": rng.normal(0, 1, 1500), "b": rng.normal(2, 1, 1500)}
    assert drift_report(ref, similar)["drifted"] is False
    report = drift_report(ref, shifted)
    assert report["drifted"] is True
    assert report["features"]["b"]["drifted"] is True


def test_extract_features_keys_and_oov():
    texts = ["love it works great"] * 20 + ["broke terrible useless refund"] * 10
    y = [1] * 20 + [0] * 10
    pipe = Pipeline([
        ("tfidf", build_vectorizer("tfidf", min_df=1)),
        ("clf", LogisticRegression(max_iter=500)),
    ]).fit(texts, y)
    feats = extract_features({"pipeline": pipe}, ["love great", "zzqq foobar neverseen"])
    assert set(feats) == {"negative_score", "review_length", "oov_rate"}
    assert feats["oov_rate"][1] > 0.0
