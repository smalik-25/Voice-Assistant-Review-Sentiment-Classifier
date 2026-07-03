"""Serving smoke test: build a tiny bundle, load the app, hit /health and /predict."""
from __future__ import annotations

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.text import build_vectorizer
from src.serving.app import create_app


def _make_bundle(path):
    texts = ["broke terrible useless refund"] * 15 + ["love great works perfect"] * 35
    y = np.array([0] * 15 + [1] * 35)
    pipe = Pipeline([
        ("tfidf", build_vectorizer("tfidf", min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]).fit(texts, y)
    joblib.dump({"pipeline": pipe, "threshold": 0.3, "classes": [0, 1]}, path)


def test_health_and_predict(tmp_path):
    bundle_path = tmp_path / "model.joblib"
    _make_bundle(bundle_path)
    client = TestClient(create_app(str(bundle_path)))

    health = client.get("/health").json()
    assert health["model_loaded"] is True
    assert health["threshold"] == 0.3

    neg = client.post("/predict", json={"text": "it broke and is useless, want a refund"}).json()
    pos = client.post("/predict", json={"text": "i love it, works great"}).json()
    assert neg["feedback"] == 0 and neg["label"] == "negative"
    assert pos["feedback"] == 1 and pos["label"] == "positive"
    assert 0.0 <= neg["negative_probability"] <= 1.0


def test_predict_without_model_returns_503(tmp_path):
    client = TestClient(create_app(str(tmp_path / "missing.joblib")))
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/predict", json={"text": "hello"}).status_code == 503
