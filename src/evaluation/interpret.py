"""Coefficient interpretation for a linear text pipeline.

Explainability, kept separate from prediction: this reads the fitted linear model's
weights to say which words push a review toward the negative class and which toward the
positive class. Negative coefficients push toward class 0 (negative). It also reports
sparsity (the share of exactly-zero coefficients), which is where an L1 penalty earns its
place: a sparse model is a shorter, more legible explanation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _find_steps(pipeline):
    vectorizer = clf = None
    for _, step in pipeline.steps:
        if hasattr(step, "get_feature_names_out"):
            vectorizer = step
        if hasattr(step, "coef_"):
            clf = step
    if vectorizer is None or clf is None:
        raise ValueError("pipeline needs a vectorizer step and a linear classifier step")
    return vectorizer, clf


def top_coefficients(pipeline, n: int = 20) -> dict:
    """Top words driving each class, plus the model's coefficient sparsity."""
    vectorizer, clf = _find_steps(pipeline)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    coef = np.asarray(clf.coef_).ravel()
    frame = pd.DataFrame({"word": vocab, "coef": coef})
    return {
        "negative_drivers": frame.nsmallest(n, "coef").reset_index(drop=True),
        "positive_drivers": frame.nlargest(n, "coef").reset_index(drop=True),
        "sparsity": float((coef == 0).mean()),
        "n_features": int(coef.size),
    }
