"""Cross-validation protocol.

The rare class makes any single split noisy, so selection runs on repeated stratified
k-fold and reports mean and standard deviation across folds. A separate held-out test
set, produced by ``make_holdout`` and never touched during selection, is the final
check.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.evaluation.metrics import evaluate, negative_scores


def make_holdout(X, y, test_size: float = 0.20, seed: int = 42):
    """Stratified train/test split. Stratifying preserves the rare-class prevalence
    (about 8.6% negative) in both parts, so the held-out check is not distorted by an
    unlucky draw."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def cross_validate_negative(
    estimator,
    X,
    y,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> dict:
    """Score an estimator with repeated stratified k-fold through the shared eval.

    A fresh clone is fit on each training fold, so no state leaks between folds. Returns
    per-fold scalar metrics plus their mean and standard deviation. PR-AUC uses the
    predicted probability of the negative class when the estimator exposes
    ``predict_proba``.
    """
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    per_fold: list[dict] = []
    for train_idx, val_idx in cv.split(X, y):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        neg_score = negative_scores(model, X[val_idx]) if hasattr(model, "predict_proba") else None
        per_fold.append(evaluate(y[val_idx], y_pred, neg_score=neg_score))

    scalar_keys = [k for k, v in per_fold[0].items() if not isinstance(v, list)]
    mean = {k: float(np.mean([f[k] for f in per_fold])) for k in scalar_keys}
    std = {k: float(np.std([f[k] for f in per_fold])) for k in scalar_keys}
    return {"per_fold": per_fold, "mean": mean, "std": std, "n_folds": len(per_fold)}
