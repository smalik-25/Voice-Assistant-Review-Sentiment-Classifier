"""Shared text representations for the vector-input model family.

Two representations, named as the class named them:
- ``tfidf``: term frequency inverse document frequency.
- ``bow``: bag-of-words, that is raw term-frequency counts.

Comparing the two is an ablation axis (Phase 3). Rare and ubiquitous terms are pruned
with ``min_df`` and ``max_df`` rather than the built-in ``english`` stop-word list,
which was built for computer-science text and drops words like "computer."

Sequence and pretrained representations (tokenized embeddings for the CNN, BiLSTM, and
DistilBERT variants) are deliberately not built here. They land in Phase 3 when a model
that needs them exists, to avoid abstractions ahead of a real consumer.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_vectorizer(
    representation: str = "tfidf",
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int | float = 2,
    max_df: int | float = 1.0,
):
    """Return an unfitted vectorizer for the chosen representation.

    The vectorizer is meant to live inside a Pipeline so it is only ever fit on
    training folds. ``ngram_range=(1, 2)`` adds bigrams, which is tested as an ablation
    (bigrams multiply the feature count, so the overfitting risk rises on this small,
    sparse dataset).
    """
    common = dict(ngram_range=tuple(ngram_range), min_df=min_df, max_df=max_df)
    if representation == "tfidf":
        return TfidfVectorizer(**common)
    if representation == "bow":
        return CountVectorizer(**common)
    raise ValueError(f"unknown representation {representation!r}; use 'tfidf' or 'bow'")
