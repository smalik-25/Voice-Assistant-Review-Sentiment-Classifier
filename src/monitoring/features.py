"""Monitored features for drift detection.

For a text model the practical drift signals are the model's own output distribution and a
couple of cheap input-text statistics:

- ``negative_score``: the model's predicted probability of the negative class. A shift here
  means the population the model sees is changing, or the model is behaving differently.
- ``review_length``: number of tokens, an input-distribution signal.
- ``oov_rate``: share of tokens not in the fitted TF-IDF vocabulary. A rising out-of-
  vocabulary rate is the clearest sign that incoming language has drifted away from the
  training data.

These are computed with the same fitted pipeline the service uses, so monitoring sees
exactly what the model sees.
"""
from __future__ import annotations

import numpy as np

NEG = 0


def extract_features(bundle: dict, texts) -> dict:
    pipeline = bundle["pipeline"]
    texts = [str(t) for t in texts]
    neg_col = list(pipeline.classes_).index(NEG)
    neg_score = pipeline.predict_proba(texts)[:, neg_col]

    tfidf = pipeline.named_steps["tfidf"]
    analyze = tfidf.build_analyzer()
    vocab = tfidf.vocabulary_

    lengths, oov = [], []
    for text in texts:
        tokens = analyze(text)
        lengths.append(len(tokens))
        oov.append(float(np.mean([tok not in vocab for tok in tokens])) if tokens else 0.0)

    return {
        "negative_score": np.asarray(neg_score, dtype=float),
        "review_length": np.asarray(lengths, dtype=float),
        "oov_rate": np.asarray(oov, dtype=float),
    }
