# Voice Assistant Review Sentiment Classifier

Predicting the binary `feedback` label (1 positive, 0 negative) from Amazon Alexa review
text, built as a full ML lifecycle: dataset curation and validation, a config-driven
harness for cross-validated cross-framework ablations, a containerized inference service,
and drift monitoring. The point is reproducible experimentation infrastructure. The
models are the payload; the harness and the honest evaluation are the point.

This README is a stub. It fills out through Phase 7 into the full lifecycle writeup.

## The one thing to know up front

The `feedback` label is a hard threshold on the star `rating`: ratings 1 and 2 are
negative, 3 through 5 are positive. So "sentiment" here is a thresholded star count, not
an independent human judgment. The dataset is small (about 3,150 rows) and heavily
imbalanced (about 92% positive), which is why accuracy is never the headline and every
result is read against a majority-class baseline, minority class first.

## Layout

- `src/` the harness: data, features, models, evaluation, serving, monitoring.
- `configs/` one file per experiment variant.
- `data/` raw (gitignored) and curated (gitignored) data, plus the data card.
- `notebooks/` the narrative report.
- `reports/` findings and figures.

## Setup

```bash
pip install -r requirements-dev.txt   # full stack for training and analysis
# the serving container installs only requirements-serve.txt
```

Place the dataset at `data/raw/amazon_alexa.tsv` (see `data/README.md`). It is never
committed.

## Lifecycle checklist

- [x] Phase 0: repo and harness skeleton (config load, seed-from-config, MLflow dummy run)
- [x] Phase 1: dataset curation and validation (label finding, Pandera, curated artifact)
- [x] Phase 2: shared preprocessing and evaluation (representations, eval fn, CV protocol)
- [x] Phase 3: model variants and cross-validated ablations (sklearn, PyTorch, TensorFlow)
- [x] Phase 4: analysis and model selection (threshold, error analysis, fairness, interpretation)
- [ ] Phase 5: containerized inference (FastAPI plus Docker)
- [ ] Phase 6: drift monitoring
- [ ] Phase 7: documentation, CI, lifecycle writeup
