# Drift monitoring (design and implementation)

This is drift monitoring I designed and implemented, not live production monitoring. There is
no real request traffic, so the reference is the training population and a batch to check is
supplied as a file. In a deployment that batch would be a window of logged request texts. I
want to be clear about that up front: the module is real and it runs, but it has never
watched live traffic.

## What I monitor

Three features, computed with the same fitted pipeline the service uses (`features.py`):

- `negative_score`: the model's predicted probability of the negative class. A shift means
  the population is changing, or the model is scoring it differently.
- `review_length`: token count, an input-side signal.
- `oov_rate`: the share of tokens missing from the fitted TF-IDF vocabulary. A rising
  out-of-vocabulary rate is the clearest sign the incoming language has moved away from the
  training data, and it is exactly what a TF-IDF model cannot represent.

## How I measure drift

Per feature, PSI is the trigger and a KS test is a second opinion (`drift.py`). PSI bins the
reference distribution and measures how much mass moved, which holds up across batch sizes. I
do not trigger on KS because it flags trivially small shifts once a batch gets large. The PSI
rule of thumb: below 0.1 nothing meaningful, 0.1 to 0.2 moderate, 0.2 and up significant. The
default alert threshold is 0.2, and a batch counts as drifted if any monitored feature crosses
it.

## How it runs

`check.py` compares a batch against the stored reference, prints a JSON report, and exits
non-zero when it finds drift, so it can run as a scheduled job with alerting hung off the exit
code. I kept it as an offline check rather than a `/drift` endpoint on purpose: the serving
container installs only `requirements-serve.txt` and stays small, while the check uses the dev
stack (scipy). Turning it into an endpoint later would be a small change.

```bash
python -m src.serving.export_model       # produces models/model.joblib
python -m src.monitoring.build_reference  # reference from the training population
python -m src.monitoring.check --batch path/to/reviews.parquet
```

## What an alert means, and when I would retrain

One flagged batch is a signal, not a verdict, because odd one-off batches happen. I would
retrain on sustained drift: the same feature crossing the PSI threshold on several
consecutive checks, weighting a rising `oov_rate` most heavily since it most directly
undermines a bag-of-words model. When that fires, the response is to re-curate the newer data
through the same validated curation step, re-run selection through the harness, compare the
new candidate against the current model on the held-out metrics, and redeploy only if it
wins. The reference gets refreshed to the new training population at each redeploy.

## The honest limits

The reference is the training data, not a production baseline. The thresholds are
conventional starting points, not values I tuned against real traffic. And because the
negative class is rare, class-conditional drift on negatives would be noisy, so the signals I
watch are population-level.
