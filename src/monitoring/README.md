# Drift monitoring (design and implementation)

This is designed and implemented drift monitoring, not live production monitoring. There is
no real request traffic here. The reference is the training population, and a batch to check
is supplied as a file; in a deployment that batch would be a window of logged request texts.
The framing matters: the module is real and runs, but it has never watched live traffic.

## What is monitored

Three features, computed with the same fitted pipeline the service uses (`features.py`):

- `negative_score`: the model's predicted probability of the negative class. A shift means
  the population is changing or the model is scoring it differently.
- `review_length`: token count, an input-distribution signal.
- `oov_rate`: the share of tokens missing from the fitted TF-IDF vocabulary. A rising
  out-of-vocabulary rate is the clearest sign that incoming language has moved away from the
  training data, and it is exactly what a TF-IDF model cannot represent.

## How drift is measured

Per feature, the population stability index (PSI) is the trigger and a KS test is reported
as a second opinion (`drift.py`). PSI bins the reference distribution and measures how much
mass moved, which is stable across batch sizes. KS is not the trigger because it flags
trivially small shifts once a batch is large. The PSI rule of thumb: below 0.1 no meaningful
shift, 0.1 to 0.2 moderate, 0.2 and above significant. The default alert threshold is 0.2.

A batch is flagged as drifted when any monitored feature crosses the PSI threshold.

## How it runs

`check.py` compares a batch against the stored reference, prints a JSON report, and exits
non-zero when drift is detected, so it runs as a scheduled job with alerting wired to the
exit code. It is kept as an offline check rather than a `/drift` endpoint on purpose: the
serving container installs only `requirements-serve.txt` and stays slim, while the check uses
the dev stack (scipy). Exposing it as an endpoint later is a small change.

```bash
python -m src.serving.export_model      # produces models/model.joblib
python -m src.monitoring.build_reference # reference from the training population
python -m src.monitoring.check --batch path/to/reviews.parquet
```

## What an alert means, and the retraining trigger

A single flagged batch is a signal, not a verdict, because a one-off odd batch happens. The
retraining trigger is sustained drift: the same feature crossing the PSI threshold on
several consecutive scheduled checks, with a rising `oov_rate` weighted most heavily since
it most directly undermines a bag-of-words model. When that fires, the response is to
re-curate the newer data (through the Pandera-validated curation step), re-run model
selection through the harness, compare the new candidate against the current model on the
held-out metrics, and redeploy only if it wins. The reference is refreshed to the new
training population at each redeploy.

## Honest limits

The reference is the training data, not a production baseline. Thresholds are conventional
starting points, not tuned against observed traffic. And because the negative class is rare,
group-level or class-conditional drift on negatives would be noisy, so the current signals
are population-level.
