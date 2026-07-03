# DEVLOG

Development log. Newest entries at the bottom.

## 2026-07-02: Dataset curation and validation

Loaded `data/raw/amazon_alexa.tsv` (3,150 rows). Established the central finding: the
`feedback` label is a deterministic function of `rating`, with `feedback == 1` if and
only if `rating >= 3`. So the label is a thresholded star rating, and the interesting
rows are the ones where text and stars disagree (69 captured to
`reports/phase1_curation_findings.md` and a companion CSV). Imbalance is about 92%
positive. Recorded the `DummyClassifier(most_frequent)` floor as the baseline every model
must beat.

Built the curation and validation layer. `src/data/load.py` strips whitespace, drops 80
rows with no review text, and drops 686 exact duplicate records (they would otherwise
leak across cross-validation folds). `src/data/schema.py` is a Pandera schema enforcing
column types, non-empty text, a binary label, and the label rule itself.
`src/data/build_curated.py` writes the versioned artifact
`data/processed/curated.parquet`: 2,384 rows, 205 negative (8.6%), content hash
`543047b2d164`. Schema passes on the curated set and rejects a flipped label.

## 2026-07-02: Experiment harness skeleton

`src/run_experiment.py` is a config-driven entrypoint: it reads a YAML config, seeds
Python and numpy from it, runs the named variant, and logs to MLflow. `configs/noop.yaml`
is a no-op smoke that logs one seeded metric and reproduces exactly across runs with the
same seed. Dependencies are split into `requirements-serve.txt` (slim runtime for the
future inference container) and `requirements-dev.txt` (full training and analysis stack).
Added a stub `README.md` with the lifecycle checklist and a `.gitignore` covering data,
model artifacts, and `mlruns/`.

## 2026-07-03: Shared preprocessing and evaluation

`src/features/text.py` builds either TF-IDF (term frequency inverse document frequency) or
bag-of-words counts, with `min_df`/`max_df` frequency pruning and configurable n-grams,
deliberately avoiding the built-in `english` stop-word list. `src/evaluation/metrics.py`
is a single `evaluate()` scoring everything for the negative class: precision, recall, F1,
specificity, PR-AUC, and a confusion matrix, with specificity and precision documented as
answering different questions. `src/evaluation/cv.py` adds a stratified holdout and a
repeated stratified k-fold protocol reporting mean and standard deviation. Tests prove a
linear pipeline and a dummy baseline are scored through the identical path; `pytest` and
`ruff` are green.

## 2026-07-03: Linear control through the harness

`configs/sklearn_logreg_tfidf.yaml` plus `src/models/sklearn_linear.py` run the linear
control: a `GridSearchCV` over the penalty (L1 vs L2 on the saga solver), an explicit `C`
log grid, and class weighting, selected by cross-validated negative-class recall, then
evaluated once on a held-out test set that selection never touched. The run logs its
params, held-out metrics, the full grid ablation table, and the fitted model to MLflow.
Early ablation favors L1 (C = 1.0, balanced), which also gives sparse coefficients for the
interpretation work later. Selecting purely on recall overshoots on precision, which is
the motivation for the threshold work next: pick an operating point tied to
negative-class recall >= 0.80 and report its precision and specificity cost. After that,
the deep-learning variants.

## 2026-07-03: Narrative report notebook

Rewrote the notebook as a thin report (`notebooks/report.ipynb`) that holds no modeling
logic of its own. It imports from `src/` for curation, preprocessing, and evaluation, and
attributes model selection to the harness and MLflow. It characterizes the label, records
the `DummyClassifier` floor through the shared `evaluate`, and presents the selected linear
control (TF-IDF, L1, balanced) via the shared cross-validation and held-out evaluation.
Verified to run top to bottom with `jupyter nbconvert --execute`. Held-out control on the
untouched test set: negative-class recall 0.63, precision 0.37, specificity 0.90,
PR-AUC 0.585, against a Dummy floor of 0. The empty `app/` placeholder is dropped (serving
lives under `src/serving/`).
