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

## 2026-07-03: Phase 3 variants and ablation infrastructure

Built the comparison scaffolding and the first neural variants, all scored through one
shared path so the ablation is apples-to-apples.

`src/models/harness.py` (`run_single`) is the shared path for any single-config variant:
stratified holdout, repeated stratified k-fold, shared `evaluate`, and MLflow logging.
`src/models/baseline.py` logs the majority-class floor as its own run so the baseline row
comes from MLflow, not by hand. `src/evaluation/ablation.py` reads the `sentiment-ablation`
experiment and prints a variant-by-metric table sorted by held-out negative-class F1 with
the baseline pinned on top. Current table (control run on a reduced grid for the sandbox):

| variant | framework | holdout_neg_recall | holdout_neg_precision | holdout_neg_f1 | holdout_specificity | holdout_pr_auc |
|---|---|---|---|---|---|---|
| baseline_dummy | sklearn | 0.000 | 0.000 | 0.000 | 1.000 | 0.086 |
| sklearn_linear | sklearn | 0.976 | 0.238 | 0.383 | 0.706 | 0.599 |

`src/models/torch_mlp.py` and `src/models/tf_mlp.py` are the framework parity pair: the same
one-hidden-layer MLP (ReLU, dropout, balanced class weights, early stopping) on TF-IDF,
wrapped as sklearn-compatible estimators so they clone into the same folds and eval as
everything else. Their configs share an identical model block; only the framework differs.
Smoke configs (subset, one epoch) are included for CI and quick local checks.

Verification: `ruff` clean, dispatch imports without torch or tensorflow (lazy), and
`pytest` passes with the two neural tests skipped (`importorskip`) since the sandbox has no
access to the torch/tensorflow wheels. The neural variants are therefore integrated and
lint-checked but not yet executed here; the DL runs are local by design. Next actions to
clear the Phase 3 bar: run the two smoke configs and the full parity configs locally, then
regenerate the ablation table. After that, the remaining variants (text-CNN, BiLSTM,
DistilBERT) and the imbalance-handling axis.

## 2026-07-03: Phase 3 results (local runs)

Ran the full parity configs plus baseline and control locally. Held-out ablation
(negative class, 0.5 threshold except where selection sets it):

| variant | framework | neg_recall | neg_precision | neg_f1 | specificity | pr_auc | cv_neg_recall |
|---|---|---|---|---|---|---|---|
| baseline_dummy | sklearn | 0.000 | 0.000 | 0.000 | 1.000 | 0.086 | 0.000 |
| pytorch_mlp | pytorch | 0.415 | 0.500 | 0.453 | 0.961 | 0.526 | 0.525 |
| tensorflow_mlp | tensorflow | 0.390 | 0.500 | 0.438 | 0.963 | 0.523 | 0.504 |
| sklearn_linear | sklearn | 0.951 | 0.150 | 0.259 | 0.493 | 0.278 | 0.812 |

Two findings. First, framework parity holds: the PyTorch and TensorFlow MLPs are within a
few thousandths on every metric, which is the reproducibility point of running both.

Second, and more important, the selection criterion, not the model family, dominated the
control. Selecting the linear model on cross-validated negative-class recall drove it to a
degenerate corner (recall 0.95 but precision 0.15, specificity 0.49, so it flags half of
all positive reviews) with the worst PR-AUC of any non-baseline model (0.278). By the
threshold-independent PR-AUC, the MLPs (~0.52) actually rank negatives better than the
recall-selected linear model. This is not "deep learning won": the linear model ranked
well under a smaller grid earlier (PR-AUC ~0.60). It is that recall-only selection over a
wide C grid picks an over-regularized model with poor ranking. The fix belongs to Phase 4:
select on a threshold-independent or precision-constrained criterion (PR-AUC, or recall
subject to a precision floor) and then sweep the threshold to a principled operating point,
rather than selecting on raw recall.

Phase 3 acceptance bar met: control plus a PyTorch and a TensorFlow variant run and log
cleanly, the parity pair is present, and the ablation table is generated from MLflow. The
text-CNN, BiLSTM, and DistilBERT variants remain as enhancements beyond the bar.

## 2026-07-03: Phase 4, selection metric and operating point

Fixed the selection defect the Phase 3 table exposed. The control now selects on PR-AUC
(threshold-independent negative-class ranking) instead of raw recall, with negative recall
kept as a recorded secondary metric. Selecting on PR-AUC picks a well-ranked model
(L2, C=10, unweighted; cv PR-AUC 0.57) instead of the degenerate recall-maximizing model
(PR-AUC 0.28). Properly selected, the linear control (PR-AUC ~0.57) now ranks negatives as
well as or better than the MLPs (~0.52), which restores the expected finding: complexity
does not earn its place here.

Added `src/evaluation/threshold.py`: `choose_threshold` picks the highest threshold whose
negative recall meets a target (default 0.80), with `evaluate_at_threshold` and a `sweep`
for plotting. The control run now reports an operating point chosen on out-of-fold TRAIN
predictions (via `cross_val_predict`), then applied to the untouched test set, so the
threshold is never tuned on the test data. Reduced-grid demonstration:

- default 0.5 threshold: negative recall 0.24, precision 0.71, specificity 0.99 (too
  conservative under imbalance).
- operating point (threshold ~0.077, chosen on OOF train): negative recall 0.76 on the
  test set, precision 0.37, specificity 0.88. The trade is explicit: catching about
  three quarters of negative reviews means most negative flags are false alarms and about
  12% of positive reviews are wrongly flagged.

`tests/test_threshold.py` covers the operating-point logic; `ruff` and `pytest` green (the
full path verified through `run_experiment` on a reduced grid). Remaining Phase 4:
disagreement-row error analysis, the separation fairness check across `variation`, L1
coefficient interpretation, and folding the PR curve and threshold sweep into the report
notebook. Locally, re-run the full control config and regenerate the ablation table so
MLflow reflects the PR-AUC selection and the operating-point metrics.

Also changed the ablation table to sort by held-out PR-AUC rather than F1 at 0.5, since
F1 at the default threshold understates a model that operates at a tuned threshold. Under
that ranking the linear control leads (PR-AUC 0.575) with the two MLPs a notch below
(~0.52), which is the honest read: on this data the neural nets do not beat a
properly-selected linear model.

## 2026-07-03: Phase 4 analysis (interpretation, fairness, disagreements)

Added `src/evaluation/interpret.py`, `fairness.py`, and `error_analysis.py`, each tested,
and ran them on the selected linear model.

Interpretation. The negative-driving words are negation and complaint terms (not, didn,
back, poor, return, sucks, meh, stopped); positive-driving are praise (love, great, easy,
works, amazing, best). The selected L2 model is dense (1,839 nonzero coefficients) and its
top lists carry filler like "my", "an", "so". An L1 model at the same C is 85% sparse (267
nonzero) with a cleaner negative list (awful, poor, stopped, useless, garbage, return,
within, siri). This is the concrete case for L1 on the explainability goal: a much shorter
list of words carries the signal, so the model is easier to read, at comparable ranking.

Fairness as separation across device `variation`, at the operating threshold. The
false-positive rate is consistent across groups (spread ~0.02). Negative recall varies more
(spread ~0.20 among the three variations with at least five negatives), but most variations
have only three to six negative reviews, so per-group recall is noisy and a couple of tiny
groups swing to 0.0 or 1.0 on three examples. The honest conclusion is that the rare
negative class makes per-device fairness estimates unreliable, reported with counts rather
than overclaimed as a disparity.

Disagreements with the star label. The model diverges from the threshold label mostly in
one direction: 56 held-out reviews are high-star but the model reads their text as negative,
versus 9 low-star reviews the model reads as positive. The high-star flags often catch real
complaint language the stars gloss over (a 5-star review whose text says the device
"stopped replying to my requests"); the low-star misses are terse or sarcastic ("like
having another kid in the house; I have to constantly repeat myself"), which a bag-of-words
model does not catch. These are the error-analysis cases where the text model is sometimes
more right than the label, and where it is predictably wrong.

Remaining Phase 4: fold the PR curve, the threshold sweep, and these three analyses into the
report notebook as figures, then Phase 4 is complete.
