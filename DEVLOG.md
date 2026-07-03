# DEVLOG

Append-only. Newest entries at the bottom.

## 2026-07-02: Repo restructure + Phase 0

**Restructure.** Moved the flat folder into the layout from PROJECT_PLAN section 8.
The original notebook is now `reference/original_notebook.ipynb` and the class report is
`reference/original_report.pdf`. Created `data/raw/` (holds the gitignored tsv),
`notebooks/`, `reports/figures/`, `app/`, and `.github/workflows/`. Added `data/README.md`
(Kaggle source, no data committed) and a minimal `.gitignore` that excludes `data/raw/`.
The full pinned `requirements.txt` and CI still belong to Phase 5.

**Phase 0.** Confirmed the core angle: `feedback` is a deterministic function of `rating`
(rating {1,2} to 0, {3,4,5} to 1, exact for all 3150 rows). The label is a thresholded
star rating, not human sentiment. Imbalance is 91.84% positive (257 negatives).
Recorded the `DummyClassifier(most_frequent)` floor: accuracy 0.9184, negative-class
precision and recall both 0, specificity 1.0. Found 80 rows with no usable review text
(1 NaN + 79 blank). Saved 69 text/star disagreement rows to
`reports/phase0_disagreements.csv` and wrote up the results in `reports/phase0_findings.md`.

Acceptance bar met. Next up is Phase 1 (rebuild the notebook as a narrative), not started.

## 2026-07-02: Phase 1 (narrative notebook)

Built `notebooks/sentiment_analysis.ipynb` as an explained analysis rather than a
script. One imports cell at the top, nothing imported later. Six markdown sections with
intent statements: Data and Label, Preprocessing, Model Selection, Evaluation, Fairness,
Further Analysis. Fairness and Further Analysis are stubbed with their intent stated;
they get delivered in Phase 4.

All modeling goes through a single `Pipeline(CountVectorizer(min_df=2),
LogisticRegression(penalty="l2"))`, so the vectorizer is fit only on training data and
the old fit-twice leakage is gone. Encoding is named as bag-of-words / term-frequency,
the L2 penalty is stated explicitly, the built-in `english` stop-word list is avoided in
favor of `min_df` pruning, and `stratify` is justified as preserving the ~8% negative
prevalence per split. The M >> N risk is shown directly: raw vocabulary 3669 vs 2456
training rows.

Honest baseline on the held-out test set (614 rows), minority class first, against the
`DummyClassifier(most_frequent)` floor:

| model | accuracy | neg precision | neg recall | specificity |
|---|---|---|---|---|
| Dummy (most_frequent) | 0.9235 | 0.000 | 0.000 | 1.000 |
| LogReg L2 (threshold 0.5) | 0.9528 | 0.846 | 0.468 | 0.993 |

The 0.468 negative-class recall (the model still misses over half the negatives at the
default threshold) is the real headline and the thing Phases 2 and 3 attack, not the
0.95 accuracy. Notebook verified to run top to bottom with no errors via
`jupyter nbconvert --execute`.

Acceptance bar met: runs clean, reads as an explained analysis, no vectorizer fit outside
the pipeline. Next up is Phase 2 (model selection with GridSearchCV), not started.

## 2026-07-02: Scope pivot to full lifecycle (branch pivot/full-lifecycle), steps 1 to 4

Scope expanded from a single-notebook classical project to a full ML lifecycle: a
config-driven MLflow harness, cross-framework ablations, containerized serving, and drift
monitoring. Phases were renumbered (new Phase 0 is the harness skeleton, new Phase 1 is
curation and validation). Audit and migration plan are in `reports/reconciliation.md`.
This batch is steps 1 to 4 of that plan.

Step 1, scaffold. Added the `src/` package tree (`data`, `features`, `models`,
`evaluation`, `serving`, `monitoring`) with `__init__.py`, plus `tests/` and `configs/`.
Purely additive; the existing notebook still runs.

Step 2, Phase 0 files. Refactored `.gitignore` (added `mlruns/`, `data/processed/`, model
artifacts, `.DS_Store`). Split dependencies into `requirements-serve.txt` (slim runtime:
sklearn, numpy, pandas, FastAPI) and `requirements-dev.txt` (full stack incl. torch,
tensorflow, transformers, mlflow, pandera). The Phase 5 Dockerfile will install serve
only. Added a stub `README.md` with the lifecycle checklist.

Step 3, harness. `src/run_experiment.py` loads a YAML config, seeds Python and numpy from
it, and logs one dummy metric to MLflow. `configs/noop.yaml` is the no-op variant. Kept
deliberately minimal: no config schema, no model abstraction until a real variant needs
them. Verified the Phase 0 bar: seed 42 reproduces `dummy_metric=0.3745401188473625`
across two runs, seed 7 gives a different value. (MLflow 3.x gates the file store, so the
harness sets `MLFLOW_ALLOW_FILE_STORE=true` to keep the gitignored `mlruns/` backend.)

Step 4, curation and validation. `src/data/load.py` curates (strip whitespace and BOM,
drop 80 blank-text rows, drop 686 exact duplicate records). `src/data/schema.py` is a
Pandera schema enforcing column types, non-empty text, a binary label, and the label rule
`feedback == 1 iff rating >= 3`. `src/data/build_curated.py` writes the versioned artifact
`data/processed/curated.parquet` (gitignored). Curated set: 2,384 rows, 205 negative
(8.6%), content hash `543047b2d164`. Verified the schema passes on the curated set and
rejects a flipped label. Renamed the findings to `reports/phase1_curation_findings.md`
(via `git mv`) and stated the exact mapping in one line at the top. `data/README.md` is
now a full data card documenting the dedup decision and its tradeoff. `ruff check src/`
passes.

Phase 0 and Phase 1 acceptance bars met. Stopping before step 5 (shared preprocessing and
evaluation) for check-in, per the agreed amendments. The dedup choice (686 rows, ~22%) is
flagged for review; it is reversible via `DEDUP_KEYS` in `src/data/load.py`.
