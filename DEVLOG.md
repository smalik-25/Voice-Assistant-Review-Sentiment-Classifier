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
