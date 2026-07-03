# Reconciliation: current repo vs the expanded PROJECT_PLAN

Audit only. No code written, moved, or deleted. Branch context:
`pivot/full-lifecycle` is checked out; `main` holds the two earlier commits (restructure
+ leak-free pipeline); this branch holds the scope-revision commit. The migration below
is designed to run on `pivot/full-lifecycle`.

Note on phase numbering: the plan was renumbered. Old Phase 0 (restructure) and old
Phase 1 (narrative notebook) do NOT correspond to new Phase 0 (harness skeleton) and new
Phase 1 (curation + validation). This report uses the new numbering throughout.

## 1. What exists today, mapped to the new plan

| Item | New phase | Verdict | Reason |
|---|---|---|---|
| `CLAUDE.md` | meta | keep | Replaced by you; governs the project. |
| `PROJECT_PLAN.md` | meta | keep | Replaced by you; the new spec. |
| `INSTRUCTOR_FEEDBACK.md` | cross (2,3,4) | keep | Rigor defects still bind the control model and all metric reporting. |
| `DEVLOG.md` | cross | keep (append) | Append-only history is fine; add a pivot entry noting the renumbering. Do not rewrite past entries. |
| `.gitignore` | 0 | refactor | Add `mlruns/`, `data/processed/`, model artifacts, `.DS_Store`. Currently only `data/raw/` + Python + local helper. |
| `data/raw/amazon_alexa.tsv` | 1 | keep | Source data, gitignored. Untouched. |
| `data/README.md` | 1 | refactor | Only a basic source note now. New Phase 1 needs a full data card (curation notes, dedup, versioned artifact). |
| `reference/original_notebook.ipynb` | ref | keep | The class control reference. Section 8 keeps `reference/`. |
| `reference/original_report.pdf` | ref | keep | Class deliverable. |
| `reports/phase0_findings.md` | 1 | keep, rename | Content (label mapping, imbalance, dummy floor) is new-Phase-1 curation work, mislabeled "phase0". Rename to `reports/phase1_curation_findings.md`. |
| `reports/phase0_disagreements.csv` | 1 | keep (rename optional) | The 69 text/rating disagreement rows are a Phase 1 deliverable and Phase 4 error-analysis input. |
| `reports/figures/` | 4,7 | keep | Empty; figures land here later. |
| `notebooks/sentiment_analysis.ipynb` | 1 -> 2/3 + notebook | refactor (split) | EDA/label/baseline half satisfies Phase 1. The embedded Pipeline modeling must leave the notebook: it becomes the sklearn control in `src/` (Phase 3), scored via the shared harness (Phase 2). Notebook is renamed `report.ipynb` and made thin (calls `src/`). |
| `app/` (empty dir) | none | discard | New layout has no `app/`; serving lives in `src/serving/`. Superseded. |
| `.github/workflows/` (empty dir) | 7 | keep | CI (`ci.yml`) lands here in Phase 7. |
| `build_notebook.py` | none | discard | Local, gitignored helper. The notebook approach is changing; carries no committed logic. |
| `.DS_Store` | none | discard | OS cruft. Add to `.gitignore`. |

Nothing is discarded that carries working logic. The three discards (`app/`,
`build_notebook.py`, `.DS_Store`) are an empty dir, a gitignored scratch script, and OS
cruft.

## 2. Gap list (required by the new plan, does not exist yet)

Foundational, called out explicitly:

- **Config-driven MLflow harness (new Phase 0): NOT in place at all.** There is no
  `src/run_experiment.py`, no config loader, no seed-from-config mechanism, no MLflow
  wiring, no `mlruns/`. Nothing logs to MLflow today.
- **`src/` package layout (section 8): NOT in place at all.** There is no `src/`
  directory. None of `src/data`, `src/features`, `src/models`, `src/evaluation`,
  `src/serving`, `src/monitoring` exist. Phases 2 through 6 all depend on this.

Everything else missing, by phase:

- Phase 0: `configs/`, `requirements.txt`, stub `README.md` (no README exists yet).
- Phase 1: Pandera schema and enforcement; deduplication step; versioned curated
  artifact in `data/processed/`; full data card. (Label finding, dummy baseline, and
  disagreement rows already exist.)
- Phase 2: shared representations (`src/features/`, TF-IDF + sequence); shared eval
  function returning precision/recall/F1/specificity/PR-AUC/confusion matrix for the
  negative class; repeated stratified k-fold protocol with an untouched held-out test
  set. (Current work has a single split and a partial inline metric helper, no PR-AUC.)
- Phase 3: model variant modules (sklearn control, PyTorch MLP/CNN/DistilBERT,
  TensorFlow MLP/BiLSTM), the PyTorch/TensorFlow parity pair, MLflow-logged runs, and an
  ablation table generated from MLflow.
- Phase 4: threshold sweep to negative-recall >= 0.80; disagreement error analysis;
  separation fairness check across `variation`; coefficient/attribution interpretation.
- Phase 5: FastAPI service (`/predict`, `/health`), `Dockerfile`, optional
  `docker-compose.yml`, reproducibility note.
- Phase 6: drift module (PSI/KS via scipy), a `/drift` or scheduled check, monitoring
  design doc.
- Phase 7: full-lifecycle `README.md` with architecture diagram, `tests/` + pytest, CI
  workflow with lint + smoke run + `docker build`.

## 3. Where the in-progress Phase 1 work lands

The old Phase 1 deliverable (the narrative notebook with a leak-free Pipeline) splits
cleanly across the new plan.

Survives as new Phase 1 (curation and validation), already satisfying much of it:
- The label-to-rating finding (deterministic threshold, documented).
- The imbalance quantification and the `DummyClassifier(most_frequent)` floor.
- The 69 captured text/rating disagreement rows.

Needs added to close new Phase 1: a Pandera schema and passing validation; a
deduplication decision; a persisted, versioned curated artifact in `data/processed/`;
and a real data card in `data/README.md`. The current notebook strips empty reviews but
does not dedup, validate, or persist a curated artifact.

Migrates forward, does NOT stay in the notebook:
- The `Pipeline(CountVectorizer(min_df=2), LogisticRegression(penalty="l2"))` becomes
  the seed of the sklearn CONTROL variant in Phase 3, rebuilt as a config plus an
  `src/models/` module and scored through the Phase 2 shared eval and CV. It stops being
  run inline.
- The single stratified train/test split is superseded by the Phase 2 repeated
  stratified k-fold protocol (with a final held-out test set no selection touches).
- The six-section markdown narrative survives as reusable prose for
  `notebooks/report.ipynb`, but that notebook becomes thin: it calls into `src/` and
  presents results and MLflow figures rather than doing the modeling.

Net: roughly the EDA/label/baseline half of the current notebook survives close to
as-is (and covers most of new Phase 1); the modeling half survives as raw material for
the Phase 3 control but must move into the harness.

## 4. Proposed migration order (preserve working code, restructure incrementally)

Principle: nothing that runs today gets deleted until its replacement exists. Each step
is additive or a safe rename, ends with a DEVLOG entry, a README checklist update once
the README exists, and a copy-paste git block. All on `pivot/full-lifecycle`.

1. **Scaffold the new layout, additively.** Create the `src/` package tree (with
   `__init__.py`), `configs/`, and `tests/`. Nothing is moved or deleted, so the current
   notebook keeps running. (New Phase 0, structural half.)
2. **Phase 0 non-harness files.** Refactor `.gitignore` (add `mlruns/`,
   `data/processed/`, artifacts, `.DS_Store`), add pinned `requirements.txt`, add a stub
   `README.md` with the lifecycle checklist.
3. **Build the harness.** `src/run_experiment.py` plus a config loader, seed-setting from
   config, and MLflow logging, with a no-op config. Meet the new Phase 0 acceptance bar
   (no-op config logs a dummy metric, reproducible across two seeded runs). This is the
   dependency for everything after it.
4. **Migrate Phase 1 content.** Rename `reports/phase0_findings.md` to
   `reports/phase1_curation_findings.md`; add `src/data/` (loading, curation, dedup,
   Pandera schema); write the curated artifact to `data/processed/`; expand
   `data/README.md` into a data card. Fold in the existing findings rather than redo
   them. Meet the new Phase 1 acceptance bar.
5. **Extract shared preprocessing and evaluation.** Pull the representation and metric
   logic out of the notebook into `src/features/` and `src/evaluation/`, add the repeated
   stratified k-fold protocol and PR-AUC. (Phase 2.)
6. **Rebuild the control through the harness, then thin the notebook.** Recreate the
   sklearn Pipeline as `src/models/` + a config, run it through the harness with MLflow
   logging, then convert `notebooks/sentiment_analysis.ipynb` to
   `notebooks/report.ipynb` calling into `src/`. Only now retire the inline modeling,
   the empty `app/` dir, and `build_notebook.py`, since their replacements exist.
   (Start of Phase 3.)
7. **Proceed Phases 3 to 7 as written:** remaining DL variants and ablations, model
   selection and analysis, containerized serving, drift monitoring, docs and CI.

Open decisions worth settling before or during step 1 (from plan section 10): whether
two infra-flavored portfolio projects (this and #01) is intended given the overlap; the
CI budget (smoke-only DL is the plan's default); and whether the live-endpoint stretch
is in scope this cycle. None of these block steps 1 to 4.
