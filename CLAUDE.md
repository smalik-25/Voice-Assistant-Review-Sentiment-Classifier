# CLAUDE.md — Voice Assistant Review Sentiment Classifier

Read this file and `PROJECT_PLAN.md` in full before doing anything. The plan has the
phases and acceptance bars. `INSTRUCTOR_FEEDBACK.md` has the rigor defect list that
still governs how the classical control is built and how all metrics are reported.

## What this project is

A full ML lifecycle project on the Amazon Alexa reviews dataset: dataset curation and
validation, cross-validated ablation studies across scikit-learn, PyTorch, and
TensorFlow, a containerized inference service, and drift monitoring. The identity is
**reproducible experimentation infrastructure**: a config-driven harness that makes
cross-framework NLP ablations repeatable. The models are the payload; the harness and
the reproducibility are the point.

Binary target `feedback` (1 positive, 0 negative), predicted from `verified_reviews`
text. The dataset is small (~3,150 rows, ~250 negatives) and imbalanced.

## Honesty guardrails (these protect interview defensibility)

- **Deep learning will probably lose to the linear baseline** on this data. That is the
  ablation finding, not a failure. Never write that a neural model improved results
  unless the logged MLflow runs show it. The plausible small-data winner is a
  fine-tuned DistilBERT.
- **Do not claim production monitoring.** There is no live traffic. Build a real drift
  detector plus a monitoring design doc and describe it as "designed and implemented
  drift monitoring for the inference service."
- **The PyTorch vs TensorFlow comparison is a parity and ablation exercise**, not a
  contest. The point is a framework-agnostic, reproducible harness.
- **Never lead with accuracy.** On a ~92% positive label it sits near the majority
  floor. Always compare against a `DummyClassifier` baseline, minority-class first.

## The analytical spine (do not lose it as scope grows)

- The `feedback` label is almost certainly a hard threshold on `rating`. Confirm first
  (Phase 1), document it, and build around it. "Sentiment" here is a thresholded star
  rating, and the interesting cases are text/rating disagreements.
- The sklearn linear model is the control every deep variant must beat. Keep it in
  every comparison.

## Environment and stack

- Core: Python 3.11+, scikit-learn, pandas, numpy.
- Deep learning: PyTorch, TensorFlow/Keras, Hugging Face Transformers.
- Validation: Pandera. Tracking: MLflow. Serving: FastAPI plus Docker. Monitoring:
  PSI/KS via scipy. Quality: pytest, GitHub Actions. Figures: plotly.
- Development is local on Sam's Mac. The dataset lives at `data/raw/amazon_alexa.tsv`
  and is gitignored. Do not download it into the repo and do not commit it. Do not add
  `kagglehub` calls to committed code; read from `data/raw/` and document the source in
  `data/README.md`.
- Deep learning training is heavy. Keep full ablation runs local. In CI, run only a
  tiny smoke variant (one epoch, subset) plus lint, tests, and `docker build`.

## Reproducibility rules

- Every experiment is a config file. No hardcoded hyperparameters in model code.
- Set all seeds (Python, numpy, framework) from the config at the start of every run.
- Every run logs params, metrics, the seed, and artifacts to MLflow. Ablation tables
  are generated from MLflow, never hand-copied.
- The selected model must be rebuildable from its config plus seed, and the container
  from a pinned `requirements.txt`.
- `mlruns/`, `data/raw/`, `data/processed/`, and model artifacts are gitignored.

## Repo structure

Follow `PROJECT_PLAN.md` section 8. Shared preprocessing and a shared evaluation
function so every variant is scored identically. Config-driven entrypoint in
`src/run_experiment.py`. The narrative report lives in `notebooks/report.ipynb`; the
harness lives in `src/`, not in a notebook.

## Git workflow (non-negotiable)

- **Never run git from the sandbox.** No `git` commands executed by you. It causes
  `.lock` files and there are no GitHub credentials here.
- Prepare and edit files, then hand Sam an exact copy-paste block for the Terminal. Sam
  runs every git command himself.
- After each meaningful chunk (a phase or a self-contained step):
  1. Append a dated `DEVLOG.md` entry.
  2. Update the `README.md` checklist once it exists.
  3. Give Sam a copy-paste block: `git add` of the specific files, a Conventional
     Commits message, `git push` only when agreed.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`), with
  a scope where useful, for example `feat(harness): ...`.
- Never assume a commit or push happened. Confirm with Sam.

Example block:

```bash
git add src/models/pytorch_mlp.py configs/pt_mlp_tfidf.yaml DEVLOG.md
git commit -m "feat(models): add PyTorch MLP variant on TF-IDF features"
git push
```

## Notebook conventions (for the report notebook)

- One imports cell at the top. Markdown section headers with a one to three sentence
  intent statement per section. Runs top to bottom with no errors given the data.
- No modeling logic lives only in the notebook. The notebook calls into `src/` and
  presents results and figures. Training and selection happen through the harness.

## Modeling standards (non-negotiable, from the feedback)

- Name the encoding: TF-IDF is term frequency inverse document frequency; counts are
  bag-of-words. Compare them as an ablation axis.
- Do not use `stop_words='english'`. Use `min_df`/`max_df` frequency pruning, justified.
- State the regularization penalty every time a `C` is reported. Compare L1 and L2
  explicitly (saga solver). L1 gives sparse interpretable coefficients, which serves
  explainability.
- M >> N is the core overfitting risk and the reason regularization matters. Do not
  claim linear models "handle high-dimensional data well" without that caveat.
- If Random Forest is kept as a variant, its rationale is variance reduction via
  bagging, and tune `max_depth`/`min_samples_leaf`, not `n_estimators`.
- Justify `stratify`: it preserves the ~8% negative prevalence in every split.
- Cross-validation is repeated stratified k-fold; report mean and standard deviation.
  A single held-out test set that no selection touches is the final check.

## Metric rules

- Report a `DummyClassifier(strategy="most_frequent")` floor and compare against it.
- Lead with minority-class precision, recall, and F1 (the resume bullet names these),
  plus specificity, PR-AUC, and a confusion matrix.
- Distinguish specificity ("how often are positive reviews wrongly flagged") from
  precision ("how trustworthy is a negative flag"). State which question each serves.
- Do not treat the 0.5 threshold as given. Sweep it and pick an operating point tied to
  the success criterion (negative-class recall >= 0.80).
- Fairness means separation: negative-class recall and false positive rate across
  device `variation` groups. Name the criterion.

## Writing conventions

- All human-facing prose (README, data/README, monitoring design doc, website copy)
  gets the-humanizer voice: concise, high signal, no filler, no buzzwords, peer-level.
- **No em dashes anywhere.** Use commas, parentheses, or restructure.
- README covers the full lifecycle (curation, validation, harness, ablation vs
  baseline, chosen model, serving, monitoring design) and ends on a real "what I
  learned" lesson.

## Never do

- Never run git from the sandbox.
- Never commit the dataset or add Kaggle download calls to committed code.
- Never claim a neural model beat the baseline unless the logged runs show it.
- Never claim live production monitoring. Say designed-and-implemented drift monitoring.
- Never report a `C` value without its penalty type.
- Never use `stop_words='english'`.
- Never lead with accuracy or omit the majority baseline.
- Never hardcode hyperparameters outside a config, or skip seeding.
- Never put full DL training in CI. Smoke runs only.
- Never reintroduce the Part II "offensive speech" content. Wrong application.
- Never use em dashes.
- Never assume a commit or push happened. Confirm with Sam.

## Reference files

- `PROJECT_PLAN.md` — phases, acceptance bars, traceability, honesty guardrails.
- `INSTRUCTOR_FEEDBACK.md` — the graded defect list.
- `reference/original_notebook.ipynb` — the original code (becomes the sklearn control).
- `reference/original_report.pdf` — the class deliverable.
