# Project Plan: Voice Assistant Review Sentiment Classifier

## 1. Goal

Build a full ML lifecycle project on the Amazon Alexa reviews dataset, from dataset
curation through containerized deployment and drift monitoring. The identity of the
project is **reproducible experimentation infrastructure**: a config-driven harness
that makes cross-validated, cross-framework NLP ablations repeatable, with the winning
model curated, served, and monitored. It targets a role focused on reproducibility and
experimentation infrastructure.

Target resume description this project must support:

> Designed a full ML training workflow from dataset curation through model deployment:
> curated and validated labeled training datasets, ran cross-validated ablation studies
> comparing PyTorch and TensorFlow architectures, measured precision, recall, and
> F1-score across model variants, and containerized the inference pipeline for
> reproducible deployment. Documented the full ML lifecycle from data preparation
> through production monitoring.

## 2. Honesty guardrails (read before writing any claim)

These keep the project defensible in an interview. Violating them turns a strong
project into one that collapses under a single follow-up question.

- **Deep learning will likely lose to the linear baseline** on this dataset (~3,150
  rows, ~250 negatives). That is the finding, not a failure. Frame it as an ablation
  result. Do not claim a neural model improved performance unless the logged runs show
  it. The one variant that can plausibly win is a fine-tuned DistilBERT (transfer
  learning is the small-data exception).
- **"Production monitoring" is not literally true** without live traffic. Build a real
  drift-monitoring module and a documented monitoring/alerting design, and describe it
  as "designed and implemented drift monitoring for the inference service," never as
  "monitored production traffic."
- **The framework comparison is a parity/ablation exercise**, not a contest to crown a
  framework. Be ready to answer "why both PyTorch and TensorFlow": to show the harness
  is framework-agnostic and the results are reproducible across backends.
- **Metrics are reported against a majority baseline**, minority-class first. Accuracy
  on a ~92% positive label is near the majority floor and is never the headline.

## 3. Positioning

- vs #01 (Sneaker Price MLOps Pipeline): that project is distributed training and
  orchestration (Ray, Spark, Airflow) on tabular data. This project is NLP,
  cross-framework ablation, containerized serving, and drift monitoring. Different
  competency axis. Keep this one framework-and-serving focused, not orchestration
  focused, so they do not blur.
- vs #09 (Privacy-Aware Fashion Review Risk Detector): that is a single classical text
  classifier. This one adds the full lifecycle (curation, cross-framework ablation,
  deployment, monitoring) and the reproducibility harness. The classical model here is
  just the control.

The analytical spine that keeps it honest, carried from the earlier scope:
- **Label skepticism.** `feedback` is almost certainly a hard threshold on `rating`,
  so "sentiment" is a thresholded star rating. Confirm in Phase 1 and build around it.
  The interesting cases are text/rating disagreements.
- **Honest baselines and minority-class metrics.** The linear model is the control
  every deep variant must beat.

## 4. Current state

What exists:
- `reference/original_notebook.ipynb`: a data-loading notebook plus one script-style
  modeling cell (train/val/test split, `CountVectorizer(stop_words='english')`, manual
  loops over logistic C and RF n_estimators, prints accuracy/precision/recall).
- The original code has a vectorizer leakage bug (fit separately for selection and
  final fit), reports accuracy on a ~92% positive label, and has no baseline, no
  threshold analysis, no PR analysis, and no markdown narrative.
- Reported class results: accuracy 0.946, precision 0.758, negative-class recall
  0.490.

Reference material in the repo:
- `INSTRUCTOR_FEEDBACK.md`: the full graded defect list. Every rigor point still
  applies to how the classical control is built and how all metrics are reported.
- `reference/original_report.pdf`: the class deliverable.

Nothing from the original modeling code is reused directly. It becomes the sklearn
control inside the new harness.

## 5. Stack

- Core: Python 3.11+, scikit-learn, pandas, numpy.
- Deep learning: PyTorch, TensorFlow/Keras, Hugging Face Transformers (DistilBERT).
- Data validation: Pandera.
- Experiment tracking: MLflow.
- Serving: FastAPI plus a Dockerfile. Optional docker-compose.
- Monitoring: a drift-detection module (PSI/KS), scipy for the tests.
- Quality: pytest, GitHub Actions, plotly for figures.

## 6. Phases

Each phase ends with a DEVLOG entry, a README checklist update, and a copy-paste git
block for Sam to run (see CLAUDE.md). Do not advance until the acceptance bar is met.

### Phase 0: Repo and harness skeleton

Steps:
1. Lay out the repo structure (section 8). Create `requirements.txt`, `.gitignore`
   (exclude `data/raw/`, `mlruns/`, model artifacts), and a stub `README.md`.
2. Create a `configs/` directory. Each experiment variant is a config file (dataset
   settings, representation, model, framework, hyperparameters, seed).
3. Create a single entrypoint (for example `src/run_experiment.py`) that reads a
   config, runs the shared pipeline, and logs to MLflow. Set global seeds from config.

Acceptance: a no-op config runs end to end, logs a dummy metric to MLflow, and is
reproducible across two runs with the same seed.

### Phase 1: Dataset curation and validation

Steps:
1. Load `data/raw/amazon_alexa.tsv`. Confirm shape, dtypes, missing values.
2. **Characterize the label.** Cross-tabulate `feedback` against `rating`, document the
   exact threshold mapping, quantify imbalance (~92% positive expected).
3. **Validate with Pandera.** Write a schema for the curated training set (column
   types, value ranges, non-null text, label in {0,1}) and enforce it. This is the
   "validated labeled training datasets" claim made real.
4. **Curation decisions, documented:** deduplicate reviews, strip empties, decide the
   imbalance strategy (class weights vs resampling, chosen per variant later), and
   produce a versioned curated dataset artifact (`data/processed/`, gitignored) with a
   data card in `data/README.md`.
5. Surface 5 to 10 text/rating disagreement rows for later error analysis.
6. Fit `DummyClassifier(strategy="most_frequent")` and record the baseline
   accuracy/precision/recall/specificity/F1.

Acceptance: label mapping documented, Pandera schema passes on the curated set, curated
artifact and data card exist, baseline recorded, disagreement examples captured.

### Phase 2: Shared preprocessing and evaluation

Steps:
1. One shared preprocessing layer per representation: TF-IDF (name it correctly, term
   frequency inverse document frequency) and a tokenized/embedded representation for
   the sequence models. Frequency pruning (`min_df`, `max_df`), not `stop_words=
   'english'`.
2. One shared evaluation function: given predictions and probabilities, return
   precision, recall, F1, specificity, PR-AUC, and a confusion matrix, always for the
   negative class as the class of interest.
3. Cross-validation protocol: repeated stratified k-fold (the rare class makes a single
   split noisy). Report mean and standard deviation across folds. Hold out a final
   test set that no model selection touches.

Acceptance: every model variant is scored through the same eval function on the same
folds, so results are comparable. Metric definitions match the feedback (specificity vs
precision distinguished, baseline-relative).

### Phase 3: Model variants and cross-validated ablations

Build variants that plug into the harness. Suggested set:

- **sklearn (control):** Logistic Regression on TF-IDF, explicit L1 and L2, `C` on a
  log grid. This is the number every deep model must beat.
- **PyTorch:** MLP on TF-IDF; text-CNN on learned embeddings; DistilBERT fine-tune (HF
  Transformers, PyTorch backend).
- **TensorFlow/Keras:** MLP on TF-IDF (parity with the PyTorch MLP); BiLSTM on learned
  embeddings.

Ablation axes to vary and log: representation (TF-IDF vs learned vs pretrained),
architecture (linear/MLP/CNN/RNN/transformer), framework (the PyTorch/TensorFlow MLP
parity pair), and imbalance handling (class weights vs focal loss vs resampling).

Steps:
1. Implement each variant as a config plus a model module, cross-validated through the
   shared harness, all runs logged to MLflow (params, metrics, seed, artifacts).
2. Keep DL models small and regularized with early stopping. Expect overfitting on ~250
   negatives, and document it.
3. Produce an ablation table: variant by precision/recall/F1 (mean and std), sorted,
   with the baseline row on top.

Acceptance: at least the control plus one PyTorch and one TensorFlow variant run and
log cleanly, the PyTorch/TensorFlow parity pair is present, and the ablation table is
generated from MLflow, not hand-copied.

### Phase 4: Analysis and model selection

Steps:
1. Select the final model honestly from the ablation table and a threshold sweep tied
   to the success criterion (negative-class recall >= 0.80), reporting the precision and
   specificity cost.
2. Error analysis on the text/rating disagreement rows: where does the chosen model
   diverge from the star label, and is it ever more right than the label.
3. Fairness as **separation**: negative-class recall and false positive rate across
   device `variation` groups, reported with the spread.
4. Interpretation: coefficients for the linear model, or a lightweight attribution for
   the chosen deep model, to explain what drives negative predictions.

Acceptance: final model chosen with a principled threshold, disagreement error analysis
present, separation check named and reported, interpretation present.

### Phase 5: Containerized inference

Steps:
1. FastAPI service that loads the selected model artifact and exposes a `/predict`
   endpoint (review text in, label plus probability out) and a `/health` endpoint.
2. Dockerfile with pinned dependencies. Build reproducibly. Optional docker-compose to
   run the service plus the monitoring component.
3. A short reproducibility note: exact steps to rebuild the image and reproduce the
   selected model from its config and seed.

Acceptance: `docker build` succeeds, the container serves predictions locally, and the
selected model is reproducible from config plus seed.

### Phase 6: Drift monitoring

Steps:
1. A monitoring module that compares incoming request features and predicted-score
   distributions against a stored training reference using PSI and/or a KS test, with
   configurable alert thresholds.
2. Expose it (a `/metrics` or `/drift` endpoint, or a scheduled check script) and log
   drift measurements.
3. A monitoring design doc: what is monitored, thresholds, what an alert means, and
   what the retraining trigger would be. Framed as designed-and-implemented drift
   monitoring, not live production monitoring.

Acceptance: drift module runs on a simulated shifted batch and flags it, the design doc
exists, and the language is honest about the absence of real production traffic.

### Phase 7: Documentation, CI, lifecycle writeup

Steps:
1. `README.md` covering the full lifecycle: question, dataset curation and validation,
   the label finding, the experiment harness, the ablation results against baseline,
   the chosen model, containerized serving, and the monitoring design. Apply
   the-humanizer voice. An architecture diagram of the lifecycle.
2. `DEVLOG.md` complete.
3. CI (`.github/workflows/ci.yml`): lint, pytest, a tiny smoke run (one small variant,
   one epoch, subset) so CI stays fast, and a `docker build`. Full ablation runs happen
   locally, not in CI.
4. Update the sam-malik.com entry and confirm the resume bullet matches what shipped.

Acceptance: a stranger can read the README, follow the data steps, run CI green, build
the container, and see how every phase connects. The resume bullet is backed by real
artifacts.

### Stretch: live endpoint

Deploy the container to Cloud Run, Fly.io, or a Hugging Face Space for a live
`/predict` link from the README and website. Optional, only after Phases 0 to 7.

## 7. Feedback to fix traceability

Every graded defect still applies, now to the control model and to how all metrics are
reported.

| Feedback item | Phase | Fix |
|---|---|---|
| Encoding never named | 2 | Name TF-IDF and bag-of-words explicitly |
| `stop_words='english'` unjustified | 2 | Frequency pruning, justified |
| Bigrams-vs-small-data hand-wave | 2, 3 | Frame as overfitting; test n-grams as an ablation axis |
| Precision vs specificity confusion | 2, 4 | Report both, name the question each answers |
| "Insights into patterns" is inferential | 4 | Split explainability from prediction |
| Fairness prompt not addressed | 4 | Name separation; check recall/FPR across `variation` |
| Logreg reason is appeal to popularity | 3 | Re-justify on the merits |
| "Logreg handles high-dim well" | 3 | Correct it: M >> N is why you regularize |
| Contenders vague, no reg type | 3 | Explicit L1/L2, stated grids in configs |
| RF rationale wrong; n_estimators not a knob | 3 | If RF is kept, variance rationale and tune depth |
| `stratify` unjustified | 2 | Justify: preserves rare-class prevalence per split |
| Notebook is a script | 0, 3 | Config-driven harness plus a narrative report notebook |
| Overengineered selection loop | 0, 2 | Shared harness and eval, not ad hoc loops |
| "C=10" meaningless | 3, 4 | State penalty type wherever C is reported |
| Proposed further analysis never done | 4 | Delivered in full |
| Part II offensive-speech content | n/a | Not carried over; wrong application |

## 8. Repo structure

```
voice-assistant-sentiment/
├── CLAUDE.md
├── PROJECT_PLAN.md
├── README.md
├── DEVLOG.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml          # optional
├── configs/                    # one file per experiment variant
├── data/
│   ├── README.md               # source, curation notes, data card
│   ├── raw/                    # amazon_alexa.tsv (gitignored)
│   └── processed/              # curated artifact (gitignored)
├── src/
│   ├── run_experiment.py       # config-driven entrypoint
│   ├── data/                   # loading, curation, Pandera schema
│   ├── features/               # TF-IDF and sequence representations
│   ├── models/                 # sklearn, pytorch, tensorflow variants
│   ├── evaluation/             # shared metrics and CV protocol
│   ├── serving/                # FastAPI app
│   └── monitoring/             # drift detection
├── notebooks/
│   └── report.ipynb            # narrative analysis and figures
├── reports/
│   └── figures/
├── tests/
├── reference/
│   ├── original_notebook.ipynb
│   └── original_report.pdf
├── .github/workflows/ci.yml
└── (INSTRUCTOR_FEEDBACK.md at repo root)
```

## 9. Definition of done

- Config-driven harness runs any variant reproducibly and logs to MLflow.
- Dataset curated, Pandera-validated, label finding documented.
- Cross-validated ablation table across sklearn, PyTorch, and TensorFlow variants,
  including the PyTorch/TensorFlow parity pair, reporting precision, recall, F1 with
  fold variance, against the majority baseline.
- Final model selected with a principled threshold; disagreement error analysis,
  separation fairness check, and interpretation present.
- FastAPI service containerized and reproducible.
- Drift monitoring implemented and documented honestly.
- README covers the full lifecycle with an architecture diagram; CI green.
- Website and resume bullet match what shipped.

## 10. Open questions

- Confirm the `feedback` to `rating` mapping early (Phase 1). The analytical spine
  depends on it.
- Confirm you want two infra-flavored projects (this and #01) in the portfolio, given
  the overlap.
- Decide the CI budget: keep DL training out of CI (smoke only) to avoid slow, flaky
  runs. Full ablations run locally.
- Decide whether the live-endpoint stretch is in scope this cycle.
