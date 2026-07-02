# CLAUDE.md — Alexa Review Sentiment Classifier

Read this file and `PROJECT_PLAN.md` in full before doing anything. `PROJECT_PLAN.md`
has the phases and acceptance bars. `INSTRUCTOR_FEEDBACK.md` has the defect list. This
file has the rules for how work happens.

## What this project is

A portfolio rebuild of a class sentiment classifier on the Amazon Alexa reviews
dataset. Binary target `feedback` (1 positive, 0 negative), predicted from the free
text in `verified_reviews`. The point of the project is **modeling judgment and label
skepticism**, not infrastructure. Keep it lean: pandas, scikit-learn, plotly, one
notebook, optionally one small Streamlit app. No Ray, no Airflow, no warehouse, no dbt.
Those belong to other projects in the portfolio.

## What must not get lost: the angle

The `feedback` label is almost certainly a hard threshold on the `rating` column.
Confirm this first (Phase 0). If confirmed, the framing for the whole project is that
"sentiment" here is really a thresholded star rating, and the interesting cases are
where the text and the stars disagree. Distrust the label, characterize it, then build
around it. Do not revert to treating `feedback` as if it were ground-truth human
sentiment.

## Current state

- `reference/original_notebook.ipynb` is a data-loading notebook plus one giant,
  script-style modeling cell.
- The original code has a vectorizer leakage/inconsistency bug (fit separately for
  selection and final fit), reports accuracy on a ~92% majority-positive label, and
  does no threshold or PR analysis.
- Reported class results: accuracy 0.946, precision 0.758, negative-class recall
  0.490. The 0.49 recall is the real story.

Full detail is in `PROJECT_PLAN.md` section 3.

## Environment and stack

- Python 3.11+, scikit-learn, pandas, numpy, plotly, jupyter, nbconvert. Streamlit
  only for the Phase 6 stretch app.
- Development happens locally on Sam's Mac. The dataset lives at
  `data/raw/amazon_alexa.tsv` and is gitignored. Do not download it into the repo and
  do not commit it.
- If the sandbox cannot reach Kaggle, assume the tsv is already present at
  `data/raw/`. Do not add `kagglehub` download calls to the committed notebook; point
  to `data/raw/amazon_alexa.tsv` and document the source in `data/README.md`.

## Repo structure

Mirror the layout in `PROJECT_PLAN.md` section 8. Notebook goes in `notebooks/`,
stretch app in `app/`, CI in `.github/workflows/`. Keep imports in a single cell at
the top of the notebook.

## Git workflow (non-negotiable)

- **Never run git from the sandbox.** No `git add`, `commit`, `push`, `init`, or any
  other git command executed by you. It causes `.lock` files and there are no GitHub
  credentials here.
- Your job: prepare and edit files, then hand Sam an exact copy-paste block for the
  Terminal. Sam runs every git command himself.
- After each meaningful chunk of work (a completed phase or a self-contained step):
  1. Append a dated entry to `DEVLOG.md` describing what changed and why.
  2. Update the checklist in `README.md` (once the README exists).
  3. Give Sam a copy-paste block: `git add` of the specific files, a Conventional
     Commits message, and `git push` only when agreed.
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`,
  `test:`). Scope where useful, for example `feat(model): ...`.
- Never assume a commit or push happened. If you need the repo in a certain state,
  ask Sam to confirm he ran the block.

Example block format:

```bash
git add notebooks/sentiment_analysis.ipynb DEVLOG.md
git commit -m "feat(eval): add PR curve and threshold sweep for negative class"
git push
```

## Notebook conventions

- Single imports cell at the top. Nothing imported further down.
- Every section starts with a short markdown header and one to three sentences of
  intent: the goal of the section, not a line-by-line description of the code.
- All modeling goes through an `sklearn` `Pipeline(vectorizer, classifier)`. The
  vectorizer is only ever fit inside training folds or on the training split, never on
  validation or test data, and never fit twice on different subsets.
- Prefer `GridSearchCV` / cross-validation over hand-written selection loops. The
  original manual loop was flagged as overengineered and obfuscating.
- The notebook must run top to bottom with no errors, assuming the tsv is present.

## Modeling standards (non-negotiable)

These come straight from the graded feedback. Do not reintroduce the mistakes.

- **Name the encoding.** CountVectorizer counts are a bag-of-words / term-frequency
  encoding. Say so. Compare against TF-IDF.
- **State the regularization penalty every time.** Reporting a `C` value without
  saying L1 or L2 is meaningless. Default to comparing L1 and L2 explicitly with the
  `saga` solver. L1 is easier to justify here because it gives sparse, interpretable
  coefficients and does feature selection, which serves the explainability goal.
- **Grids are stated in advance.** For `C`, use a log grid such as
  `np.logspace(-3, 2, 12)`. If the best value lands on an edge, extend the grid.
- **M >> N is the core overfitting risk.** The vocabulary is much larger than the row
  count. This is why regularization matters and why the "logreg handles high-dim data
  well" claim is wrong.
- **Do not use `stop_words='english'`.** The built-in list was built for
  computer-science text and drops words like "computer." Use frequency pruning
  (`min_df`, `max_df`) instead and justify it.
- **Random Forest rationale is variance reduction via bagging**, not "capturing
  complex patterns" and not "finding less complex things." Tune `max_depth` and
  `min_samples_leaf`, not `n_estimators` (which is not a bias/variance knob).
- **Justify `stratify`.** It preserves the roughly 8% negative prevalence in every
  split so estimates are not distorted by an unlucky draw of a rare class.

## Metric rules

- Never lead with accuracy. On a ~92% positive label it is barely above a majority
  baseline. Always report a `DummyClassifier(strategy="most_frequent")` floor and
  compare against it.
- Lead with minority-class metrics: negative-class precision, recall, specificity, a
  confusion matrix, a precision-recall curve, and PR-AUC.
- Distinguish specificity from precision explicitly. Specificity answers "how often
  are positive reviews wrongly flagged as negative." Precision answers "how
  trustworthy is a negative flag." State which question each metric serves.
- Do not treat the 0.5 decision threshold as given. Sweep it and choose an operating
  point tied to the stated success criterion (negative-class recall >= 0.80).
- Fairness in this project means **separation**: check whether negative-class recall
  and false positive rate are roughly equal across device `variation` groups. Name the
  criterion when you report it.

## Writing and prose conventions

- All human-facing prose (README, data/README, website copy) gets the-humanizer voice:
  concise, high signal, no filler, no corporate buzzwords, peer-level confident.
- **No em dashes anywhere.** Use commas, parentheses, or restructure the sentence.
- README structure: the question, the approach, the label finding, honest results
  against baseline, what I learned. The "what I learned" should contain a real
  engineering or statistical lesson, not a summary.
- Internal docs (DEVLOG, this file, the plan) use the same plain register.

## Definition of done

See `PROJECT_PLAN.md` section 7. In short: clean runnable narrative notebook, every
graded defect closed or consciously cut, baseline-relative minority-class metrics, the
label finding used, fairness-as-separation and coefficient interpretation present,
full repo hygiene with green CI, website copy updated. The stretch app is a bonus.

## Never do

- Never run git from the sandbox.
- Never commit the dataset or add Kaggle download calls to the committed notebook.
- Never report a `C` value without its penalty type.
- Never use `stop_words='english'`.
- Never lead with accuracy or omit the majority baseline.
- Never fit the vectorizer outside a training fold.
- Never reintroduce the Part II "offensive speech" content. That was from the wrong
  application and does not belong here.
- Never use em dashes.
- Never claim a commit or push happened. Confirm with Sam.

## Reference files

- `PROJECT_PLAN.md` — phases, acceptance bars, traceability table.
- `INSTRUCTOR_FEEDBACK.md` — the full defect list.
- `reference/original_notebook.ipynb` — the current notebook.
- `reference/original_report.pdf` — the class deliverable.
