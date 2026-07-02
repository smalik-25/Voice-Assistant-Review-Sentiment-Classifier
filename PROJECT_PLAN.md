# Project Plan: Alexa Review Sentiment Classifier (portfolio rebuild)

## 1. Goal

Turn a thin INFO 371 class assignment into a portfolio project that matches the
rigor of the top projects on sam-malik.com. The identity of this project is
**modeling judgment and label skepticism**, not infrastructure. It is the project
that proves I understand what the metrics mean, what the label actually is, and why a
regularized linear model is the right call. It deliberately stays lean: no Ray, no
Airflow, no warehouse. Project #01 (Sneaker Price MLOps Pipeline) already owns the
infra story.

## 2. Positioning

Two existing projects create overlap risk, so the differentiation has to be explicit.

- vs #01 (MLOps pipeline): that one is about distributed training and orchestration.
  This one is about statistical reasoning on a hard, imbalanced problem. Keep it
  single-notebook plus a thin app, no pipeline tooling.
- vs #09 (Privacy-Aware Fashion Review Risk Detector): same technique family (sparse
  text features into a regularized linear model, recall on the minority class). The
  three things that make this project distinct:
  1. **Label interrogation.** The `feedback` label is almost certainly a hard
     threshold on the `rating` column, so "sentiment" is really "high star rating."
     The interesting cases are where text and stars disagree. This is the same move
     as the fitness project (distrust the number, then decompose it).
  2. **Encoding and regularization as a deliberate comparison**, named correctly,
     with explicit penalty types.
  3. **Fairness framed as separation across device variations**, which also doubles
     as the further analysis the class version never delivered.

## 3. Current state (read this before touching anything)

What exists today:

- **Data-loading notebook** (11 cells). Cells 0 to 9 download the Kaggle Amazon Alexa
  reviews dataset via `kagglehub`, load `amazon_alexa.tsv`, print shape and class
  balance, drop rows with missing `verified_reviews`, and draw one Plotly histogram of
  the label. This part is fine as scaffolding.
- **One giant modeling cell** (cell 10). It re-loads the data, does a train/val/test
  split with `stratify`, vectorizes with `CountVectorizer(stop_words='english')`,
  loops logistic C over {0.1, 1, 10} and RF `n_estimators` over {50, 100, 200}
  selecting on negative-class recall, then refits the winner and prints accuracy,
  precision, recall.

Known problems in the current code (beyond the graded feedback):

- **Leakage / inconsistency in vectorization.** The vectorizer is `fit_transform` on
  `X_train` for the selection phase, then `fit_transform` again on `X_train_full`
  before the test transform. Selection and final fit therefore run on different
  feature spaces. This needs to become a single `Pipeline` fit only on training folds.
- **Accuracy is reported on a ~92% majority-positive label**, so 0.946 accuracy is
  barely above a majority-class baseline. There is no `DummyClassifier` floor.
- **Only negative-class recall drives selection**, with no precision/specificity
  tradeoff, no threshold tuning, no PR curve.
- **No markdown narrative.** It reads as a script.

Reported final results from the class report: accuracy 0.946, precision 0.758, recall
0.490 on the negative class. The recall of 0.49 (the model misses half the negatives)
is the real headline and the thing to fix or at least honestly explain.

Reference material in the repo:
- `reference/original_notebook.ipynb` — the current notebook.
- `reference/original_report.pdf` — the class deliverable.
- `INSTRUCTOR_FEEDBACK.md` — the full defect list.

## 4. The core angle (do not lose this)

Confirm and then build around one finding: **`feedback` is a deterministic function
of `rating`.** If true (verify in Phase 0), the label is not human sentiment, it is a
thresholded star rating. Everything follows from that:

- The task is really "recover the star-rating threshold from free text."
- The genuinely interesting rows are the disagreements: gushing text with a low star
  rating, or a terse "it broke" with a high one. These are where a text model can add
  something a rating cannot, and they are the error-analysis material.
- Heavy imbalance plus a proxy label is exactly why accuracy is the wrong headline
  metric and why the precision/recall/specificity distinction matters.

## 5. Phases

Each phase has concrete steps and an acceptance bar. After each phase: write a DEVLOG
entry, update the README checklist, and I get a copy-paste git block (see CLAUDE.md
git workflow). Do not move to the next phase until the acceptance bar is met.

### Phase 0: Label and baseline investigation

Steps:
1. Load `data/raw/amazon_alexa.tsv`. Confirm shape, dtypes, missing values.
2. Cross-tabulate `feedback` against `rating`. Determine the exact mapping and state
   it plainly (expected: rating in {1,2} maps to 0, {3,4,5} maps to 1, or similar).
3. Quantify imbalance (expected roughly 92% positive).
4. Fit `DummyClassifier(strategy="most_frequent")` and record its accuracy, precision,
   recall, and specificity as the floor.
5. Surface 5 to 10 rows where text sentiment and star rating visibly disagree.

Acceptance: the label-to-rating mapping is documented, the majority baseline numbers
are recorded, and disagreement examples are captured for later.

### Phase 1: Rebuild the notebook as a narrative

Steps:
1. One imports cell at the top. Nothing imported later.
2. Markdown sections with intent statements (one to three sentences each): Data and
   Label, Preprocessing, Model Selection, Evaluation, Fairness, Further Analysis.
3. Move all modeling into an `sklearn` `Pipeline(vectorizer, classifier)` so the
   vectorizer is only ever fit on training folds. This removes the leakage and the
   overengineered manual loop in one move.

Acceptance: notebook runs top to bottom with no errors, reads as an explained
analysis rather than a script, no vectorizer is fit outside a pipeline/CV fold.

### Phase 2: Model selection, done correctly

Steps:
1. `GridSearchCV` (stratified k-fold) over a real grid:
   - penalty in {l1, l2} with the `saga` solver.
   - `C` on a log grid, for example `np.logspace(-3, 2, 12)`. If the best C sits at an
     edge of the grid, extend the grid.
   - `class_weight` in {None, "balanced"}.
2. Encoding comparison, each named correctly:
   - bag-of-words / term-frequency counts vs TF-IDF (term frequency, inverse document
     frequency).
   - unigram vs unigram+bigram (this directly tests the "not good" interaction claim
     from Part I).
   - stop-word handling: replace the built-in `english` list with frequency pruning
     (`min_df`, `max_df`) and justify.
3. RF contender: tune `max_depth` and `min_samples_leaf` (real bias/variance knobs),
   not `n_estimators`. State the correct rationale: bagging reduces variance.

Acceptance: selection uses cross-validation, penalty type is explicit, grids are
stated in advance, and the RF rationale and knobs are corrected. Expect the linear
model to win. If it does, say so and tie it to "complexity has to earn its place."

### Phase 3: Honest evaluation on the minority class

Steps:
1. Confusion matrix on the held-out test set.
2. Precision-recall curve and PR-AUC (more informative than ROC under imbalance).
3. Threshold sweep instead of the default 0.5 cutoff. Choose an operating point tied
   to the stated criterion (negative-class recall >= 0.80) and report what it costs in
   precision and specificity.
4. Report **specificity and precision side by side** and state which question each
   answers: specificity = "how often are positive reviews wrongly flagged,"
   precision = "how trustworthy is a negative flag." This closes the Part I metric
   error directly.

Acceptance: the report leads with baseline-relative, minority-class-aware metrics; the
0.5 threshold is no longer treated as given; specificity vs precision is explained.

### Phase 4: The analyses that were promised

Steps:
1. Plotly variation-level view: predicted negative rate (and/or error rate) by device
   `variation`. This is the further analysis the class version skipped.
2. Reframe it as a **separation** fairness check: is negative-class recall and false
   positive rate roughly equal across variations? Report the spread.
3. Coefficient interpretation from the L1 model: top positive and negative words. Use
   this to argue why L1 beats L2 for the explainability goal here.

Acceptance: one clear Plotly figure per analysis, the fairness check names the
statistical criterion (separation), and coefficients are interpreted, not just dumped.

### Phase 5: Repository hygiene

Steps:
1. `README.md` in the house style: question, approach, the label finding, honest
   results vs baseline, what I learned. Apply the-humanizer to the prose.
2. `requirements.txt` pinned, `.gitignore` (exclude `data/raw/`), `data/README.md`
   with the Kaggle source and download instructions (no data committed).
3. `.github/workflows/ci.yml` that runs the notebook end to end (`jupyter nbconvert
   --execute`) or a smoke test, on push.
4. `DEVLOG.md` up to date.

Acceptance: a stranger can clone, read the README, follow the data instructions, and
run CI green. Structure mirrors sneaker-intel.

### Phase 6 (stretch): thin live demo

Steps:
1. A small Streamlit app: paste a review, see the prediction, the words driving it
   (coefficient contributions), and the recall/precision tradeoff at the chosen
   threshold.
2. Deploy on Streamlit Community Cloud, link from the README and the website.

Acceptance: optional. Only start once Phases 0 to 5 are done and committed. The
notebook plus README already stands alone without this.

### Wrap-up

Update the sam-malik.com project entry so the copy describes the label-skepticism
angle and the honest metric story, not a generic "compared logreg vs RF."

## 6. Feedback to fix traceability

Every graded defect maps to a phase. Nothing is dropped silently.

| Feedback item | Phase | Fix |
|---|---|---|
| Encoding never named | 2 | Call it bag-of-words / term-frequency; compare vs TF-IDF |
| `stop_words='english'` unjustified, class warned against it | 2 | Frequency pruning (`min_df`/`max_df`), compared |
| Bigrams-vs-small-data hand-wave | 2 | Frame as overfitting: bigrams explode M with sparse counts |
| Precision vs specificity confusion | 3 | Report both, name the question each answers |
| "Insights into patterns" is inferential | 4 | Split explainability (coefficients) from prediction |
| Fairness prompt not addressed | 4 | Name it separation; check recall/FPR across `variation` |
| Logreg reason is appeal to popularity | 1 | Re-justify: sparse high-dim text is near linearly separable |
| "Logreg handles high-dim well" | 2 | Correct it: M >> N is why you regularize |
| Contenders vague, no reg type | 2 | Explicit L1/L2, stated `C` grid |
| RF rationale wrong; n_estimators not a real knob | 2 | Variance-reduction rationale; tune `max_depth` |
| `stratify` unjustified | 1 | Justify: preserves ~8% negative prevalence per split |
| Notebook is a script | 1 | Markdown structure, one imports block, per-section intent |
| Overengineered selection loop | 1, 2 | `Pipeline` + `GridSearchCV` |
| "C=10" meaningless | 2, 3 | State penalty type everywhere C is reported |
| Proposed further analysis never done | 4 | Delivered in full |
| Part II answers from wrong application | n/a | Cut the class Part II/VI scaffolding entirely |

## 7. Definition of done

- Notebook runs clean top to bottom, reads as an explained analysis.
- Every graded defect above is closed or consciously cut.
- Results are reported against the majority baseline, minority-class first.
- The label-to-rating finding is stated and used.
- Fairness-as-separation and coefficient interpretation are present.
- README, requirements, .gitignore, data/README, CI all in place; CI green.
- Website copy updated.
- Stretch app is a bonus, not a gate.

## 8. Repo structure

```
voice-assistant-sentiment/
├── CLAUDE.md
├── PROJECT_PLAN.md
├── README.md                 # Phase 5
├── DEVLOG.md                 # append-only
├── requirements.txt          # Phase 5
├── .gitignore                # Phase 5
├── data/
│   ├── README.md             # source + download steps, no data committed
│   └── raw/                  # amazon_alexa.tsv (gitignored)
├── notebooks/
│   └── sentiment_analysis.ipynb
├── reports/
│   └── figures/              # exported figures if needed
├── app/                      # Phase 6 stretch
│   └── app.py
├── reference/
│   ├── original_notebook.ipynb
│   ├── original_report.pdf
│   └── (INSTRUCTOR_FEEDBACK.md lives at repo root)
└── .github/workflows/ci.yml  # Phase 5
```

## 9. Open questions to resolve early

- Confirm the exact `feedback` to `rating` mapping (Phase 0). The whole angle depends
  on it. If for some reason it is not a clean threshold, adjust the framing.
- Decide the primary operating metric before Phase 3: negative-class recall at a fixed
  minimum specificity/precision, or the reverse. Tie it to the stated success
  criterion (recall >= 0.80) so the threshold choice is principled, not post hoc.
- Decide whether the stretch app is in scope for this cycle or a later one.
