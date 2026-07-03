# Dev log

Notes to myself as I built this, newest at the bottom.

## The label

First thing I checked was what `feedback` actually is. It turned out to be a deterministic
function of `rating`: every 1 or 2 star review is negative, every 3 to 5 star review is
positive, no exceptions across all 3,150 rows. So the label is a thresholded star count, not
a human judgment, and the reviews worth studying are the ones where the text and the stars
pull in opposite directions. I pulled 69 of those disagreement rows aside for later. The
data is about 92% positive, so I fixed the majority baseline (`DummyClassifier`) as the
floor everything gets measured against.

## Curation and validation

I load the raw TSV, drop the 80 rows with no usable text, and drop 686 exact duplicate
records. The duplicates matter because identical rows in different CV folds would leak. That
leaves 2,384 reviews, 205 negative. I wrote a Pandera schema that enforces the column types,
non-empty text, a binary label, and the label rule itself, so if a data refresh ever broke
`feedback == 1 iff rating >= 3` it would fail validation instead of slipping through. The
curated set is a versioned parquet artifact; I kept the data card in `data/README.md`.

## The harness

I did not want modeling logic scattered across a notebook, so every experiment is a config
file and `src/run_experiment.py` runs it: load config, seed everything from it, run the
variant, log to MLflow. A no-op config was the first thing that worked, and I checked it
reproduces the same seeded number across runs. Then I built the shared pieces: TF-IDF and
bag-of-words in `src/features`, one evaluation function that always reports the negative
class (precision, recall, F1, specificity, PR-AUC, confusion matrix), and a repeated
stratified k-fold protocol. Every variant goes through the same path, so the comparison is
honest.

## The control, and a thinner notebook

The sklearn control is logistic regression on TF-IDF, with an explicit L1 vs L2 and `C`
grid, selected by cross-validation and logged to MLflow with its grid table. I moved the old
script-style notebook into `notebooks/report.ipynb` as a thin narrative that calls into
`src/` and presents results, rather than doing the modeling itself.

## PyTorch and TensorFlow parity

I wrapped a small MLP as a scikit-learn-compatible estimator in both PyTorch and TensorFlow,
same architecture and hyperparameters, so they run through the exact same folds and
evaluation as the control. Ran locally (the deep-learning wheels are heavy, so those stay off
CI). The two came out within a few thousandths of each other, which is the whole point of
running both: the harness is framework-agnostic and the result reproduces across backends.

## The selection mistake, and the fix

The first ablation embarrassed the control: selecting on cross-validated negative recall
drove the linear model into a corner where it predicted "negative" for almost everything
(recall 0.95, precision 0.15, PR-AUC 0.28, the worst of any real model). The neural nets
looked better only because the linear model's selection was broken. Switching selection to
PR-AUC, which does not care about the 0.5 threshold, fixed it: the linear model jumped to
PR-AUC 0.57 and back to the top, with the MLPs around 0.52. The lesson is that selection and
the decision threshold are two different choices. I also made the ablation table rank by
PR-AUC instead of F1 at 0.5, since F1 at a fixed threshold understates a model that operates
at a tuned one.

For the operating point I sweep the threshold on out-of-fold training predictions (never the
test set) to reach negative recall of 0.80, then apply it to the held-out test. On unseen
data that is recall 0.76, precision 0.37, specificity 0.88. It is a real trade and I report
it as one.

## Analysis

Three things on the selected model. The coefficients read the way you would hope: negation
and complaint words drive the negative class, praise drives the positive one. The L2 model is
dense (1,839 words); an L1 model at the same `C` is 85% sparse (267 words) for the same
ranking, which is the concrete argument for L1 when you want a short explanation. Fairness as
separation across device `variation` showed a consistent false-positive rate but noisy
per-group recall, and the honest reason is that most variations have only a handful of
negatives, so I report it with counts rather than claiming a disparity. The disagreement rows
are the fun part: the model mostly flags complaint language in high-star reviews (a 5-star
review whose text says the device "stopped replying to my requests"), and it misses terse or
sarcastic low-star reviews, which a bag-of-words model just cannot see. Those are the cases
where the text model is sometimes more right than the label. All of it is in the report
notebook.

## Serving

`src/serving/export_model.py` fits the selected model on the curated data, computes the
operating threshold, and saves a plain scikit-learn bundle plus the threshold, so the service
loads it with scikit-learn alone. `app.py` is a small FastAPI service with `/health` and
`/predict` that applies the operating threshold, not 0.5. The Dockerfile installs only
`requirements-serve.txt`, so the image carries none of the training stack. Confirmed the
image builds and serves locally: "it stopped working ... support was useless" comes back
negative, "love my echo, works great" positive.

## Monitoring

This is designed-and-implemented drift monitoring, not live monitoring, because there is no
real traffic. It compares a batch against the training reference on the negative-score
distribution, review length, and out-of-vocabulary rate, using PSI as the trigger and a KS
test as a second opinion. It runs as a scheduled check that exits non-zero on drift. I tested
it on a batch with out-of-vocabulary tokens appended and it flagged the length and OOV
shifts while correctly leaving the score distribution alone. The design doc in
`src/monitoring/README.md` covers thresholds and the retraining trigger.

## Docs and CI

Wrote the full lifecycle README with an architecture diagram, and a CI workflow that lints,
runs the tests, does a smoke experiment, and builds the Docker image. CI installs a lean set
without torch or tensorflow so it stays fast; the neural tests skip themselves when those
are not installed, and the full ablation runs stay local.

## Deploy

I added a small HTML demo page at `/` on the service (paste a review, get the prediction) so
a live link is usable without curl, on top of the FastAPI `/docs`. For hosting I set up a
Hugging Face Docker Space: `deploy/hf_space/` has the Space Dockerfile (same slim runtime,
port 7860), the Space README with its frontmatter, and `DEPLOY.md` with the assemble-and-push
steps. The Space repo carries the small model bundle; the raw dataset never ships. The plan
is to push it, then link the live URL from this README and the site.

## How solid is the finding

The whole story leans on the linear model's PR-AUC (0.575) beating the MLPs (about 0.52), so I
went back and checked whether that gap is real. The test set has only 41 negatives, so I
bootstrapped the held-out PR-AUC: resample the test rows with replacement, recompute, take a
95% interval. `src/evaluation/bootstrap.py` does the per-variant interval and a paired
resample (same rows for both models) to compare two variants; each variant now saves its
held-out predictions through the harness so the comparison runs on identical rows, and
`bootstrap_report.py` prints it.

The control's PR-AUC interval is [0.42, 0.72], which is wide, and the MLPs sit inside it. The
paired comparison confirms it: control minus PyTorch is +0.049 with a 95% interval of
[-0.026, +0.124], and control minus TensorFlow is +0.052 [-0.024, +0.128]. Both straddle zero,
so neither gap is significant, though the control wins about 90% of resamples so it does lean
better. Against the baseline it is unambiguous (+0.489 [+0.346, +0.623], every resample). So I
stopped saying the linear model beats the neural nets and started saying it matches them at a
fraction of the complexity, which is the defensible claim on data this small.
