# Instructor Feedback (verbatim, reorganized)

Source: graded INFO 371 class deliverable. This is the raw defect list. Every item
here must be addressed or consciously dropped in the portfolio rebuild. Do not
silently ignore any of it.

Total deductions: Part I (-0.25), Part IV (-0.25), Part V (-0.25).

---

## Part I: General Planning (-0.25)

**Non-linearities / interactions.** On "starting with simple word counts is a
reasonable approach": the reviewer asks *why* this is more reasonable for a small
dataset. Is it to avoid overfitting? Also note there are fewer tokens to find. The
original answer hand-waved this.

**Preprocessing.** CountVectorizer is referenced, but the resulting encoding is
never named in the terms used in class.

**Metrics.** Prompt not addressed. Should: copy a snippet from the proposal as
evidence the metric matters, then write 1 to 3 sentences on why it is a good metric
for this use case.

**Fairness.** Prompt not really addressed. What approach to *statistical* fairness
is this? How would you check it?

**Success criteria.** The proposers gave specific numeric values (which they were
not really supposed to).
- "The model achieves high precision for negative reviews so that positive reviews
  are not incorrectly flagged as negative." If the goal is "positive reviews are not
  incorrectly flagged as negative," the correct metric is **specificity**, not
  precision.
- "The model provides useful insights into patterns in the review data (device
  variations that receive more negative feedback)" is an **inferential** goal, not a
  prediction goal, unless it is reframed as explainability (how the model works)
  rather than what is happening in the world.

## Part II: Proposal Q&A

The submitted Q&A answers talk about **offensive speech**, which belongs to a
different application. The answers appear to have been taken from the wrong 2025
proposal link and are not responses the in-person group would give. This whole
section is class-process scaffolding and does not belong in the portfolio version.

## Part III: Modelling Plan

**Logistic Regression rationale.** "logistic regression is commonly used for text
classification and works well with high-dimensional word features" is *argumentum ad
populum* (appeal to popularity). Fallacious rationale.

**Overfitting.** "logistic regression generally handles high-dimensional data well"
is wrong in many senses. This is exactly why sklearn defaults to regularization.
This is a prime case where overfitting is a major concern: M (features) will be much,
much bigger than N (rows).

**Contenders (logreg).** No concrete values listed to try (for example 0.1, 1, 10,
100). No regularization type chosen. Must say L1, L2, or other, and why.

**Random Forest rationale.** Wrong. RF does not "capture more complex patterns" or
"find less complex things." Its advantage is lowering variance / avoiding overfitting
relative to a single deep tree.

**Contenders (RF).** Number of trees is not a hyperparameter with a U-shaped
performance curve and was not practiced in class. Use something that was practiced
and accomplishes the stated goal (for example max depth).

## Part IV: Execute the Analysis (-0.25)

The notebook is a script pasted into a notebook file, not a readable notebook. No
document structure or markdown to justify and interpret work. Import statements
should all be in one place.

**Splits.** The `stratify` argument was used but not justified. What are the pros and
cons?

**Encoding.** `CountVectorizer(stop_words='english')` was used without trying other
options. The encoding is never named (it is a term-frequency encoding). The class
did not use the built-in `english` stop-word list, and warned against it: that list
was built for computer-science text and drops words like "computer," so it is not
generally appropriate.

**Model selection.** The code is overengineered, obfuscates what is happening, and is
not designed for a notebook. The C values were not specified in advance. If the best
value is the largest tried, why not try larger? The regularization type was not
chosen and defaults to L2, which is harder to justify than L1 for this task.

## Part V: Model Report (-0.25)

**Final approach.** "using regularization with C = 10" is nearly meaningless without
stating the regularization penalty type.

**Further analysis.** The proposed further analyses (variation-level insight) were
never actually done.
