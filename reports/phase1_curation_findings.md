# Phase 1: Label and baseline investigation

**Exact label mapping: `feedback == 1` if and only if `rating >= 3` (ratings 1 to 2 map
to 0, ratings 3 to 5 map to 1), with no exceptions in the data.**

Data: `data/raw/amazon_alexa.tsv`, 3150 rows, 5 columns (`rating`, `date`,
`variation`, `verified_reviews`, `feedback`).

## The label is a thresholded star rating (confirmed)

`feedback` is a deterministic function of `rating`. Every rating maps to exactly one
feedback value, with no exceptions:

| rating | feedback | count |
|---|---|---|
| 1 | 0 | 161 |
| 2 | 0 | 96 |
| 3 | 1 | 152 |
| 4 | 1 | 455 |
| 5 | 1 | 2286 |

So the rule is: rating in {1, 2} gives `feedback = 0`, rating in {3, 4, 5} gives
`feedback = 1`. The check `(rating in {1,2}) == (feedback == 0)` holds for all 3150
rows. "Sentiment" here is not a human judgment of the text, it is a hard threshold on
the star count. The task is really "recover the star-rating threshold from free text,"
and the rows worth studying are the ones where the text and the stars disagree.

## Imbalance

Positive share is 0.9184 (2893 positive, 257 negative). The negative class is 8.2% of
the data. This is why accuracy is the wrong headline metric and why `stratify` matters
later: an unlucky split could badly distort estimates for a class this rare.

## Data quality note

`verified_reviews` has 1 true NaN and 79 blank (empty-string) entries, so 80 rows carry
no usable text. These get dropped before modeling.

## Majority baseline floor (DummyClassifier, most_frequent)

Predicting "positive" for everything:

| metric | value |
|---|---|
| accuracy | 0.9184 |
| negative-class precision | 0.0000 |
| negative-class recall | 0.0000 |
| specificity | 1.0000 |

This is the floor every real model must beat. For comparison, an accuracy of 0.946
sits only about 3 points above this do-nothing baseline, which is the whole reason
accuracy cannot lead. The floor gets negative-class recall of 0, so any recall on the minority
class is progress that accuracy alone hides.

## Disagreement examples captured

69 candidate rows where text and star rating pull in opposite directions are saved to
`reports/phase0_disagreements.csv` (30 gushing text at 1 to 2 stars, 39 complaint-laden
text at 4 to 5 stars). These are the error-analysis material for later phases. A few:

- `[1 star, feedback 0]` "great product, but useless overall. Too many unnecessary
  features."
- `[1 star, feedback 0]` "Great product but returning for new Alexa Dot. Refurbished is
  already giving me problems with connection."
- `[4 stars, feedback 1]` "We really only use this as a speaker to stream music. We've
  had it 7 months and it's just kinda useless."
- `[4 stars, feedback 1]` "The outlet does not work with it. Was disappointed in that."

## Acceptance bar

Met: the label-to-rating mapping is documented, the majority baseline numbers are
recorded, and disagreement examples are captured for later phases.
