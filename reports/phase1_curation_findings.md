# Label and baseline notes

The one thing to know before anything else: `feedback == 1` if and only if `rating >= 3`.
Ratings 1 and 2 map to 0, ratings 3 to 5 map to 1, with no exceptions in the data.

Source: `data/raw/amazon_alexa.tsv`, 3,150 rows, 5 columns (`rating`, `date`, `variation`,
`verified_reviews`, `feedback`).

## The label is a thresholded star rating

`feedback` is a deterministic function of `rating`. Every rating maps to exactly one feedback
value:

| rating | feedback | count |
|---|---|---|
| 1 | 0 | 161 |
| 2 | 0 | 96 |
| 3 | 1 | 152 |
| 4 | 1 | 455 |
| 5 | 1 | 2286 |

The check `(rating in {1,2}) == (feedback == 0)` holds for all 3,150 rows. So "sentiment"
here is not a judgment of the text, it is a hard cut on the star count. The real task is
recovering that threshold from free text, and the rows I care about are the ones where the
text and the stars disagree.

## Imbalance

Positive share is 0.9184 (2,893 positive, 257 negative), so the negative class is about 8.2%
of the data. That is why accuracy is the wrong headline, and why I stratify the splits: an
unlucky random split could badly distort the estimates for a class this rare.

## A data-quality note

`verified_reviews` has 1 true NaN and 79 blank entries, so 80 rows carry no usable text. I
drop them before modeling.

## The majority baseline

Predicting "positive" for everything:

| metric | value |
|---|---|
| accuracy | 0.9184 |
| negative-class precision | 0.0000 |
| negative-class recall | 0.0000 |
| specificity | 1.0000 |

This is the floor every real model has to beat. An accuracy of 0.946 sits only about 3 points
above it, which is the whole reason accuracy can't lead. The floor catches zero negatives, so
any recall on the minority class is progress that accuracy alone would hide.

## Disagreement rows I set aside

I saved 69 rows where the text and the stars pull in opposite directions to
`reports/phase0_disagreements.csv` (30 with gushing text at 1 to 2 stars, 39 with complaint
text at 4 to 5 stars). These are the error-analysis material for later. A few:

- `[1 star, feedback 0]` "great product, but useless overall. Too many unnecessary features."
- `[1 star, feedback 0]` "Great product but returning for new Alexa Dot. Refurbished is
  already giving me problems with connection."
- `[4 stars, feedback 1]` "We really only use this as a speaker to stream music. We've had it
  7 months and it's just kinda useless."
- `[4 stars, feedback 1]` "The outlet does not work with it. Was disappointed in that."
