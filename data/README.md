# Data card: Amazon Alexa reviews

## Source

Kaggle, "Amazon Alexa Reviews" (`sid321axn/amazon-alexa-reviews`). I keep the raw file at
`data/raw/amazon_alexa.tsv` and never commit it. There is no download code in the repo;
grab it from Kaggle by hand.

Raw shape: 3,150 rows, 5 columns.

| column | type | notes |
|---|---|---|
| `rating` | int | star rating, 1 to 5 |
| `date` | str | review date, for example `31-Jul-18` |
| `variation` | str | device variant, for example `Charcoal Fabric` |
| `verified_reviews` | str | free-text review |
| `feedback` | int | binary label, 1 positive, 0 negative |

## The label is a thresholded rating

`feedback == 1` if and only if `rating >= 3`. Ratings 1 and 2 are negative, 3 through 5 are
positive, with no exceptions in the raw data. So the label carries nothing beyond the star
threshold, and "sentiment" here is really a thresholded star count. The Pandera schema in
`src/data/schema.py` enforces this, so a future refresh that broke the rule would fail
validation instead of passing quietly.

## What curation does

`python -m src.data.build_curated` loads the raw file, curates it, validates it, and writes
the artifact. On the current data:

| step | rows removed | rows remaining |
|---|---|---|
| raw | . | 3,150 |
| drop rows with no review text (NaN or blank) | 80 | 3,070 |
| drop exact duplicate records | 686 | 2,384 |

Curated set: 2,384 rows, 205 negative (8.6%), content hash `543047b2d164`.

The dedup drops rows that are identical across every column. I do it to prevent leakage: an
identical record showing up in two different cross-validation folds would let a model see
test data during training. The cost is that a few of these might be different customers who
wrote the same short text ("Love it!") on the same day for the same device, and I am
treating them as duplicates. The effect on balance is small (positive share goes from 92.3%
to 91.4%, and only 32 of the 686 removed rows are negative). If I ever want to revisit it, it
is one line: `DEDUP_KEYS` in `src/data/load.py`.

I leave the imbalance alone in the curated set. Whether to reweight or resample is a
per-variant choice I make during the experiments, not something baked into the data.

## The artifact

`data/processed/curated.parquet`, gitignored. Rebuild it any time from raw with the command
above; the content hash tells me whether two builds match. Nothing under `data/` is
committed.
