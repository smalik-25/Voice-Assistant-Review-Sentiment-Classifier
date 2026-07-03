# Data card: Amazon Alexa reviews

## Source

Kaggle, "Amazon Alexa Reviews" (`sid321axn/amazon-alexa-reviews`). Place the raw file
at `data/raw/amazon_alexa.tsv`. It is gitignored and never committed. No download code
lives in the repo; fetch it manually from Kaggle.

Raw shape: 3,150 rows, 5 columns.

| column | type | notes |
|---|---|---|
| `rating` | int | star rating, 1 to 5 |
| `date` | str | review date, for example `31-Jul-18` |
| `variation` | str | device variant, for example `Charcoal Fabric` |
| `verified_reviews` | str | free-text review |
| `feedback` | int | binary label, 1 positive, 0 negative |

## The label is a thresholded rating

`feedback == 1` if and only if `rating >= 3`. Ratings 1 and 2 are negative, 3 through 5
are positive, with no exceptions in the raw data. The label carries no information beyond
the star threshold, so "sentiment" here is a thresholded star count. The Pandera schema
(`src/data/schema.py`) enforces this rule, so a future data refresh that broke it would
fail validation rather than pass silently.

## Curation

Built by `python -m src.data.build_curated`, which loads raw, curates, validates, and
writes the artifact. Steps and their effect on the current raw file:

| step | rows removed | rows remaining |
|---|---|---|
| raw | . | 3,150 |
| drop rows with no review text (NaN or blank) | 80 | 3,070 |
| drop exact duplicate records | 686 | 2,384 |

Curated set: 2,384 rows, 205 negative (8.6%), content hash `543047b2d164`.

Deduplication drops rows that are identical across every column (`rating`, `date`,
`variation`, `verified_reviews`, `feedback`). The reason is leakage: identical records
that land in different cross-validation folds would let a model see test rows during
training and inflate the scores. The tradeoff is that a few of these may be distinct
customers who wrote the same short text (for example "Love it!") on the same day for the
same device; treating them as duplicates removes them. The effect on balance is small:
positive share moves from 92.3% to 91.4%, and 32 of the 686 removed rows are negative.
This decision is reversible by changing `DEDUP_KEYS` in `src/data/load.py`.

Imbalance is left intact in the curated set. The imbalance strategy (class weights vs
resampling) is a per-variant choice made during the ablations, not baked into the data.

## Artifact

`data/processed/curated.parquet` (gitignored). Rebuild it any time from raw with the
command above; the content hash lets you confirm two builds match. Nothing under `data/`
is committed.
