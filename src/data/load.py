"""Load and curate the raw Amazon Alexa reviews dataset.

Curation is deliberate and documented: strip whitespace and the UTF-8 BOM from the
header, drop rows with no usable review text, and drop exact duplicate records
(identical across every column). Deduplication is what keeps identical rows from
straddling cross-validation folds later, which would leak. The class imbalance is left
intact; the imbalance strategy (class weights vs resampling) is a per-variant choice
made in later phases, not baked into the curated set.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/amazon_alexa.tsv")
DEDUP_KEYS = ["rating", "date", "variation", "verified_reviews", "feedback"]


@dataclass
class CurationReport:
    raw_rows: int
    dropped_blank: int
    dropped_duplicates: int
    curated_rows: int
    negatives: int


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw TSV and normalize the header (strip whitespace and BOM)."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def curate(df: pd.DataFrame) -> tuple[pd.DataFrame, CurationReport]:
    """Return the curated frame and a report of what each step removed."""
    raw_rows = len(df)
    out = df.copy()
    out["verified_reviews"] = out["verified_reviews"].astype("string").str.strip()
    out["variation"] = out["variation"].astype("string").str.strip()

    has_text = out["verified_reviews"].notna() & (out["verified_reviews"].str.len() > 0)
    dropped_blank = int((~has_text).sum())
    out = out[has_text]

    before_dedup = len(out)
    out = out.drop_duplicates(subset=DEDUP_KEYS).reset_index(drop=True)
    dropped_duplicates = before_dedup - len(out)

    out["rating"] = out["rating"].astype(int)
    out["feedback"] = out["feedback"].astype(int)
    out["variation"] = out["variation"].astype(str)
    out["verified_reviews"] = out["verified_reviews"].astype(str)
    out["date"] = out["date"].astype(str)

    report = CurationReport(
        raw_rows=raw_rows,
        dropped_blank=dropped_blank,
        dropped_duplicates=dropped_duplicates,
        curated_rows=len(out),
        negatives=int((out["feedback"] == 0).sum()),
    )
    return out, report
