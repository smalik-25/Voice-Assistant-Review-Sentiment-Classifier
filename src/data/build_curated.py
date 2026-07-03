"""Build the versioned curated dataset artifact.

Loads raw, curates, validates against the Pandera schema, writes
``data/processed/curated.parquet``, and prints a content hash so the artifact is
versioned. The output is gitignored and rebuilt from raw on demand.

Usage:
    python -m src.data.build_curated
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.load import curate, load_raw
from src.data.schema import validate

OUT = Path("data/processed/curated.parquet")


def content_hash(df: pd.DataFrame) -> str:
    """A short, order-sensitive content hash for versioning the curated set."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()[:12]


def main() -> None:
    raw = load_raw()
    curated, report = curate(raw)
    validate(curated)  # raises on any violation
    OUT.parent.mkdir(parents=True, exist_ok=True)
    curated.to_parquet(OUT, index=False)
    print("curation report:", report)
    print(f"wrote {OUT} rows={len(curated)} sha256[:12]={content_hash(curated)}")


if __name__ == "__main__":
    main()
