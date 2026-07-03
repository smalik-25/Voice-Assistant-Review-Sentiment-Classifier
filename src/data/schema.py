"""Pandera schema for the curated training set.

Enforces column types and ranges, non-empty text, a binary label, and the project's
central finding as a hard rule: ``feedback == 1`` if and only if ``rating >= 3``. If a
future data refresh ever breaks that rule, validation fails loudly rather than letting a
silent label change through.
"""
from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

curated_schema = DataFrameSchema(
    columns={
        "rating": Column(int, Check.isin([1, 2, 3, 4, 5])),
        "date": Column(str, Check.str_length(min_value=1)),
        "variation": Column(str, Check.str_length(min_value=1)),
        "verified_reviews": Column(str, Check.str_length(min_value=1)),
        "feedback": Column(int, Check.isin([0, 1])),
    },
    checks=Check(
        lambda df: df["feedback"] == (df["rating"] >= 3).astype(int),
        error="feedback must equal 1 iff rating >= 3 (the thresholded-rating label rule)",
    ),
    strict=True,
    coerce=False,
)


def validate(df):
    """Validate the curated frame; raises on any violation (lazy, reports all failures)."""
    return curated_schema.validate(df, lazy=True)
