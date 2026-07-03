"""Where the model and the star-threshold label disagree.

The label is a thresholded star rating, so the revealing cases are the ones where the text
model reads a review differently from its stars. This surfaces those rows, split into the
two directions: the model flags negative on a high-star review, or the model reads positive
on a low-star review. Some of these are places the model is arguably more right than the
label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEG = 0


def disagreements(frame: pd.DataFrame, y_pred, neg_score, text_col="verified_reviews") -> pd.DataFrame:
    """Rows where the model prediction differs from the star-threshold label."""
    out = frame.copy()
    out["model_pred"] = np.asarray(y_pred)
    out["neg_score"] = np.asarray(neg_score)
    out["direction"] = np.where(
        (out["feedback"] == 1) & (out["model_pred"] == NEG), "model_flags_negative_high_star",
        np.where((out["feedback"] == NEG) & (out["model_pred"] == 1), "model_reads_positive_low_star", ""),
    )
    disagreeing = out[out["model_pred"] != out["feedback"]]
    cols = ["direction", "rating", "feedback", "model_pred", "neg_score", "variation", text_col]
    return disagreeing[cols].sort_values("neg_score", ascending=False).reset_index(drop=True)
