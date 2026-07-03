"""Fairness as separation across device variation groups.

Separation (equalized odds) asks whether the model's error rates are the same regardless
of group. Here the groups are device `variation`. For each group we report negative-class
recall (of genuinely negative reviews, how many the model catches) and the false-positive
rate (of genuinely positive reviews, how many are wrongly flagged negative), then the
spread across groups. A large spread means the model treats some devices' reviews
differently, which is the thing to surface. Groups with very few negatives give noisy
rates, so counts are reported alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NEG = 0


def separation_by_group(y_true, y_pred, groups, min_negatives: int = 5) -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame({"y": np.asarray(y_true), "pred": np.asarray(y_pred), "group": np.asarray(groups)})
    rows = []
    for name, sub in df.groupby("group"):
        neg = sub["y"] == NEG
        pos = sub["y"] == 1
        rows.append({
            "variation": name,
            "n": len(sub),
            "n_negatives": int(neg.sum()),
            "neg_recall": (sub.loc[neg, "pred"] == NEG).mean() if neg.any() else np.nan,
            "fpr": (sub.loc[pos, "pred"] == NEG).mean() if pos.any() else np.nan,
        })
    table = pd.DataFrame(rows).sort_values("n_negatives", ascending=False).reset_index(drop=True)

    # Spread is computed only over groups with enough negatives to be meaningful.
    solid = table[table["n_negatives"] >= min_negatives]
    spread = {
        "groups_scored": int(len(solid)),
        "neg_recall_spread": float(solid["neg_recall"].max() - solid["neg_recall"].min()) if len(solid) else np.nan,
        "fpr_spread": float(solid["fpr"].max() - solid["fpr"].min()) if len(solid) else np.nan,
    }
    return table, spread
