"""Bootstrap PR-AUC intervals and pairwise significance across variants.

Reads the held-out predictions each variant saved (`models/predictions/*.parquet`), reports a
95% bootstrap confidence interval on each one's negative-class PR-AUC, and does a paired
comparison of the top model against every other on the same test rows. If a difference
interval straddles zero, the gap is not significant on this test set.

Usage:
    python -m src.evaluation.bootstrap_report
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from src.evaluation.bootstrap import PRED_DIR, paired_diff, pr_auc_ci


def main() -> None:
    files = sorted(PRED_DIR.glob("*.parquet"))
    if not files:
        print(f"no predictions in {PRED_DIR}; run the variants first")
        return
    preds = {f.stem: pd.read_parquet(f) for f in files}

    rows = [{"variant": name, **pr_auc_ci(df["y_true"], df["neg_score"])} for name, df in preds.items()]
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    print("PR-AUC with 95% bootstrap CI (negative class):")
    for _, r in table.iterrows():
        print(f"  {r['variant']:16s} {r['pr_auc']:.3f}  [{r['lo']:.3f}, {r['hi']:.3f}]  (n={int(r['n'])})")

    top = table.loc[0, "variant"]
    base = preds[top]
    print(f"\nPaired comparison against {top} (same test rows):")
    for name, df in preds.items():
        if name == top:
            continue
        if len(df) != len(base) or not np.array_equal(df["y_true"].values, base["y_true"].values):
            print(f"  {name}: skipped (predictions not aligned to the same test set)")
            continue
        d = paired_diff(base["y_true"], base["neg_score"], df["neg_score"])
        verdict = "significant" if (d["lo"] > 0 or d["hi"] < 0) else "not significant"
        print(f"  {top} - {name}: {d['diff']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]  "
              f"P({top}>{name})={d['prob_a_gt_b']:.2f}  ({verdict})")


if __name__ == "__main__":
    main()
