"""Distribution drift: PSI and a KS test per monitored feature.

The population stability index (PSI) is the drift decision: it bins the reference
distribution and measures how much probability mass moved, which is stable across sample
sizes. A KS test is reported alongside as a second opinion, but it is not the trigger
because it flags trivially small shifts once the batch is large.

PSI reading, the usual rule of thumb: below 0.1 no meaningful shift, 0.1 to 0.2 moderate,
0.2 and above significant.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_KS_ALPHA = 0.05


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index of ``current`` against ``reference`` using reference
    quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant reference; fall back to a fixed span
        lo, hi = float(ref.min()), float(ref.max())
        edges = np.linspace(lo, hi + 1e-9, bins + 1)
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_frac = np.histogram(cur, bins=edges)[0] / max(len(cur), 1)
    eps = 1e-6
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def feature_drift(reference, current, psi_threshold=DEFAULT_PSI_THRESHOLD, ks_alpha=DEFAULT_KS_ALPHA) -> dict:
    psi_value = psi(reference, current)
    ks = ks_2samp(np.asarray(reference, float), np.asarray(current, float))
    return {
        "psi": psi_value,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "ks_significant": bool(ks.pvalue < ks_alpha),
        "drifted": bool(psi_value >= psi_threshold),  # PSI is the trigger
    }


def drift_report(reference: dict, current: dict,
                 psi_threshold=DEFAULT_PSI_THRESHOLD, ks_alpha=DEFAULT_KS_ALPHA) -> dict:
    features = {
        name: feature_drift(reference[name], current[name], psi_threshold, ks_alpha)
        for name in reference if name in current
    }
    n_drifted = sum(f["drifted"] for f in features.values())
    return {
        "features": features,
        "n_drifted_features": int(n_drifted),
        "drifted": bool(n_drifted > 0),
        "psi_threshold": psi_threshold,
        "ks_alpha": ks_alpha,
    }
