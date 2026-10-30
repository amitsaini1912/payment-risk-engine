"""
PSI (population stability index) drift: does live traffic still look like the training data? (Portfolio Day 17)

At training time we store, per key feature, decile bin edges and the share of TRAINING rows in each bin. At runtime
each new decision adds 1 to its bin (Redis counters, risk_v2/governance.py). PSI compares the two:

    PSI = sum over bins (actual% - expected%) * ln(actual% / expected%)

Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift (watch), > 0.25 significant (retrain / investigate).
"""

from __future__ import annotations

import math

import numpy as np

# raw model inputs (+ one engineered ratio) whose shift would hurt the model most
DRIFT_FEATURES = ["amount", "avg_order_value", "amount_to_avg_ratio", "account_age_days",
                  "txn_count_1h", "txn_count_24h", "hour"]
STABLE, MODERATE = 0.1, 0.25
EPS = 1e-4            # an empty bin would make ln() blow up
MIN_SAMPLES = 100     # fewer live decisions than this: PSI is noise, not drift


def feature_value(txn: dict, feature: str) -> float:
    if feature == "amount_to_avg_ratio":
        return float(txn["amount"]) / max(float(txn["avg_order_value"]), 1.0)
    return float(txn[feature])


def reference(values) -> dict:
    """Decile edges (deduplicated: count features have few distinct values) + expected share per bin."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return {"edges": [float(e) for e in edges], "expected": [float(c) / len(values) for c in counts]}


def bin_of(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def psi(expected: list[float], actual_counts: list[int]) -> float:
    total = sum(actual_counts)
    out = 0.0
    for e, c in zip(expected, actual_counts):
        a = max(c / total, EPS)
        e = max(e, EPS)
        out += (a - e) * math.log(a / e)
    return out


def level(value: float) -> str:
    return "stable" if value < STABLE else "moderate" if value < MODERATE else "significant"
