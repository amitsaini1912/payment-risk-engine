"""
Load the trained pipeline and score a single raw transaction.

Shared by the reasoning agent and the API so both use identical scoring logic.
"""

from __future__ import annotations

import os
from functools import lru_cache

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits  # ships with scikit-learn

from features import MODEL_FEATURES, REQUIRED_RAW_FIELDS, engineer_features

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model.pkl"),
)


# Threads for the model's OpenMP pool. A risk call scores 1-11 rows: more threads don't help, and when the machine
# is busy their hand-off made p95 jump to ~665 ms (vs ~73 ms with 1 thread, measured Day 15). 1 by default.
MODEL_THREADS = int(os.environ.get("MODEL_THREADS", "1"))


class ScoringError(ValueError):
    """Raised when a transaction cannot be scored (missing / unusable fields)."""


class Scorer:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.threshold: float = bundle["threshold"]
        self.model_name: str = bundle.get("model_name", "unknown")
        self.costs: dict = bundle.get("costs_inr", {})
        self.metrics: dict = bundle.get("metrics", {})

    def _validate(self, txn: dict) -> None:
        missing = [f for f in REQUIRED_RAW_FIELDS if f not in txn or txn[f] is None]
        if missing:
            raise ScoringError(f"missing/null required field(s): {missing}")
        for numeric in ("amount", "account_age_days", "avg_order_value",
                        "past_chargebacks", "past_orders", "txn_count_1h",
                        "txn_count_24h", "hour"):
            try:
                float(txn[numeric])
            except (TypeError, ValueError):
                raise ScoringError(f"field {numeric!r} is not numeric: {txn[numeric]!r}")
        if float(txn["amount"]) <= 0:
            raise ScoringError(f"amount must be positive, got {txn['amount']!r}")

    def score(self, txn: dict) -> float:
        """Return P(chargeback) in [0, 1]. Raises ScoringError on bad input."""
        self._validate(txn)
        try:
            row = pd.DataFrame([txn])
            engineered = engineer_features(row)[MODEL_FEATURES]
            return float(self.pipeline.predict_proba(engineered)[:, 1][0])
        except ScoringError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transform failure = bad input
            raise ScoringError(f"could not score transaction: {exc}") from exc

    def score_batch(self, txns: list[dict]) -> list[float]:
        """Score several transactions in one model call (same cost as one). Raises ScoringError on bad input."""
        for txn in txns:
            self._validate(txn)
        try:
            engineered = engineer_features(pd.DataFrame(txns))[MODEL_FEATURES]
            return [float(p) for p in self.pipeline.predict_proba(engineered)[:, 1]]
        except Exception as exc:  # noqa: BLE001 - any transform failure = bad input
            raise ScoringError(f"could not score transactions: {exc}") from exc

    def feature_snapshot(self, txn: dict) -> dict:
        """Human-readable derived features, for the LLM prompt and audit log."""
        engineered = engineer_features(pd.DataFrame([txn])).iloc[0]
        keys = [
            "amount", "amount_to_avg_ratio", "account_age_days",
            "historical_chargeback_rate", "past_chargebacks", "txn_count_1h",
            "txn_count_24h", "is_odd_hour", "is_new_account", "is_high_amount",
            "ip_billing_country_mismatch", "is_new_device", "is_international_card",
            "email_is_disposable", "card_type", "payment_method",
            "merchant_category",
        ]
        return {k: _plain(engineered[k]) for k in keys if k in engineered}


def _plain(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else round(f, 4)
    except (TypeError, ValueError):
        return str(v)


@lru_cache(maxsize=1)
def load_scorer(path: str = MODEL_PATH) -> Scorer:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run `python model/train.py` first."
        )
    threadpool_limits(limits=MODEL_THREADS, user_api="openmp")  # process-wide, stays in force
    return Scorer(joblib.load(path))
