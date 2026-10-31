"""
Load the trained pipeline and score a single raw transaction.

Shared by the reasoning agent and the API so both use identical scoring logic.
"""

from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits  # ships with scikit-learn

from features import MODEL_FEATURES, REQUIRED_RAW_FIELDS, engineer_features

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model.pkl"),
)
# Optional challenger: scores live /v2 traffic next to the real model, never affects a decision (Day 17).
SHADOW_PATH = os.environ.get(
    "SHADOW_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "shadow.pkl"),
)


# Threads for the model's OpenMP pool. A risk call scores 1-11 rows: more threads don't help, and when the machine
# is busy their hand-off made p95 jump to ~665 ms (vs ~73 ms with 1 thread, measured Day 15). 1 by default.
MODEL_THREADS = int(os.environ.get("MODEL_THREADS", "1"))


class ScoringError(ValueError):
    """Raised when a transaction cannot be scored (missing / unusable fields)."""


class Scorer:
    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        # models trained with n_jobs=-1 (random forest) would start a worker per core for every 1-11 row predict:
        # p50 39 ms vs 15 ms with 1 job (Day 17). Same reasoning as MODEL_THREADS for OpenMP.
        for _name, step in self.pipeline.steps:
            if hasattr(step, "n_jobs"):
                step.n_jobs = MODEL_THREADS
        self.threshold: float = bundle["threshold"]
        self.model_name: str = bundle.get("model_name", "unknown")
        self.costs: dict = bundle.get("costs_inr", {})
        self.metrics: dict = bundle.get("metrics", {})
        # bundles from before Day 17 have no version: name it after the model so decisions still carry something
        self.model_version: str = bundle.get("model_version", f"{self.model_name}-unversioned")
        self.role: str = bundle.get("role", "champion")
        self.drift_reference: dict = bundle.get("drift_reference", {})
        self.fast_forest: bool = _fast_forest_ok(self.pipeline.steps[-1][1])

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
        return self.score_engineered(self.engineer(txns))

    def engineer(self, txns: list[dict]) -> pd.DataFrame:
        """Validated raw transactions -> engineered frame (all columns). Row by row, so a row's features don't depend
        on the rest of the batch: one frame can be reused for the shadow model and the snapshot (Day 22)."""
        for txn in txns:
            self._validate(txn)
        try:
            return engineer_features(pd.DataFrame(txns))
        except Exception as exc:  # noqa: BLE001 - any transform failure = bad input
            raise ScoringError(f"could not score transactions: {exc}") from exc

    def score_engineered(self, engineered: pd.DataFrame) -> list[float]:
        """P(chargeback) per row of an engineer() frame."""
        try:
            X = engineered[MODEL_FEATURES]
            if not self.fast_forest:
                return [float(p) for p in self.pipeline.predict_proba(X)[:, 1]]
            Xt = self.pipeline[:-1].transform(X) if len(self.pipeline) > 1 else X
            return [float(p) for p in _forest_proba(self.pipeline.steps[-1][1], Xt)[:, 1]]
        except Exception as exc:  # noqa: BLE001 - any transform failure = bad input
            raise ScoringError(f"could not score transactions: {exc}") from exc

    def feature_snapshot(self, txn: dict, engineered: pd.DataFrame | None = None) -> dict:
        """Human-readable derived features, for the LLM prompt and audit log. `engineered`: txn's engineer() row,
        if the caller has it already."""
        engineered = (engineered if engineered is not None else engineer_features(pd.DataFrame([txn]))).iloc[0]
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


def _forest_proba(forest, X) -> np.ndarray:
    """RandomForestClassifier.predict_proba without scikit-learn's per-tree Python overhead (Day 22 load test).

    sklearn predicts tree by tree through joblib + input checks: ~33 us per tree, 9.8 ms for our 300 trees on one
    row - the largest single cost of a /v2 decision, and the engine has one worker. Calling each tree's compiled
    predict directly = the same arithmetic (average of per-tree class fractions) in ~3 ms."""
    X = np.ascontiguousarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    total = np.zeros((X.shape[0], forest.n_classes_))
    for tree in forest.estimators_:
        p = tree.tree_.predict(X).reshape(X.shape[0], -1)[:, :forest.n_classes_]
        norm = p.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        total += p / norm
    return total / len(forest.estimators_)


def _fast_forest_ok(model) -> bool:
    """Use _forest_proba only for a plain single-output random forest, and only if it gives sklearn's answer on
    random inputs (a different sklearn version changing tree internals -> back to sklearn, never wrong scores)."""
    from sklearn.ensemble import RandomForestClassifier
    if os.environ.get("MODEL_FAST_FOREST", "true").lower() == "false" or type(model) is not RandomForestClassifier \
            or getattr(model, "n_outputs_", 1) != 1:
        return False
    try:
        rng = np.random.default_rng(22)
        X = rng.normal(0, 3, size=(256, model.n_features_in_)).astype(np.float32)
        X[::4] = rng.integers(0, 2, size=X[::4].shape)               # 0/1 columns too (one-hot)
        return bool(np.allclose(_forest_proba(model, X), model.predict_proba(X), rtol=0, atol=1e-9))
    except Exception:  # noqa: BLE001 - anything unexpected: use sklearn
        return False


@lru_cache(maxsize=1)
def load_scorer(path: str = MODEL_PATH) -> Scorer:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run `python model/train.py` first."
        )
    threadpool_limits(limits=MODEL_THREADS, user_api="openmp")  # process-wide, stays in force
    return Scorer(joblib.load(path))


@lru_cache(maxsize=1)
def load_shadow(path: str = SHADOW_PATH) -> Scorer | None:
    """The shadow model, or None when there is none (no file = no shadow scoring)."""
    if not os.path.exists(path):
        return None
    load_scorer()  # same thread limits
    return Scorer(joblib.load(path))
