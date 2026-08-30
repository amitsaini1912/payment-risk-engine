"""
Feature engineering shared by training, the API, and the reasoning agent.

Keeping this in one place means the exact same transformation is applied at
train time and at inference time - a common source of silent bugs otherwise.

`engineer_features` takes a DataFrame (or a single-row dict wrapped in one) of
RAW transaction fields and returns a DataFrame with the derived columns added.
`build_pipeline` returns an unfitted sklearn Pipeline that one-hot encodes the
categoricals, scales the numerics, and ends in a classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Raw fields the caller must supply for a prediction.
REQUIRED_RAW_FIELDS = [
    "amount", "payment_method", "card_type", "is_international_card",
    "hour", "is_new_device", "ip_billing_country_mismatch",
    "account_age_days", "past_chargebacks", "past_orders",
    "avg_order_value", "txn_count_1h", "txn_count_24h", "email_is_disposable",
    "merchant_category",
]

NUMERIC_FEATURES = [
    "amount", "log_amount", "amount_to_avg_ratio", "account_age_days",
    "log_account_age", "past_chargebacks", "past_orders",
    "historical_chargeback_rate", "txn_count_1h", "txn_count_24h",
    "is_international_card", "is_new_device", "ip_billing_country_mismatch",
    "email_is_disposable", "hour", "is_odd_hour", "is_new_account",
    "is_high_amount",
]
CATEGORICAL_FEATURES = ["payment_method", "card_type", "merchant_category"]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Pure function - does not mutate the input."""
    d = df.copy()

    d["amount"] = d["amount"].astype(float)
    d["avg_order_value"] = d["avg_order_value"].astype(float).clip(lower=1.0)
    d["account_age_days"] = d["account_age_days"].astype(float).clip(lower=0)

    d["log_amount"] = np.log1p(d["amount"])
    d["log_account_age"] = np.log1p(d["account_age_days"])
    d["amount_to_avg_ratio"] = d["amount"] / d["avg_order_value"]
    d["historical_chargeback_rate"] = (
        d["past_chargebacks"].astype(float) / (d["past_orders"].astype(float) + 1.0)
    )

    d["is_odd_hour"] = ((d["hour"] >= 23) | (d["hour"] <= 5)).astype(int)
    d["is_new_account"] = (d["account_age_days"] <= 30).astype(int)
    d["is_high_amount"] = (d["amount"] >= 15000).astype(int)

    for col in ("is_international_card", "is_new_device",
                "ip_billing_country_mismatch", "email_is_disposable"):
        d[col] = d[col].astype(int)

    return d


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the columns the model consumes, in a stable order."""
    engineered = engineer_features(df)
    return engineered[MODEL_FEATURES]


def build_pipeline(classifier) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("clf", classifier)])
