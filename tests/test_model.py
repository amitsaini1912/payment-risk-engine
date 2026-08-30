"""
Model sanity: it loads, it predicts, feature engineering is stable, and the
saved metrics meet the acceptance bar.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from features import MODEL_FEATURES, engineer_features
from scorer import ScoringError, load_scorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model", "model.pkl")
METRICS_PATH = os.path.join(ROOT, "model", "metrics.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="model/model.pkl not built - run `python model/train.py`",
)


def test_model_loads_and_has_threshold():
    scorer = load_scorer(MODEL_PATH)
    assert 0.0 < scorer.threshold < 1.0
    assert scorer.model_name


def test_scores_a_known_low_risk_transaction(sample_transaction):
    scorer = load_scorer(MODEL_PATH)
    safe = dict(sample_transaction,
                amount=500.0, avg_order_value=600.0, account_age_days=1200,
                past_chargebacks=0, is_new_device=0, txn_count_1h=0,
                hour=13, email_is_disposable=0)
    score = scorer.score(safe)
    assert 0.0 <= score <= 1.0
    assert score < scorer.threshold


def test_scores_a_known_high_risk_transaction(high_risk_transaction):
    scorer = load_scorer(MODEL_PATH)
    score = scorer.score(high_risk_transaction)
    assert 0.0 <= score <= 1.0
    assert score > scorer.threshold


def test_scorer_rejects_missing_field(sample_transaction):
    scorer = load_scorer(MODEL_PATH)
    broken = dict(sample_transaction)
    del broken["amount"]
    with pytest.raises(ScoringError):
        scorer.score(broken)


def test_scorer_rejects_non_numeric_amount(sample_transaction):
    scorer = load_scorer(MODEL_PATH)
    with pytest.raises(ScoringError):
        scorer.score(dict(sample_transaction, amount="lots"))


def test_engineer_features_is_pure_and_complete(sample_transaction):
    df = pd.DataFrame([sample_transaction])
    before = df.copy()
    out = engineer_features(df)
    pd.testing.assert_frame_equal(df, before)  # input not mutated
    for col in MODEL_FEATURES:
        assert col in out.columns


def test_metrics_meet_acceptance_bar():
    with open(METRICS_PATH) as fh:
        metrics = json.load(fh)
    sel = metrics["selected"]
    assert sel["recall"] >= 0.75, f"recall {sel['recall']} below 0.75 bar"
    assert sel["precision"] >= 0.25, f"precision {sel['precision']} implausibly low"
    assert 0.0 < sel["threshold"] < 1.0
    assert "cost" in sel and sel["cost"]["total_cost_at_threshold_inr"] > 0
