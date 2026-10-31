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
    """Since Day 17 the threshold is tuned on VALIDATION and these numbers are from the untouched TEST set (#15)."""
    with open(METRICS_PATH) as fh:
        metrics = json.load(fh)
    sel = metrics["selected"]
    assert sel["recall"] >= 0.75, f"test recall {sel['recall']} below 0.75 bar"
    assert sel["precision"] >= 0.25, f"precision {sel['precision']} implausibly low"
    assert 0.0 < sel["threshold"] < 1.0
    assert "cost" in sel and sel["cost"]["total_cost_at_threshold_inr"] > 0
    split = metrics["dataset"]["split"]
    assert split["train"] == 9000 and split["validation"] == 3000 and split["test"] == 3000


def test_bundle_carries_version_and_drift_reference():
    scorer = load_scorer(MODEL_PATH)
    assert scorer.model_version and scorer.model_version.split("-")[0] in ("lr", "rf", "hgb")
    assert set(scorer.drift_reference) >= {"amount", "txn_count_1h", "account_age_days"}
    for ref in scorer.drift_reference.values():
        assert len(ref["expected"]) == len(ref["edges"]) + 1 and abs(sum(ref["expected"]) - 1) < 1e-9


def test_training_data_velocity_has_no_future_payments():
    """Day 14 finding, fixed Day 17: txn_count_1h counted the customer's LATER payments (1h > 24h)."""
    data = os.path.join(ROOT, "data", "transactions.csv")
    if not os.path.exists(data):
        pytest.skip("data/transactions.csv not generated")
    df = pd.read_csv(data)
    assert (df["txn_count_1h"] <= df["txn_count_24h"]).all()
    assert df["txn_count_1h"].mean() <= df["txn_count_24h"].mean()


def test_serving_uses_one_job_per_model():
    """A forest trained with n_jobs=-1 must not spread a single payment over every core at serving time."""
    from scorer import MODEL_THREADS
    for _name, step in load_scorer(MODEL_PATH).pipeline.steps:
        assert getattr(step, "n_jobs", MODEL_THREADS) == MODEL_THREADS


# ---- Day 22: fast forest inference (the engine's largest cost per decision, found by the k6 load test) ----

DATA_PATH = os.path.join(ROOT, "data", "transactions.csv")


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="data/transactions.csv not generated")
def test_fast_forest_gives_sklearns_scores_on_all_training_rows():
    import numpy as np
    from scorer import _forest_proba
    scorer = load_scorer(MODEL_PATH)
    assert scorer.fast_forest is True                      # the shipped model takes the fast path
    raw = pd.read_csv(DATA_PATH).drop(columns=["is_chargeback"], errors="ignore")
    frame = scorer.engineer(raw.to_dict("records"))
    X = frame[MODEL_FEATURES]
    expected = scorer.pipeline.predict_proba(X)[:, 1]
    got = _forest_proba(scorer.pipeline.steps[-1][1], scorer.pipeline[:-1].transform(X))[:, 1]
    assert np.abs(expected - got).max() < 1e-9
    assert np.allclose(scorer.score_engineered(frame.head(500)), expected[:500], rtol=0, atol=1e-9)


def test_fast_path_only_for_a_plain_random_forest():
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from scorer import _fast_forest_ok
    X, y = np.random.default_rng(1).normal(size=(200, 4)), np.arange(200) % 2
    assert _fast_forest_ok(RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)) is True
    assert _fast_forest_ok(ExtraTreesClassifier(n_estimators=5, random_state=0).fit(X, y)) is False
    assert _fast_forest_ok(LogisticRegression().fit(X, y)) is False


def test_fast_forest_can_be_switched_off(monkeypatch):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from scorer import _fast_forest_ok
    X, y = np.random.default_rng(1).normal(size=(200, 4)), np.arange(200) % 2
    monkeypatch.setenv("MODEL_FAST_FOREST", "false")
    assert _fast_forest_ok(RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)) is False


def test_engineered_once_equals_engineered_per_call(sample_transaction):
    scorer = load_scorer(MODEL_PATH)
    other = dict(sample_transaction, amount=99_000.0, is_new_device=1)
    frame = scorer.engineer([sample_transaction, other])
    assert scorer.score_engineered(frame.iloc[[0]]) == scorer.score_batch([sample_transaction])
    assert scorer.score_engineered(frame)[1] == scorer.score_batch([other])[0]
    assert scorer.feature_snapshot(sample_transaction, frame.iloc[[0]]) == scorer.feature_snapshot(sample_transaction)
