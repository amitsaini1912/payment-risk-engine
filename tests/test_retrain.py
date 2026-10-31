"""
Day 18: scripts/retrain.py - labeled decisions from the feedback store -> a candidate model in its own folder, compared
with the live champion on the newest labeled payments. The live model files are never touched.
Uses rows of data/transactions.csv as "live" decisions (skipped if the data or the model isn't built).
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

from features import REQUIRED_RAW_FIELDS
from risk_v2.feedback import DAY_MS, OutcomeStore
from scripts import retrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "transactions.csv")
CHAMPION = os.path.join(ROOT, "model", "model.pkl")
NOW = time.time()    # retrain.py judges label maturity on the real clock


@pytest.fixture
def data():
    if not (os.path.exists(DATA) and os.path.exists(CHAMPION)):
        pytest.skip("data/transactions.csv or model/model.pkl not built")
    return pd.read_csv(DATA)


def fill(path: str, rows: pd.DataFrame, *, age_days: float, action: str = "allow", captured: bool = True) -> OutcomeStore:
    """Each row becomes a /v2 decision `age_days` old (one ms apart), billing's `captured` outcome if `captured`, and,
    if is_chargeback, a chargeback."""
    store = OutcomeStore(path, clock=lambda: NOW)
    start = int(NOW * 1000 - age_days * DAY_MS)
    for i, row in enumerate(rows.to_dict("records")):
        txn = f"live-{action}-{'c' if captured else 'n'}-{int(age_days)}-{i}"
        model_input = {k: row[k] for k in REQUIRED_RAW_FIELDS} | {"transaction_id": txn, "customer_id": "c"}
        store.record_decision({"transaction_id": txn, "action": action, "model_score": 0.1, "threshold": 0.4,
                               "model_version": "old", "rules_version": "r1", "model_input": model_input,
                               "signals": {"observed_at_ms": start + i}})
        if captured:
            store._conn().execute("INSERT INTO payment_outcomes VALUES (?,?,?,?,?,?,?,?)",
                                  (f"out-{txn}", txn, "captured", 100, "INR", "2026-01-01T00:00:00+00:00", 0, 1))
        if row["is_chargeback"]:
            store._conn().execute("INSERT INTO chargebacks VALUES (?,?,?,?,?,?,?,?)",
                                  (f"evt-{txn}", txn, 100, "INR", "fraud", "2026-01-01T00:00:00+00:00", 0, 1))
    return store


def run(tmp_path, *extra) -> int:
    return retrain.main(["--db", str(tmp_path / "feedback.db"), "--base", str(tmp_path / "base.csv"),
                         "--out", str(tmp_path / "candidate"), "--champion", CHAMPION, *extra])


def test_retrain_writes_a_candidate_and_compares_it_on_the_newest_labels(data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                         # anything written relative to the cwd lands here
    data.iloc[:4000].to_csv(tmp_path / "base.csv", index=False)
    live = data.iloc[4000:5000]
    fill(str(tmp_path / "feedback.db"), live, age_days=200)
    fill(str(tmp_path / "feedback.db"), data.iloc[5000:5300], age_days=200, action="block", captured=False)  # never trained on
    fill(str(tmp_path / "feedback.db"), data.iloc[6000:6200], age_days=200, captured=False)   # allowed, not captured
    fill(str(tmp_path / "feedback.db"), data.iloc[5300:5600], age_days=30)                    # not labeled yet
    champion_mtime = os.path.getmtime(CHAMPION)

    assert run(tmp_path) == 0
    out = tmp_path / "candidate"
    for name in ("model.pkl", "shadow.pkl", "metrics.json", "comparison.json", "training_data.csv"):
        assert (out / name).exists(), name
    training = pd.read_csv(out / "training_data.csv")
    assert (training["source"] == "live").sum() == 800 and (training["source"] == "synthetic").sum() == 4000
    comparison = json.loads((out / "comparison.json").read_text())
    assert comparison["holdout"]["rows"] == 200                          # newest 20% of the 1,000 matured allows
    assert comparison["holdout"]["chargebacks"] == int(live.iloc[800:]["is_chargeback"].sum())
    assert comparison["holdout"]["from_ms"] == int(NOW * 1000 - 200 * DAY_MS) + 800    # the NEWEST payments
    assert comparison["candidate"]["model_version"] != comparison["champion"]["model_version"]
    for role in ("champion", "candidate"):
        cm = comparison[role]["confusion_matrix"]
        assert sum(cm.values()) == 200 and comparison[role]["total_cost_inr"] > 0
    assert os.path.getmtime(CHAMPION) == champion_mtime                  # live model untouched


def test_not_enough_labels_writes_nothing(data, tmp_path):
    data.iloc[:500].to_csv(tmp_path / "base.csv", index=False)
    fill(str(tmp_path / "feedback.db"), data.iloc[500:600], age_days=200)
    assert run(tmp_path) == 2                                            # 100 < 200 labeled
    assert run(tmp_path, "--min-labeled", "50", "--min-chargebacks", "1000") == 2
    assert not (tmp_path / "candidate").exists()


def test_the_label_window_decides_what_counts_as_labeled(data, tmp_path):
    data.iloc[:500].to_csv(tmp_path / "base.csv", index=False)
    fill(str(tmp_path / "feedback.db"), data.iloc[500:900], age_days=30)
    assert run(tmp_path, "--min-labeled", "100", "--min-chargebacks", "1") == 2          # 30 days < 120
    assert run(tmp_path, "--min-labeled", "100", "--min-chargebacks", "1", "--label-window-days", "0",
               "--min-holdout-chargebacks", "1") == 0


def test_a_holdout_without_chargebacks_is_refused_before_training(data, tmp_path):
    data.iloc[:500].to_csv(tmp_path / "base.csv", index=False)
    rows = data.iloc[500:1500].sort_values("is_chargeback", ascending=False)   # every chargeback on OLD payments
    fill(str(tmp_path / "feedback.db"), rows, age_days=200)
    assert run(tmp_path) == 2
    assert not (tmp_path / "candidate").exists()


def test_no_feedback_store_is_exit_2_not_a_crash(tmp_path):
    assert run(tmp_path) == 2


def test_population_allow_is_the_day_18_behaviour_without_outcomes(data, tmp_path):
    data.iloc[:500].to_csv(tmp_path / "base.csv", index=False)
    fill(str(tmp_path / "feedback.db"), data.iloc[500:900], age_days=200, captured=False)   # billing sent no outcomes
    args = ("--min-labeled", "100", "--min-chargebacks", "1", "--min-holdout-chargebacks", "1")
    assert run(tmp_path, *args) == 2                                                    # captured: nothing labeled
    assert run(tmp_path, *args, "--population", "allow") == 0
