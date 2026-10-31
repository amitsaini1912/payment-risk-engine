"""
Day 18: the feedback loop without Kafka - decisions kept in SQLite, chargebacks as labels, the Redis profile counter,
precision/recall over time, /feedback endpoints. Real Redis (Testcontainers) + the real model; fixtures in
conftest.py. The Kafka side is in test_chargeback_kafka.py.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
import redis
from prometheus_client import REGISTRY

from risk_v2.feedback import DAY_MS, ChargebackEvent, FeedbackLoop
from v2_support import facts

_events = iter(range(1, 10**9))


def chargeback(txn: str, **overrides) -> ChargebackEvent:
    body = {"event_id": f"evt-{next(_events)}", "transaction_id": txn, "amount_minor": 150_000,
            "currency": "INR", "reason": "fraud", "created_at": "2026-11-20T10:15:00+05:30", "schema_version": 1}
    body.update(overrides)
    return ChargebackEvent.model_validate(body)


def decision(txn: str, action: str, *, score=0.2, threshold=0.4, at_ms: int, version="m1", customer="cust-1") -> dict:
    """A /v2 decision as record_decision() sees it (only the fields it keeps)."""
    return {"transaction_id": txn, "action": action, "model_action": action, "model_score": score,
            "threshold": threshold, "model_version": version, "rules_version": "r1",
            "signals": {"observed_at_ms": at_ms},
            "model_input": {"customer_id": customer, "amount": 1500.0, "transaction_id": txn}}


def metric(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ---- /v2 decisions are kept for the labels ----

def test_a_v2_decision_is_kept_with_its_model_input(v2_client, outcomes):
    body = v2_client.post("/v2/assess", json=facts()).json()
    kept = outcomes.decision(body["transaction_id"])
    assert kept["action"] == body["action"] and kept["model_score"] == body["model_score"]
    assert kept["model_version"] == body["model_version"] and kept["rules_version"] == body["rules_version"]
    assert kept["threshold"] == body["threshold"] and kept["customer_id"] == "cust-1"
    assert kept["decided_at_ms"] == body["signals"]["observed_at_ms"]
    assert kept["model_input"] == body["model_input"]


def test_a_replay_is_not_kept_twice(v2_client, outcomes):
    body = facts()
    v2_client.post("/v2/assess", json=body)
    assert v2_client.post("/v2/assess", json=body).json()["replayed"] is True
    assert outcomes._conn().execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


def test_v1_assess_is_not_kept(v2_client, outcomes, sample_transaction):
    assert v2_client.post("/assess", json=sample_transaction).status_code == 200
    assert outcomes._conn().execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


@pytest.mark.parametrize("broken", ["raises", "unavailable"])
def test_feedback_store_trouble_never_changes_the_decision(v2_client, outcomes, monkeypatch, broken):
    from api.main import app, get_outcome_store
    if broken == "raises":
        def fail(_decision):
            raise sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(outcomes, "record_decision", fail)
    else:
        app.dependency_overrides[get_outcome_store] = lambda: None
    before = metric("risk_feedback_write_failures_total")
    response = v2_client.post("/v2/assess", json=facts())
    assert response.status_code == 200 and response.json()["action"] in ("allow", "hold_for_review", "block")
    assert metric("risk_feedback_write_failures_total") - before == 1


# ---- chargeback -> label + profile counter ----

def test_a_chargeback_feeds_the_customers_past_chargebacks(v2_client, outcomes, redis_client):
    first = v2_client.post("/v2/assess", json=facts(customer_id="cust-cb")).json()
    assert first["model_input"]["past_chargebacks"] == 0
    applied = FeedbackLoop(outcomes, redis_client).apply(chargeback(first["transaction_id"]))
    assert applied.outcome == "stored" and applied.customer_chargebacks == 1
    later = v2_client.post("/v2/assess", json=facts(customer_id="cust-cb")).json()
    assert later["signals"]["customer_chargebacks"] == 1 and later["model_input"]["past_chargebacks"] == 1
    other = v2_client.post("/v2/assess", json=facts(customer_id="cust-clean")).json()
    assert other["model_input"]["past_chargebacks"] == 0


def test_redelivery_and_a_second_dispute_count_once_per_payment(v2_client, outcomes, redis_client):
    loop = FeedbackLoop(outcomes, redis_client)
    a = v2_client.post("/v2/assess", json=facts(customer_id="cust-dup")).json()["transaction_id"]
    b = v2_client.post("/v2/assess", json=facts(customer_id="cust-dup")).json()["transaction_id"]
    event = chargeback(a)
    assert loop.apply(event).outcome == "stored"
    assert loop.apply(event).outcome == "duplicate"                              # same event_id again
    assert loop.apply(chargeback(a, reason="duplicate")).customer_chargebacks == 1  # 2nd dispute, same payment
    assert len(outcomes.chargebacks(a)) == 2
    assert loop.apply(chargeback(b)).customer_chargebacks == 2                     # another payment


def test_a_chargeback_without_a_decision_is_kept_as_unmatched(store, outcomes, redis_client):   # store: empty Redis
    applied = FeedbackLoop(outcomes, redis_client).apply(chargeback("pay-unknown"))
    assert applied.outcome == "unmatched" and applied.customer_chargebacks is None
    assert len(outcomes.chargebacks("pay-unknown")) == 1
    assert outcomes.performance()["unmatched_chargebacks"] == 1
    assert redis_client.keys("risk:cust:*") == []


def test_redis_down_raises_so_the_consumer_retries_and_the_retry_counts_it(v2_client, outcomes, redis_client):
    txn = v2_client.post("/v2/assess", json=facts(customer_id="cust-retry")).json()["transaction_id"]
    event = chargeback(txn)
    dead = redis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)
    with pytest.raises(redis.RedisError):
        FeedbackLoop(outcomes, dead).apply(event)
    assert len(outcomes.chargebacks(txn)) == 1                 # the label is already stored...
    applied = FeedbackLoop(outcomes, redis_client).apply(event)
    assert applied.outcome == "duplicate" and applied.customer_chargebacks == 1   # ...and the retry counts it


# ---- the event contract (pydantic side; the JSON schema file is checked in test_chargeback_kafka.py) ----

@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": "1"}, {"amount_minor": "150000"}, {"amount_minor": 0},
    {"amount_minor": 1.5}, {"currency": "inr"}, {"reason": "Fraud!"}, {"reason": "4111111111111111"},
    {"created_at": "2026-11-20T10:15:00"}, {"transaction_id": "pay 1"}, {"event_id": ""},
])
def test_events_outside_the_contract_are_rejected(change):
    with pytest.raises(ValueError):
        chargeback("pay-1", **change)


def test_unknown_fields_are_ignored():
    event = chargeback("pay-1", new_optional_field="later", network_reason_code="10.4")
    assert event.transaction_id == "pay-1" and not hasattr(event, "new_optional_field")


# ---- precision / recall over time ----

def test_precision_and_recall_per_day(outcomes, clock):
    now = outcomes.now_ms()
    day1, day2 = (now // DAY_MS - 3) * DAY_MS, now // DAY_MS * DAY_MS     # 3 days ago, today
    rows = [  # txn, action, model score (threshold 0.4), chargeback?
        ("d1-a", "allow", 0.10, True), ("d1-b", "allow", 0.10, False), ("d1-c", "allow", 0.50, False),
        ("d1-d", "hold_for_review", 0.60, True), ("d1-e", "block", 0.90, False),
    ]
    for i, (txn, action, score, cb) in enumerate(rows):
        outcomes.record_decision(decision(txn, action, score=score, at_ms=day1 + i * 1000))
        if cb:
            outcomes.record_chargeback(chargeback(txn))
    outcomes.record_decision(decision("d2-a", "allow", score=None, at_ms=day2))   # rules_only: no model score
    report = outcomes.performance(bucket="day", days=7, label_window_days=1)
    first, second = report["buckets"]
    assert first["start"] == datetime.fromtimestamp(day1 / 1000, tz=timezone.utc).isoformat()
    assert (first["decisions"], first["chargebacks"]) == (5, 2)
    assert first["final"] == {"flagged": 2, "tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5,
                              "held": 1, "held_with_chargeback": 1, "blocked": 1, "blocked_with_chargeback": 0}
    # model band: flagged d1-c, d1-d, d1-e (>= 0.4); tp d1-d; fn d1-a
    assert first["model"] == {"scored": 5, "flagged": 3, "tp": 1, "fp": 2, "fn": 1,
                              "precision": 0.3333, "recall": 0.5}
    assert first["mature"] is True and second["mature"] is False   # today's chargebacks are still coming
    assert second["model"] == {"scored": 0, "flagged": 0, "tp": 0, "fp": 0, "fn": 0, "precision": None,
                               "recall": None}
    assert second["final"]["precision"] is None and second["final"]["recall"] is None
    assert first["model_versions"] == ["m1"]


def test_weeks_window_and_model_version_filter(outcomes):
    now = outcomes.now_ms()
    outcomes.record_decision(decision("old", "allow", at_ms=now - 60 * DAY_MS))
    outcomes.record_decision(decision("new-1", "allow", at_ms=now - 1000, version="m1"))
    outcomes.record_decision(decision("new-2", "block", at_ms=now - 2000, version="m2"))
    weekly = outcomes.performance(bucket="week", days=14)
    assert sum(b["decisions"] for b in weekly["buckets"]) == 2                  # "old" is outside 14 days
    assert weekly["buckets"][-1]["model_versions"] == ["m1", "m2"]
    only_m2 = outcomes.performance(days=14, model_version="m2")
    assert [b["decisions"] for b in only_m2["buckets"]] == [1]
    assert outcomes.performance(days=90)["buckets"][0]["decisions"] == 1        # "old" is back


def test_labeled_decisions_are_matured_allows_only(outcomes):
    now = outcomes.now_ms()
    for txn, action, age_days in [("a1", "allow", 200), ("a2", "allow", 150), ("h1", "hold_for_review", 200),
                                  ("b1", "block", 200), ("young", "allow", 10)]:
        outcomes.record_decision(decision(txn, action, at_ms=now - age_days * DAY_MS))
    outcomes.record_chargeback(chargeback("a2"))
    rows = outcomes.labeled_decisions(decided_before_ms=now - 120 * DAY_MS)
    assert [(r["transaction_id"], r["is_chargeback"]) for r in rows] == [("a1", 0), ("a2", 1)]   # oldest first
    assert rows[0]["amount"] == 1500.0 and rows[0]["action"] == "allow"


# ---- endpoints ----

def test_performance_endpoint(v2_client, outcomes, redis_client):
    txn = v2_client.post("/v2/assess", json=facts()).json()["transaction_id"]
    FeedbackLoop(outcomes, redis_client).apply(chargeback(txn))
    report = v2_client.get("/feedback/performance", params={"bucket": "week", "days": 7}).json()
    assert report["bucket"] == "week" and report["buckets"][0]["chargebacks"] == 1
    assert "lower bounds" in report["note"] and report["buckets"][0]["mature"] is False
    assert v2_client.get("/feedback/performance", params={"bucket": "month"}).status_code == 422
    assert v2_client.get("/feedback/performance", params={"days": 0}).status_code == 422


def test_feedback_for_one_payment_label_is_unknown_until_the_window_passes(v2_client, outcomes, redis_client, clock):
    paid = v2_client.post("/v2/assess", json=facts()).json()["transaction_id"]
    disputed = v2_client.post("/v2/assess", json=facts()).json()["transaction_id"]
    FeedbackLoop(outcomes, redis_client).apply(chargeback(disputed))
    assert v2_client.get(f"/feedback/{paid}").json()["label"] is None             # chargebacks may still come
    body = v2_client.get(f"/feedback/{disputed}").json()
    assert body["label"] == 1 and body["chargebacks"][0]["reason"] == "fraud"
    clock.advance(121 * 86400)
    assert v2_client.get(f"/feedback/{paid}").json()["label"] == 0
    assert v2_client.get("/feedback/pay-never-seen").status_code == 404


def test_feedback_endpoints_need_a_key(v2_client):
    from fastapi.testclient import TestClient

    from api.main import app
    client = TestClient(app)
    assert client.get("/feedback/performance").status_code == 401
    assert client.get("/feedback/pay-1").status_code == 401
