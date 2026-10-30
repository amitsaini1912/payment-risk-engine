"""
Day 16 production behaviour of /v2/assess: stored decisions (idempotency), the 300 ms budget, the bounded LLM pool
(#18), API keys, Prometheus metrics, /ready, and the /audit count fix (#20).
Real Redis (Testcontainers) + the real model; fixtures in conftest.py.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import risk_v2.budget as budget
from api.main import app, get_decision_store
from llm_pool import LLMBusy, LLMPool, LLMTimeout
from risk_v2.auth import ApiKeys, parse_keys
from risk_v2.decisions import DecisionStore
from risk_v2.store import FeaturesUnavailable
from v2_support import API_KEY, facts

AFTERNOON_IST = "2026-10-29T08:30:00+00:00"


def metric(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def post(client, body, **kw):
    return client.post("/v2/assess", json=body, **kw)


# ---- idempotency: same transaction_id -> the same stored decision ----

def test_retry_after_a_rules_change_gets_the_original_decision(v2_client, rules_path):
    body = facts(timestamp=AFTERNOON_IST)
    first = post(v2_client, body).json()
    assert first["action"] == "allow" and first["replayed"] is False

    with open(rules_path, "a") as fh:      # an incident rule that would hold this payment now
        fh.write("""
  - id: freeze_all
    action: hold_for_review
    reasons: [R02_AMOUNT_ANOMALY]
    when: {field: amount_inr, op: ">", value: 0}
""")
    again = post(v2_client, body).json()
    assert again["replayed"] is True
    assert again["action"] == "allow" and again["rules_version"] == first["rules_version"]
    assert post(v2_client, facts(timestamp=AFTERNOON_IST)).json()["action"] == "hold_for_review"   # new ids: new rules


def test_a_retry_does_not_run_the_pipeline_again(v2_client, monkeypatch):
    import risk_v2.decision as decision_mod
    body = facts(timestamp=AFTERNOON_IST)
    first = post(v2_client, body).json()

    def must_not_run(*_a, **_kw):
        raise AssertionError("retry recomputed the decision")
    monkeypatch.setattr(decision_mod, "load_scorer", must_not_run)
    monkeypatch.setattr("api.main.decide", must_not_run)
    again = post(v2_client, body)
    assert again.status_code == 200 and again.json()["replayed"] is True
    assert again.json()["model_score"] == first["model_score"]


def test_a_retry_is_not_counted_again(v2_client):
    body = facts(timestamp=AFTERNOON_IST)
    for _ in range(3):
        post(v2_client, body)
    assert post(v2_client, facts(timestamp=AFTERNOON_IST)).json()["signals"]["card_txn_count_1h"] == 1


def test_same_id_with_different_facts_is_409(v2_client):
    body = facts(timestamp=AFTERNOON_IST)
    assert post(v2_client, body).status_code == 200
    response = post(v2_client, {**body, "amount_minor": body["amount_minor"] + 1})
    assert response.status_code == 409 and "different facts" in response.json()["detail"]


def test_concurrent_copies_of_one_request_get_one_decision(v2_client):
    body = facts(timestamp=AFTERNOON_IST)
    before = metric("risk_decisions_total", api_version="v2", action="allow", path_used="rules+model")
    results, errors = [], []

    def call():
        try:
            r = post(v2_client, body)
            results.append(r.json() if r.status_code == 200 else {"status": r.status_code, "body": r.text})
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors and len(results) == 10
    assert [r for r in results if "status" in r] == []
    assert len({(r["action"], tuple(r["reason_codes"]), r["decided_in_ms"]) for r in results}) == 1
    assert sum(not r["replayed"] for r in results) == 1                     # one decided, nine replayed
    assert metric("risk_decisions_total", api_version="v2", action="allow", path_used="rules+model") - before == 1
    audit = v2_client.get("/audit", params={"limit": 1000}).json()["entries"]
    assert sum(e["transaction_id"] == body["transaction_id"] for e in audit) == 1


def test_decided_but_not_stored_is_503(v2_client, redis_client):
    class FailingStore(DecisionStore):
        def put_first(self, *args):
            raise FeaturesUnavailable("redis went away")
    app.dependency_overrides[get_decision_store] = lambda: FailingStore(redis_client)
    try:
        response = post(v2_client, facts())
    finally:
        app.dependency_overrides.pop(get_decision_store, None)
    assert response.status_code == 503 and "fallback" in response.json()["detail"]


# ---- the 300 ms budget ----

def test_a_slow_llm_never_delays_the_decision(v2_client, explainer):
    explainer["fn"] = lambda _prompt: time.sleep(3) or '{"explanation": "late"}'
    started = time.perf_counter()
    body = post(v2_client, facts(timestamp=AFTERNOON_IST)).json()
    took = time.perf_counter() - started

    assert took < 1.5                                   # not the LLM's 3 s (generous: CI/busy machines)
    assert body["explanation_source"] == "template"
    # usually the LLM is tried with the ~250 ms left and times out; on a very busy machine there's no time left
    assert body["failure_mode"] in ("llm_timeout", "llm_no_budget")


def test_a_fast_llm_inside_the_budget_writes_the_explanation(v2_client, explainer, monkeypatch):
    monkeypatch.setattr(budget, "BUDGET_S", 5.0)        # "there is time left" - not a race against a busy machine
    explainer["fn"] = lambda _prompt: '{"explanation": "Normal order from a known pattern."}'
    body = post(v2_client, facts(timestamp=AFTERNOON_IST)).json()
    assert body["explanation_source"] == "llm" and body["failure_mode"] is None


def test_no_budget_left_means_the_llm_is_not_even_called(v2_client, explainer, monkeypatch):
    calls = []
    explainer["fn"] = lambda prompt: calls.append(prompt) or '{"explanation": "x"}'
    monkeypatch.setattr(budget, "BUDGET_S", 0.0)
    body = post(v2_client, facts(timestamp=AFTERNOON_IST)).json()
    assert calls == [] and body["failure_mode"] == "llm_no_budget"
    assert body["action"] in ("allow", "hold_for_review", "block")          # the decision is still made


# ---- bounded LLM pool (#18) ----

def test_the_llm_pool_is_bounded_and_refuses_instead_of_queueing():
    pool, release = LLMPool(size=2), threading.Event()
    for _ in range(2):
        with pytest.raises(LLMTimeout):
            pool.run(lambda _a: release.wait(5), None, 0.05)   # hung calls keep their slots
    assert pool.in_flight() == 2
    started = time.perf_counter()
    with pytest.raises(LLMBusy):
        pool.run(lambda _a: "never runs", None, 1.0)
    assert time.perf_counter() - started < 0.05                # refused at once, not queued

    release.set()
    for _ in range(100):
        if pool.in_flight() == 0:
            break
        time.sleep(0.01)
    assert pool.in_flight() == 0 and pool.run(lambda a: a * 2, 21, 1.0) == 42


def test_many_timeouts_do_not_pile_up_threads():
    pool, release = LLMPool(size=3), threading.Event()
    for _ in range(50):
        try:
            pool.run(lambda _a: release.wait(5), None, 0.001)
        except (LLMTimeout, LLMBusy):
            pass
    assert sum(t.name == "llm-call" and t.is_alive() for t in threading.enumerate()) <= 3 + 4   # ours + global pool
    release.set()


def test_llm_errors_come_back_to_the_caller():
    with pytest.raises(ZeroDivisionError):
        LLMPool(1).run(lambda _a: 1 / 0, None, 1.0)


# ---- API keys ----

@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key-0123456789"}, {"X-API-Key": API_KEY[:-1]}])
def test_missing_or_wrong_key_is_401_and_nothing_is_recorded(store, redis_client, headers):
    client = TestClient(app, headers=headers)
    body = facts()
    before = metric("risk_auth_failures_total")
    assert post(client, body).status_code == 401
    assert client.get("/audit").status_code == 401
    assert client.get(f"/audit/{body['transaction_id']}").status_code == 401
    assert client.post("/assess", json={}).status_code == 401
    assert redis_client.exists(f"risk:txn:{body['transaction_id']}") == 0
    assert metric("risk_auth_failures_total") - before == 4


def test_the_audit_trail_names_the_caller(v2_client):
    body = post(v2_client, facts()).json()
    assert v2_client.get(f"/audit/{body['transaction_id']}").json()["caller"] == "billing"


def test_no_keys_configured_refuses_everything():
    from api.main import get_api_keys
    app.dependency_overrides[get_api_keys] = lambda: ApiKeys("")
    response = TestClient(app, headers={"X-API-Key": API_KEY}).post("/v2/assess", json=facts())
    assert response.status_code == 503 and "auth not configured" in response.json()["detail"]


@pytest.mark.parametrize("raw", ["billing", "billing:short", ":abcdefghijklmnopq"])
def test_malformed_key_config_is_an_error(raw):
    with pytest.raises(ValueError):
        parse_keys(raw)


def test_health_and_metrics_need_no_key(store):
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/metrics/prometheus").status_code == 200


# ---- metrics ----

def test_prometheus_exposes_latency_histogram_and_decision_counts(v2_client):
    post(v2_client, facts(timestamp=AFTERNOON_IST))
    text = v2_client.get("/metrics/prometheus").text
    assert 'risk_request_duration_seconds_bucket{endpoint="/v2/assess",le="0.3",status="200"}' in text
    assert 'risk_decisions_total{action="allow",api_version="v2",path_used="rules+model"}' in text
    assert "risk_llm_calls_in_flight" in text


def test_503s_are_counted_as_fallbacks(v2_client, rules_path):
    before = metric("risk_unavailable_total", reason="rules")
    with open(rules_path, "w") as fh:
        fh.write("rules: [")
    from api.main import get_rule_loader
    from risk_v2.rules import RuleLoader
    app.dependency_overrides[get_rule_loader] = lambda: RuleLoader(rules_path)
    assert post(v2_client, facts()).status_code == 503
    assert metric("risk_unavailable_total", reason="rules") - before == 1


# ---- readiness, audit count ----

def test_ready_only_when_everything_is_usable(v2_client, rules_path):
    assert v2_client.get("/ready").status_code == 200
    with open(rules_path, "w") as fh:
        fh.write("rules: [")
    from api.main import get_rule_loader
    from risk_v2.rules import RuleLoader
    app.dependency_overrides[get_rule_loader] = lambda: RuleLoader(rules_path)
    response = v2_client.get("/ready")
    assert response.status_code == 503 and response.json()["checks"]["rules"] != "ok"


def test_audit_count_is_the_real_number_of_entries(v2_client):
    for _ in range(3):
        post(v2_client, facts())
    listing = v2_client.get("/audit", params={"limit": 25}).json()
    assert listing["count"] == len(listing["entries"]) == 3          # was "count": 25 (#20)
