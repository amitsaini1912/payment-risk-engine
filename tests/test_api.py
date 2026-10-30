"""
API contract: a normal request returns a valid verdict, malformed requests are
rejected cleanly, and the audit endpoints work.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "model", "model.pkl")),
    reason="model/model.pkl not built - run `python model/train.py`",
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    # isolate the audit trail per test
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.jsonl"))
    import audit
    monkeypatch.setattr(audit, "DEFAULT_PATH", str(tmp_path / "audit.jsonl"))

    # keep API tests offline + deterministic: force the LLM path to be
    # unavailable so /assess exercises the fallback. The live LLM path is
    # covered by an injected fake in test_failure_modes.py.
    import reasoning_agent
    from llm_client import LLMUnavailable

    def _no_llm(_prompt):
        raise LLMUnavailable("disabled in tests")

    monkeypatch.setattr(reasoning_agent, "call_gemini", _no_llm)
    from v2_support import AUTH
    return TestClient(app, headers=AUTH)


def _body(**over):
    base = dict(
        merchant_id="M1", merchant_category="electronics", customer_id="C1",
        amount=6000.0, payment_method="card", card_type="credit",
        is_international_card=0, hour=14, is_new_device=0,
        ip_billing_country_mismatch=0, account_age_days=300, past_chargebacks=0,
        past_orders=10, avg_order_value=3000.0, txn_count_1h=0, txn_count_24h=1,
        email_is_disposable=0,
    )
    base.update(over)
    return base


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_assess_normal_request(client):
    r = client.post("/assess", json=_body())
    assert r.status_code == 200
    body = r.json()
    assert body["action"] in ("allow", "hold_for_review", "block")
    assert body["path_used"] in ("llm", "fallback")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["transaction_id"].startswith("TXN")
    assert isinstance(body["feature_snapshot"], dict)


def test_assess_auto_generates_transaction_id(client):
    a = client.post("/assess", json=_body()).json()["transaction_id"]
    b = client.post("/assess", json=_body()).json()["transaction_id"]
    assert a != b


@pytest.mark.parametrize("bad", [
    {"amount": -10},
    {"amount": 0},
    {"hour": 25},
    {"payment_method": "bitcoin"},
    {"card_type": "amex"},
    {"past_chargebacks": -1},
    {"extra_unexpected_field": 1},
])
def test_assess_rejects_malformed_request(client, bad):
    r = client.post("/assess", json=_body(**bad))
    assert r.status_code == 422


def test_assess_missing_required_field(client):
    body = _body()
    del body["merchant_id"]
    assert client.post("/assess", json=body).status_code == 422


def test_audit_roundtrip(client):
    tid = client.post("/assess", json=_body()).json()["transaction_id"]
    r = client.get(f"/audit/{tid}")
    assert r.status_code == 200
    assert r.json()["transaction_id"] == tid
    assert r.json()["path_used"] in ("llm", "fallback")


def test_audit_unknown_id_is_404(client):
    assert client.get("/audit/TXN-does-not-exist").status_code == 404


@pytest.mark.parametrize("mode", ["llm_timeout", "llm_unreachable", "llm_malformed", "bad_input"])
def test_assess_simulate_forces_fallback(client, mode):
    r = client.post("/assess", params={"simulate": mode}, json=_body(amount=40000, account_age_days=3))
    assert r.status_code == 200
    body = r.json()
    assert body["path_used"] == "fallback"
    assert body["action"] in ("allow", "hold_for_review", "block")


def test_assess_unknown_simulate_mode_is_400(client):
    assert client.post("/assess", params={"simulate": "nope"}, json=_body()).status_code == 400
