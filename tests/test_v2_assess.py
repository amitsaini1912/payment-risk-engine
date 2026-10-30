"""
/v2/assess (Day 14): the contract, and velocity/profile features the ENGINE computes in a real Redis 7
(Testcontainers). Time is a fake clock we move forward, so "an hour later" takes no time.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone

import pytest
import redis as redis_lib
from fastapi.testclient import TestClient

from api.main import app, get_feature_store
from fallback_rules import VALID_ACTIONS
from risk_v2.assembler import amount_in_inr_minor
from risk_v2.store import FeatureStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOUR = 3600


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(scope="module")
def redis_client():
    try:
        from testcontainers.redis import RedisContainer
        container = RedisContainer("redis:7.4-alpine")
        container.start()
    except Exception as exc:  # noqa: BLE001 - no Docker on this machine
        pytest.skip(f"Docker/Testcontainers not available: {exc}")
    host, port = container.get_container_host_ip(), container.get_exposed_port(6379)
    client = redis_lib.Redis(host=host, port=int(port), decode_responses=True)
    yield client
    container.stop()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(redis_client, clock):
    redis_client.flushdb()
    return FeatureStore(redis_client, clock=clock)


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    if not os.path.exists(os.path.join(ROOT, "model", "model.pkl")):
        pytest.skip("model/model.pkl not built - run `python model/train.py`")
    import audit
    import reasoning_agent
    from llm_client import LLMUnavailable

    monkeypatch.setattr(audit, "DEFAULT_PATH", str(tmp_path / "audit.jsonl"))

    def _no_llm(_prompt):
        raise LLMUnavailable("disabled in tests")

    monkeypatch.setattr(reasoning_agent, "call_gemini", _no_llm)
    app.dependency_overrides[get_feature_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


_seq = iter(range(1, 10**9))


def facts(**overrides) -> dict:
    body = {
        "transaction_id": f"pay-{next(_seq)}",
        "amount_minor": 150_000,                    # INR 1,500.00
        "currency": "INR",
        "card_token": "tok_abcdefgh23",
        "card_fingerprint": "fp" + "a" * 62,
        "last4": "4242",
        "customer_id": "cust-1",
        "merchant_id": "M1001",
        "ip": "203.0.113.7",
        "device_id": "dev-1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    body.update(overrides)
    return body


def observe(store, body: dict):
    return store.observe(transaction_id=body["transaction_id"], customer_id=body["customer_id"],
                         card_fingerprint=body["card_fingerprint"], device_id=body["device_id"], ip=body["ip"],
                         amount_inr_minor=amount_in_inr_minor(body["amount_minor"], body["currency"]))


# ---- contract ----

@pytest.mark.parametrize("change", [
    {"past_chargebacks": 0},                      # the caller can't send risk features any more (weakness #14)
    {"txn_count_1h": 0},
    {"amount_minor": 1500.5},                     # money is never a float
    {"amount_minor": "150000"},                   # nor a string
    {"amount_minor": 0},
    {"currency": "inr"},
    {"ip": "999.1.1.1"},
    {"last4": "42"},
    {"timestamp": "2026-10-30T10:00:00"},         # no UTC offset
    {"timestamp": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},  # from the future
    {"card_type": "platinum"},
])
def test_invalid_facts_are_422(client, change):
    assert client.post("/v2/assess", json=facts(**change)).status_code == 422


def test_unsupported_currency_is_422(client):
    response = client.post("/v2/assess", json=facts(currency="GBP"))
    assert response.status_code == 422
    assert "GBP" in response.json()["detail"]


# ---- the whole path ----

def test_first_payment_gets_a_verdict_and_engine_computed_features(client):
    response = client.post("/v2/assess", json=facts(merchant_category="electronics", card_type="credit"))

    assert response.status_code == 200
    body = response.json()
    assert body["api_version"] == "v2"
    assert body["action"] in VALID_ACTIONS     # allow | hold_for_review | block
    assert 0.0 <= body["model_score"] <= 1.0
    model_input = body["model_input"]
    assert model_input["amount"] == 1500.0        # 150000 paise
    assert model_input["txn_count_1h"] == 0 and model_input["past_orders"] == 0
    assert model_input["account_age_days"] == 0   # never seen this customer
    assert model_input["avg_order_value"] == 1500.0  # no history: the current amount
    assert model_input["is_new_device"] == 1
    assert model_input["merchant_category"] == "electronics" and model_input["card_type"] == "credit"
    assert body["signals"]["card_is_new"] is True and body["signals"]["replayed"] is False


def test_card_testing_burst_raises_velocity_and_the_score(client, clock):
    scores = []
    for i in range(8):                            # small payments, same card, a minute apart
        body = client.post("/v2/assess", json=facts(amount_minor=50_000, device_id="dev-x")).json()
        scores.append(body["model_score"])
        clock.advance(60)

    assert body["signals"]["card_txn_count_1h"] == 7
    assert body["model_input"]["txn_count_1h"] == 7
    assert scores[-1] > scores[0]                 # the model reacts to velocity the engine counted itself


def test_unknown_redis_is_503_never_a_guess(client, monkeypatch):
    dead = FeatureStore(redis_lib.Redis(host="localhost", port=1, socket_timeout=0.2, socket_connect_timeout=0.2))
    app.dependency_overrides[get_feature_store] = lambda: dead
    response = client.post("/v2/assess", json=facts())
    assert response.status_code == 503
    assert "fallback" in response.json()["detail"]


def test_old_assess_is_untouched(client, sample_transaction):
    assert client.post("/assess", json=sample_transaction).status_code == 200


# ---- velocity windows (store level) ----

def test_counts_are_from_before_this_transaction_and_age_out(store, clock):
    for _ in range(3):
        observe(store, facts())
        clock.advance(10 * 60)                    # every 10 minutes

    s = observe(store, facts())
    assert (s.card_txn_count_1h, s.card_txn_count_24h) == (3, 3)
    assert (s.customer_txn_count_1h, s.customer_txn_count_24h) == (3, 3)
    assert s.card_amount_sum_24h_minor == 3 * 150_000

    clock.advance(61 * 60)                        # all four are now > 1h old
    s = observe(store, facts())
    assert (s.card_txn_count_1h, s.card_txn_count_24h) == (0, 4)

    clock.advance(25 * HOUR)                      # and > 24h
    s = observe(store, facts())
    assert (s.card_txn_count_1h, s.card_txn_count_24h, s.card_amount_sum_24h_minor) == (0, 0, 0)


def test_distinct_devices_and_ips_per_card(store, clock):
    observe(store, facts(device_id="d1", ip="198.51.100.1"))
    observe(store, facts(device_id="d2", ip="198.51.100.2"))
    s = observe(store, facts(device_id="d3", ip="198.51.100.2"))
    assert (s.card_distinct_devices_24h, s.card_distinct_ips_24h) == (3, 2)


def test_same_card_different_token_shares_velocity_other_card_does_not(store):
    observe(store, facts(card_token="tok_first111"))
    s = observe(store, facts(card_token="tok_second22"))   # same fingerprint
    assert s.card_txn_count_1h == 1 and not s.card_is_new
    other = observe(store, facts(card_fingerprint="fp" + "b" * 62))
    assert other.card_txn_count_1h == 0 and other.card_is_new


def test_profile_age_orders_average_and_new_device(store, clock):
    observe(store, facts(amount_minor=100_000, device_id="phone"))
    clock.advance(10 * 24 * HOUR)
    observe(store, facts(amount_minor=300_000, device_id="phone"))
    clock.advance(2 * 24 * HOUR)

    s = observe(store, facts(device_id="phone"))
    assert s.customer_prior_orders == 2
    assert s.customer_prior_amount_sum_minor == 400_000          # avg INR 2,000.00
    assert (s.observed_at_ms - s.customer_first_seen_ms) // 86_400_000 == 12
    assert s.customer_device_known
    assert not observe(store, facts(device_id="laptop")).customer_device_known


def test_a_retried_transaction_is_counted_once_and_gets_the_same_signals(store, clock):
    body = facts()
    first = observe(store, body)
    clock.advance(60)
    again = observe(store, body)                  # billing retries the same payment

    assert again.replayed and not first.replayed
    assert {k: v for k, v in again.as_dict().items() if k != "replayed"} == \
           {k: v for k, v in first.as_dict().items() if k != "replayed"}
    assert observe(store, facts()).card_txn_count_1h == 1      # still one earlier payment, not two


def test_concurrent_payments_on_one_card_see_distinct_counts(store):
    seen, errors = [], []

    def pay():
        try:
            seen.append(observe(store, facts()).card_txn_count_1h)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=pay) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert sorted(seen) == list(range(20))        # atomic read-then-record: nobody saw the same "earlier" count


# ---- mapping details ----

def test_hour_is_ist_and_currency_is_converted_for_scoring(client):
    body = client.post("/v2/assess", json=facts(timestamp="2026-10-29T21:00:00+00:00",   # 02:30 IST
                                                currency="USD", amount_minor=10_000,      # USD 100.00
                                                card_country="US", ip_country="SG", billing_country="US",
                                                email_domain="mailinator.com")).json()
    snap = body["model_input"]
    assert snap["hour"] == 2
    assert snap["amount"] == 8300.0
    assert snap["is_international_card"] == 1
    assert snap["ip_billing_country_mismatch"] == 1
    assert snap["email_is_disposable"] == 1
