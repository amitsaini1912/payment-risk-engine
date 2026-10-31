"""Shared test fixtures and path setup."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT first so `import api.main` resolves when pytest is invoked as `pytest`
# (not just `python -m pytest`, which adds the cwd for you).
sys.path.insert(0, ROOT)
for sub in ("agent", "model", "api"):
    sys.path.insert(0, os.path.join(ROOT, sub))


@pytest.fixture(autouse=True)
def api_keys():
    """Every test runs with known API keys (v2_support.API_KEYS) instead of the environment's."""
    from api.main import app, get_api_keys
    from risk_v2.auth import ApiKeys
    from v2_support import API_KEYS
    app.dependency_overrides[get_api_keys] = lambda: ApiKeys(API_KEYS)
    yield
    app.dependency_overrides.pop(get_api_keys, None)


@pytest.fixture(autouse=True)
def _feedback_db_outside_the_repo(tmp_path, monkeypatch):
    """A test that reaches the real get_outcome_store() gets a private file, never the repo's data/feedback.db."""
    import api.main
    monkeypatch.setenv("FEEDBACK_DB_PATH", str(tmp_path / "feedback-default.db"))
    monkeypatch.setattr(api.main, "_outcome_store", None)


@pytest.fixture
def sample_transaction() -> dict:
    """A mid-risk, fully-populated transaction."""
    return {
        "transaction_id": "TXN-TEST-0001",
        "merchant_id": "M1001",
        "merchant_category": "electronics",
        "customer_id": "C-TEST",
        "amount": 8200.0,
        "payment_method": "card",
        "card_type": "credit",
        "is_international_card": 0,
        "hour": 14,
        "is_new_device": 1,
        "ip_billing_country_mismatch": 0,
        "account_age_days": 210,
        "past_chargebacks": 0,
        "past_orders": 12,
        "avg_order_value": 3000.0,
        "txn_count_1h": 1,
        "txn_count_24h": 2,
        "email_is_disposable": 0,
    }


@pytest.fixture
def high_risk_transaction() -> dict:
    return {
        "transaction_id": "TXN-TEST-HIGH",
        "merchant_id": "M1002",
        "merchant_category": "jewellery",
        "customer_id": "C-TEST-2",
        "amount": 45000.0,
        "payment_method": "card",
        "card_type": "prepaid",
        "is_international_card": 1,
        "hour": 3,
        "is_new_device": 1,
        "ip_billing_country_mismatch": 1,
        "account_age_days": 5,
        "past_chargebacks": 0,
        "past_orders": 1,
        "avg_order_value": 2000.0,
        "txn_count_1h": 1,
        "txn_count_24h": 1,
        "email_is_disposable": 1,
    }


@pytest.fixture
def audit_path(tmp_path) -> str:
    return str(tmp_path / "audit_trail.jsonl")


# ---- /v2: real Redis 7 (Testcontainers, one per test session), fake clock, a copy of rules.yaml ----

@pytest.fixture(scope="session")
def redis_client():
    import redis as redis_lib
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


@pytest.fixture(scope="session")
def kafka_bootstrap():
    """A real single-node Kafka (KRaft) - the same apache/kafka image as billing's docker-compose.yml.

    Not testcontainers' KafkaContainer: that one is written for Confluent's cp-kafka image. The broker must advertise
    the host port, so a free port is picked first and bound 1:1.
    """
    import socket
    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.waiting_utils import wait_for_logs
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        container = (DockerContainer("apache/kafka:3.9.1").with_bind_ports(9092, port)
                     .with_env("KAFKA_NODE_ID", "1").with_env("KAFKA_PROCESS_ROLES", "broker,controller")
                     .with_env("KAFKA_LISTENERS", "PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093")
                     .with_env("KAFKA_ADVERTISED_LISTENERS", f"PLAINTEXT://localhost:{port}")
                     .with_env("KAFKA_LISTENER_SECURITY_PROTOCOL_MAP", "PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT")
                     .with_env("KAFKA_CONTROLLER_LISTENER_NAMES", "CONTROLLER")
                     .with_env("KAFKA_CONTROLLER_QUORUM_VOTERS", "1@localhost:9093")
                     .with_env("KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR", "1")
                     .with_env("KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR", "1")
                     .with_env("KAFKA_TRANSACTION_STATE_LOG_MIN_ISR", "1")
                     .with_env("KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS", "0"))
        container.start()
        wait_for_logs(container, "Kafka Server started", timeout=90)
    except Exception as exc:  # noqa: BLE001 - no Docker on this machine
        pytest.skip(f"Docker/Kafka not available: {exc}")
    yield f"localhost:{port}"
    container.stop()


@pytest.fixture
def clock():
    from v2_support import FakeClock
    return FakeClock()


@pytest.fixture
def store(redis_client, clock):
    from risk_v2.store import FeatureStore
    redis_client.flushdb()
    return FeatureStore(redis_client, clock=clock)


@pytest.fixture
def rules_path(tmp_path) -> str:
    """A private copy of the real rules.yaml, so a test can edit it (hot reload) without touching the repo."""
    import shutil
    path = tmp_path / "rules.yaml"
    shutil.copy(os.path.join(ROOT, "rules.yaml"), path)
    return str(path)


@pytest.fixture
def explainer():
    """The LLM is off in tests: explanations come from the template. Tests that need an LLM replace it."""
    from llm_client import LLMUnavailable

    def _no_llm(_prompt):
        raise LLMUnavailable("disabled in tests")
    return {"fn": _no_llm}


@pytest.fixture
def outcomes(tmp_path, clock):
    """The feedback loop's SQLite file (Day 18), private to the test, on the same fake clock as the feature store."""
    from risk_v2.feedback import OutcomeStore
    return OutcomeStore(str(tmp_path / "feedback.db"), clock=clock)


@pytest.fixture
def v2_client(store, outcomes, rules_path, explainer, tmp_path, monkeypatch):
    if not os.path.exists(os.path.join(ROOT, "model", "model.pkl")):
        pytest.skip("model/model.pkl not built - run `python model/train.py`")
    from fastapi.testclient import TestClient

    import audit
    from api.main import app, get_explainer, get_feature_store, get_outcome_store, get_rule_loader
    from risk_v2.rules import RuleLoader

    monkeypatch.setattr(audit, "DEFAULT_PATH", str(tmp_path / "audit.jsonl"))
    loader = RuleLoader(rules_path)
    app.dependency_overrides[get_feature_store] = lambda: store
    app.dependency_overrides[get_rule_loader] = lambda: loader
    app.dependency_overrides[get_explainer] = lambda: explainer["fn"]
    app.dependency_overrides[get_outcome_store] = lambda: outcomes
    from v2_support import AUTH
    yield TestClient(app, headers=AUTH)
    for dep in (get_feature_store, get_rule_loader, get_explainer, get_outcome_store):
        app.dependency_overrides.pop(dep, None)
