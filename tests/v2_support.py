"""Helpers for the /v2 tests (the fixtures are in conftest.py)."""

from __future__ import annotations

from datetime import datetime, timezone

from risk_v2.assembler import amount_in_inr_minor

HOUR = 3600
API_KEY = "test-billing-key-0123456789"
API_KEYS = f"billing:{API_KEY},dashboard:test-dashboard-key-0123"
AUTH = {"X-API-Key": API_KEY}


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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
