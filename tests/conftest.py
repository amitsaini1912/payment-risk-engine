"""Shared test fixtures and path setup."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("agent", "model", "api"):
    sys.path.insert(0, os.path.join(ROOT, sub))


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
