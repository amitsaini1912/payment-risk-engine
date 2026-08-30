"""
Named failure simulators.

Each entry forces one distinct failure mode through the real `assess()` pipeline
so the deterministic fallback can be demonstrated on demand - from the CLI here,
and from the "simulate failure" buttons on the Day-4 dashboard.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning_agent import assess  # noqa: E402


def _slow_llm(_prompt: str) -> str:
    time.sleep(30)
    return '{"action": "allow", "explanation": "never returned in time", "confidence": 0.9}'


def _unreachable_llm(_prompt: str) -> str:
    raise ConnectionError("could not connect to generativelanguage.googleapis.com")


def _chatty_llm(_prompt: str) -> str:
    return ("I'd probably let this one through, it doesn't look too bad to me, "
            "but you might want a human to double-check.")


def sim_llm_timeout(transaction: dict, **kw) -> dict:
    """Failure mode 1a: the LLM call exceeds its wall-clock timeout."""
    return assess(transaction, llm_fn=_slow_llm, timeout_s=2.0, **kw)


def sim_llm_unreachable(transaction: dict, **kw) -> dict:
    """Failure mode 1b: the LLM API is unreachable."""
    return assess(transaction, llm_fn=_unreachable_llm, **kw)


def sim_llm_malformed(transaction: dict, **kw) -> dict:
    """Failure mode 2: the LLM returns prose instead of the required JSON."""
    return assess(transaction, llm_fn=_chatty_llm, **kw)


def sim_bad_input(transaction: dict, **kw) -> dict:
    """Failure mode 3: a required field is missing / corrupt."""
    broken = dict(transaction)
    broken["amount"] = None
    return assess(broken, llm_fn=lambda p: "unused", **kw)


SIMULATIONS = {
    "llm_timeout": sim_llm_timeout,
    "llm_unreachable": sim_llm_unreachable,
    "llm_malformed": sim_llm_malformed,
    "bad_input": sim_bad_input,
}


_DEMO_TXN = {
    "transaction_id": "TXN-SIM-0001", "merchant_id": "M1001",
    "merchant_category": "electronics", "customer_id": "C-SIM",
    "amount": 12500.0, "payment_method": "card", "card_type": "credit",
    "is_international_card": 0, "hour": 2, "is_new_device": 1,
    "ip_billing_country_mismatch": 1, "account_age_days": 20,
    "past_chargebacks": 1, "past_orders": 3, "avg_order_value": 2500.0,
    "txn_count_1h": 1, "txn_count_24h": 2, "email_is_disposable": 0,
}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(SIMULATIONS) if which == "all" else [which]
    for i, name in enumerate(names):
        txn = dict(_DEMO_TXN, transaction_id=f"TXN-SIM-{name.upper()}")
        print(f"\n=== simulating: {name} ===")
        result = SIMULATIONS[name](txn)
        print(json.dumps({k: result[k] for k in
                          ("transaction_id", "path_used", "failure_mode",
                           "action", "confidence", "explanation")}, indent=2))
