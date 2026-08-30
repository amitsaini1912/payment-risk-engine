"""
Multi-mode failure handling for the reasoning agent.

A finance service fails in more than one way. Each test below drives one distinct
failure into `assess()` and proves:
  * it does not crash,
  * it still returns a valid verdict (action in the allowed set),
  * it fell back to the deterministic engine,
  * the audit log records which failure path was taken.
"""

from __future__ import annotations

import json
import time

import pytest

from fallback_rules import VALID_ACTIONS
from reasoning_agent import assess


def _read_audit(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _assert_valid_fallback_entry(result: dict, path: str, expected_failure: str):
    assert result["action"] in VALID_ACTIONS
    assert isinstance(result["explanation"], str) and result["explanation"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["path_used"] == "fallback"
    assert result["failure_mode"] == expected_failure

    entries = _read_audit(path)
    assert len(entries) == 1
    logged = entries[0]
    assert logged["transaction_id"] == result["transaction_id"]
    assert logged["path_used"] == "fallback"
    assert logged["failure_mode"] == expected_failure
    assert logged["action"] in VALID_ACTIONS


# --- Failure mode 1: LLM call times out / API unreachable -------------------

def test_llm_timeout_falls_back(sample_transaction, audit_path):
    def slow_llm(_prompt):
        time.sleep(3.0)
        return '{"action": "allow", "explanation": "x", "confidence": 0.9}'

    result = assess(sample_transaction, llm_fn=slow_llm, timeout_s=0.5,
                    audit_path=audit_path)
    _assert_valid_fallback_entry(result, audit_path, "llm_timeout")


def test_llm_unreachable_falls_back(sample_transaction, audit_path):
    def unreachable_llm(_prompt):
        raise ConnectionError("could not resolve generativelanguage.googleapis.com")

    result = assess(sample_transaction, llm_fn=unreachable_llm, audit_path=audit_path)
    _assert_valid_fallback_entry(result, audit_path, "llm_error")
    assert "error_detail" in _read_audit(audit_path)[0]


# --- Failure mode 2: LLM returns malformed / non-JSON output ----------------

def test_llm_non_json_falls_back(sample_transaction, audit_path):
    def chatty_llm(_prompt):
        return "Sure! I think this transaction looks mostly fine, allow it."

    result = assess(sample_transaction, llm_fn=chatty_llm, audit_path=audit_path)
    _assert_valid_fallback_entry(result, audit_path, "llm_malformed")
    assert _read_audit(audit_path)[0]["llm_raw_output"].startswith("Sure!")


def test_llm_json_with_bad_action_falls_back(sample_transaction, audit_path):
    def bad_schema_llm(_prompt):
        return '{"action": "definitely_fraud", "explanation": "bad enum", "confidence": 2}'

    result = assess(sample_transaction, llm_fn=bad_schema_llm, audit_path=audit_path)
    _assert_valid_fallback_entry(result, audit_path, "llm_malformed")


# --- Failure mode 3: missing / corrupt input data --------------------------

def test_missing_required_field_falls_back(sample_transaction, audit_path):
    broken = dict(sample_transaction)
    broken["amount"] = None

    called = {"llm": False}

    def tracking_llm(_prompt):
        called["llm"] = True
        return '{"action": "allow", "explanation": "x", "confidence": 0.5}'

    result = assess(broken, llm_fn=tracking_llm, audit_path=audit_path)
    _assert_valid_fallback_entry(result, audit_path, "bad_input")
    assert result["model_score"] is None
    assert called["llm"] is False  # never even attempt the LLM on corrupt input


def test_corrupt_field_type_falls_back(sample_transaction, audit_path):
    broken = dict(sample_transaction)
    broken["account_age_days"] = "not-a-number"

    result = assess(broken, llm_fn=lambda p: "unused", audit_path=audit_path)
    _assert_valid_fallback_entry(result, audit_path, "bad_input")


# --- Happy path: the LLM path actually works when it behaves ----------------

def test_llm_happy_path_uses_llm(high_risk_transaction, audit_path):
    def good_llm(_prompt):
        return json.dumps({
            "action": "block",
            "explanation": "New account, prepaid intl card, odd hour, geo mismatch.",
            "confidence": 0.88,
        })

    result = assess(high_risk_transaction, llm_fn=good_llm, audit_path=audit_path)
    assert result["path_used"] == "llm"
    assert result["failure_mode"] is None
    assert result["action"] == "block"
    assert result["model_score"] is not None

    logged = _read_audit(audit_path)[0]
    assert logged["path_used"] == "llm"
    assert logged["failure_mode"] is None


def test_llm_fenced_json_is_parsed(sample_transaction, audit_path):
    def fenced_llm(_prompt):
        return '```json\n{"action": "hold_for_review", "explanation": "mid risk", "confidence": 0.6}\n```'

    result = assess(sample_transaction, llm_fn=fenced_llm, audit_path=audit_path)
    assert result["path_used"] == "llm"
    assert result["action"] == "hold_for_review"


@pytest.mark.parametrize("n", [1, 2, 3])
def test_never_raises(high_risk_transaction, audit_path, n):
    """Whatever the LLM does, assess() returns a dict - it never propagates."""
    behaviours = {
        1: lambda p: (_ for _ in ()).throw(RuntimeError("boom")),
        2: lambda p: None,
        3: lambda p: "",
    }
    result = assess(high_risk_transaction, llm_fn=behaviours[n], audit_path=audit_path)
    assert isinstance(result, dict)
    assert result["action"] in VALID_ACTIONS
    assert result["path_used"] == "fallback"
