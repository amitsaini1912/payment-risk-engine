"""
The reasoning agent.

Given a raw transaction, it:
  1. Scores it with the Day-2 model.
  2. Asks an LLM (Gemini flash-lite) for a STRUCTURED verdict - action +
     explanation + confidence - passing the score and the derived risk features
     as context. The LLM *decides*; it is not a chatbot.
  3. Enforces a hard ~8s wall-clock timeout on that call.
  4. Validates the LLM's JSON against a strict schema.
  5. On ANY failure (timeout, unreachable, malformed output, un-scorable input)
     falls back to the deterministic rule engine.
  6. Logs every decision - LLM or fallback - to the audit trail, recording which
     failure (if any) triggered the fallback.

Nothing here raises to the caller. A finance service must always return a
verdict.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "model"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit import write_entry  # noqa: E402
from fallback_rules import VALID_ACTIONS, decide as fallback_decide  # noqa: E402
from llm_client import LLMError, LLMUnavailable, call_gemini  # noqa: E402
from scorer import ScoringError, load_scorer  # noqa: E402

DEFAULT_TIMEOUT_S = 8.0

_PROMPT = """You are a payment-risk decision agent for a payment gateway. Decide \
what to do with ONE transaction.

You are given a machine-learning risk score (probability this transaction will be \
charged back, 0-1) and the derived risk features. All amounts are in INR (Rs). \
The operating threshold is {threshold:.2f}: at or above it the transaction \
normally needs review.

Transaction id: {txn_id}
Model risk score: {score:.3f}  (threshold {threshold:.2f})
Derived risk features (JSON):
{features}

Choose ONE action:
  - "allow": low risk, let it through
  - "hold_for_review": send to a human fraud analyst
  - "block": decline outright (only for very strong fraud signals)

Respond with ONLY a JSON object, no prose, no code fences:
{{"action": "allow|hold_for_review|block", "explanation": "<=40 words, cite the \
specific features that drove your decision", "confidence": <float 0-1>}}"""


def _build_prompt(txn_id, score, threshold, features) -> str:
    return _PROMPT.format(
        txn_id=txn_id, score=score, threshold=threshold,
        features=json.dumps(features, indent=2),
    )


def _extract_json(text: str) -> dict:
    """Parse the first JSON object in the text. Raises ValueError if none/invalid."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in LLM output")
    return json.loads(match.group(0))


def _validate_verdict(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    action = obj.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action {action!r}")
    explanation = obj.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("missing/empty explanation")
    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        raise ValueError("confidence is not a number")
    confidence = max(0.0, min(1.0, confidence))
    return {"action": action, "explanation": explanation.strip(),
            "confidence": round(confidence, 3)}


class LLMTimeout(RuntimeError):
    """The LLM call did not return within the wall-clock budget."""


def _run_with_timeout(fn, arg, timeout_s: float):
    """Run fn(arg) with a hard wall-clock timeout.

    Uses a daemon thread and does NOT join it back if it overruns - so a hung
    LLM call cannot block the decision path. The orphaned thread dies with the
    process; a real Gemini timeout surfaces as a network error well before this.
    """
    box: dict = {}

    def target():
        try:
            box["result"] = fn(arg)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise LLMTimeout(f"no response within {timeout_s:.1f}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def assess(transaction: dict, *, llm_fn=None, timeout_s: float = DEFAULT_TIMEOUT_S,
           audit_path: str | None = None) -> dict:
    """Assess one transaction. Always returns a verdict dict; never raises."""
    txn = dict(transaction or {})
    txn_id = txn.get("transaction_id", "UNKNOWN")
    llm_fn = llm_fn or call_gemini

    path_used = "fallback"
    failure_mode = None
    model_score = None
    threshold = None
    features: dict = {}
    llm_raw: str | None = None

    # --- step 1: score (failure mode 3: bad / corrupt input) ---
    try:
        scorer = load_scorer()
        threshold = scorer.threshold
        model_score = scorer.score(txn)
        features = scorer.feature_snapshot(txn)
    except (ScoringError, FileNotFoundError) as exc:
        failure_mode = "bad_input"
        verdict = fallback_decide(txn, None, f"bad_input: {exc}")
        return _finalize(txn_id, None, threshold, "fallback", failure_mode,
                         verdict, {}, None, str(exc), audit_path)

    # --- step 2: LLM verdict (failure modes 1 & 2) ---
    prompt = _build_prompt(txn_id, model_score, threshold, features)
    error_detail = None
    try:
        llm_raw = _run_with_timeout(llm_fn, prompt, timeout_s)
    except LLMTimeout as exc:
        failure_mode = "llm_timeout"
        error_detail = str(exc)
    except LLMUnavailable as exc:
        failure_mode = "llm_unavailable"
        error_detail = str(exc)
    except LLMError as exc:
        failure_mode = "llm_error"
        error_detail = str(exc)
    except Exception as exc:  # noqa: BLE001 - anything else the LLM path throws
        failure_mode = "llm_error"
        error_detail = f"{type(exc).__name__}: {exc}"

    if failure_mode is None and not llm_raw:
        failure_mode = "llm_malformed"
        error_detail = "LLM returned an empty response"

    if llm_raw and failure_mode is None:
        try:
            verdict = _validate_verdict(_extract_json(llm_raw))
            verdict.setdefault("rules_fired", [])
            path_used = "llm"
        except (ValueError, json.JSONDecodeError) as exc:
            failure_mode = "llm_malformed"
            error_detail = str(exc)
            verdict = fallback_decide(txn, model_score, f"llm_malformed: {exc}")
    else:
        verdict = fallback_decide(
            txn, model_score, failure_mode or "llm_unavailable"
        )

    return _finalize(txn_id, model_score, threshold, path_used, failure_mode,
                     verdict, features, llm_raw, error_detail, audit_path)


def _finalize(txn_id, model_score, threshold, path_used, failure_mode,
              verdict, features, llm_raw, error_detail, audit_path) -> dict:
    result = {
        "transaction_id": txn_id,
        "model_score": round(model_score, 4) if model_score is not None else None,
        "threshold": threshold,
        "path_used": path_used,               # "llm" | "fallback"
        "failure_mode": failure_mode,         # None | llm_timeout | llm_unavailable
                                              # | llm_error | llm_malformed | bad_input
        "action": verdict["action"],
        "explanation": verdict["explanation"],
        "confidence": verdict["confidence"],
        "rules_fired": verdict.get("rules_fired", []),
        "feature_snapshot": features,
    }
    entry = dict(result)
    if error_detail:
        entry["error_detail"] = error_detail
    if llm_raw is not None:
        entry["llm_raw_output"] = llm_raw[:2000]
    write_entry(entry, path=audit_path) if audit_path else write_entry(entry)
    return result


if __name__ == "__main__":
    demo = {
        "transaction_id": "TXN-DEMO-1", "merchant_id": "M1001",
        "merchant_category": "electronics", "customer_id": "C1",
        "amount": 42000, "payment_method": "card", "card_type": "prepaid",
        "is_international_card": 1, "hour": 3, "is_new_device": 1,
        "ip_billing_country_mismatch": 1, "account_age_days": 4,
        "past_chargebacks": 0, "past_orders": 1, "avg_order_value": 1500,
        "txn_count_1h": 1, "txn_count_24h": 2, "email_is_disposable": 1,
    }
    print(json.dumps(assess(demo), indent=2))
