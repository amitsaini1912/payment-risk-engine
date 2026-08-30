"""
Deterministic fallback decision engine.

Used automatically whenever the LLM path cannot produce a valid verdict
(timeout, unreachable, malformed output) or when the input is too corrupt to
score. It is intentionally simple, fully explainable, and has no external
dependencies - so it cannot itself fail for the reasons the LLM path can.

Design: score bands anchored on the cost-based threshold from Day 2 (0.42),
plus a few hard rules that fire regardless of the model score.
"""

from __future__ import annotations

VALID_ACTIONS = ("allow", "hold_for_review", "block")

# Score bands. `review_at` is the model's shipped threshold; `block_at` is a
# higher band where we skip manual review and decline outright.
REVIEW_AT = 0.42
BLOCK_AT = 0.80


def decide(transaction: dict, model_score: float | None,
           reason_for_fallback: str) -> dict:
    """Return a verdict dict: {action, explanation, confidence, rules_fired}."""
    rules_fired: list[str] = []
    txn = transaction or {}

    def g(key, default=0):
        v = txn.get(key, default)
        return default if v is None else v

    # --- hard rules (independent of the score) ---
    hard_action = None

    past_cb = _num(g("past_chargebacks"))
    if past_cb >= 3:
        rules_fired.append(f"customer has {int(past_cb)} prior chargebacks (>=3)")
        hard_action = "block"
    elif past_cb == 2:
        rules_fired.append("customer has 2 prior chargebacks")
        hard_action = _max_action(hard_action, "hold_for_review")

    amount = _num(g("amount"))
    acct_age = _num(g("account_age_days"), default=9999)
    hour = _num(g("hour"), default=12)
    odd_hour = hour >= 23 or hour <= 5
    if acct_age <= 30 and amount >= 15000 and odd_hour:
        rules_fired.append(
            "new account (<=30d) + high amount (>=Rs15k) + odd hour - stolen-card signature"
        )
        hard_action = "block"

    if _num(g("txn_count_1h")) >= 6 and 0 < amount <= 900:
        rules_fired.append("6+ tiny transactions in the last hour - card-testing signature")
        hard_action = _max_action(hard_action, "hold_for_review")

    if _num(g("ip_billing_country_mismatch")) == 1 and _num(g("is_new_device")) == 1 and amount >= 5000:
        rules_fired.append("IP/billing country mismatch + new device on a high-value order")
        hard_action = _max_action(hard_action, "hold_for_review")

    # --- score-band rule ---
    if model_score is None:
        rules_fired.append(
            "transaction could not be scored - defaulting to manual review (conservative)"
        )
        band_action = "hold_for_review"
        confidence = 0.4
    else:
        if model_score >= BLOCK_AT:
            band_action = "block"
        elif model_score >= REVIEW_AT:
            band_action = "hold_for_review"
        else:
            band_action = "allow"
        rules_fired.append(
            f"model score {model_score:.3f} -> band '{band_action}' "
            f"(review>={REVIEW_AT}, block>={BLOCK_AT})"
        )
        # confidence: distance from the nearest band edge
        edges = [0.0, REVIEW_AT, BLOCK_AT, 1.0]
        nearest = min(abs(model_score - e) for e in edges)
        confidence = round(min(0.95, 0.55 + nearest), 2)

    action = _max_action(hard_action, band_action)

    explanation = (
        f"Deterministic fallback ({reason_for_fallback}). "
        f"Action: {action.upper()}. "
        + "; ".join(rules_fired)
        + "."
    )
    return {
        "action": action,
        "explanation": explanation,
        "confidence": confidence,
        "rules_fired": rules_fired,
    }


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _max_action(a: str | None, b: str | None) -> str:
    """Return the more restrictive of two actions."""
    order = {None: -1, "allow": 0, "hold_for_review": 1, "block": 2}
    return a if order.get(a, -1) >= order.get(b, -1) else b
