"""
Why did the MODEL score this payment high? Per-reason contributions without SHAP (docs/adr/0002).

The model is gradient-boosted trees (no coefficients to read). For each reason code we take the payment, put that
reason's inputs back to a normal value ("what if this customer weren't new?"), and score again:

    contribution(R) = score(payment) - score(payment with R's inputs made normal)

All variants are scored in ONE batch with the real payment (~7 ms for 1 or 11 rows), so it costs almost nothing.
It is a leave-one-out counterfactual, not SHAP: contributions don't add up to the score, and two signals that each
explain the risk on their own (an OR) both look small. Good enough to name the top drivers; documented as such.
"""

from __future__ import annotations

from typing import Callable

# A normal order for a customer we have no history for (INR): training data median amount ~2,100,
# median avg_order_value ~1,600.
REFERENCE_AMOUNT_INR = 1500.0
NORMAL_HOUR = 14
SETTLED_ACCOUNT_DAYS = 365


def _velocity(t: dict) -> None:
    t["txn_count_1h"] = 0
    t["txn_count_24h"] = 0


def _amount(t: dict) -> None:
    usual = t["avg_order_value"] if t["past_orders"] > 0 else REFERENCE_AMOUNT_INR
    if t["amount"] > usual:
        if t["past_orders"] == 0:
            t["avg_order_value"] = usual  # no history: the average IS the amount, keep the ratio at 1
        t["amount"] = usual


def _set(field: str, value) -> Callable[[dict], None]:
    def apply(t: dict) -> None:
        t[field] = value
    return apply


def _account_age(t: dict) -> None:
    t["account_age_days"] = max(t["account_age_days"], SETTLED_ACCOUNT_DAYS)


def _card(t: dict) -> None:
    t["is_international_card"] = 0
    if t["card_type"] == "prepaid":
        t["card_type"] = "debit"


# reason code -> how to make its inputs normal (model features only: R10 is a card signal the model doesn't have)
NORMALIZERS: dict[str, Callable[[dict], None]] = {
    "R01_VELOCITY": _velocity,
    "R02_AMOUNT_ANOMALY": _amount,
    "R03_NEW_DEVICE": _set("is_new_device", 0),
    "R04_NEW_ACCOUNT": _account_age,
    "R05_CHARGEBACK_HISTORY": _set("past_chargebacks", 0),
    "R06_GEO_MISMATCH": _set("ip_billing_country_mismatch", 0),
    "R07_ODD_HOUR": _set("hour", NORMAL_HOUR),
    "R08_DISPOSABLE_EMAIL": _set("email_is_disposable", 0),
    "R09_HIGH_RISK_CARD": _card,
}


def score_with_contributions(score_batch: Callable[[list[dict]], list[float]],
                             txn: dict) -> tuple[float, dict[str, float]]:
    """Return (score, {reason code: contribution}) for the codes whose inputs are not already normal."""
    variants: list[tuple[str, dict]] = []
    for code, normalize in NORMALIZERS.items():
        variant = dict(txn)
        normalize(variant)
        if variant != txn:
            variants.append((code, variant))
    scores = score_batch([txn] + [v for _, v in variants])
    score = scores[0]
    return score, {code: round(score - s, 4) for (code, _), s in zip(variants, scores[1:])}
