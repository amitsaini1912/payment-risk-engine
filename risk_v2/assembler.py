"""
Facts (AssessV2Request) + engine signals (Signals) -> the raw transaction dict the Day-2 model expects.

Mapping to the model's feature names (model/features.py REQUIRED_RAW_FIELDS):
  amount                       amount_minor converted to INR (rupees, float) - the model was trained on rupees
  payment_method               "card" (/v2 is the card path)
  card_type                    request, else "na"
  is_international_card        card_country given and not IN
  hour                         the caller's timestamp, in IST (Asia/Kolkata), like the training data
  is_new_device                device not seen for this CUSTOMER in 90 days
  ip_billing_country_mismatch  ip_country and billing_country both given and different
  account_age_days             days since the engine first saw this customer (0 = first time)
  past_chargebacks             engine's counter (0 until the Day 18 feedback loop feeds it)
  past_orders                  orders the engine assessed for this customer before
  avg_order_value              their average (INR); the current amount when there's no history
  txn_count_1h / txn_count_24h the CUSTOMER's earlier transactions in the window (the training data's meaning)
  email_is_disposable          email_domain in a small known list
  merchant_category            request, else "other" (unseen by the model -> ignored by its one-hot encoder)
Card-level signals (fingerprint velocity, distinct devices/IPs, new card) go to the response as `signals` for the
Day 15 rules; the current model doesn't use them.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from zoneinfo import ZoneInfo

from risk_v2.contract import AssessV2Request
from risk_v2.store import Signals

IST = ZoneInfo("Asia/Kolkata")
DAY_MS = 86_400_000

# Approximate INR value of a foreign amount, for SCORING only (the model thinks in rupees). Not for money movement:
# billing's ledger uses its own locked rate (ADR 0012).
SCORING_FX_TO_INR = {"INR": Decimal("1"), "USD": Decimal("83.00"), "EUR": Decimal("90.00")}

DISPOSABLE_EMAIL_DOMAINS = frozenset({
    "mailinator.com", "guerrillamail.com", "10minutemail.com", "tempmail.com", "yopmail.com", "trashmail.com",
    "sharklasers.com", "getnada.com", "dispostable.com", "throwawaymail.com",
})


class UnsupportedCurrency(ValueError):
    pass


def amount_in_inr_minor(amount_minor: int, currency: str) -> int:
    """Both INR/USD/EUR have 2 decimals, so minor -> INR minor is just the rate, HALF_EVEN to the paisa."""
    rate = SCORING_FX_TO_INR.get(currency)
    if rate is None:
        raise UnsupportedCurrency(f"currency {currency} is not supported for scoring "
                                  f"(supported: {sorted(SCORING_FX_TO_INR)})")
    return int((Decimal(amount_minor) * rate).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def to_model_transaction(req: AssessV2Request, signals: Signals, amount_inr_minor: int) -> dict:
    amount_inr = amount_inr_minor / 100
    prior_orders = signals.customer_prior_orders
    avg_order_value = (signals.customer_prior_amount_sum_minor / prior_orders / 100) if prior_orders else amount_inr
    account_age_days = max(0, signals.observed_at_ms - signals.customer_first_seen_ms) / DAY_MS
    mismatch = int(bool(req.ip_country and req.billing_country and req.ip_country != req.billing_country))
    return {
        "transaction_id": req.transaction_id,
        "merchant_id": req.merchant_id,
        "customer_id": req.customer_id,
        "merchant_category": req.merchant_category or "other",
        "amount": amount_inr,
        "payment_method": "card",
        "card_type": req.card_type or "na",
        "is_international_card": int(bool(req.card_country and req.card_country != "IN")),
        "hour": req.timestamp.astimezone(IST).hour,
        "is_new_device": 0 if signals.customer_device_known else 1,
        "ip_billing_country_mismatch": mismatch,
        "account_age_days": round(account_age_days, 2),
        "past_chargebacks": signals.customer_chargebacks,
        "past_orders": prior_orders,
        "avg_order_value": round(max(avg_order_value, 1.0), 2),
        "txn_count_1h": signals.customer_txn_count_1h,
        "txn_count_24h": signals.customer_txn_count_24h,
        "email_is_disposable": int(req.email_domain in DISPOSABLE_EMAIL_DOMAINS) if req.email_domain else 0,
    }
