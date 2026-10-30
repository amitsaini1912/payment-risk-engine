"""
Fixed reason codes for /v2/assess decisions (docs/adr/0002).

Billing, analysts and dashboards key on these codes, so they never change meaning: add new ones, never rename or
reuse. Rules (rules.yaml) and the model's per-feature contributions (risk_v2/contributions.py) both map to them.
"""

from __future__ import annotations

REASON_CODES: dict[str, str] = {
    "R01_VELOCITY": "many recent payments by this customer or on this card",
    "R02_AMOUNT_ANOMALY": "amount is high, or far above this customer's usual order",
    "R03_NEW_DEVICE": "device not seen for this customer before",
    "R04_NEW_ACCOUNT": "customer is new to us",
    "R05_CHARGEBACK_HISTORY": "customer has disputed payments before",
    "R06_GEO_MISMATCH": "IP country differs from the billing country",
    "R07_ODD_HOUR": "payment made at an unusual hour (23:00-05:59 IST)",
    "R08_DISPOSABLE_EMAIL": "throwaway email domain",
    "R09_HIGH_RISK_CARD": "prepaid or international card",
    "R10_CARD_SHARED": "one card used from many devices or IPs",
    "R90_MODEL_SCORE": "model score is high without one dominant factor",
    "R99_NOT_SCORED": "the model could not score this payment; held for review to be safe",
}
