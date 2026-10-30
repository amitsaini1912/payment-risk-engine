"""
The /v2/assess request: FACTS about one card payment, as billing-service knows them (docs/adr/0001).

Nothing in here is a risk feature. Counts, averages, account age, chargeback history are computed by the engine
from what it has seen (risk_v2/store.py), so a caller can't make a transaction look safe by sending
`past_chargebacks=0` (weakness #14). Extra fields are rejected.

Money is an integer in minor units (paise, cents) plus an ISO 4217 code, like billing's Money - never a float.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from ipaddress import IPv4Address, IPv6Address
from typing import Literal, Optional, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, field_validator

ID = r"^[A-Za-z0-9_.:\-]+$"
MAX_CLOCK_SKEW = timedelta(minutes=5)


class AssessV2Request(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # --- required facts ---
    transaction_id: str = Field(min_length=1, max_length=64, pattern=ID,
                                description="billing's payment id; the same id again = the same transaction")
    amount_minor: StrictInt = Field(gt=0, le=10**12, description="integer minor units, e.g. 150000 = INR 1,500.00")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    card_token: str = Field(min_length=4, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    card_fingerprint: str = Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_\-]+$",
                                  description="same card -> same value across tokens (card-vault HMAC)")
    last4: str = Field(pattern=r"^\d{4}$")
    customer_id: str = Field(min_length=1, max_length=64, pattern=ID)
    merchant_id: str = Field(min_length=1, max_length=64, pattern=ID)
    ip: Union[IPv4Address, IPv6Address]
    device_id: str = Field(min_length=1, max_length=128, pattern=ID)
    timestamp: AwareDatetime = Field(description="when the payment was made, with a UTC offset")

    # --- optional context the engine can't work out itself (safe defaults when absent) ---
    merchant_category: Optional[str] = Field(default=None, max_length=40, pattern=r"^[a-z_]+$")
    card_type: Optional[Literal["credit", "debit", "prepaid"]] = None
    card_country: Optional[str] = Field(default=None, pattern=r"^[A-Z]{2}$")
    billing_country: Optional[str] = Field(default=None, pattern=r"^[A-Z]{2}$")
    ip_country: Optional[str] = Field(default=None, pattern=r"^[A-Z]{2}$")
    email_domain: Optional[str] = Field(default=None, max_length=255, pattern=r"^[a-z0-9.\-]+\.[a-z]{2,}$")

    @field_validator("timestamp")
    @classmethod
    def _not_in_the_future(cls, v: datetime) -> datetime:
        if v - datetime.now(timezone.utc) > MAX_CLOCK_SKEW:
            raise ValueError("timestamp is more than 5 minutes in the future")
        return v
