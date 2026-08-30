"""
FastAPI service exposing the risk agent.

  POST /assess              transaction -> verdict (also written to the audit trail)
  GET  /audit/{txn_id}      retrieve the logged decision for a transaction
  GET  /audit               recent decisions (dashboard convenience)
  GET  /metrics             Day-2 model metrics
  GET  /health              liveness

Run:
  uvicorn api.main:app --reload --port 8000     # from the repo root
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("agent", "model"):
    sys.path.insert(0, os.path.join(_ROOT, sub))

from audit import get_by_transaction_id, recent  # noqa: E402
from failure_sim import SIMULATIONS  # noqa: E402
from llm_client import DEFAULT_MODEL, is_configured  # noqa: E402
from reasoning_agent import assess  # noqa: E402

app = FastAPI(
    title="Razorpay Risk Agent",
    version="1.0.0",
    description="Chargeback-risk scoring + LLM reasoning agent with deterministic fallback.",
)

PAYMENT_METHODS = {"card", "upi", "netbanking", "wallet", "emi"}
CARD_TYPES = {"credit", "debit", "prepaid", "na"}


class TransactionIn(BaseModel):
    """One transaction to assess. Extra fields are rejected."""
    model_config = {"extra": "forbid"}

    transaction_id: Optional[str] = Field(default=None)
    merchant_id: str
    merchant_category: str
    customer_id: str
    amount: float = Field(gt=0, description="INR, must be positive")
    payment_method: str
    card_type: str = "na"
    is_international_card: int = Field(default=0, ge=0, le=1)
    hour: int = Field(ge=0, le=23)
    is_new_device: int = Field(default=0, ge=0, le=1)
    ip_billing_country_mismatch: int = Field(default=0, ge=0, le=1)
    account_age_days: float = Field(ge=0)
    past_chargebacks: int = Field(ge=0)
    past_orders: int = Field(ge=0)
    avg_order_value: float = Field(gt=0)
    txn_count_1h: int = Field(ge=0)
    txn_count_24h: int = Field(ge=0)
    email_is_disposable: int = Field(default=0, ge=0, le=1)

    @field_validator("payment_method")
    @classmethod
    def _pm(cls, v: str) -> str:
        if v not in PAYMENT_METHODS:
            raise ValueError(f"payment_method must be one of {sorted(PAYMENT_METHODS)}")
        return v

    @field_validator("card_type")
    @classmethod
    def _ct(cls, v: str) -> str:
        if v not in CARD_TYPES:
            raise ValueError(f"card_type must be one of {sorted(CARD_TYPES)}")
        return v


class Verdict(BaseModel):
    transaction_id: str
    model_score: Optional[float]
    threshold: Optional[float]
    path_used: str
    failure_mode: Optional[str]
    action: str
    explanation: str
    confidence: float
    rules_fired: list[str]
    feature_snapshot: dict


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": is_configured(),
        "llm_model": DEFAULT_MODEL if is_configured() else None,
    }


@app.get("/metrics")
def metrics() -> dict:
    import json
    path = os.path.join(_ROOT, "model", "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="model not trained yet")
    with open(path) as fh:
        return json.load(fh)


@app.post("/assess", response_model=Verdict)
def assess_transaction(txn: TransactionIn, simulate: Optional[str] = None) -> dict:
    """Assess a transaction.

    `?simulate=llm_timeout|llm_unreachable|llm_malformed|bad_input` forces that
    failure through the real pipeline - a demo affordance for the dashboard.
    """
    payload = txn.model_dump()
    if not payload.get("transaction_id"):
        payload["transaction_id"] = f"TXN-{uuid.uuid4().hex[:12]}"
    try:
        if simulate:
            if simulate not in SIMULATIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown simulate mode; choose from {sorted(SIMULATIONS)}",
                )
            return SIMULATIONS[simulate](payload)
        return assess(payload)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - assess() shouldn't raise, but never 500 silently
        raise HTTPException(status_code=500, detail=f"assessment failed: {exc}") from exc


@app.get("/audit/{transaction_id}")
def get_audit(transaction_id: str) -> dict:
    entry = get_by_transaction_id(transaction_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no audit entry for {transaction_id}")
    return entry


@app.get("/audit")
def list_audit(limit: int = 25) -> dict:
    return {"count": limit, "entries": recent(limit)}
