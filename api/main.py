"""
FastAPI service exposing the risk agent.

  POST /assess              transaction -> verdict (also written to the audit trail)
  POST /v2/assess           FACTS from billing -> engine computes velocity/profile itself (Redis) -> rules (rules.yaml)
                            + model -> verdict with reason codes
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
from typing import Callable, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)  # the risk_v2 package
for sub in ("agent", "model"):
    sys.path.insert(0, os.path.join(_ROOT, sub))

from audit import get_by_transaction_id, recent  # noqa: E402
from failure_sim import SIMULATIONS  # noqa: E402
from llm_client import DEFAULT_MODEL, is_configured  # noqa: E402
from reasoning_agent import assess  # noqa: E402
from risk_v2.assembler import UnsupportedCurrency, amount_in_inr_minor, to_model_transaction  # noqa: E402
from risk_v2.contract import AssessV2Request  # noqa: E402
from risk_v2.decision import decide  # noqa: E402
from risk_v2.rules import RuleBook, RuleLoader, RulesUnavailable  # noqa: E402
from risk_v2.store import FeatureStore, FeaturesUnavailable  # noqa: E402

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


_rule_loader = RuleLoader(os.environ.get("RULES_PATH", os.path.join(_ROOT, "rules.yaml")))


def get_rule_loader() -> RuleLoader:
    """rules.yaml (RULES_PATH), reloaded when the file changes. Tests override this."""
    return _rule_loader


@app.get("/health")
def health(loader: RuleLoader = Depends(get_rule_loader)) -> dict:
    try:
        rules_version = loader.get().version
    except RulesUnavailable:
        rules_version = None
    return {
        "status": "ok",
        "llm_configured": is_configured(),
        "llm_model": DEFAULT_MODEL if is_configured() else None,
        "rules_version": rules_version,   # the version /v2 decisions use right now
        "rules_error": loader.last_error,  # why the newest rules.yaml was rejected (the old version keeps running)
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


class VerdictV2(BaseModel):
    api_version: str
    transaction_id: str
    action: str                      # allow | hold_for_review | block - decided by rules + model, never the LLM
    reason_codes: list[str]          # fixed codes (risk_v2/reason_codes.py), most decisive first
    reasons: list[dict]              # each code with its source (rule / model / engine) and evidence
    rules_fired: list[str]
    rules_version: str
    model_score: Optional[float]
    threshold: Optional[float]       # review band start = the model bundle's threshold
    block_at: float
    model_contributions: dict        # reason code -> how much it raised the score (risk_v2/contributions.py)
    path_used: str                   # rules+model | rules_only (model could not score)
    failure_mode: Optional[str]
    explanation: str
    explanation_source: str          # llm | template | codes (the template itself failed)
    confidence: float
    feature_snapshot: dict
    signals: dict      # what the engine computed itself (velocity, profile, card) - see docs/adr/0001
    model_input: dict  # the exact raw transaction the model scored (facts + signals mapped to its features)


_feature_store: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """One Redis-backed store per process. REDIS_URL, default redis://localhost:6379/0. Tests override this."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    return _feature_store


def get_explainer() -> Optional[Callable[[str], str]]:
    """The LLM call for explanations; None = Gemini if a key is set, else the template. Tests override this."""
    return None


def get_rulebook(loader: RuleLoader = Depends(get_rule_loader)) -> RuleBook:
    try:
        return loader.get()
    except RulesUnavailable as exc:
        raise HTTPException(status_code=503, detail="risk rules unavailable, use fallback") from exc


@app.post("/v2/assess", response_model=VerdictV2)
def assess_v2(req: AssessV2Request, store: FeatureStore = Depends(get_feature_store),
              book: RuleBook = Depends(get_rulebook),
              explainer: Optional[Callable[[str], str]] = Depends(get_explainer)) -> dict:
    """Billing sends facts; every risk feature is computed here (weakness #14). Old /assess is unchanged.

    Pipeline (docs/adr/0002): rules.yaml -> model + per-reason contributions -> strictest action -> explanation.
    422: invalid facts or an unsupported currency. 503: the feature store (Redis) or the rules are unavailable -
    we never score with made-up "0 earlier payments" velocity; the caller uses its own fallback.
    """
    try:
        inr_minor = amount_in_inr_minor(req.amount_minor, req.currency)
    except UnsupportedCurrency as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        signals = store.observe(transaction_id=req.transaction_id, customer_id=req.customer_id,
                                card_fingerprint=req.card_fingerprint, device_id=req.device_id,
                                ip=str(req.ip), amount_inr_minor=inr_minor)
    except FeaturesUnavailable as exc:
        raise HTTPException(status_code=503, detail="risk features unavailable, use fallback") from exc
    model_input = to_model_transaction(req, signals, inr_minor)
    verdict = decide(req, signals, model_input, book, llm_fn=explainer)
    return {**verdict, "api_version": "v2", "signals": signals.as_dict(), "model_input": model_input}


@app.get("/audit/{transaction_id}")
def get_audit(transaction_id: str) -> dict:
    entry = get_by_transaction_id(transaction_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no audit entry for {transaction_id}")
    return entry


@app.get("/audit")
def list_audit(limit: int = 25) -> dict:
    return {"count": limit, "entries": recent(limit)}
