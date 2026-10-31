"""
FastAPI service exposing the risk agent.

  POST /assess              transaction -> verdict (also written to the audit trail)
  POST /v2/assess           FACTS from billing -> engine computes velocity/profile itself (Redis) -> rules (rules.yaml)
                            + model -> verdict with reason codes
  GET  /audit/{txn_id}      retrieve the logged decision for a transaction
  GET  /audit               recent decisions (dashboard convenience)
  GET  /metrics             Day-2 model metrics
  GET  /metrics/prometheus  service metrics for Prometheus (latency histogram, decisions, 503s, ...)
  GET  /health              liveness
  GET  /ready               readiness: API keys, rules, model and Redis all usable
  GET  /feedback/performance  precision/recall over time from chargebacks (Day 18, docs/adr/0005)
  GET  /feedback/{txn_id}     a /v2 decision with its chargebacks (the label)

/assess, /v2/assess, /audit, /governance and /feedback need `X-API-Key` (RISK_API_KEYS, risk_v2/auth.py).
KAFKA_ENABLED=true also starts the feedback consumer: chargeback.created + payment.outcome (risk_v2/feedback_consumer.py).

Run:
  uvicorn api.main:app --reload --port 8000     # from the repo root
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Callable, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, field_validator, model_validator

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)  # the risk_v2 package
for sub in ("agent", "model"):
    sys.path.insert(0, os.path.join(_ROOT, sub))

import llm_pool  # noqa: E402
from audit import get_by_transaction_id, recent, write_entry  # noqa: E402
from failure_sim import SIMULATIONS  # noqa: E402
from llm_client import DEFAULT_MODEL, is_configured  # noqa: E402
from pci import install_log_masking, mask, reject_pans  # noqa: E402
from reasoning_agent import assess  # noqa: E402
from risk_v2 import metrics as m  # noqa: E402
from risk_v2.assembler import UnsupportedCurrency, amount_in_inr_minor, to_model_transaction  # noqa: E402
from risk_v2.auth import ApiKeys, AuthNotConfigured  # noqa: E402
from risk_v2.budget import Deadline  # noqa: E402
from risk_v2.contract import AssessV2Request  # noqa: E402
from risk_v2.decision import decide  # noqa: E402
from risk_v2.feedback_consumer import FeedbackConsumer  # noqa: E402
from risk_v2.decisions import DecisionConflict, DecisionStore, request_hash  # noqa: E402
from risk_v2.feedback import DAY_MS, LABEL_WINDOW_DAYS, FeedbackLoop, OutcomeStore  # noqa: E402
from risk_v2.governance import Governance  # noqa: E402
from risk_v2.rules import RuleBook, RuleLoader, RulesUnavailable  # noqa: E402
from risk_v2 import tracing  # noqa: E402
from risk_v2.store import FeatureStore, FeaturesUnavailable  # noqa: E402
from scorer import load_scorer, load_shadow  # noqa: E402


install_log_masking()   # every log line, incl. uvicorn's access log: PANs/emails masked (Day 17)
tracing.setup()         # OTLP export only when OTEL_EXPORTER_OTLP_ENDPOINT is set (Day 20a)
log = logging.getLogger(__name__)
_consumer: Optional[FeedbackConsumer] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model and rules before the first request: the first payment must not pay ~1 s of joblib load."""
    for warm in (load_scorer, get_rule_loader().get):
        try:
            warm()
        except Exception as exc:  # noqa: BLE001 - /ready reports it; the process still starts
            print(f"[startup] warm-up {getattr(warm, '__name__', warm)} failed: {exc}")
    global _consumer
    outcomes = get_outcome_store()
    if os.environ.get("KAFKA_ENABLED", "false").lower() == "true" and outcomes is not None:
        _consumer = FeedbackConsumer.from_env(FeedbackLoop(outcomes, get_feature_store().client))
        _consumer.start()
    yield
    if _consumer is not None:
        _consumer.stop()
        _consumer = None


app = FastAPI(
    title="Razorpay Risk Agent",
    version="1.0.0",
    description="Chargeback-risk scoring + LLM reasoning agent with deterministic fallback.",
    lifespan=lifespan,
)
tracing.instrument(app)   # server spans; billing's traceparent joins /v2/assess to the payment's trace


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing what was sent: FastAPI's default puts each bad value in "input" - a PAN in a wrong field
    would be sent back (and logged by whatever logs responses). Only where + why."""
    errors = [{"loc": e.get("loc"), "type": e.get("type"), "msg": mask(str(e.get("msg", "")))} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


@app.middleware("http")
async def _timing(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    m.REQUEST_SECONDS.labels(getattr(route, "path", "unmatched"), str(response.status_code)).observe(
        time.perf_counter() - started)
    return response


@lru_cache(maxsize=1)
def get_api_keys() -> ApiKeys:
    """RISK_API_KEYS, read once. Tests override this."""
    return ApiKeys()


def require_caller(x_api_key: Optional[str] = Header(default=None),
                   keys: ApiKeys = Depends(get_api_keys)) -> str:
    try:
        caller = keys.caller(x_api_key)
    except AuthNotConfigured as exc:
        raise HTTPException(status_code=503, detail="risk engine auth not configured") from exc
    if caller is None:
        m.AUTH_FAILURES.inc()
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key",
                            headers={"WWW-Authenticate": "ApiKey"})
    return caller

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

    @model_validator(mode="before")
    @classmethod
    def _no_card_numbers(cls, values):
        reject_pans(values)   # Day 17: a PAN in any field -> 422 (agent/pci.py)
        return values

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
    model_version: Optional[str] = None    # Day 17: which model / prompt produced this verdict
    prompt_version: Optional[str] = None


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
        "feedback_consumer": _consumer.status() if _consumer else None,   # KAFKA_ENABLED=false -> null
    }


@app.get("/metrics")
def metrics() -> dict:
    path = os.path.join(_ROOT, "model", "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="model not trained yet")
    with open(path) as fh:
        return json.load(fh)


@app.post("/assess", response_model=Verdict)
def assess_transaction(txn: TransactionIn, simulate: Optional[str] = None,
                       _caller: str = Depends(require_caller)) -> dict:
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
            verdict = SIMULATIONS[simulate](payload)
        else:
            verdict = assess(payload)
        m.DECISIONS.labels("v1", verdict["action"], verdict["path_used"]).inc()
        return verdict
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - assess() shouldn't raise, but never 500 silently
        raise HTTPException(status_code=500, detail=f"assessment failed: {exc}") from exc


class VerdictV2(BaseModel):
    api_version: str
    transaction_id: str
    replayed: bool                   # true = the stored decision for this transaction_id (Day 16), nothing recomputed
    decided_in_ms: float             # time to the decision on the first request (budget: RISK_BUDGET_MS, 300)
    action: str                      # allow | hold_for_review | block - decided by rules + model, never the LLM
    reason_codes: list[str]          # fixed codes (risk_v2/reason_codes.py), most decisive first
    reasons: list[dict]              # each code with its source (rule / model / engine) and evidence
    rules_fired: list[str]
    rules_version: str
    model_version: Optional[str]     # Day 17: model, rules and prompt versions on every decision
    prompt_version: str
    model_action: Optional[str]      # the model band alone (before rules)
    shadow: Optional[dict]           # shadow model's score/actions - recorded only, never used (docs/adr/0004)
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


def get_decision_store(store: FeatureStore = Depends(get_feature_store)) -> DecisionStore:
    return DecisionStore(store.client)


def get_governance(store: FeatureStore = Depends(get_feature_store)) -> Governance:
    return Governance(store.client)


_outcome_store: Optional[OutcomeStore] = None


def get_outcome_store() -> Optional[OutcomeStore]:
    """Decisions + chargebacks for the feedback loop (FEEDBACK_DB_PATH). None if the file can't be opened: /v2
    decisions go on without it (counted in risk_feedback_write_failures_total). Tests override this."""
    global _outcome_store
    if _outcome_store is None:
        try:
            _outcome_store = OutcomeStore.from_env()
        except (sqlite3.Error, OSError) as exc:
            log.warning("feedback store unavailable: %s", exc)
            return None
    return _outcome_store


def get_explainer() -> Optional[Callable[[str], str]]:
    """The LLM call for explanations; None = Gemini if a key is set, else the template. Tests override this."""
    return None


def get_rulebook(loader: RuleLoader = Depends(get_rule_loader)) -> RuleBook:
    try:
        return loader.get()
    except RulesUnavailable as exc:
        m.UNAVAILABLE.labels("rules").inc()
        raise HTTPException(status_code=503, detail="risk rules unavailable, use fallback") from exc


def _features_unavailable(exc: Exception) -> HTTPException:
    m.UNAVAILABLE.labels("features").inc()
    return HTTPException(status_code=503, detail="risk features unavailable, use fallback")


@app.post("/v2/assess", response_model=VerdictV2)
def assess_v2(req: AssessV2Request, caller: str = Depends(require_caller),
              store: FeatureStore = Depends(get_feature_store),
              decisions: DecisionStore = Depends(get_decision_store),
              book: RuleBook = Depends(get_rulebook),
              governance: Governance = Depends(get_governance),
              outcomes: Optional[OutcomeStore] = Depends(get_outcome_store),
              explainer: Optional[Callable[[str], str]] = Depends(get_explainer)) -> dict:
    """Billing sends facts; every risk feature is computed here (weakness #14). Old /assess is unchanged.

    Pipeline (docs/adr/0002): rules.yaml -> model + per-reason contributions -> strictest action -> explanation.
    Same transaction_id again -> the stored decision (docs/adr/0003); with different facts -> 409.
    401: no/wrong X-API-Key. 422: invalid facts or an unsupported currency. 503: Redis or the rules are
    unavailable - we never score with made-up "0 earlier payments" velocity; the caller uses its own fallback.
    """
    deadline = Deadline()
    req_hash = request_hash(req)
    try:
        stored = decisions.get(req.transaction_id, req_hash)
    except DecisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FeaturesUnavailable as exc:
        raise _features_unavailable(exc) from exc
    if stored is not None:
        m.REPLAYS.inc()
        return {**stored, "replayed": True}

    try:
        inr_minor = amount_in_inr_minor(req.amount_minor, req.currency)
    except UnsupportedCurrency as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        with tracing.tracer.start_as_current_span("risk.features"):
            signals = store.observe(transaction_id=req.transaction_id, customer_id=req.customer_id,
                                    card_fingerprint=req.card_fingerprint, device_id=req.device_id,
                                    ip=str(req.ip), amount_inr_minor=inr_minor)
    except FeaturesUnavailable as exc:
        raise _features_unavailable(exc) from exc
    model_input = to_model_transaction(req, signals, inr_minor)
    with tracing.tracer.start_as_current_span("risk.decide") as span:
        verdict = decide(req, signals, model_input, book, llm_fn=explainer, deadline=deadline)
        span.set_attributes({"risk.action": verdict["action"], "risk.path": verdict["path_used"],
                             "risk.model_version": verdict.get("model_version") or "none",
                             "risk.rules_version": verdict["rules_version"]})
    decision = {**verdict, "api_version": "v2", "replayed": False, "signals": signals.as_dict(),
                "model_input": model_input}
    try:
        with tracing.tracer.start_as_current_span("risk.store"):
            kept = decisions.put_first(req.transaction_id, req_hash, decision)
    except DecisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FeaturesUnavailable as exc:  # decided but can't store it: a retry might differ -> let billing fall back
        raise _features_unavailable(exc) from exc
    if kept is not decision:            # a concurrent copy of this request stored first: answer with that one
        m.REPLAYS.inc()
        return {**kept, "replayed": True}

    write_entry({**decision, "caller": caller, "trace_id": tracing.current_trace_id()})   # audit -> trace (Day 20a)
    _keep_for_feedback(outcomes, decision)
    if decision["model_score"] is not None:
        governance.record(decision, model_input, load_scorer().drift_reference)
    m.DECISIONS.labels("v2", decision["action"], decision["path_used"]).inc()
    m.EXPLANATIONS.labels(decision["explanation_source"], decision["failure_mode"] or "none").inc()
    if decision["action"] != "allow":
        for code in decision["reason_codes"]:
            m.REASON_CODES.labels(code).inc()
    if decision["decided_in_ms"] > deadline.budget_s * 1000:
        m.BUDGET_EXCEEDED.inc()
    return decision


def _keep_for_feedback(outcomes: Optional[OutcomeStore], decision: dict) -> None:
    """Like the audit trail: a failure here must not change or fail the decision (it only loses a future label)."""
    try:
        if outcomes is None:
            raise sqlite3.OperationalError("feedback store not available")
        outcomes.record_decision(decision)
    except sqlite3.Error as exc:
        m.FEEDBACK_WRITE_FAILURES.inc()
        log.warning("decision %s not kept for the feedback loop: %s", decision["transaction_id"], exc)


@app.get("/audit/{transaction_id}")
def get_audit(transaction_id: str, _caller: str = Depends(require_caller)) -> dict:
    entry = get_by_transaction_id(transaction_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no audit entry for {mask(transaction_id)}")   # no PAN echo
    return entry


@app.get("/audit")
def list_audit(limit: int = 25, _caller: str = Depends(require_caller)) -> dict:
    entries = recent(limit)
    return {"count": len(entries), "entries": entries}   # was `"count": limit` (weakness #20)


@app.get("/ready")
def ready(loader: RuleLoader = Depends(get_rule_loader), keys: ApiKeys = Depends(get_api_keys),
          store: FeatureStore = Depends(get_feature_store)) -> Response:
    """Readiness (k8s/compose): 200 only when a /v2 decision can actually be made, else 503 with what is wrong."""
    checks = {}
    for name, check in (("api_keys", lambda: keys.caller("") is None),
                        ("rules", lambda: loader.get().version),
                        ("model", lambda: load_scorer().threshold),
                        ("redis", lambda: store.client.ping())):
        try:
            check()
            checks[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks[name] = f"{type(exc).__name__}: {exc}"[:200]
    ok = all(v == "ok" for v in checks.values())
    return Response(content=json.dumps({"ready": ok, "checks": checks}),
                    status_code=200 if ok else 503, media_type="application/json")


@app.get("/governance/shadow")
def shadow_report(_caller: str = Depends(require_caller), governance: Governance = Depends(get_governance)) -> dict:
    """Champion vs shadow on live /v2 traffic since this champion went live (docs/adr/0004)."""
    shadow = load_shadow()
    return governance.shadow_report(load_scorer().model_version, shadow.model_version if shadow else None)


@app.get("/governance/drift")
def drift_report(_caller: str = Depends(require_caller), governance: Governance = Depends(get_governance)) -> dict:
    """PSI of live /v2 traffic against the champion's training data, per key feature (model/drift.py)."""
    scorer = load_scorer()
    return governance.drift_report(scorer.model_version, scorer.drift_reference)


@app.get("/metrics/prometheus")
def prometheus_metrics(governance: Governance = Depends(get_governance)) -> Response:
    m.LLM_IN_FLIGHT.set(llm_pool.in_flight())
    try:   # drift + shadow agreement as gauges, from the shared Redis counters
        scorer, shadow = load_scorer(), load_shadow()
        for feature, row in governance.drift_report(scorer.model_version, scorer.drift_reference)["features"].items():
            if row["psi"] is not None:
                m.DRIFT_PSI.labels(scorer.model_version, feature).set(row["psi"])
        if shadow is not None:
            report = governance.shadow_report(scorer.model_version, shadow.model_version)
            if report["n"]:
                m.SHADOW_AGREEMENT.labels(scorer.model_version, shadow.model_version).set(
                    report["final_action_agreement"])
    except Exception:  # noqa: BLE001 - Redis or model trouble shows in /ready; metrics still served
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _require_outcomes(outcomes: Optional[OutcomeStore] = Depends(get_outcome_store)) -> OutcomeStore:
    if outcomes is None:
        raise HTTPException(status_code=503, detail="feedback store unavailable")
    return outcomes


@app.get("/feedback/performance")
def feedback_performance(bucket: Literal["day", "week"] = "day", days: int = Query(default=30, ge=1, le=730),
                         model_version: Optional[str] = Query(default=None, max_length=64),
                         _caller: str = Depends(require_caller),
                         outcomes: OutcomeStore = Depends(_require_outcomes)) -> dict:
    """Precision/recall of /v2 decisions per day or week, from the chargebacks received so far (docs/adr/0005).
    Lower bounds (blocked payments can't charge back); buckets younger than the label window are not `mature`."""
    return outcomes.performance(bucket=bucket, days=days, model_version=model_version)


@app.get("/feedback/{transaction_id}")
def feedback_for(transaction_id: str, _caller: str = Depends(require_caller),
                 outcomes: OutcomeStore = Depends(_require_outcomes)) -> dict:
    decision, chargebacks = outcomes.decision(transaction_id), outcomes.chargebacks(transaction_id)
    if decision is None and not chargebacks:
        raise HTTPException(status_code=404, detail=f"no decision or chargeback for {mask(transaction_id)}")
    if chargebacks:
        label = 1
    else:   # "no chargeback" is only a label once the payment is older than the window chargebacks can arrive in
        label = 0 if decision["decided_at_ms"] + LABEL_WINDOW_DAYS * DAY_MS <= outcomes.now_ms() else None
    return {"transaction_id": transaction_id, "decision": decision, "chargebacks": chargebacks, "label": label}
