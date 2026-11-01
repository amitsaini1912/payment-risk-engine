"""
Prometheus metrics (Day 16), served at GET /metrics/prometheus (GET /metrics stays the model's training metrics).

Percentiles are NOT computed here: the histogram exports buckets and Prometheus computes p50/p95/p99, e.g.
  histogram_quantile(0.95, sum by (le) (rate(risk_request_duration_seconds_bucket{endpoint="/v2/assess"}[5m])))
Fallback rate (billing had to use its own rules) = rate(risk_unavailable_total[5m]) / rate(requests).
Several uvicorn workers per container (ADR 0011): with PROMETHEUS_MULTIPROC_DIR set, every process writes its
values to files there and a scrape (answered by any one worker) adds them all up - without it a scrape would show one
random worker's numbers. Gauges say how to combine: in-flight LLM calls are summed over live workers, drift/shadow
gauges (the same Redis-backed value in every worker) take the most recent write. The directory must be empty when the
container starts (the Dockerfile's CMD clears it).
"""

from __future__ import annotations

import os

from prometheus_client import (CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry, Counter, Gauge, Histogram,
                               generate_latest)

__all__ = ["CONTENT_TYPE_LATEST", "exposition"]

REQUEST_SECONDS = Histogram(
    "risk_request_duration_seconds", "Time to answer, by endpoint and status",
    ["endpoint", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0),
)
DECISIONS = Counter("risk_decisions_total", "Decisions made (replays not counted)",
                    ["api_version", "action", "path_used"])
REPLAYS = Counter("risk_replays_total", "Requests answered with a stored decision")
UNAVAILABLE = Counter("risk_unavailable_total", "503s: the caller must use its fallback", ["reason"])
REASON_CODES = Counter("risk_reason_codes_total", "Reason codes on hold/block decisions", ["code"])
EXPLANATIONS = Counter("risk_explanations_total", "How /v2 explanations were produced", ["source", "failure"])
BUDGET_EXCEEDED = Counter("risk_budget_exceeded_total", "/v2 decisions that took longer than the budget")
AUTH_FAILURES = Counter("risk_auth_failures_total", "Requests refused for a missing or wrong API key")
LLM_IN_FLIGHT = Gauge("risk_llm_calls_in_flight", "LLM calls running on the bounded pool",
                      multiprocess_mode="livesum")
DRIFT_PSI = Gauge("risk_drift_psi", "PSI of live /v2 inputs vs training data (updated on scrape)",
                  ["model_version", "feature"], multiprocess_mode="mostrecent")
SHADOW_AGREEMENT = Gauge("risk_shadow_final_action_agreement", "Share of /v2 decisions where the shadow model "
                         "would have led to the same final action", ["champion", "shadow"],
                         multiprocess_mode="mostrecent")
FEEDBACK_EVENTS = Counter("risk_feedback_events_total", "Feedback events handled: chargeback.created (Day 18), "
                          "payment.outcome (Day 19b)",
                          ["topic", "outcome"])   # outcome: stored | duplicate | unmatched | dead_lettered | retried
FEEDBACK_WRITE_FAILURES = Counter("risk_feedback_write_failures_total",
                                  "/v2 decisions not kept for the feedback loop (SQLite error); the decision stood")


def exposition() -> bytes:
    """The /metrics/prometheus body: all workers' values in multiprocess mode, this process's otherwise."""
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry)
    return generate_latest(REGISTRY)
