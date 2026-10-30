"""
Prometheus metrics (Day 16), served at GET /metrics/prometheus (GET /metrics stays the model's training metrics).

Percentiles are NOT computed here: the histogram exports buckets and Prometheus computes p50/p95/p99, e.g.
  histogram_quantile(0.95, sum by (le) (rate(risk_request_duration_seconds_bucket{endpoint="/v2/assess"}[5m])))
Fallback rate (billing had to use its own rules) = rate(risk_unavailable_total[5m]) / rate(requests).
One uvicorn worker per container: the default registry is per process (scale with replicas, not workers).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

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
LLM_IN_FLIGHT = Gauge("risk_llm_calls_in_flight", "LLM calls running on the bounded pool")
DRIFT_PSI = Gauge("risk_drift_psi", "PSI of live /v2 inputs vs training data (updated on scrape)",
                  ["model_version", "feature"])
SHADOW_AGREEMENT = Gauge("risk_shadow_final_action_agreement", "Share of /v2 decisions where the shadow model "
                         "would have led to the same final action", ["champion", "shadow"])
