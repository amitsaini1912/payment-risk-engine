# ADR 0008 — JSON logs with trace ids

- Status: Accepted
- Date: 2026-10-31
- Other side: billing-service ADR 0020 (metrics, dashboard, ECS logs for the Java services).

## Decision
`LOG_FORMAT=json` (set in the Dockerfile and compose) → `risk_v2/json_logs.py` writes one JSON object per line for the
engine's loggers and uvicorn's (error + access): `@timestamp`, `log.level`, `log.logger`, `service.name`, `message`,
`trace_id` / `span_id` of the current span (Day 20a), `error.type` / `error.message` / `error.stack_trace`. Same
names as the Java services' ECS logs, so one query works for all of them. Messages and exceptions are masked again
(the record factory from Day 17 already masks message and args). Unset → plain text, as before.
Metrics: unchanged (`/metrics/prometheus` since Day 16), now scraped by the compose Prometheus and on billing's
Grafana dashboard (`/v2/assess` p95, decisions by action, feedback events).

## Results (real, 2026-10-31)
- `pytest -q`: 231 → **235** (`tests/test_json_logs.py` 4: fields + current trace id, none outside a span, PAN masked in
  message and stack trace, only with LOG_FORMAT=json - the test restores pytest's own handlers afterwards).
  Also in reverse file order.
- By hand: 400 recent container log lines all JSON, 182 with trace_id; 0 card numbers in them.
