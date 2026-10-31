# ADR 0007 — Tracing (OpenTelemetry) and UUIDs are never card numbers

- Status: Accepted
- Date: 2026-10-31
- Other side: billing-service ADR 0019 (agent + outbox trace context).

## Context
billing-service sends a W3C `traceparent` with `/v2/assess` and puts the payment request's context on every Kafka
event (its outbox stores it). The engine dropped it: its part of a payment was invisible.

## Decision
1. `risk_v2/tracing.py`: OpenTelemetry SDK 1.28.2 + FastAPI instrumentation 0.49b2, OTLP/HTTP; exports only when
   `OTEL_EXPORTER_OTLP_ENDPOINT` is set (compose: `http://jaeger:4318`), otherwise the API stays a no-op.
   - server span per request (`/health`, `/ready`, `/metrics*` left out); inside `/v2/assess`: `risk.features`,
     `risk.decide` (action, path, model + rules version as attributes), `risk.store`; the audit entry gets `trace_id`;
   - every consumed feedback event: a CONSUMER span, parent = the context in its Kafka headers.
2. **PAN guard** (`agent/pci.py`): text in canonical UUID form (8-4-4-4-12 hex) is never read as a card number.
   Day 17's rule (13–19 digits, single spaces/dashes, Luhn) matched digit groups INSIDE UUIDs: billing's payment id
   `f2fdd154-9879-3051-96c0-3a2147155a8d` → `154-9879-3051-96` passes Luhn → 422 → billing's fallback. 0.22% of
   random UUIDs (two UUID fields per payment → ~1 payment in 225). Found in the Day 20a traces (a `/v2/assess`
   without the engine's spans). A PAN glued to other text (`pay_5555…`, `M3782…`) or next to a UUID is still found.

## Results (real, 2026-10-31)
- `pytest -q`: 224 → **231** passed, also in reverse file order. New: `tests/test_tracing.py` 5 (billing's
  traceparent → server span parent + `risk.*` spans in the same trace + audit trace_id; no header → new trace; probes
  not traced; no endpoint → no exporter; Kafka header → consumer span in the producer's trace, no header → own trace),
  `test_governance.py` +2 (the real UUID accepted in transaction_id/customer_id, 0 of 20,000 random UUIDs flagged,
  masking keeps UUIDs; a PAN next to a UUID still 422). Break check: consumer ignoring the header → its test fails.
- By hand: see billing ADR 0019 (one trace id across 7 services); the failing UUID → 200 after the fix, with
  `risk.features` / `risk.decide` / `risk.store` in the caller's trace.

## Consequences
- `protobuf` (6 → 5.29) and `wrapt` (2 → 1.17) were downgraded by the OTel packages (transitive; tests unchanged).
- Every request is traced (no sampling) when an endpoint is set.
