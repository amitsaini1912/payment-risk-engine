# ADR 0006 — Payment outcomes from billing: labels on captured payments

- Status: Accepted
- Date: 2026-10-31
- Builds on: ADR 0005 (feedback loop). Other side: billing-service ADR 0017 (risk check) and 0018 (events).

## Context
ADR 0005 left precision/recall as lower bounds and trained on "final action allow", assuming those were captured.
The engine didn't know which payments were captured (only those can charge back), nor what a person decided about
the ones it held. Since Day 19b billing calls `/v2/assess` before authorization and reports what happened next.

## Decision
1. **Contract** `docs/contracts/payment.outcome.v1.schema.json`: event_id, transaction_id, `outcome` ∈ captured |
   review_approved | review_rejected, amount_minor, currency, occurred_at, schema_version 1. JSON, key =
   transaction_id, unknown fields ignored. Pydantic model == schema file (test, field by field, like chargebacks).
2. **One consumer, two topics** (`risk_v2/feedback_consumer.py`, was `chargeback_consumer.py`): chargeback.created
   and payment.outcome, each with its own model, handler and `<topic>.DLT`; counts per topic in `/health`,
   `risk_feedback_events_total{topic,outcome}` (replaces `risk_chargeback_events_total`). Outcomes are stored once
   per event_id (`payment_outcomes` table); for a payment the engine never decided → `unmatched`.
3. **Metadata refresh 10 s** (`KAFKA_METADATA_REFRESH_MS`, librdkafka default 5 min): the engine usually starts
   before billing creates its topics or adds partitions; until a refresh those are not read.
4. **`/feedback/performance`** adds per bucket `outcomes` (captured / review_approved / review_rejected) and
   `captured.final` / `captured.model`: precision/recall over CAPTURED payments only — every row there could charge
   back, so its labels are real. Flagged-and-captured = held, approved by a person, captured.
5. **Retraining population** (`scripts/retrain.py --population`): `captured` (default) = decisions billing reported
   captured — allowed ones and held-then-approved ones; `allow` = the Day 18 behaviour for an engine without
   outcomes. Never blocked or rejected payments (they can't charge back).
6. `scripts/produce_chargeback.py` → `scripts/produce_feedback.py` (`--outcome` for payment.outcome events).

## Results (real, 2026-10-31)
- `pytest -q`: 210 → **224** passed, also in reverse file order. New: outcome contract (schema == model, examples,
  billing's exact JSON incl. Java's ms + `Z` times), outcomes through Kafka (stored/duplicate/unmatched, own DLT),
  topics created after the consumer started (fails with the 5-minute default, passes with 10 s — checked both),
  6 contract violations, captured label population, captured precision/recall, retrain `--population allow`.
- By hand with billing (see billing ADR 0018): chargeback → label 1 within ~2 s of the fixed consumer starting;
  the customer's next payment held with `R05_CHARGEBACK_HISTORY`; review_approved + captured outcomes stored.

## Consequences
- Still blind to what was blocked or rejected: the `captured` numbers are real labels for a population the engine
  itself selected. Measuring blocks needs a small random allow-holdout — not done (it costs fraud losses).
- Reviewers' rejections are stored but not used as labels (a person's suspicion is not a chargeback).
- Metric name changed: `risk_chargeback_events_total` → `risk_feedback_events_total` (no dashboards used it yet).
