# ADR 0009 — Property-based tests (hypothesis)

- Status: Accepted
- Date: 2026-10-31
- Other side: billing-service ADR 0021 (storm tests, jqwik).

## Decision
`tests/test_properties.py` (`hypothesis==6.118.8`, test dependency only). Where possible the engine is compared with a
tiny pure-Python model of what it must do:
- scoring FX: integer result, exact for INR, ≤ half a paisa off, monotonic; unknown currencies refused;
- PCI: any Luhn-valid 13–19 digit number (plain, spaced, dashed) inside any digit-free text is found and masked, the
  masked text is clean; UUIDs (any case) are never card numbers and are kept by masking; < 13 digits never are;
- decisions (real rules.yaml, model score pinned, LLM off): final action = strictest of the model band and every
  fired rule; holds/blocks always have a reason; for the same facts a higher score is never a looser action;
- velocity (real Redis, 150 random histories of ≤ 25 payments / retries / clock jumps + pinned window edges at
  exactly 1 h and 24 h): every counter equals a list-based model; a retry returns exactly the first snapshot;
- feedback: any mix of redelivered events and second disputes → stored once per event, counted once per payment.

## Results (real, 2026-10-31)
- `pytest -q`: 235 → **243** passed, also in reverse file order.
- Break checks (5): 1 h window made exclusive, 24 h cleanup off by one, strictest-action replaced by "first", the
  chargeback counter without its SADD guard, the PAN pattern without dashes — all caught. The 24 h one survived 60
  random examples at first (random histories rarely place two payments exactly 86,400 s apart) → window edges are now
  explicit `@example`s and examples raised to 150.
