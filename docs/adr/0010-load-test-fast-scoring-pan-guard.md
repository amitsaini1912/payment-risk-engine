# ADR 0010 — Load test findings: faster scoring per decision, PAN guard false positives

- Status: Accepted
- Date: 2026-10-31
- Other side: billing-service ADR 0022 (k6 checkout load test), `docs/PERFORMANCE.md` in that repo (all numbers).

## Context
The Day 22 k6 test of the whole checkout found two things in this engine:
1. **Capacity.** One uvicorn worker spent ~45 ms of CPU per `/v2/assess`. From 20 checkouts/s it queued, billing's
   300 ms budget ran out, billing's breaker opened, and **66% of payments were decided by billing's fallback rules**.
   k6 saw no errors. Profile: RandomForest (300 trees) 17.4 ms with the reason-code variants (the forest alone 9.8 ms,
   mostly scikit-learn's per-tree Python overhead for one row). Pandas feature engineering ran 3 times (score batch,
   snapshot, shadow model).
2. **422s for values that were not card numbers** (`agent/pci.py`), so those payments fell back too:
   `dev-<uuid>` device ids; card-vault fingerprints (64 hex chars) holding a Luhn-valid digit run: **0.44% of cards,
   on every payment**. And a pre-existing gap: a PAN after a short number (`order 12 4111…`) or two PANs in a row
   were not found, so not masked either.

## Decision
- **Fast forest inference** (`model/scorer.py`): average each tree's compiled `tree_.predict`, the same arithmetic as
  `predict_proba` (max difference over the 15,000 training rows: 1e-15). Only for a plain single-output
  `RandomForestClassifier`, and only if a self-check on random inputs matches scikit-learn when the model loads;
  otherwise scikit-learn. `MODEL_FAST_FOREST=false` switches it off.
- **Features engineered once per decision** (`risk_v2/decision.py`): the snapshot and the shadow model reuse the
  frame of the contribution batch (feature engineering is row by row). A shadow only needs `score_batch`; the
  shared frame is used when it also has `score_engineered`.
- **PAN guard**: a UUID next to a dash is still a UUID; a whole 64-character lowercase hex token is a digest, never a PAN;
  inside a run of digit groups every window of whole groups with 13–19 digits is checked, longest first.

## Results
38.6 → ~23 ms per decision in-process. In the checkout step test the engine answered every check up to 30/s
(before: 15/s); 72% of all checks answered (before: 33%). 0 false 422s in 601 checkouts.

## Consequences
- Accepted blind spot: a PAN hidden inside an exactly-64-lowercase-hex token is not found.
- The window scan finds a few more false positives in long, spaced or dashed numbers (more windows are Luhn-checked).
  Callers must not put long numbers in ids (as before).
- `_forest_proba` relies on `Tree.predict` (a public attribute of a fitted tree, but internal-ish). The load-time
  self-check and the all-rows test catch a change.
- Still one worker. More capacity = more processes: needs Prometheus multiprocess mode and a real DB instead of
  SQLite for the feedback store (one Kafka consumer per process is fine: consumer group).
