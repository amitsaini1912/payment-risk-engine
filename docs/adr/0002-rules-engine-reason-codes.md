# ADR 0002 — `/v2` decisions: rules.yaml + model, fixed reason codes, LLM only explains

- Status: Accepted
- Date: 2026-10-30
- Builds on: ADR 0001 (`/v2/assess` facts + engine-computed signals)

## Context
On `/assess` (v1) the LLM decides and the deterministic rules are only a fallback. A payment decision must be
repeatable, explainable with stable codes (billing, analysts, disputes) and changeable by risk ops without a
deploy. The Day-14 card signals (velocity per fingerprint, devices/IPs per card) are not model features, so nothing
used them. The fallback's review threshold was a hand-copied 0.42 next to the bundle's 0.4237 (weakness #19).

## Decision
1. **Pipeline** (`risk_v2/decision.py`): `rules.yaml` → model score + per-reason contributions → **strictest** of
   (fired rules, model band) → explanation. Bands: score ≥ bundle threshold → `hold_for_review`, ≥ `block_at`
   (rules.yaml, 0.80) → `block`. Rules can only tighten (`hold_for_review` / `block`), never allow.
   The model scores even when a rule blocks (score recorded for Day 17/18).
2. **Rules in YAML** (`rules.yaml`, `risk_v2/rules.py`): condition trees (`all` / `any` / leaf
   `{field, op, value}`) over a fixed list of context fields; `yaml.safe_load`; strict pydantic schema (unknown
   field / operator / reason code / key, wrong value type, duplicate id → the whole file is rejected).
   - Hot reload: `stat()` per request; reload when (mtime_ns, size, inode) changes. Bad file → last good version
     keeps running, error in `/health` (`rules_error`), parsed once per change. Deploy by write + `mv` (atomic).
   - `rules_version` = first 12 hex of the file's SHA-256, on every decision and in `/health`.
   - No valid version ever loaded → **503** "use fallback" (never "no rules = allow"), and nothing is recorded
     in Redis (the rules dependency runs before the feature store).
3. **Fixed reason codes** (`risk_v2/reason_codes.py`): R01_VELOCITY … R10_CARD_SHARED, R90_MODEL_SCORE (model band
   without one dominant factor), R99_NOT_SCORED (model unavailable → at least hold). Never renamed or reused.
   Every hold/block carries ≥ 1 code. Response: `reason_codes` (unique, rules first, then model by size) and
   `reasons` (each with source rule/model/engine and evidence, e.g. `card_txn_count_1h=5 >= 5`).
4. **Model reasons without SHAP** (`risk_v2/contributions.py`): the model is gradient-boosted trees (no
   coefficients). Per reason: put its inputs back to normal, score again, contribution = drop in score. All
   variants in one batch with the real row (same cost as one row). Reported if ≥ `min_contribution` (0.02), top 3.
   No new dependency (SHAP not approved).
5. **The LLM only words the explanation** (`risk_v2/explain.py`): it is told the decision is final; whatever it
   returns, the action doesn't change. Any LLM failure → template from the reason codes; a crash in the template
   → `"BLOCK: R04_…, R02_…"` (`explanation_error`). No key → template, no failure.
6. **#19**: v1 fallback takes `review_at` from the bundle (required with a score); `/v2` uses the bundle
   threshold too.
7. Model OpenMP threads limited to 1 (`MODEL_THREADS`): under CPU contention the thread hand-off made a 5-row
   score p95 665 ms vs 73 ms (Day 15, busy machine).

## Consequences
- New dependency: `PyYAML==6.0.2`.
- Contributions are leave-one-out counterfactuals: they don't sum to the score; two redundant signals (an OR) both
  look small; "normal" values are fixed (₹1,500, 14:00, 365 days, debit card) and change with the next model.
- The current model gives low scores to patterns the rules catch (card testing across customers scored 0.072; a
  first ₹45k payment at 03:00 between 0.31 and 0.60): the rules carry these until the Day 17 retrain.
- Reason codes are also reported on allow decisions (informational).
- A half-written file that happens to be valid YAML would load as fewer rules: only atomic replace is safe.
  Validation can't catch a valid-but-wrong rule (e.g. `>= 1` instead of `>= 10`) — no shadow mode for rules yet.
- Hot reload is per process: N replicas reload independently (each stats its own file/ConfigMap).
- `BLOCK_AT` 0.80 still exists twice: v1 fallback (code) and `/v2` (rules.yaml).
- The explanation still waits for the LLM up to `LLM_EXPLAIN_TIMEOUT_S` (5 s) when a key is set — Day 16 moves it
  inside the 300 ms budget.
