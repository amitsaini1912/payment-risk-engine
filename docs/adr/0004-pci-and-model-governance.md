# ADR 0004 — PCI-aware inputs/outputs and model governance (versions, shadow, drift, honest metrics)

- Status: Accepted
- Date: 2026-10-31
- Builds on: ADR 0001–0003

## Context
- The engine needs card-vault tokens and fingerprints only. A PAN sent by mistake would end up in Redis keys, the
  audit trail and logs, and put this service in PCI DSS scope. FastAPI's default 422 even echoes the bad value back.
- Nobody could say which model, rules or prompt produced a decision; there was no way to try a new model safely or
  to see that live traffic no longer looks like the training data.
- The threshold was tuned on the test set the metrics came from (#15), and the training data's `txn_count_1h`
  counted later payments (found Day 14).

## Decision
1. **PANs** (`agent/pci.py`): 13–19 digits (single spaces/dashes allowed) passing Luhn.
   - Any text field of `/v2/assess` or `/assess` containing one → **422** naming only the field; the request never
     reaches Redis. All 422s drop FastAPI's `input` echo (custom handler).
   - Audit entries are scrubbed before writing (`[PAN ****1234]`, emails `j***@domain`); every log record (all
     loggers incl. uvicorn's access log) is masked in the log-record factory, arguments one by one.
   - Accepted false positive: a long Luhn-valid number inside an id (1 in 10 of such numbers). Billing ids are UUIDs.
2. **Versions on every decision** (v1 and v2, response + audit): `model_version` (`rf-<sha256(data+model+seed)>`),
   `rules_version` (rules.yaml hash, Day 15), `prompt_version` (`v1-`/`v2-` + prompt hash).
3. **Honest training** (`model/train.py`): 60/20/20 stratified split; fit on train; threshold per candidate on
   validation (cost-minimising with the 0.3 precision floor); **champion = lowest validation cost** (tie-break
   AP); test used once for the reported numbers. Generator fixed: events are generated in time order, velocity
   counts only earlier payments.
4. **Shadow model**: the runner-up is saved as `model/shadow.pkl`. On each new `/v2` decision (budget permitting,
   ≥ 20 ms left) it scores the same row; its band and "final action with the same rules" are returned under
   `shadow` and counted in Redis (`risk:shadow:<champion>:<shadow>`), never used for the action. A broken or missing
   shadow → `shadow.skipped` / `null`. Report: `GET /governance/shadow` (agreement, action matrices, mean score
   difference) and gauge `risk_shadow_final_action_agreement`.
5. **PSI drift** (`model/drift.py`): training deciles per key feature stored in the bundle; each new decision adds
   to its bin in Redis (`risk:drift:<model>:<feature>`). `GET /governance/drift` + gauge `risk_drift_psi`.
   < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant; < 100 decisions = "insufficient data".
6. Every model is served with `n_jobs = MODEL_THREADS` (1): the forest was trained with `n_jobs=-1`.

## Results (real, 2026-10-31)
| | old Day-2 claim | Day 17 champion (test) | Day 17 shadow (test) |
|---|---|---|---|
| model | hist_gradient_boosting | random_forest `rf-b677801468` | logistic_regression `lr-2dc4403f6c` |
| threshold | 0.4237 (tuned on test) | 0.404 (validation) | 0.670 (validation) |
| precision / recall | 0.313 / 0.925 | 0.294 / 0.813 | 0.338 / 0.723 |
| AP | 0.457 | 0.407 | 0.468 |
| cost (3,000 test payments) | — | ₹305,314 | ₹380,197 |

- Selecting by AP (Days 2–16) would have picked logistic regression: validation cost ₹317,645 vs the forest's
  ₹265,274. **Disclosure**: the switch to cost-based selection was made after I had also looked at test-set costs
  of all candidates; the validation numbers alone give the same choice.
- The old Day-2 model scored on the NEW test set: recall 0.898, precision 0.283 (below the 0.3 floor), cost
  ₹223,575 — cheaper than the new champion. It was trained on 80% of a different sample (vs 60% now) and its
  threshold breaks the floor; the cost gap is a reason to look at training on train+validation after selection.

By hand (image `risk-engine:day17` + Redis, 300 mixed payments + scenarios): shadow final-action agreement 0.914
(the logistic regression always scores higher but its threshold is higher too: 19 rf-holds it would allow);
drift: amount 0.07 stable, avg_order_value 0.13 moderate, account_age_days 8.3 and amount_to_avg_ratio 8.3
significant (no history in the engine yet), hour 1.04 (my synthetic traffic only covered part of the day);
p50 46 ms / p95 68 ms per decision (Day 16: 21 / 33 ms - forest + shadow). PAN in a field → 422 naming the field;
0 full PANs in the container log (access log shows `[PAN ****1111]`); the `/audit/<id>` 404 still echoed the id —
fixed + tested after that run.

## Consequences
- The champion changed (hgb → rf) and with it the leave-one-out contributions; one Day-15 test asserted the old
  model's reason order and now checks the set instead.
- Latency roughly doubled (p50 21 → 46 ms): random forest (11-row batch for contributions) + shadow; capacity per
  process ~halves — Day 22.
- Shadow scoring costs ~5–20 ms per new decision inside the 300 ms budget (skipped when short of time).
- Drift counters grow with traffic (one hash per feature, ≤ 10 fields) and are per model version.
- `/v2` profiles start empty, so `account_age_days` drift is "significant" from day one (every customer is 0 days
  old to the engine) — expected until history accumulates or is backfilled.
- Promotion (shadow → champion) is still manual: retrain, compare, swap files. No registry.
