# ADR 0005 — Feedback loop: chargebacks as labels, precision/recall over time, retraining

- Status: Accepted
- Date: 2026-10-31
- Builds on: ADR 0001 (facts in, engine-owned profile), 0003 (stored decisions), 0004 (governance, manual promotion)

## Context
- The engine decides at payment time; whether it was right is only known when a chargeback arrives, days to
  months later (card networks allow ~120 days for most reasons). Nothing connected the two.
- `/v2` decisions live in Redis for 25 h (ADR 0003): far too short to attach a chargeback to.
- `past_chargebacks` was engine-owned since Day 14 but nothing ever increased it: always 0.
- billing-service becomes the producer of `chargeback.created` only on Day 19 (PLAN section 3 item 9), so the
  contract has to be fixed and tested before the real producer exists.

## Decision
1. **Contract**: `docs/contracts/chargeback.created.v1.schema.json` — `event_id`, `transaction_id` (billing's id
   sent to `/v2/assess`), `amount_minor` (integer), `currency`, `reason` (`^[a-z0-9_]+$`), `created_at` (ISO-8601
   with offset), `schema_version` = 1. JSON value, key = `transaction_id` (per-payment order). Unknown fields are
   ignored (adding an optional field is not breaking); a breaking change is `schema_version` 2. The pydantic model
   (`risk_v2/feedback.py`) must equal the schema file — a test compares them field by field.
2. **Durable decisions** (`risk_v2/feedback.py`, SQLite at `FEEDBACK_DB_PATH`): every NEW `/v2` decision is also
   written to `decisions` (action, score, threshold, versions, engine-clock time, the exact `model_input`); first
   decision wins like Redis. A write failure never changes or fails the decision
   (`risk_feedback_write_failures_total`), like the audit trail.
3. **Consumer** (`risk_v2/chargeback_consumer.py`, `confluent-kafka==2.6.1`), a thread in the API process when
   `KAFKA_ENABLED=true` (default off: the engine still runs alone):
   - at-least-once: offset committed only after the event is stored; `event_id` primary key → a redelivery is
     stored once;
   - SQLite/Redis error → no commit, seek back, retry after 2 s (later events of that partition wait);
   - bad event (not JSON, schema, `schema_version` ≠ 1, a PAN) → `chargeback.created.DLT` with the value masked
     and the error + source offset in headers, then committed (retrying can't fix it);
   - a chargeback for a payment with no `/v2` decision is kept and reported as `unmatched`.
4. **Label** = the payment has at least one chargeback. It also feeds the customer's Redis profile counter
   (`past_chargebacks`) **once per payment** (Lua: `SADD` the transaction id, `HINCRBY` only if new): redeliveries,
   retries after a half-done attempt and a second dispute on the same payment don't add again.
5. **`GET /feedback/performance?bucket=day|week&days=&model_version=`**: per decision-date bucket, final-action
   precision/recall (flagged = hold or block) and model-band precision/recall (score ≥ threshold), counts of
   held/blocked with chargebacks, model versions in the bucket, `mature` (bucket older than the label window,
   `FEEDBACK_LABEL_WINDOW_DAYS` = 120). `GET /feedback/{transaction_id}`: decision + chargebacks + label
   (1 / 0 only after the window / `null` = not known yet).
6. **`scripts/retrain.py`** writes a CANDIDATE (`model/candidate/`, gitignored), never the live model:
   - labeled rows = matured decisions with final action **allow** only;
   - the newest 20% (by decision time) is a holdout for champion vs candidate (`comparison.json`); with fewer than
     5 chargebacks in it → exit 2 before training;
   - training data = synthetic base set + the older labeled rows, through the unchanged `model/train.py`;
   - promotion stays manual (ADR 0004): candidate → `model/shadow.pkl` → `/governance/shadow` → champion.

## Why these choices
| Choice | Alternative | Why |
|---|---|---|
| SQLite file | Postgres; the JSONL audit trail | No new server for a stand-alone repo; real SQL for the joins. The audit trail has the data but a scan per chargeback is O(all decisions). |
| Consumer thread in the API process | Separate consumer process/container | It must write the same SQLite file the API writes; one image, one switch. |
| Train on `allow` only | All decisions | A blocked payment is never captured, so it never charges back. As "no chargeback" rows they would teach the model that what we block is safe, and each retrain would forget a bit more (selection bias / label censoring). |
| Keep the synthetic base set | Live labels only | Live labels only cover what we let through: without the base set the model loses the patterns we block. |
| Label window before "0" | Label 0 at once | A young payment without a chargeback is "not yet", not "fine". |
| Time-ordered holdout | Random split | The question is "is the new model better on the future?"; a random split leaks later behaviour into training. |

## Results (real, 2026-10-31)
- Tests: 172 → **210** passed (`pytest -q`, also in reverse file order). New: `tests/test_feedback.py` (27),
  `tests/test_chargeback_kafka.py` (6, real Kafka `apache/kafka:3.9.1` via Testcontainers + the producer script),
  `tests/test_retrain.py` (5). Break checks (8 mutations: commit on failure, no seek-back, counter not idempotent,
  blocked TPs dropped, blocks in training data, no DLT masking, holdout in training, write error not caught):
  each caught by a test.
- By hand (image `risk-engine:day18` + billing's compose Kafka/Redis, `KAFKA_ENABLED=true`): the consumer started
  before the topic existed (reports `UNKNOWN_TOPIC_OR_PART`, keeps running) and picked the topic up ~8 s after the
  first event. 300 payments (141 allow / 157 hold / 2 block) + 29 events from `scripts/produce_chargeback.py`:
  25 stored, 1 duplicate, 1 unmatched, 2 dead-lettered (PAN masked in the DLT, 0 PANs in the container log);
  the customer's next payment had `past_chargebacks` 1. `/feedback/performance`: final precision 0.031 / recall 0.2,
  `mature: false`. After a container restart: 301 decisions still there, nothing re-read.
- Found by hand and fixed: `retrain.py` trained and exited 0 with a holdout containing 0 chargebacks (all
  chargebacks were on older payments) → the comparison said nothing; now exit 2 before training. Also: the DLT
  error header had an empty field name for whole-event errors; `/health` kept showing a stale consumer error
  (all three fixed + tested after that run).

## Consequences
- Precision and recall are **lower bounds**: a blocked fraud is never captured, so it counts as a false positive
  and is missing from recall's true positives. Measuring them properly needs billing to report capture outcomes
  (Day 19 can add it) or a small random holdout that is allowed through — not done.
- SQLite is per container: with more than one replica, decisions would be split across files → Postgres (same
  SQL). One writer at a time; the decision path waits at most 0.25 s for the lock.
- Every new `/v2` decision now also writes one SQLite row (WAL, no fsync per commit).
- A chargeback that arrives before the decision is stored (or for a v1 payment) is kept as `unmatched` and never
  counted in the Redis profile.
- Live rows are a small share of the training data next to 15,000 synthetic rows; no re-weighting yet.
- Kafka offsets are per consumer group (`KAFKA_GROUP_ID`); a new group re-reads the topic from the start, which
  is safe (idempotent) but slow on a big topic.
