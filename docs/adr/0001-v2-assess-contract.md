# ADR 0001 — `/v2/assess`: the caller sends facts, the engine computes risk features (Redis)

- Status: Accepted
- Date: 2026-10-30
- Callers: billing-service (patient-management-microservices), from Day 19

## Context
`POST /assess` takes the model's features from the caller: `past_chargebacks`, `txn_count_1h`, `avg_order_value` …
A fraudster or a buggy client can send `past_chargebacks=0` and look safe (weakness #14). Amounts are float rupees,
while billing works in integer minor units with a currency (ADR 0001 of the Java repo). A real risk engine computes
its own history.

## Decision
1. **New endpoint, old one untouched**: `POST /v2/assess`. Extra fields → 422, so risk features *can't* be sent.
2. **Request = facts** (Pydantic v2):
   | Field | Type | Notes |
   |---|---|---|
   | `transaction_id` | string ≤ 64 | billing's payment id; the same id again = the same transaction |
   | `amount_minor` | **strict int** > 0 | paise/cents; a float or a string is 422 |
   | `currency` | `[A-Z]{3}` | INR, USD, EUR for scoring (others 422) |
   | `card_token`, `card_fingerprint`, `last4` | strings | from card-vault; never a PAN |
   | `customer_id`, `merchant_id`, `device_id` | ids | |
   | `ip` | IPv4/IPv6 | |
   | `timestamp` | ISO-8601 **with offset** | ≤ 5 min in the future |
   | optional: `merchant_category`, `card_type` (credit/debit/prepaid), `card_country`, `billing_country`, `ip_country`, `email_domain` | | context the engine can't know; safe defaults |
3. **Engine-owned features in Redis** (`risk_v2/velocity.lua`, one atomic script per request: read history BEFORE
   this transaction, then record it):
   - customer: transactions in the last 1h / 24h (sorted sets + `ZREMRANGEBYSCORE` + TTL) → the model's
     `txn_count_1h/24h`; devices seen (90 days) → `is_new_device`; profile hash: first seen → `account_age_days`,
     orders + amount sum → `past_orders`, `avg_order_value`; chargebacks → `past_chargebacks` (0 until Day 18).
   - card, keyed by **fingerprint** (not token — a new token per tokenization would reset it): 1h/24h counts,
     24h amount sum, distinct devices/IPs in 24h, new-card flag → returned as `signals` for the Day 15 rules.
   - windows use the engine's clock (arrival time); the caller's timestamp only gives the hour (IST).
   - the same `transaction_id` again returns the first snapshot and counts nothing twice.
4. **Money**: `amount_minor` → INR for the model at fixed scoring rates (USD 83, EUR 90, HALF_EVEN). Scoring only;
   billing's ledger uses its own locked rate.
5. **Response** = the v1 verdict (`action` ∈ `allow | hold_for_review | block`, `model_score`, `explanation`, …) +
   `api_version: "v2"`, `signals` (what the engine computed), `model_input` (the exact row the model scored).
   Billing maps the actions to its own APPROVE/HOLD/DECLINE on Day 19.
6. **Redis down or slow** (200 ms timeouts) → **503 "risk features unavailable, use fallback"**. We never score with
   invented "0 earlier payments" velocity — that's exactly what a fraudster wants. Billing's fallback rules +
   circuit breaker handle 503 (Day 19).

## Consequences
- New dependencies: `redis` (client), `testcontainers[redis]` (tests, real Redis 7 in Docker).
- The engine's history starts empty: until it has seen a customer, account age and orders are 0 (no backfill).
- `past_orders` counts orders *assessed*, not approved/settled (no outcome feedback yet: Day 18/19).
- Not cluster-safe as is: one script touches customer and card keys in different hash slots (fine on one Redis;
  Redis Cluster needs hash tags or two scripts).
- The training data's `txn_count_1h` is inflated by a generator bug (it also counts the customer's *later*
  transactions); `/v2` computes true prior counts, so the inputs differ from the training distribution until the
  Day 17 retrain fixes the generator.
- Replays recompute the verdict from the same features (deterministic on the fallback path); storing decisions per
  `transaction_id` is Day 16. No PAN check on the fields yet (Day 17).
