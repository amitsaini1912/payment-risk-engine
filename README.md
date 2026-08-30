# Razorpay Risk Agent

An AI Risk Manager for payment chargeback / fraud risk: a classical ML risk
scorer wrapped in an LLM reasoning agent that explains the verdict and picks an
action (`allow` / `hold_for_review` / `block`), with a deterministic rule-based
fallback whenever the LLM path fails.

> **Status:** in development (Buildathon). This README grows day by day; the full
> architecture diagram, metrics, failure-handling section and end-to-end run
> guide land on Day 5.

## Problem (AI Risk Manager track)

Card-not-present payments carry chargeback risk: a transaction is approved now,
but weeks later the cardholder disputes it and the merchant (and PSP) eats the
loss plus a dispute fee. A good risk manager has to do three things at once:
score the risk accurately on a heavily imbalanced signal (real chargeback rates
sit around 2-8%, not 50/50), decide an action against a real business cost
trade-off (a missed fraud costs far more than a manual review), and stay up even
when a dependency fails. This project builds all three.

## Repo layout

```
data/        synthetic transaction generator + generated CSV
model/       training, cost-based threshold analysis, saved model
agent/       LLM reasoning agent + deterministic fallback rules
api/         FastAPI service (POST /assess, GET /audit/{id})
dashboard/   Streamlit live demo dashboard
logs/        audit_trail.jsonl (gitignored; sample entry below)
tests/       model, fallback-path, and API tests
```

## Data

### Why synthetic

- **Control of signal.** The generator buries known fraud patterns in the data,
  so we can verify the model learns real structure rather than an artefact.
- **No licensing / PII risk** from real payment data.
- **Tunable imbalance.** We target a realistic 2-8% chargeback rate.

The generator never flips a random label. For each transaction it builds a
"true" chargeback probability from a small base rate times a set of hand-crafted
pattern multipliers, then draws the label from that probability.

### Hand-crafted fraud patterns

| ID | Pattern | Intuition |
|----|---------|-----------|
| P1 | New account (<=30d) + high amount (>=15k) + odd hour (23:00-05:00) | Stolen card, cash out fast |
| P2 | Customer has prior chargebacks | Recidivism / friendly fraud / compromised account |
| P3 | >=6 transactions in the last hour, each small | Card-number testing |
| P4 | IP country != billing country, or new device on a high-value buy | Account takeover |
| P5 | Prepaid or international card + high amount | Anonymous / hard-to-recover instrument |
| P6 | Amount >> customer's average order value | Spend-spike after takeover |
| P7 | Disposable email domain | Throwaway identity |

Run `python data/generate_data.py` to see each pattern's observed prevalence and
chargeback lift printed to the console.

### Regenerate the dataset

```bash
python data/generate_data.py --rows 15000 --seed 42 --out data/transactions.csv
```

Deterministic given `--seed`. Debug columns (`_true_prob`, `_patterns_fired`)
are dropped from the CSV by default so they cannot leak the label into the
model; pass `--keep-debug-cols` to inspect them.

### Data dictionary (`data/transactions.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | str | Unique transaction identifier |
| `merchant_id` | str | Merchant identifier (~100 merchants) |
| `merchant_category` | str | electronics, fashion, travel, gaming, grocery, digital_goods, jewellery, food_delivery |
| `customer_id` | str | Customer identifier (~3.7k customers, attributes stable across their transactions) |
| `amount` | float | Transaction amount in INR |
| `payment_method` | str | card, upi, netbanking, wallet, emi |
| `card_type` | str | credit, debit, prepaid, or `na` for non-card methods |
| `is_international_card` | int | 1 if the card was issued outside India |
| `timestamp` | str | ISO 8601 transaction time (90-day window from 2025-05-01) |
| `hour` | int | Hour of day 0-23 (derived from timestamp) |
| `is_new_device` | int | 1 if this device has not been seen for this customer |
| `ip_address` | str | Transaction IP |
| `ip_country` | str | Country the IP geolocates to |
| `billing_country` | str | Customer's billing country |
| `ip_billing_country_mismatch` | int | 1 if `ip_country` != `billing_country` |
| `account_age_days` | int | Age of the customer account at transaction time |
| `past_chargebacks` | int | Count of prior chargebacks by this customer |
| `past_orders` | int | Count of prior completed orders by this customer |
| `avg_order_value` | float | Customer's historical average order value (INR) |
| `txn_count_1h` | int | Transactions by this customer in the preceding 1 hour |
| `txn_count_24h` | int | Transactions by this customer in the preceding 24 hours |
| `email_is_disposable` | int | 1 if the signup email domain is a known throwaway provider |
| `is_chargeback` | int | **Label.** 1 if the transaction was later charged back |

## Model (Day 2)

Baseline-first: Logistic Regression → Random Forest → Histogram Gradient
Boosting, all with balanced class weights. The model with the best average
precision (area under the PR curve) is selected — **HistGradientBoosting**,
AP ≈ 0.46, ROC-AUC ≈ 0.92 on a stratified 20% held-out test set.

Regenerate:

```bash
python model/train.py                 # writes model/model.pkl + model/metrics.json
python model/threshold_analysis.py    # standalone cost-vs-threshold report + plot
```

### Cost-based decision threshold

The threshold is not eyeballed. `model/threshold_analysis.py` sweeps every
threshold and computes total expected rupee cost:

| Mistake | Cost |
|---------|------|
| Missed chargeback (FN) | transaction amount + ₹1,500 dispute fee + ₹500 ops |
| False alarm (FP) | ₹400 manual review + friction |

The **unconstrained** cost minimum is threshold ≈ 0.27 (recall 0.95, precision
0.28). Because manual-review capacity is finite and precision below ~30% wastes
reviewer time, we impose a **0.30 precision floor**. The cost-minimising
threshold under that constraint is **0.42**, costing only ~8% more than the
unconstrained optimum.

### Metrics at the shipped threshold (0.42)

| Metric | Value |
|--------|-------|
| Recall | **0.92** (acceptance bar ≥ 0.75) |
| Precision | 0.31 |
| F1 | 0.47 |
| Confusion (test, n=3000) | TP 184 · FP 404 · FN 15 · TN 2397 |
| Expected cost | ₹88 / txn vs ₹936 / txn if we approve everything (−91%) |

Top drivers (permutation importance): transaction `amount`,
`historical_chargeback_rate`, `amount_to_avg_ratio`, `txn_count_1h` (velocity),
`past_chargebacks`, `ip_billing_country_mismatch`. A tree model is chosen partly
so a flagged transaction can be explained by naming these inputs — a neural net
would not give that.

Full numbers: `model/metrics.json`. Cost curve: `model/threshold_cost_curve.png`.

## Reasoning agent (Day 3)

`agent/reasoning_agent.assess(transaction)` is the single entry point:

1. Score the transaction with the Day-2 model.
2. Send the score + derived risk features to **Gemini** (`gemini-flash-lite`,
   ~1-2s) and ask for a structured verdict — `{action, explanation, confidence}`
   where `action ∈ {allow, hold_for_review, block}`. The LLM **decides**; it is
   not a chatbot. Falls through a list of model ids if one is deprecated/overloaded.
3. Enforce a hard ~8s wall-clock timeout on that call (daemon thread — a hung
   call cannot block the decision).
4. Validate the LLM's JSON against a strict schema.
5. On **any** failure, fall back to `agent/fallback_rules.py` — a deterministic
   score-band + hard-rule engine with no external dependencies.
6. Log every decision to `logs/audit_trail.jsonl`, recording which failure (if
   any) triggered the fallback.

`assess()` never raises — a payments service must always return a verdict.

The agent is **key-optional**: with no `GEMINI_API_KEY` it logs
`failure_mode: "llm_unavailable"` and uses the fallback. To enable live LLM
reasoning, create `.env` (gitignored) in the repo root:

```
GEMINI_API_KEY=your-key-here
# optional: GEMINI_MODEL=gemini-flash-lite-latest
```

## Failure handling

Three distinct failure modes, each with a dedicated test in
`tests/test_failure_modes.py` and a one-click simulation on the dashboard
(`agent/failure_sim.py`):

| # | Failure | Detection | Response | `failure_mode` logged |
|---|---------|-----------|----------|-----------------------|
| 1 | LLM call times out / API unreachable | wall-clock timeout; connection exception | deterministic fallback | `llm_timeout`, `llm_error`, `llm_unavailable` |
| 2 | LLM returns malformed / non-JSON output | JSON parse fails, or schema/enum invalid | deterministic fallback | `llm_malformed` |
| 3 | Missing / corrupt input field | scorer validation (`ScoringError`) | fallback → conservative `hold_for_review`; LLM never attempted | `bad_input` |

Run the simulations: `python agent/failure_sim.py all`

## Audit trail

Every decision (LLM path or fallback) is appended to `logs/audit_trail.jsonl`
(gitignored). Sample entry — an LLM timeout that fell back to the rule engine:

```json
{
  "logged_at": "2026-08-30T16:38:47.246751+00:00",
  "transaction_id": "TXN-SIM-LLM_TIMEOUT",
  "model_score": 0.9396,
  "threshold": 0.4237,
  "path_used": "fallback",
  "failure_mode": "llm_timeout",
  "action": "block",
  "explanation": "Deterministic fallback (llm_timeout). Action: BLOCK. IP/billing country mismatch + new device on a high-value order; model score 0.940 -> band 'block' (review>=0.42, block>=0.8).",
  "confidence": 0.61,
  "rules_fired": ["IP/billing country mismatch + new device on a high-value order", "model score 0.940 -> band 'block'"],
  "feature_snapshot": { "amount": 12500, "historical_chargeback_rate": 0.25, "...": "..." },
  "error_detail": "no response within 2.0s"
}
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
