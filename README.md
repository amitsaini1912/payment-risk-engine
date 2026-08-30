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

## Audit trail

Every decision (LLM path or fallback) is appended to `logs/audit_trail.jsonl`.
The file is gitignored; a sample entry will be shown here from Day 3.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
