# Razorpay Risk Agent

**Razorpay AI Buildathon — AI Risk Manager track.**

A chargeback-risk system in three layers: a classical ML **risk scorer**, an LLM
**reasoning agent** that turns the score into an action and a plain-English
justification, and a deterministic **rule-based fallback** that takes over
whenever the LLM path fails. Every decision is written to an audit trail, and
the whole thing is driven from a live dashboard.

---

**Demo video:** https://drive.google.com/file/d/10b1Q4O3ro8oGWHSt1Qu3MaBtCnVIqIvw/view?usp=sharing

## The problem, and why it's real

The AI Risk Manager track asks for an agent that assesses transaction risk,
decides an action, and handles failure gracefully. Chargebacks are the sharp end
of that:

A card-not-present payment is approved in milliseconds, but for weeks afterwards
the cardholder can dispute it. When they do, the money is pulled back from the
merchant, a dispute fee is charged on top, and the payment gateway wears the
operational cost and the network scrutiny. For a PSP operating at Razorpay's
scale, basis points of chargeback rate are a large absolute number.

What makes it a genuine ML problem rather than a checkbox:

- **Severe class imbalance.** Real chargeback rates are ~2–8%, not 50/50. A
  model that predicts "never fraud" is already ~95% accurate and worthless.
- **Asymmetric costs.** A missed chargeback costs the transaction amount + fees;
  a false alarm costs a few minutes of analyst time. The decision threshold is a
  business trade-off, not a default.
- **Reviewers need reasons.** A fraud analyst acting on a flag needs to know
  *why* it fired — which is why the scorer is a tree model and the agent emits an
  explanation.
- **It has to stay up.** A risk check sits in the payment path. If the LLM is
  slow, down, or wrong, the transaction still needs a verdict *now*.

## Architecture

```
                       data/generate_data.py
                    (synthetic txns + planted
                       fraud patterns, ~15k)
                               │
                               ▼
                        model/train.py
             LogReg → RandomForest → HistGradientBoosting
             pick best by PR-AUC · cost-based threshold
                               │
                               ▼
                        model/model.pkl
                  (sklearn pipeline + threshold)
                               │
        ┌──────────────────────┼───────────────────────┐
        │           agent/reasoning_agent.assess()      │
        │                      │                        │
        │        1. score  ──► model.pkl                │
        │        2. ask LLM (Gemini) for                │
        │           {action, explanation, confidence}   │
        │           · hard ~8s timeout                  │
        │           · strict JSON-schema validation     │
        │                      │                        │
        │            success   │   any failure          │
        │              ▼       │       ▼                │
        │        LLM verdict   │   agent/fallback_rules │
        │                      │   (score bands +       │
        │                      │    hard rules,         │
        │                      │    no dependencies)    │
        │                      ▼                        │
        │              agent/audit.py  ──► logs/audit_trail.jsonl
        └──────────────────────┬───────────────────────┘
                               │
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
      api/main.py  (FastAPI)            dashboard/app.py  (Streamlit)
   POST /assess · GET /audit/{id}       submit · simulate failures ·
   GET /audit · /metrics · /health      live audit table · model metrics
```

## Quickstart

Runs end to end on a clean clone. Python 3.9+.

```bash
git clone <this-repo> && cd razorpay-risk-agent

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. generate the dataset (deterministic, ~15k rows, ~6% chargebacks)
python data/generate_data.py --rows 15000 --seed 42 --out data/transactions.csv

# 2. train the model + choose the cost-based threshold
python model/train.py                 # -> model/model.pkl, model/metrics.json
python model/threshold_analysis.py    # -> cost-vs-threshold report + plot

# 3. run the tests
pytest -q                             # 36 tests, no skips once model.pkl exists

# 4. (optional) enable the live LLM path
cp .env.example .env && echo 'GEMINI_API_KEY=your-key' >> .env

# 5. launch the demo (API on :8000, dashboard on :8501)
./run.sh
# then open http://localhost:8501
```

Without a `GEMINI_API_KEY` everything still works — the agent logs
`failure_mode: "llm_unavailable"` and uses the deterministic fallback.

## Repo layout

```
data/generate_data.py       synthetic transaction generator
data/transactions.csv       generated dataset (committed, regenerable)
model/features.py           shared feature engineering (train == inference)
model/train.py              model selection + cost-based threshold
model/threshold_analysis.py standalone cost-vs-threshold analysis
model/scorer.py             load model.pkl, validate + score one transaction
model/model.pkl             saved pipeline (gitignored — run train.py)
model/metrics.json          all metrics for every candidate + the selected model
agent/reasoning_agent.py    assess(): score → LLM → validate → fallback → log
agent/llm_client.py         Gemini wrapper, key-optional
agent/fallback_rules.py     deterministic score-band + hard-rule engine
agent/audit.py              append-only JSONL audit trail + readers
agent/failure_sim.py        named failure simulators (CLI + dashboard)
api/main.py                 FastAPI service
dashboard/app.py            Streamlit demo dashboard
logs/audit_trail.jsonl      decision log (gitignored — sample below)
tests/                      model, fallback-path, and API tests
run.sh                      start API + dashboard together
```

## Data

### Why synthetic

- **Control of signal.** The generator plants known fraud patterns, so we can
  verify the model learns real structure rather than an artefact.
- **No licensing / PII risk** from real payment data.
- **Tunable imbalance.** We target a realistic 2–8% chargeback rate (~6% here).

The label is never a coin flip. For each transaction the generator builds a
"true" chargeback probability = small base rate × pattern multipliers, with some
patterns imposing a probability floor, then draws the label from it. Run
`python data/generate_data.py` to print each pattern's prevalence and its
observed chargeback lift.

### Hand-crafted fraud patterns

| ID | Pattern | Intuition | Observed lift |
|----|---------|-----------|---------------|
| P1 | New account (≤30d) + high amount (≥₹15k) + odd hour | Stolen card, cash out fast | ~8× |
| P2 | Customer has prior chargebacks | Recidivism / compromised account | ~4× |
| P3 | ≥6 tiny transactions in the last hour | Card-number testing | ~5× |
| P4 | IP country ≠ billing country, or new device on a high-value buy | Account takeover | ~4× |
| P5 | Prepaid or international card + high amount | Anonymous / hard-to-recover instrument | ~8× |
| P6 | Amount ≫ customer's average order value | Spend spike after takeover | ~5× |
| P7 | Disposable email domain | Throwaway identity | ~2× |

### Regenerate

```bash
python data/generate_data.py --rows 15000 --seed 42 --out data/transactions.csv
```

Deterministic given `--seed`. Debug columns (`_true_prob`, `_patterns_fired`)
are dropped from the CSV by default so they cannot leak the label; pass
`--keep-debug-cols` to inspect them.

### Data dictionary (`data/transactions.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | str | Unique transaction identifier |
| `merchant_id` | str | Merchant identifier (~100 merchants) |
| `merchant_category` | str | electronics, fashion, travel, gaming, grocery, digital_goods, jewellery, food_delivery |
| `customer_id` | str | Customer identifier (~3.7k, attributes stable across their transactions) |
| `amount` | float | Transaction amount in INR |
| `payment_method` | str | card, upi, netbanking, wallet, emi |
| `card_type` | str | credit, debit, prepaid, or `na` for non-card methods |
| `is_international_card` | int | 1 if the card was issued outside India |
| `timestamp` | str | ISO 8601 transaction time (120-day window from 2025-05-01) |
| `hour` | int | Hour of day 0–23 (derived from timestamp) |
| `is_new_device` | int | 1 if this device has not been seen for this customer |
| `ip_address` | str | Transaction IP |
| `ip_country` | str | Country the IP geolocates to |
| `billing_country` | str | Customer's billing country |
| `ip_billing_country_mismatch` | int | 1 if `ip_country` ≠ `billing_country` |
| `account_age_days` | int | Age of the customer account at transaction time |
| `past_chargebacks` | int | Count of prior chargebacks by this customer |
| `past_orders` | int | Count of prior completed orders by this customer |
| `avg_order_value` | float | Customer's historical average order value (INR) |
| `txn_count_1h` | int | Transactions by this customer in the preceding 1 hour |
| `txn_count_24h` | int | Transactions by this customer in the preceding 24 hours |
| `email_is_disposable` | int | 1 if the signup email domain is a known throwaway provider |
| `is_chargeback` | int | **Label.** 1 if the transaction was later charged back |

## Model

Baseline-first: Logistic Regression → Random Forest → HistGradientBoosting, all
class-balanced. Selected by best average precision (PR-AUC) on a stratified 20%
held-out test set — **HistGradientBoosting**, AP ≈ 0.46, ROC-AUC ≈ 0.92.

Feature engineering lives in `model/features.py` and is imported by both
training and inference, so the transformation can't drift between them.

### Cost-based decision threshold

`model/threshold_analysis.py` sweeps every threshold and computes total expected
rupee cost:

| Mistake | Cost |
|---------|------|
| Missed chargeback (FN) | transaction amount + ₹1,500 dispute fee + ₹500 ops |
| False alarm (FP) | ₹400 manual review + friction |

The **unconstrained** cost minimum is threshold ≈ 0.27 (recall 0.95, precision
0.28). Manual-review capacity is finite and precision below ~30% wastes reviewer
time, so we impose a **0.30 precision floor**. The cost-minimising threshold
under that constraint is **≈ 0.42**, costing only ~8% more than the unconstrained
optimum. That is the number we ship — and it's a defensible sentence, not a
guess.

### Metrics at the shipped threshold (≈ 0.42)

| Metric | Value |
|--------|-------|
| Recall | **≈ 0.92** (acceptance bar ≥ 0.75) |
| Precision | ≈ 0.31 |
| F1 | ≈ 0.47 |
| Confusion (test, n = 3000) | TP 184 · FP 404 · **FN 15** · TN 2397 |
| Expected cost | ≈ ₹88 / txn vs ≈ ₹936 / txn if we approve everything (−91%) |

Top drivers (permutation importance): `amount`, `historical_chargeback_rate`,
`amount_to_avg_ratio`, `txn_count_1h`, `past_chargebacks`,
`ip_billing_country_mismatch`. A tree model is used partly so a flag can be
explained by naming these inputs — a neural net would not give that, and this
dataset (15k rows, tabular) is exactly where trees win.

Full numbers: `model/metrics.json`. Cost curve: `model/threshold_cost_curve.png`.
(Exact figures move slightly if you regenerate the data with a different seed.)

## Reasoning agent

`agent/reasoning_agent.assess(transaction)` is the single entry point:

1. Score the transaction with the model.
2. Send the score + derived risk features to **Gemini** (`gemini-flash-lite`,
   ~1–2s) and ask for a structured verdict — `{action, explanation, confidence}`,
   `action ∈ {allow, hold_for_review, block}`. The LLM **decides**; it is not a
   chatbot. It falls through a list of model ids if one is deprecated/overloaded.
3. Enforce a hard ~8s wall-clock timeout (daemon thread — a hung call cannot
   block the decision path).
4. Validate the JSON against a strict schema (`action` enum, non-empty
   explanation, numeric confidence).
5. On **any** failure, fall back to `agent/fallback_rules.py` — score bands
   anchored on the cost-based threshold (review ≥ 0.42, block ≥ 0.80) plus hard
   rules (≥3 prior chargebacks → block, stolen-card signature → block,
   card-testing → review). No external dependencies, so it can't fail the way
   the LLM path can.
6. Log every decision to `logs/audit_trail.jsonl`.

`assess()` never raises.

### Enabling the LLM

```
cp .env.example .env       # then set GEMINI_API_KEY (or GOOGLE_API_KEY)
```

`.env` is gitignored. Config: `GEMINI_MODEL` (default `gemini-flash-lite-latest`),
`GEMINI_FALLBACK_MODELS` (comma-separated).

## API + dashboard

```bash
./run.sh                                  # API :8000 + dashboard :8501
# or separately:
uvicorn api.main:app --port 8000
streamlit run dashboard/app.py            # works stand-alone (in-process fallback)
```

**FastAPI** (`api/main.py`):

| Endpoint | Purpose |
|----------|---------|
| `POST /assess` | validate body → verdict → log. Bad body → `422`. |
| `POST /assess?simulate=<mode>` | force a failure mode through the real pipeline (demo) |
| `GET /audit/{transaction_id}` | the logged decision (`404` if none) |
| `GET /audit?limit=` | recent decisions |
| `GET /metrics` | Day-2 model metrics |
| `GET /health` | liveness + whether the LLM is configured |

**Streamlit dashboard** (`dashboard/app.py`) — the primary demo surface:

| Panel | What it shows |
|-------|---------------|
| Assess a transaction | Preset + editable form → live verdict, LLM explanation, score, path, latency |
| Simulate a failure | One button per failure mode — forced through the real pipeline, fallback verdict shown on screen, no crash |
| Audit trail | Live table of recent decisions (time, score, path, failure_mode, action) |
| Sidebar | Model recall / precision / F1 / threshold / confusion matrix / cost-per-txn |

## Failure handling

Three distinct failure classes — a dependency problem, a bad-output problem, and
a bad-input problem — each with a dedicated test in `tests/test_failure_modes.py`
and a one-click dashboard simulation.

| # | Failure | Detection | Response | `failure_mode` logged |
|---|---------|-----------|----------|-----------------------|
| 1 | LLM call times out / API unreachable / not configured | wall-clock timeout; connection/SDK exception | deterministic fallback | `llm_timeout` · `llm_error` · `llm_unavailable` |
| 2 | LLM returns malformed / non-JSON / off-schema output | JSON parse fails, or `action` not in enum, or confidence not numeric | deterministic fallback | `llm_malformed` |
| 3 | Missing / corrupt input field | `scorer` validation raises `ScoringError` before the LLM is called | fallback → conservative `hold_for_review` | `bad_input` |

Beyond these: a retired Gemini model id (falls through a model list), and an
audit-log write failure (caught, logged to stderr, never breaks the decision).

Run all simulations from the CLI: `python agent/failure_sim.py all`

## Audit trail

Every decision (LLM or fallback) is appended to `logs/audit_trail.jsonl`
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

Who reads it: a compliance reviewer, a support agent handling a disputed
payment, or an incident post-mortem.

## Testing

```bash
pytest -q
```

- `tests/test_model.py` — model loads, scores a known low- and high-risk
  transaction correctly, rejects bad input, feature engineering is pure, saved
  metrics meet the acceptance bar.
- `tests/test_failure_modes.py` — each of the 3 fallback paths triggers under
  simulated failure and produces a valid logged verdict; `assess()` never
  raises whatever the LLM does; the LLM happy path works via an injected fake.
- `tests/test_api.py` — normal request returns a valid verdict, 7 kinds of
  malformed request return `422`, audit round-trip, `simulate` forces fallback.

API tests run offline (the LLM is monkeypatched to unavailable) so the suite is
deterministic and needs no key.

## Scoped out (and what I'd do with more time)

- **No feature store / real-time aggregation.** `txn_count_1h` etc. are columns
  in the dataset; in production they'd come from a streaming aggregation keyed on
  customer/card. First thing I'd add — it's where most of the signal is.
- **Threshold is fixed at train time.** Real systems re-fit it as the fraud mix
  and review-team capacity change; I'd make it a monitored, re-tunable parameter.
- **The LLM is trusted within the schema.** It can't return an invalid action,
  but it *could* return a confident, well-formed, wrong explanation. I'd add a
  consistency check (does the explanation cite features that actually moved the
  score?) and sample LLM verdicts for human audit.
- **Single model, no drift monitoring.** I'd track score distribution and
  realised chargeback rate over time and alert on drift.
- **Fallback rules are hand-tuned.** They're deliberately simple; with more data
  I'd calibrate the bands and hard-rule thresholds rather than set them by hand.
