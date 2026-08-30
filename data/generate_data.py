"""
Synthetic transaction data generator for the Razorpay AI Risk Manager project.

Why synthetic (not a scraped / public dataset):
  * Full control over the signal we bury in the data, so we can prove the model
    actually learns known fraud patterns rather than memorising an artefact.
  * No licensing / PII risk from real payment data.
  * We can dial the class imbalance to a realistic 2-8% chargeback rate.

The generator does NOT flip a random label. It builds, for every transaction, a
"true" chargeback probability from a small base rate plus a set of HAND-CRAFTED
fraud patterns (see PATTERNS below), then draws the label from that probability.
That means there is real, learnable structure in the data.

Run:
    python data/generate_data.py --rows 8000 --seed 42 --out data/transactions.csv
"""

from __future__ import annotations

import argparse
import ipaddress
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_CHARGEBACK_PROB = 0.003          # a transaction with NO pattern (~0.3%)
TARGET_RATE_RANGE = (0.02, 0.08)      # acceptance bar for the overall rate

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "emi"]
PAYMENT_METHOD_WEIGHTS = [0.46, 0.30, 0.12, 0.08, 0.04]
CARD_TYPES = ["credit", "debit", "prepaid"]
CARD_TYPE_WEIGHTS = [0.45, 0.45, 0.10]

MERCHANT_CATEGORIES = [
    "electronics", "fashion", "travel", "gaming", "grocery",
    "digital_goods", "jewellery", "food_delivery",
]
# Some categories are structurally riskier (easy to resell / hard to recover).
CATEGORY_RISK = {
    "electronics": 1.6, "gaming": 1.8, "jewellery": 2.0, "digital_goods": 1.7,
    "travel": 1.3, "fashion": 1.0, "grocery": 0.6, "food_delivery": 0.6,
}

COUNTRIES = ["IN", "US", "GB", "AE", "SG", "NG", "RU", "CN"]
COUNTRY_WEIGHTS = [0.82, 0.05, 0.03, 0.03, 0.02, 0.02, 0.015, 0.015]


# ---------------------------------------------------------------------------
# Hand-crafted fraud patterns
# ---------------------------------------------------------------------------
# Each pattern is a function row -> (lift, floor):
#   lift  - multiplicative bump on the running chargeback probability
#   floor - a minimum probability this pattern imposes (0.0 = none)
# The floor is what makes strong patterns genuinely LEARNABLE: when a
# stolen-card signature fires, the transaction really is high-risk, not just
# "base rate times a number". These (plus the base rate and a little noise) are
# the ONLY things that drive the label.

def pattern_new_account_high_value_odd_hour(r):
    """P1: brand-new account buys something expensive in the middle of the night.
    Classic stolen-card 'cash out fast' behaviour."""
    new = r["account_age_days"] <= 30
    big = r["amount"] >= 15000
    odd = r["hour"] >= 23 or r["hour"] <= 5
    if new and big and odd:
        return 15.0, 0.80
    if new and big:
        return 5.0, 0.35
    return 1.0, 0.0


def pattern_repeat_chargeback_customer(r):
    """P2: this customer has disputed transactions before. Strong recidivism
    signal - both for genuine 'friendly fraud' and for compromised accounts."""
    if r["past_chargebacks"] >= 3:
        return 9.0, 0.65
    if r["past_chargebacks"] == 2:
        return 6.0, 0.45
    if r["past_chargebacks"] == 1:
        return 3.0, 0.18
    return 1.0, 0.0


def pattern_card_testing_velocity(r):
    """P3: many transactions from the same customer in the last hour, each small.
    Looks like an attacker testing a list of stolen card numbers."""
    if r["txn_count_1h"] >= 6 and r["amount"] <= 900:
        return 8.0, 0.65
    if r["txn_count_1h"] >= 4 and r["amount"] <= 1500:
        return 3.5, 0.28
    return 1.0, 0.0


def pattern_geo_mismatch_new_device(r):
    """P4: the IP country does not match the billing country, and/or the purchase
    is from a device we've never seen for this customer on a high-value buy."""
    lift, floor = 1.0, 0.0
    if r["ip_billing_country_mismatch"] == 1 and r["is_new_device"] == 1 and r["amount"] >= 5000:
        return 8.0, 0.65
    if r["ip_billing_country_mismatch"] == 1:
        lift *= 4.0
        floor = max(floor, 0.32)
    if r["is_new_device"] == 1 and r["amount"] >= 5000:
        lift *= 2.5
        floor = max(floor, 0.18)
    return lift, floor


def pattern_high_risk_instrument(r):
    """P5: prepaid or international card + high value. Prepaid cards are anonymous
    and international cards are slow/hard to recover from."""
    lift, floor = 1.0, 0.0
    if r["card_type"] == "prepaid" and r["amount"] >= 8000:
        lift *= 3.5
        floor = max(floor, 0.50)
    if r["is_international_card"] == 1 and r["amount"] >= 8000:
        lift *= 3.0
        floor = max(floor, 0.50)
    return lift, floor


def pattern_amount_spike_vs_history(r):
    """P6: the transaction is far larger than this customer's normal order value.
    Sudden spend spikes are a takeover tell."""
    ratio = r["amount"] / r["avg_order_value"] if r["avg_order_value"] > 0 else 1.0
    if ratio >= 12:
        return 4.0, 0.45
    if ratio >= 6:
        return 2.2, 0.16
    return 1.0, 0.0


def pattern_disposable_email(r):
    """P7: signed up with a throwaway email domain. On its own a weak signal;
    compounds sharply with a high-value order."""
    if r["email_is_disposable"] == 1 and r["amount"] >= 6000:
        return 3.5, 0.40
    if r["email_is_disposable"] == 1:
        return 2.2, 0.08
    return 1.0, 0.0


PATTERNS = [
    ("P1_new_account_high_value_odd_hour", pattern_new_account_high_value_odd_hour),
    ("P2_repeat_chargeback_customer", pattern_repeat_chargeback_customer),
    ("P3_card_testing_velocity", pattern_card_testing_velocity),
    ("P4_geo_mismatch_new_device", pattern_geo_mismatch_new_device),
    ("P5_high_risk_instrument", pattern_high_risk_instrument),
    ("P6_amount_spike_vs_history", pattern_amount_spike_vs_history),
    ("P7_disposable_email", pattern_disposable_email),
]


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def _random_ip(rng: random.Random) -> str:
    return str(ipaddress.IPv4Address(rng.randint(0x0B000000, 0xDF000000)))


def _make_customers(n_customers: int, rng: random.Random) -> dict:
    """Pre-build stable per-customer attributes so a customer looks consistent
    across their transactions."""
    customers = {}
    for i in range(n_customers):
        cid = f"C{100000 + i}"
        account_age = int(np.clip(rng.lognormvariate(4.8, 1.1), 1, 3000))
        # Most customers never charge back; a few are repeat offenders.
        past_cb = np.random.poisson(0.06)
        if rng.random() < 0.02:
            past_cb += rng.randint(1, 4)
        past_orders = int(np.clip(rng.lognormvariate(1.6, 1.0), 0, 400))
        avg_order_value = round(float(np.clip(rng.lognormvariate(7.4, 0.7), 100, 60000)), 2)
        customers[cid] = {
            "account_age_days": account_age,
            "past_chargebacks": int(past_cb),
            "past_orders": past_orders,
            "avg_order_value": avg_order_value,
            "email_is_disposable": 1 if rng.random() < 0.03 else 0,
            "home_country": np.random.choice(COUNTRIES, p=COUNTRY_WEIGHTS),
        }
    return customers


def _make_merchants(n_merchants: int, rng: random.Random) -> dict:
    merchants = {}
    for i in range(n_merchants):
        mid = f"M{1000 + i}"
        merchants[mid] = {"category": rng.choice(MERCHANT_CATEGORIES)}
    return merchants


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    n_customers = max(200, rows // 4)
    n_merchants = max(20, rows // 150)
    customers = _make_customers(n_customers, rng)
    merchants = _make_merchants(n_merchants, rng)
    customer_ids = list(customers)
    merchant_ids = list(merchants)

    start = datetime(2025, 5, 1)
    # Track per-customer recent-transaction timestamps for velocity features.
    recent: dict[str, list[datetime]] = {}

    # Deliberately inject card-testing bursts: ~2% of the volume is generated as
    # rapid clusters of tiny transactions from a single customer within minutes.
    # Without this, random timestamps almost never produce a high 1h velocity,
    # so pattern P3 would have nothing to learn from.
    burst_queue: list[tuple[str, datetime, float]] = []
    n_bursts = max(1, int(rows * 0.02 / 7))
    for _ in range(n_bursts):
        bcid = rng.choice(customer_ids)
        bstart = start + timedelta(seconds=rng.randint(0, 120 * 24 * 3600))
        for k in range(rng.randint(5, 9)):
            burst_queue.append((
                bcid,
                bstart + timedelta(seconds=k * rng.randint(20, 120)),
                round(rng.uniform(50, 700), 2),
            ))

    records = []
    for n in range(rows):
        forced = burst_queue.pop() if (burst_queue and rng.random() < 0.5) else None
        cid = forced[0] if forced else rng.choice(customer_ids)
        mid = rng.choice(merchant_ids)
        cust = customers[cid]
        merch = merchants[mid]

        ts = forced[1] if forced else start + timedelta(seconds=rng.randint(0, 120 * 24 * 3600))
        hour = ts.hour

        method = np.random.choice(PAYMENT_METHODS, p=PAYMENT_METHOD_WEIGHTS)
        card_type = np.random.choice(CARD_TYPES, p=CARD_TYPE_WEIGHTS) if method in ("card", "emi") else "na"
        is_intl_card = 1 if (card_type != "na" and rng.random() < 0.06) else 0

        # Amount: mostly near the customer's average, occasionally a big spike.
        base_amt = cust["avg_order_value"]
        if forced:
            amount = forced[2]
        elif rng.random() < 0.06:
            amount = base_amt * rng.uniform(4, 18)
        else:
            amount = base_amt * rng.uniform(0.2, 2.5)
        amount = round(float(np.clip(amount, 50, 500000)), 2)

        # Velocity: how many txns has this customer had in the last hour / 24h?
        hist = recent.setdefault(cid, [])
        txn_count_1h = sum(1 for t in hist if (ts - t).total_seconds() <= 3600)
        txn_count_24h = sum(1 for t in hist if 0 <= (ts - t).total_seconds() <= 86400)
        hist.append(ts)

        # Device / geo
        is_new_device = 1 if rng.random() < 0.18 else 0
        home = cust["home_country"]
        if rng.random() < 0.05:
            ip_country = np.random.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
        else:
            ip_country = home
        billing_country = home
        mismatch = 1 if ip_country != billing_country else 0

        row = {
            "transaction_id": f"TXN{2025000000 + n}",
            "merchant_id": mid,
            "merchant_category": merch["category"],
            "customer_id": cid,
            "amount": amount,
            "payment_method": method,
            "card_type": card_type,
            "is_international_card": is_intl_card,
            "timestamp": ts.isoformat(),
            "hour": hour,
            "is_new_device": is_new_device,
            "ip_address": _random_ip(rng),
            "ip_country": ip_country,
            "billing_country": billing_country,
            "ip_billing_country_mismatch": mismatch,
            "account_age_days": cust["account_age_days"],
            "past_chargebacks": cust["past_chargebacks"],
            "past_orders": cust["past_orders"],
            "avg_order_value": cust["avg_order_value"],
            "txn_count_1h": txn_count_1h,
            "txn_count_24h": txn_count_24h,
            "email_is_disposable": cust["email_is_disposable"],
        }

        # --- build the true chargeback probability from the patterns ---
        prob = BASE_CHARGEBACK_PROB * CATEGORY_RISK[merch["category"]]
        floor = 0.0
        fired = []
        for name, fn in PATTERNS:
            lift, patt_floor = fn(row)
            if lift > 1.0 or patt_floor > 0.0:
                fired.append(name)
            prob *= lift
            floor = max(floor, patt_floor)
        prob = max(prob, floor)
        # a little irreducible noise so the problem isn't perfectly separable
        prob *= rng.uniform(0.85, 1.15)
        prob = float(np.clip(prob, 0.0, 0.95))

        row["_true_prob"] = round(prob, 5)
        row["_patterns_fired"] = ";".join(fired)
        row["is_chargeback"] = 1 if rng.random() < prob else 0
        records.append(row)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(df: pd.DataFrame) -> None:
    rate = df["is_chargeback"].mean()
    print("=" * 70)
    print(f"rows                : {len(df):,}")
    print(f"unique customers    : {df['customer_id'].nunique():,}")
    print(f"unique merchants    : {df['merchant_id'].nunique():,}")
    print(f"chargeback count    : {int(df['is_chargeback'].sum()):,}")
    print(f"chargeback rate     : {rate:.2%}   (target {TARGET_RATE_RANGE[0]:.0%}-{TARGET_RATE_RANGE[1]:.0%})")
    lo, hi = TARGET_RATE_RANGE
    print(f"rate in target range: {lo <= rate <= hi}")
    print("-" * 70)
    print("Hand-crafted pattern prevalence and observed chargeback lift:")
    overall = rate
    for name, _ in PATTERNS:
        mask = df["_patterns_fired"].str.contains(name, regex=False)
        if mask.sum() == 0:
            print(f"  {name:38s}  n=0")
            continue
        patt_rate = df.loc[mask, "is_chargeback"].mean()
        lift = patt_rate / overall if overall else float("nan")
        print(f"  {name:38s}  n={mask.sum():5d}  cb_rate={patt_rate:6.2%}  lift x{lift:4.1f}")
    print("-" * 70)
    print("Class balance:")
    print(df["is_chargeback"].value_counts().to_string())
    print("=" * 70)
    print("Sample rows (mixed classes):")
    cols = [
        "transaction_id", "amount", "payment_method", "card_type",
        "account_age_days", "past_chargebacks", "txn_count_1h",
        "ip_billing_country_mismatch", "hour", "_patterns_fired", "is_chargeback",
    ]
    sample = pd.concat([
        df[df.is_chargeback == 1].head(5),
        df[df.is_chargeback == 0].head(5),
    ])[cols]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(sample.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/transactions.csv")
    ap.add_argument(
        "--keep-debug-cols", action="store_true",
        help="keep _true_prob / _patterns_fired columns in the CSV (default: drop, "
             "they would leak the label to the model)",
    )
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    report(df)

    out_df = df if args.keep_debug_cols else df.drop(columns=["_true_prob", "_patterns_fired"])
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {len(out_df):,} rows -> {args.out}"
          f"  ({'with' if args.keep_debug_cols else 'without'} debug columns)")


if __name__ == "__main__":
    main()
