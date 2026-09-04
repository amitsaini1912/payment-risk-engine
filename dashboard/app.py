"""
Live demo dashboard for the Razorpay Risk Agent.

  streamlit run dashboard/app.py

It talks to the FastAPI service at $API_URL (default http://localhost:8000) when
that is reachable, and otherwise calls the agent in-process - so the dashboard
works whether or not the API is running. The status badge shows which mode it is
in.

Panels:
  * Assess a transaction  - form (with presets) -> live verdict + explanation
  * Simulate a failure    - one click forces each Day-3 failure mode
  * Audit trail           - live table of recent decisions
  * Model metrics         - precision / recall / threshold from Day 2
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
import requests
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("agent", "model"):
    sys.path.insert(0, os.path.join(_ROOT, sub))

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Razorpay Risk Agent", page_icon="🛡️", layout="wide")

# --------------------------------------------------------------------------
# Backend: API if reachable, else in-process
# --------------------------------------------------------------------------


@st.cache_data(ttl=5)
def _api_alive() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        if r.ok:
            return r.json()
    except requests.RequestException:
        return None
    return None


def _in_process():
    from failure_sim import SIMULATIONS
    from reasoning_agent import assess
    return assess, SIMULATIONS


def call_assess(payload: dict, simulate: str | None = None) -> dict:
    health = _api_alive()
    if health is not None:
        params = {"simulate": simulate} if simulate else None
        r = requests.post(f"{API_URL}/assess", json=payload, params=params, timeout=45)
        r.raise_for_status()
        return r.json()
    assess, sims = _in_process()
    if simulate:
        return sims[simulate](payload)
    return assess(payload)


def load_audit(limit: int = 30) -> list[dict]:
    health = _api_alive()
    if health is not None:
        try:
            return requests.get(f"{API_URL}/audit", params={"limit": limit},
                                timeout=3).json().get("entries", [])
        except requests.RequestException:
            pass
    from audit import recent
    return recent(limit)


def load_metrics() -> dict:
    health = _api_alive()
    if health is not None:
        try:
            return requests.get(f"{API_URL}/metrics", timeout=3).json()
        except requests.RequestException:
            pass
    path = os.path.join(_ROOT, "model", "metrics.json")
    return json.load(open(path)) if os.path.exists(path) else {}


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

PRESETS = {
    "Low risk – returning customer": dict(
        merchant_id="M1005", merchant_category="grocery", customer_id="C-REG-1",
        amount=780.0, payment_method="upi", card_type="na", is_international_card=0,
        hour=13, is_new_device=0, ip_billing_country_mismatch=0,
        account_age_days=900, past_chargebacks=0, past_orders=64,
        avg_order_value=850.0, txn_count_1h=0, txn_count_24h=1, email_is_disposable=0,
    ),
    "High risk – stolen-card signature": dict(
        merchant_id="M1002", merchant_category="jewellery", customer_id="C-NEW-9",
        amount=44000.0, payment_method="card", card_type="prepaid",
        is_international_card=1, hour=3, is_new_device=1,
        ip_billing_country_mismatch=1, account_age_days=4, past_chargebacks=0,
        past_orders=1, avg_order_value=1900.0, txn_count_1h=1, txn_count_24h=2,
        email_is_disposable=1,
    ),
    "Card-testing burst": dict(
        merchant_id="M1010", merchant_category="digital_goods", customer_id="C-BURST",
        amount=120.0, payment_method="card", card_type="credit",
        is_international_card=0, hour=4, is_new_device=1,
        ip_billing_country_mismatch=0, account_age_days=45, past_chargebacks=0,
        past_orders=2, avg_order_value=1500.0, txn_count_1h=8, txn_count_24h=15,
        email_is_disposable=0,
    ),
    "Repeat disputer": dict(
        merchant_id="M1004", merchant_category="electronics", customer_id="C-REPEAT",
        amount=6500.0, payment_method="card", card_type="credit",
        is_international_card=0, hour=16, is_new_device=0,
        ip_billing_country_mismatch=0, account_age_days=520, past_chargebacks=3,
        past_orders=9, avg_order_value=4000.0, txn_count_1h=0, txn_count_24h=1,
        email_is_disposable=0,
    ),
}

ACTION_STYLE = {
    "allow": ("✅", "#1a7f37"),
    "hold_for_review": ("🟡", "#9a6700"),
    "block": ("⛔", "#cf222e"),
}
FAILURE_LABEL = {
    None: "none", "llm_timeout": "LLM timed out", "llm_error": "LLM unreachable",
    "llm_unavailable": "LLM not configured", "llm_malformed": "LLM returned non-JSON",
    "bad_input": "corrupt input",
}


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

st.title("🛡️ Razorpay Risk Agent — live demo")

health = _api_alive()
col_a, col_b = st.columns([3, 2])
with col_a:
    if health:
        llm = "LLM: " + (health.get("llm_model") or "not configured")
        st.success(f"Backend: FastAPI at {API_URL}  ·  {llm}")
    else:
        st.warning("Backend: in-process (FastAPI not reachable — start it with "
                   "`uvicorn api.main:app --port 8000` for the full demo)")


def render_verdict(result: dict) -> None:
    action = result["action"]
    icon, colour = ACTION_STYLE.get(action, ("•", "#57606a"))
    path = result["path_used"]
    fm = result.get("failure_mode")

    st.markdown(
        f"<h2 style='color:{colour};margin-bottom:0'>{icon} {action.upper()}</h2>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    score = result.get("model_score")
    c1.metric("Model risk score", f"{score:.3f}" if score is not None else "—")
    c2.metric("Threshold", f"{result.get('threshold'):.2f}" if result.get("threshold") else "—")
    c3.metric("Decision path", "LLM agent" if path == "llm" else "Fallback rules")
    c4.metric("Confidence", f"{result.get('confidence', 0):.2f}")

    if path == "llm":
        st.info(f"**LLM explanation:** {result['explanation']}")
    else:
        st.error(
            f"**Fallback triggered — {FAILURE_LABEL.get(fm, fm)}.** "
            f"{result['explanation']}"
        )
    if result.get("rules_fired"):
        with st.expander("Rules fired"):
            for r in result["rules_fired"]:
                st.write("•", r)
    with st.expander("Feature snapshot"):
        st.json(result.get("feature_snapshot", {}))


tab_assess, tab_fail, tab_audit = st.tabs(
    ["Assess a transaction", "Simulate a failure", "Audit trail"]
)

with tab_assess:
    left, right = st.columns([2, 3])
    with left:
        preset_name = st.selectbox("Preset", list(PRESETS))
        p = PRESETS[preset_name]
        with st.form("txn"):
            amount = st.number_input("Amount (₹)", value=float(p["amount"]), min_value=1.0)
            cc1, cc2 = st.columns(2)
            payment_method = cc1.selectbox(
                "Payment method", ["card", "upi", "netbanking", "wallet", "emi"],
                index=["card", "upi", "netbanking", "wallet", "emi"].index(p["payment_method"]),
            )
            card_type = cc2.selectbox(
                "Card type", ["na", "credit", "debit", "prepaid"],
                index=["na", "credit", "debit", "prepaid"].index(p["card_type"]),
            )
            hour = st.slider("Hour of day", 0, 23, int(p["hour"]))
            account_age_days = st.number_input("Account age (days)", value=int(p["account_age_days"]), min_value=0)
            avg_order_value = st.number_input("Customer avg order value (₹)", value=float(p["avg_order_value"]), min_value=1.0)
            past_chargebacks = st.number_input("Past chargebacks", value=int(p["past_chargebacks"]), min_value=0)
            past_orders = st.number_input("Past orders", value=int(p["past_orders"]), min_value=0)
            txn_count_1h = st.number_input("Txns in last 1h", value=int(p["txn_count_1h"]), min_value=0)
            txn_count_24h = st.number_input("Txns in last 24h", value=int(p["txn_count_24h"]), min_value=0)
            fc1, fc2, fc3 = st.columns(3)
            is_new_device = 1 if fc1.checkbox("New device", bool(p["is_new_device"])) else 0
            ip_billing_country_mismatch = 1 if fc2.checkbox("Geo mismatch", bool(p["ip_billing_country_mismatch"])) else 0
            is_international_card = 1 if fc3.checkbox("Intl card", bool(p["is_international_card"])) else 0
            email_is_disposable = 1 if st.checkbox("Disposable email", bool(p["email_is_disposable"])) else 0
            submitted = st.form_submit_button("Assess transaction", type="primary")

        if submitted:
            payload = dict(
                merchant_id=p["merchant_id"], merchant_category=p["merchant_category"],
                customer_id=p["customer_id"], amount=amount,
                payment_method=payment_method, card_type=card_type,
                is_international_card=is_international_card, hour=hour,
                is_new_device=is_new_device,
                ip_billing_country_mismatch=ip_billing_country_mismatch,
                account_age_days=account_age_days, past_chargebacks=past_chargebacks,
                past_orders=past_orders, avg_order_value=avg_order_value,
                txn_count_1h=txn_count_1h, txn_count_24h=txn_count_24h,
                email_is_disposable=email_is_disposable,
            )
            t0 = time.time()
            with st.spinner("Scoring + asking the reasoning agent…"):
                try:
                    st.session_state["last"] = call_assess(payload)
                    st.session_state["last_latency"] = time.time() - t0
                except Exception as exc:  # noqa: BLE001
                    st.session_state["last"] = None
                    st.error(f"Request failed: {exc}")

    with right:
        if st.session_state.get("last"):
            render_verdict(st.session_state["last"])
            st.caption(f"Round-trip: {st.session_state.get('last_latency', 0):.2f}s  ·  "
                       f"logged to audit trail as "
                       f"`{st.session_state['last']['transaction_id']}`")
        else:
            st.info("Pick a preset, tweak the fields, and hit **Assess transaction**.")

with tab_fail:
    st.write(
        "Each button forces one real failure through the pipeline. The verdict "
        "still comes back — from the deterministic fallback — and is logged with "
        "the failure that caused it."
    )
    sim_labels = {
        "llm_timeout": "⏱️ LLM call times out",
        "llm_unreachable": "🔌 LLM API unreachable",
        "llm_malformed": "🧩 LLM returns non-JSON",
        "bad_input": "🚫 Corrupt input (amount = null)",
    }
    base = PRESETS["High risk – stolen-card signature"]
    cols = st.columns(len(sim_labels))
    for col, (mode, label) in zip(cols, sim_labels.items()):
        if col.button(label, width="stretch"):
            payload = dict(base, customer_id="C-SIM")
            t0 = time.time()
            with st.spinner(f"forcing {mode}…"):
                try:
                    st.session_state["last_sim"] = call_assess(payload, simulate=mode)
                    st.session_state["last_sim_latency"] = time.time() - t0
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{exc}")
                    st.session_state["last_sim"] = None

    if st.session_state.get("last_sim"):
        st.divider()
        render_verdict(st.session_state["last_sim"])
        st.caption(f"Round-trip: {st.session_state.get('last_sim_latency', 0):.2f}s "
                   f"(note: no crash, valid verdict, fallback path)")

with tab_audit:
    if st.button("↻ Refresh"):
        _api_alive.clear()
    entries = load_audit(30)
    if not entries:
        st.info("No decisions logged yet — assess a transaction or run a simulation.")
    else:
        df = pd.DataFrame(entries)
        show = pd.DataFrame({
            "time": pd.to_datetime(df.get("logged_at")).dt.strftime("%H:%M:%S"),
            "transaction_id": df.get("transaction_id"),
            "score": df.get("model_score"),
            "path": df.get("path_used"),
            "failure_mode": df.get("failure_mode").fillna("—") if "failure_mode" in df else "—",
            "action": df.get("action"),
            "confidence": df.get("confidence"),
        })
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption(f"{len(entries)} most recent decisions · "
                   f"{(show['path'] == 'fallback').sum()} via fallback")

# --------------------------------------------------------------------------
# Sidebar: model metrics
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Model metrics")
    m = load_metrics()
    sel = m.get("selected", {})
    if sel:
        st.metric("Recall", f"{sel.get('recall', 0):.2f}", help="acceptance bar ≥ 0.75")
        st.metric("Precision", f"{sel.get('precision', 0):.2f}")
        st.metric("F1", f"{sel.get('f1', 0):.2f}")
        st.metric("Decision threshold", f"{sel.get('threshold', 0):.2f}")
        cm = sel.get("confusion_matrix", {})
        if cm:
            st.write("**Confusion matrix (test set)**")
            st.table(pd.DataFrame(
                [[cm.get("tn"), cm.get("fp")], [cm.get("fn"), cm.get("tp")]],
                index=["actual: legit", "actual: chargeback"],
                columns=["pred: legit", "pred: flag"],
            ))
        cost = sel.get("cost", {})
        if cost:
            st.metric("Cost / txn at threshold",
                      f"₹{cost.get('cost_per_txn_inr', 0):.0f}",
                      help=f"vs ₹{cost.get('baseline_cost_approve_all_inr', 0) / max(1, m.get('dataset', {}).get('test_rows', 1)):.0f}/txn if we approve everything")
        st.caption(f"model: {m.get('selected_model', '—')}  ·  "
                   f"{m.get('dataset', {}).get('rows', '—')} rows, "
                   f"{m.get('dataset', {}).get('chargeback_rate', 0):.1%} chargebacks")
    else:
        st.info("Run `python model/train.py` to populate metrics.")
