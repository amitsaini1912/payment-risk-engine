"""
/v2/assess decision pipeline (Day 15): rules.yaml -> model + reason contributions -> strictest action ->
explanation. Real Redis (Testcontainers), the real model, a private copy of rules.yaml (fixtures in conftest.py).
"""

from __future__ import annotations

import os

import pytest

import reasoning_agent
import risk_v2.decision as decision
from fallback_rules import decide as fallback_decide
from risk_v2.reason_codes import REASON_CODES
from scorer import load_scorer
from v2_support import facts

AFTERNOON_IST = "2026-10-29T08:30:00+00:00"    # 14:00 IST: not an odd hour
NIGHT_IST = "2026-10-28T21:30:00+00:00"        # 03:00 IST


def assess(client, **overrides) -> dict:
    response = client.post("/v2/assess", json=facts(**{"timestamp": AFTERNOON_IST, **overrides}))
    assert response.status_code == 200, response.text
    return response.json()


def write(path, text: str) -> None:
    with open(f"{path}.tmp", "w") as fh:
        fh.write(text)
    os.replace(f"{path}.tmp", path)


def known_customer(client, clock, customer="cust-known"):
    """Five normal orders over 50 days on one phone: a settled customer the model trusts."""
    for _ in range(5):
        assess(client, customer_id=customer, device_id="phone", amount_minor=150_000,
               card_fingerprint="fp" + "k" * 62)
        clock.advance(10 * 86_400)


# ---- decisions ----

def test_quiet_known_customer_is_allowed_with_rules_version_and_no_rule(v2_client, clock, rules_path):
    known_customer(v2_client, clock)
    body = assess(v2_client, customer_id="cust-known", device_id="phone", card_fingerprint="fp" + "k" * 62)

    assert body["action"] == "allow"
    assert body["rules_fired"] == []
    assert body["path_used"] == "rules+model" and body["model_score"] < body["threshold"]
    assert body["threshold"] == load_scorer().threshold and body["block_at"] == 0.80
    assert body["rules_version"] == v2_client.get("/health").json()["rules_version"]
    assert body["explanation_source"] == "template" and body["explanation"].startswith("ALLOW")


def test_card_testing_across_customers_is_caught_by_rules_the_model_cannot_see(v2_client, clock):
    """A fresh customer, token, device and IP each time; only the card fingerprint repeats. The model's velocity is
    per customer (training data), so it sees six first-time ₹500 payments - the card rules see the attack."""
    actions = []
    for i in range(6):
        body = assess(v2_client, amount_minor=50_000, customer_id=f"mule-{i}", card_token=f"tok_burst{i:04d}",
                      device_id=f"dev-{i}", ip=f"198.51.100.{i + 1}")
        actions.append(body["action"])
        clock.advance(60)

    assert actions == ["allow", "allow", "hold_for_review", "hold_for_review", "hold_for_review", "block"]
    assert body["rules_fired"] == ["card_testing", "card_velocity", "card_shared"]
    assert body["reason_codes"][:2] == ["R01_VELOCITY", "R10_CARD_SHARED"]
    assert body["model_score"] < body["threshold"]            # the model alone would have allowed it
    first = body["reasons"][0]
    assert first == {"code": "R01_VELOCITY", "source": "rule", "rule": "card_testing",
                     "detail": "card_txn_count_1h=5 >= 5 and amount_inr=500 <= 2000"}


def test_stolen_card_cash_out_is_blocked_by_the_rule_and_the_score_is_still_recorded(v2_client):
    body = assess(v2_client, amount_minor=4_500_000, timestamp=NIGHT_IST)   # new customer, ₹45,000, 03:00 IST
    assert body["action"] == "block"
    assert "stolen_card_cash_out" in body["rules_fired"]
    assert body["reason_codes"][:3] == ["R04_NEW_ACCOUNT", "R02_AMOUNT_ANOMALY", "R07_ODD_HOUR"]
    assert body["model_score"] is not None                    # scored anyway: logged for Day 17/18 analysis
    assert body["explanation"].startswith("BLOCK: rule stolen_card_cash_out [R04_NEW_ACCOUNT, R02_AMOUNT_ANOMALY, ")
    assert body["explanation"].count("amount_inr=45000 >= 15000") == 1       # evidence once, not once per code


def test_model_only_decision_names_the_features_that_drove_it(v2_client):
    # new customer, ₹7,000, throwaway email: no rule matches, the model holds it
    body = assess(v2_client, amount_minor=700_000, email_domain="mailinator.com")
    assert body["rules_fired"] == []
    assert body["action"] in ("hold_for_review", "block")
    assert body["reason_codes"][0] == "R08_DISPOSABLE_EMAIL"
    assert all(r["source"] == "model" for r in body["reasons"])
    contributions = [r["contribution"] for r in body["reasons"]]
    assert contributions == sorted(contributions, reverse=True) and contributions[0] >= 0.02
    assert len(body["reasons"]) <= 3


def test_model_velocity_is_reason_r01(v2_client, clock):
    for _ in range(7):                                        # one customer, small payments, 5 minutes apart
        body = assess(v2_client, amount_minor=50_000, customer_id="cust-fast")
        clock.advance(300)
    assert body["model_contributions"]["R01_VELOCITY"] == max(body["model_contributions"].values())
    assert body["action"] == "block"


@pytest.mark.parametrize("overrides", [
    {"amount_minor": 4_500_000, "timestamp": NIGHT_IST},
    {"amount_minor": 700_000, "email_domain": "mailinator.com"},
    {"amount_minor": 800_000, "ip_country": "SG", "billing_country": "IN"},
    {"amount_minor": 1_200_000, "card_type": "prepaid", "card_country": "US"},
    {"amount_minor": 6_000_000},
])
def test_every_hold_or_block_has_a_known_reason_code(v2_client, overrides):
    body = v2_client.post("/v2/assess", json=facts(**overrides)).json()
    if body["action"] != "allow":
        assert body["reason_codes"]
    assert all(code in REASON_CODES for code in body["reason_codes"])
    assert len(body["reason_codes"]) == len(set(body["reason_codes"]))


def test_threshold_without_a_dominant_feature_is_r90(v2_client, clock, monkeypatch):
    known_customer(v2_client, clock)                          # nothing unusual left to name
    monkeypatch.setattr(load_scorer(), "threshold", 0.0)      # ...but everything is >= the review threshold
    body = assess(v2_client, customer_id="cust-known", device_id="phone", card_fingerprint="fp" + "k" * 62)
    assert body["action"] == "hold_for_review" and body["rules_fired"] == []
    assert "R90_MODEL_SCORE" in body["reason_codes"]


def test_a_retry_gets_the_same_decision(v2_client):
    body = facts(timestamp=AFTERNOON_IST, amount_minor=50_000)
    first = v2_client.post("/v2/assess", json=body).json()
    again = v2_client.post("/v2/assess", json=body).json()
    assert again["signals"]["replayed"] is True
    for key in ("action", "reason_codes", "rules_fired", "model_score"):
        assert again[key] == first[key]


def test_decision_is_in_the_audit_trail_with_its_reasons(v2_client):
    body = assess(v2_client, amount_minor=4_500_000, timestamp=NIGHT_IST)
    entry = v2_client.get(f"/audit/{body['transaction_id']}").json()
    assert entry["api_version"] == "v2" and entry["action"] == "block"
    assert entry["reason_codes"] == body["reason_codes"] and entry["rules_version"] == body["rules_version"]


# ---- hot reload through the API ----

def test_editing_rules_yaml_changes_decisions_without_a_restart(v2_client, rules_path):
    before = assess(v2_client, customer_id="cust-hot-1")
    assert before["action"] == "allow"

    with open(rules_path) as fh:
        text = fh.read()
    write(rules_path, text + """
  - id: big_ticket_freeze
    description: incident - hold everything over INR 1,000 for now
    action: hold_for_review
    reasons: [R02_AMOUNT_ANOMALY]
    when: {field: amount_inr, op: ">=", value: 1000}
""")
    after = assess(v2_client, customer_id="cust-hot-2")
    assert after["action"] == "hold_for_review" and after["rules_fired"] == ["big_ticket_freeze"]
    assert after["rules_version"] != before["rules_version"]

    write(rules_path, text.replace("op: \">=\", value: 5}", "op: \">=\", value: five}"))   # a typo'd edit
    kept = assess(v2_client, customer_id="cust-hot-3")
    assert kept["rules_version"] == after["rules_version"]    # the last GOOD version is still running
    health = v2_client.get("/health").json()
    assert health["rules_version"] == after["rules_version"] and "numeric" in health["rules_error"]


def test_no_valid_rules_is_503_and_the_payment_is_not_counted(v2_client, rules_path, redis_client):
    write(rules_path, "rules: [")
    body = facts()
    response = v2_client.post("/v2/assess", json=body)
    assert response.status_code == 503 and "fallback" in response.json()["detail"]
    assert redis_client.exists(f"risk:txn:{body['transaction_id']}") == 0     # nothing recorded in Redis


# ---- failure paths ----

def test_model_unavailable_holds_with_r99_never_allows(v2_client, monkeypatch):
    def gone():
        raise FileNotFoundError("model.pkl not found")
    monkeypatch.setattr(decision, "load_scorer", gone)
    body = assess(v2_client, customer_id="cust-nomodel")
    assert body["action"] == "hold_for_review" and body["path_used"] == "rules_only"
    assert body["model_score"] is None and body["failure_mode"] == "model_unavailable"
    assert body["reason_codes"] == ["R99_NOT_SCORED"]


def test_the_llm_words_the_explanation_but_cannot_change_the_action(v2_client, explainer):
    prompts = []

    def persuasive_llm(prompt):
        prompts.append(prompt)
        return '{"explanation": "Customer looks fine.", "action": "allow"}'
    explainer["fn"] = persuasive_llm

    body = assess(v2_client, amount_minor=4_500_000, timestamp=NIGHT_IST)
    assert body["action"] == "block"                          # decided before the LLM was asked
    assert body["explanation"] == "Customer looks fine." and body["explanation_source"] == "llm"
    assert "Decision: block" in prompts[0] and "R04_NEW_ACCOUNT" in prompts[0]


@pytest.mark.parametrize("reply, failure", [("no json here", "llm_malformed"), ('{"explanation": ""}', "llm_malformed"),
                                            (None, "llm_malformed")])
def test_a_bad_llm_answer_falls_back_to_the_template(v2_client, explainer, reply, failure):
    explainer["fn"] = lambda _prompt: reply
    body = assess(v2_client, amount_minor=4_500_000, timestamp=NIGHT_IST)
    assert body["action"] == "block" and body["explanation_source"] == "template"
    assert body["failure_mode"] == failure and "stolen_card_cash_out" in body["explanation"]


def test_a_crash_while_explaining_still_returns_the_decision(v2_client, monkeypatch):
    def broken(*_args, **_kwargs):
        raise KeyError("contribution")                        # the real Day 15 bug: R90 had no contribution
    monkeypatch.setattr(decision, "explain", broken)
    body = assess(v2_client, amount_minor=4_500_000, timestamp=NIGHT_IST)
    assert body["action"] == "block" and body["failure_mode"] == "explanation_error"
    assert body["explanation"].startswith("BLOCK: R04_NEW_ACCOUNT")


# ---- weakness #19: one threshold, from the model bundle ----

def test_v1_fallback_uses_the_bundle_threshold(monkeypatch, high_risk_transaction, audit_path):
    scorer = load_scorer()
    monkeypatch.setattr(scorer, "threshold", 0.99)

    def no_llm(_prompt):
        raise reasoning_agent.LLMUnavailable("off")
    verdict = reasoning_agent.assess(high_risk_transaction, llm_fn=no_llm, audit_path=audit_path)
    assert verdict["threshold"] == 0.99
    assert any("review>=0.99" in r for r in verdict["rules_fired"])


def test_v1_fallback_refuses_a_score_without_a_threshold(sample_transaction):
    with pytest.raises(ValueError):
        fallback_decide(sample_transaction, 0.5, "test")
