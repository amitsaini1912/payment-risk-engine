"""
Rules engine (Day 15): rules.yaml validation, evaluation, hot reload, "bad file -> keep the last good version".
No Redis, no model: pure functions over a context dict.
"""

from __future__ import annotations

import os

import pytest

from risk_v2.reason_codes import REASON_CODES
from risk_v2.rules import CONTEXT_FIELDS, RuleLoader, RulesInvalid, RulesUnavailable, parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ONE_RULE = b"""
rules:
  - id: card_velocity
    action: hold_for_review
    reasons: [R01_VELOCITY]
    when: {field: card_txn_count_1h, op: ">=", value: 3}
"""


def ctx(**overrides) -> dict:
    """A quiet, known customer: no rule in rules.yaml should fire."""
    base = {f: 0 for f in CONTEXT_FIELDS}
    base.update(amount_inr=1500.0, avg_order_value_inr=1500.0, amount_to_avg_ratio=1.0, currency="INR",
                card_type="credit", merchant_category="electronics", account_age_days=400.0, past_orders=12,
                hour=14, card_distinct_devices_24h=1, card_distinct_ips_24h=1)
    base.update(overrides)
    return base


def write(path, data: bytes) -> None:
    """Change the file the way the rules.yaml header asks: write a copy, then rename it over (atomic)."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


@pytest.fixture(scope="module")
def shipped():
    with open(os.path.join(ROOT, "rules.yaml"), "rb") as fh:
        return parse(fh.read())


# ---- the shipped rules.yaml ----

def test_shipped_rules_load_and_a_quiet_payment_fires_nothing(shipped):
    assert len(shipped.rules) == 9 and len(shipped.version) == 12
    assert shipped.policy.block_at == 0.80
    assert shipped.evaluate(ctx()) == []


@pytest.mark.parametrize("overrides, rule, action", [
    ({"card_txn_count_1h": 5, "amount_inr": 500.0}, "card_testing", "block"),
    ({"card_txn_count_1h": 3}, "card_velocity", "hold_for_review"),
    ({"customer_txn_count_1h": 6}, "customer_velocity", "hold_for_review"),
    ({"card_distinct_devices_24h": 3}, "card_shared", "hold_for_review"),
    ({"card_distinct_ips_24h": 4}, "card_shared", "hold_for_review"),
    ({"past_chargebacks": 3}, "repeat_chargebacks", "block"),
    ({"past_chargebacks": 2}, "chargebacks_review", "hold_for_review"),
    ({"account_age_days": 3.0, "amount_inr": 45000.0, "is_odd_hour": 1, "hour": 3}, "stolen_card_cash_out", "block"),
    ({"ip_billing_country_mismatch": 1, "is_new_device": 1, "amount_inr": 8000.0}, "geo_mismatch_new_device",
     "hold_for_review"),
    ({"card_is_new": 1, "amount_inr": 60000.0}, "new_card_large_amount", "hold_for_review"),
])
def test_each_shipped_rule_fires_on_its_pattern(shipped, overrides, rule, action):
    fired = {f.id: f for f in shipped.evaluate(ctx(**overrides))}
    assert rule in fired and fired[rule].action == action
    assert all(code in REASON_CODES for code in fired[rule].reasons)


def test_boundaries_are_exactly_as_written(shipped):
    assert "card_velocity" not in {f.id for f in shipped.evaluate(ctx(card_txn_count_1h=2))}
    # 5 earlier small payments = the 6th attempt -> block; a large one is "only" velocity
    assert "card_testing" not in {f.id for f in shipped.evaluate(ctx(card_txn_count_1h=5, amount_inr=2000.01))}
    stolen = dict(account_age_days=30.0, amount_inr=15000.0, is_odd_hour=1)
    assert "stolen_card_cash_out" in {f.id for f in shipped.evaluate(ctx(**stolen))}
    assert "stolen_card_cash_out" not in {f.id for f in shipped.evaluate(ctx(**{**stolen, "account_age_days": 30.5}))}


def test_a_fired_rule_says_which_values_made_it_fire(shipped):
    fired = {f.id: f for f in shipped.evaluate(ctx(card_distinct_devices_24h=4, card_distinct_ips_24h=1))}
    assert fired["card_shared"].detail == "card_distinct_devices_24h=4 >= 3"   # only the true branch of the `any`
    stolen = {f.id: f for f in shipped.evaluate(ctx(account_age_days=2.0, amount_inr=20000.0, is_odd_hour=1))}
    assert stolen["stolen_card_cash_out"].detail == \
        "account_age_days=2 <= 30 and amount_inr=20000 >= 15000 and is_odd_hour=1 == 1"


def test_text_fields_in_and_not_in_and_disabled_rules():
    book = parse(b"""
rules:
  - id: risky_category
    action: hold_for_review
    reasons: [R02_AMOUNT_ANOMALY]
    when:
      all:
        - {field: merchant_category, op: in, value: [jewellery, electronics]}
        - {field: currency, op: not_in, value: [INR]}
  - id: switched_off
    enabled: false
    action: block
    reasons: [R01_VELOCITY]
    when: {field: amount_inr, op: ">", value: 0}
""")
    assert [f.id for f in book.evaluate(ctx(merchant_category="jewellery", currency="USD"))] == ["risky_category"]
    assert book.evaluate(ctx(merchant_category="jewellery", currency="INR")) == []
    assert book.evaluate(ctx(merchant_category="grocery", currency="USD")) == []


# ---- a broken file is refused as a whole ----

@pytest.mark.parametrize("bad, why", [
    (b"rules: [", "not valid YAML"),
    (b"- just a list", "top level"),
    (ONE_RULE.replace(b"card_txn_count_1h", b"card_txn_count_1hr"), "unknown field"),        # typo in a field
    (ONE_RULE.replace(b'">="', b'"=>"'), "op"),                                              # typo in an operator
    (ONE_RULE.replace(b"R01_VELOCITY", b"R01_VELOCTY"), "unknown reason code"),
    (ONE_RULE.replace(b"hold_for_review", b"allow"), "action"),                              # rules only tighten
    (ONE_RULE.replace(b"value: 3", b'value: "3"'), "numeric"),                               # "3" >= 3 would raise
    (ONE_RULE.replace(b"op: \">=\", value: 3", b"op: in, value: 3"), "list"),
    (ONE_RULE.replace(b"reasons: [R01_VELOCITY]", b"reasons: []"), "reasons"),
    (ONE_RULE.replace(b"    when:", b"    priority: 1\n    when:"), "priority"),                 # unknown key
    (ONE_RULE + ONE_RULE.replace(b"rules:\n", b""), "duplicate rule id"),
    (b"model: {block_at: 1.5}\n" + ONE_RULE, "block_at"),
    (b"!!python/object/apply:builtins.len [[1, 2]]", "not valid YAML"),     # safe_load: a tag never runs code
])
def test_invalid_rule_files_are_rejected_with_a_reason(bad, why):
    with pytest.raises(RulesInvalid) as err:
        parse(bad)
    assert why in str(err.value)


# ---- hot reload ----

def test_reload_on_change_and_keep_the_last_good_version_on_a_bad_one(tmp_path):
    path = tmp_path / "rules.yaml"
    write(path, ONE_RULE)
    loader = RuleLoader(str(path))
    first = loader.get()
    assert loader.get() is first                              # unchanged file: not parsed again

    write(path, ONE_RULE.replace(b"value: 3", b"value: 10"))   # risk ops raise the limit, no restart
    second = loader.get()
    assert second.version != first.version
    assert second.evaluate(ctx(card_txn_count_1h=5)) == []

    write(path, b"rules: [")                                  # a broken edit
    assert loader.get() is second                             # still the last good rules
    assert "not valid YAML" in loader.last_error

    os.remove(path)                                           # file gone (bad deploy)
    assert loader.get() is second and "cannot read" in loader.last_error

    write(path, ONE_RULE)                                     # fixed
    assert loader.get().version == first.version and loader.last_error is None


def test_no_good_version_ever_is_unavailable_not_empty_rules(tmp_path):
    path = tmp_path / "rules.yaml"
    write(path, ONE_RULE.replace(b"R01_VELOCITY", b"NOPE"))
    loader = RuleLoader(str(path))
    with pytest.raises(RulesUnavailable):
        loader.get()                                          # never "no rules = allow everything"
    with pytest.raises(RulesUnavailable):
        RuleLoader(str(tmp_path / "missing.yaml")).get()


def test_a_bad_file_is_parsed_once_not_on_every_request(tmp_path, monkeypatch):
    import risk_v2.rules as rules_mod
    path = tmp_path / "rules.yaml"
    write(path, ONE_RULE)
    loader = RuleLoader(str(path))
    loader.get()
    write(path, b"rules: [")
    calls = []
    real_parse = rules_mod.parse
    monkeypatch.setattr(rules_mod, "parse", lambda raw: calls.append(1) or real_parse(raw))
    for _ in range(50):
        loader.get()
    assert len(calls) == 1



def test_a_typo_reports_one_error_at_its_place():
    with pytest.raises(RulesInvalid) as err:
        parse(ONE_RULE.replace(b"value: 3", b"value: three"))
    assert str(err.value).startswith("1 validation error") and "rules.0.when" in str(err.value)
