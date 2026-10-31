"""
Day 21: property-based tests (hypothesis). Instead of a few hand-picked examples, hypothesis generates hundreds of
inputs - edge cases first - and shrinks any failure to the smallest one. Where possible the engine is compared with a
tiny pure-Python model of what it must do.

  - money: scoring FX conversion is integer, exact for INR, monotonic, at most half a paisa off;
  - PCI: any Luhn-valid 13-19 digit number is found in any surrounding text, UUIDs never are, masking removes it;
  - decisions: the final action is the strictest of the model band and every fired rule; a higher score is never
    a looser decision (real rules.yaml, model score pinned to the generated value);
  - velocity (real Redis): for ANY sequence of payments, retries and clock jumps, the engine's counters equal a
    simple list-based model - and a retry returns exactly the first snapshot;
  - feedback: any mix of duplicate / repeated chargeback events counts each payment once.
"""

from __future__ import annotations

import os
from decimal import ROUND_HALF_EVEN, Decimal
from unittest import mock

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from llm_client import LLMUnavailable
from pci import contains_pan, luhn_ok, mask
from risk_v2 import decision as decision_mod
from risk_v2.assembler import SCORING_FX_TO_INR, UnsupportedCurrency, amount_in_inr_minor, to_model_transaction
from risk_v2.contract import AssessV2Request
from risk_v2.feedback import ChargebackEvent, FeedbackLoop, OutcomeStore
from risk_v2.rules import RuleLoader
from risk_v2.store import FeatureStore, Signals
from v2_support import FakeClock, facts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOUR_MS, DAY_MS = 3_600_000, 86_400_000
STRICT = {"allow": 0, "hold_for_review": 1, "block": 2}
REDIS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


# ---- money ----

@given(st.integers(min_value=0, max_value=10**12), st.sampled_from(sorted(SCORING_FX_TO_INR)))
def test_fx_to_inr_is_an_integer_rounded_once(minor, currency):
    inr = amount_in_inr_minor(minor, currency)
    exact = Decimal(minor) * SCORING_FX_TO_INR[currency]
    assert isinstance(inr, int)
    assert abs(Decimal(inr) - exact) <= Decimal("0.5")
    assert inr == int(exact.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
    assert amount_in_inr_minor(minor + 1, currency) >= inr                    # monotonic
    if currency == "INR":
        assert inr == minor


@given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=3, max_size=3))
def test_unsupported_currencies_are_refused(currency):
    assume(currency not in SCORING_FX_TO_INR)
    with pytest.raises(UnsupportedCurrency):
        amount_in_inr_minor(100, currency)


# ---- PCI ----

@st.composite
def pans(draw):
    """A Luhn-valid 13-19 digit number: random digits + the check digit that makes it valid."""
    body = draw(st.text(alphabet="0123456789", min_size=12, max_size=18))
    for check in "0123456789":
        if luhn_ok(body + check):
            return body + check
    raise AssertionError("unreachable: one check digit always works")


NO_DIGITS = st.text(alphabet=st.characters(blacklist_categories=("Nd",), blacklist_characters="-"), max_size=20)


@given(pans(), NO_DIGITS, NO_DIGITS, st.sampled_from(["", " ", "-"]))
def test_any_card_number_in_any_text_is_found_and_masked(pan, before, after, separator):
    spaced = separator.join(pan[i:i + 4] for i in range(0, len(pan), 4)) if separator else pan
    text = before + spaced + after
    assert contains_pan(text)
    masked = mask(text)
    assert pan not in masked.replace(" ", "").replace("-", "") and f"****{pan[-4:]}" in masked
    assert not contains_pan(masked)


@given(st.uuids(), NO_DIGITS, st.sampled_from(["", "-"]))
def test_uuids_are_never_card_numbers(value, around, dash):
    text = f"{around}{dash}{value}{dash}{around}"      # Day 22: "dev-<uuid>" was missed (NO_DIGITS has no dash)
    assert not contains_pan(text) and not contains_pan(str(value).upper())
    assert str(value) in mask(text)                                            # masking leaves it alone


@given(st.binary(min_size=32, max_size=32))
def test_sha256_hex_digests_are_never_card_numbers(raw):
    # Day 22: card-vault fingerprints; 0.44% of them were read as a PAN
    assert not contains_pan(raw.hex()) and not contains_pan(f"fp {raw.hex()}.")


@given(st.text(alphabet="0123456789", max_size=12), NO_DIGITS)
def test_short_numbers_are_not_card_numbers(digits, around):
    assert not contains_pan(around + digits + around)


# ---- decisions: strictest action wins, higher score never looser ----

def _no_llm(_prompt):
    raise LLMUnavailable("disabled in tests")


signals_strategy = st.builds(
    Signals, replayed=st.just(False), observed_at_ms=st.just(1_790_000_000_000),
    customer_txn_count_1h=st.integers(0, 30), customer_txn_count_24h=st.integers(0, 80),
    customer_device_known=st.booleans(), customer_first_seen_ms=st.integers(1_700_000_000_000, 1_790_000_000_000),
    customer_prior_orders=st.integers(0, 200), customer_prior_amount_sum_minor=st.integers(0, 10**9),
    customer_chargebacks=st.integers(0, 4), card_txn_count_1h=st.integers(0, 30), card_txn_count_24h=st.integers(0, 80),
    card_amount_sum_24h_minor=st.integers(0, 10**9), card_distinct_devices_24h=st.integers(1, 8),
    card_distinct_ips_24h=st.integers(1, 8), card_is_new=st.booleans())


@settings(max_examples=150, deadline=None)
@given(signals_strategy, st.integers(100, 10_000_000), st.floats(0, 1), st.floats(0, 1))
def test_the_strictest_of_model_and_rules_wins_and_more_risk_is_never_looser(signals, amount_minor, s1, s2):
    book = RuleLoader(os.path.join(ROOT, "rules.yaml")).get()
    req = AssessV2Request.model_validate(facts(amount_minor=amount_minor))
    model_input = to_model_transaction(req, signals, amount_in_inr_minor(amount_minor, "INR"))
    results = []
    for score in sorted((s1, s2)):
        with mock.patch.object(decision_mod, "score_with_contributions", return_value=(score, {})), \
                mock.patch.object(decision_mod, "_shadow", return_value=None):
            results.append(decision_mod.decide(req, signals, model_input, book, llm_fn=_no_llm))
    rule_actions = {r.id: r.action for r in book.rules}
    for d in results:
        expected = max([d["model_action"], *(rule_actions[r] for r in d["rules_fired"])], key=STRICT.get)
        assert d["action"] == expected
        assert all(STRICT[d["action"]] >= STRICT[rule_actions[r]] for r in d["rules_fired"])
        assert d["action"] == "allow" or d["reason_codes"]                  # every hold/block says why
    low, high = results
    assert low["rules_fired"] == high["rules_fired"]                      # same facts, same rules
    assert STRICT[high["action"]] >= STRICT[low["action"]]
    assert STRICT[high["model_action"]] >= STRICT[low["model_action"]]


# ---- velocity in Redis vs a list-based model ----

ops = st.lists(st.tuples(
    st.sampled_from(["new", "new", "new", "retry"]),     # mostly new payments, some retries of an earlier one
    st.integers(0, 2), st.integers(0, 2), st.integers(0, 2),   # customer, card, device
    st.integers(1, 5_000_00),                              # amount (INR minor)
    st.sampled_from([0, 1, 60, 1_800, 3_599, 3_600, 3_601, 20_000, 86_399, 86_400, 90_000])),   # seconds to wait first
    min_size=1, max_size=25)


@REDIS
@given(ops)
# the window edges, always: an earlier payment exactly 1 h / 24 h before (random histories rarely hit them; a break
# check showed a 24 h off-by-one surviving 60 random examples)
@example([("new", 0, 0, 0, 100, 0), ("new", 0, 0, 0, 100, 3_600)])
@example([("new", 0, 0, 0, 100, 0), ("new", 0, 0, 0, 100, 86_400)])
@example([("new", 0, 0, 0, 100, 0), ("new", 0, 0, 0, 100, 86_399), ("new", 0, 0, 0, 100, 1)])
def test_velocity_counters_match_a_simple_model_for_any_history(redis_client, history):
    redis_client.flushdb()
    clock = FakeClock()
    store = FeatureStore(redis_client, clock=clock)
    seen = []          # model: (txn, customer, card, device, amount, at_ms) in order
    snapshots = {}
    for kind, cust, card, device, amount, wait in history:
        clock.advance(wait)
        now = int(clock() * 1000)
        if kind == "retry" and seen:
            txn = seen[len(seen) // 2][0]
            again = store.observe(transaction_id=txn, customer_id="", card_fingerprint="", device_id="", ip="0.0.0.0",
                                  amount_inr_minor=1)
            first = snapshots[txn]
            assert again.replayed and {**again.as_dict(), "replayed": False} == first.as_dict()   # same answer
            continue
        txn = f"t{len(seen)}"
        got = store.observe(transaction_id=txn, customer_id=f"c{cust}", card_fingerprint=f"fp{card}" + "x" * 20,
                            device_id=f"d{device}", ip="203.0.113.1", amount_inr_minor=amount)
        mine = [s for s in seen if s[1] == cust]
        cards = [s for s in seen if s[2] == card]
        within = lambda rows, window: [s for s in rows if s[5] > now - DAY_MS and s[5] >= now - window]  # noqa: E731
        assert not got.replayed
        assert got.customer_txn_count_1h == len(within(mine, HOUR_MS))
        assert got.customer_txn_count_24h == len(within(mine, DAY_MS))
        assert got.card_txn_count_1h == len(within(cards, HOUR_MS))
        assert got.card_txn_count_24h == len(within(cards, DAY_MS))
        assert got.card_amount_sum_24h_minor == sum(s[4] for s in within(cards, DAY_MS))
        assert got.customer_prior_orders == len(mine)
        assert got.customer_prior_amount_sum_minor == sum(s[4] for s in mine)
        assert got.customer_device_known == any(s[3] == device and s[5] > now - 90 * DAY_MS for s in mine)
        assert got.card_is_new == (not cards)
        seen.append((txn, cust, card, device, amount, now))
        snapshots[txn] = got


# ---- feedback: duplicates never count twice ----

@REDIS
@given(st.lists(st.tuples(st.integers(0, 5), st.integers(0, 3)), min_size=1, max_size=30))
def test_chargebacks_count_once_per_payment_whatever_is_redelivered(redis_client, tmp_path_factory, deliveries):
    redis_client.flushdb()
    outcomes = OutcomeStore(str(tmp_path_factory.mktemp("fb") / "feedback.db"))
    for txn in range(4):   # 4 payments of one customer
        outcomes.record_decision({"transaction_id": f"p{txn}", "action": "allow", "model_score": 0.1,
                                  "threshold": 0.4, "model_version": "m", "rules_version": "r",
                                  "signals": {"observed_at_ms": 1}, "model_input": {"customer_id": "cust", "amount": 1.0}})
    loop = FeedbackLoop(outcomes, redis_client)
    for event_no, txn in deliveries:          # event_no repeats = redeliveries; txn repeats = second disputes
        loop.apply(ChargebackEvent.model_validate({
            "event_id": f"e{event_no}-p{txn}", "transaction_id": f"p{txn}", "amount_minor": 100, "currency": "INR",
            "reason": "fraud", "created_at": "2026-10-31T10:00:00+00:00", "schema_version": 1}))
    disputed = {txn for _, txn in deliveries}
    assert int(redis_client.hget("risk:cust:cust:profile", "chargebacks")) == len(disputed)
    stored = outcomes._conn().execute("SELECT count(*) FROM chargebacks").fetchone()[0]
    assert stored == len(set(deliveries))
    for txn in range(4):
        assert bool(outcomes.chargebacks(f"p{txn}")) == (txn in disputed)

