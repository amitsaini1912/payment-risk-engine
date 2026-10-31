"""
Day 17: PCI-aware inputs/outputs (PAN -> 422, masked audit + logs) and model governance (versions on every decision,
shadow model, PSI drift). Real Redis (Testcontainers) + the real model; fixtures in conftest.py.
"""

from __future__ import annotations

import json
import logging

import pytest

import audit
import risk_v2.budget as budget
import risk_v2.decision as decision_mod
from drift import level, psi, reference
from pci import contains_pan, luhn_ok, mask
from scorer import load_scorer
from v2_support import facts

AFTERNOON_IST = "2026-10-29T08:30:00+00:00"
VISA = "4111111111111111"
MASTERCARD = "5555555555554444"


def post(client, body):
    return client.post("/v2/assess", json=body)


# ---- PCI: no card numbers in, none out ----

@pytest.mark.parametrize("digits, ok", [(VISA, True), ("4111111111111112", False), ("378282246310005", True),
                                        ("5555555555554444", True), ("6011111111111117", True), ("1234", False)])
def test_luhn(digits, ok):
    assert luhn_ok(digits) is ok


@pytest.mark.parametrize("field, value", [
    ("customer_id", VISA),
    ("device_id", "dev-4111-1111-1111-1111"),         # dashes
    ("merchant_id", "M378282246310005"),                # Amex, glued to a prefix
    ("transaction_id", "pay_5555555555554444"),
    ("card_token", "tok_6011111111111117"),
])
def test_a_pan_in_any_field_is_422_and_never_echoed(v2_client, redis_client, field, value):
    body = facts(**{field: value})
    response = post(v2_client, body)
    assert response.status_code == 422
    assert "card number" in response.text and field in response.text
    digits = "".join(ch for ch in value if ch.isdigit())
    assert digits not in response.text.replace("-", "")       # neither the PAN...
    assert value not in response.text                          # ...nor the field as sent
    assert redis_client.dbsize() == 0                          # and nothing was recorded


# found in the Day 20a traces: billing's payment id f2fdd154-9879-3051-96c0-... was refused (422) as a "card number"
REAL_UUID = "f2fdd154-9879-3051-96c0-3a2147155a8d"


def test_uuids_are_never_card_numbers(v2_client):
    import uuid
    assert contains_pan("154-9879-3051-96") is True                                  # the run inside it passes Luhn
    assert contains_pan(REAL_UUID) is False
    assert not any(contains_pan(str(uuid.uuid4())) for _ in range(20_000))          # was 0.22%
    body = post(v2_client, facts(transaction_id=REAL_UUID, customer_id=REAL_UUID.upper()))
    assert body.status_code == 200
    assert mask(f"payment {REAL_UUID} card {VISA}") == f"payment {REAL_UUID} card [PAN ****1111]"


# found by the Day 22 load test: billing sent X-Device-Id "dev-<uuid>"; the dash before the UUID hid it from the
# UUID rule and "6047-1682-4885-9" inside it passed Luhn -> 422 -> the payment fell back to billing's rules
PREFIXED = "dev-f14f6047-1682-4885-9abc-0bf3b5c4e9be"


def test_uuids_with_a_prefix_or_suffix_are_not_card_numbers(v2_client):
    assert contains_pan("6047-1682-4885-9") is True                                  # the run inside it passes Luhn
    for value in (PREFIXED, f"android-{REAL_UUID}", f"{REAL_UUID}-v2", f"x-{REAL_UUID}-y"):
        assert contains_pan(value) is False, value
    assert post(v2_client, facts(device_id=PREFIXED)).status_code == 200


def test_a_card_number_joined_to_a_uuid_by_a_dash_is_still_found(v2_client):
    assert contains_pan(f"{VISA}-{REAL_UUID}") is True
    assert contains_pan(f"dev-{REAL_UUID}-{VISA}") is True
    assert post(v2_client, facts(device_id=f"{VISA}-{REAL_UUID}")).status_code == 422


# found by the Day 22 load test: card-vault's fingerprint of a random test card; "65357899197801886" inside it passes
# Luhn, so every payment on that card got 422 and fell back to billing's rules (0.44% of random fingerprints)
FINGERPRINT = "a6bc59a65357899197801886d9436ddd7491fefc61f4e366fdd0cd3e9160d383"


def test_card_fingerprints_are_never_card_numbers(v2_client):
    import os
    assert contains_pan("65357899197801886") is True
    assert contains_pan(FINGERPRINT) is False
    assert not any(contains_pan(os.urandom(32).hex()) for _ in range(20_000))        # was 0.44%
    assert post(v2_client, facts(card_fingerprint=FINGERPRINT)).status_code == 200
    assert mask(f"fp {FINGERPRINT}") == f"fp {FINGERPRINT}"


def test_only_a_whole_sha256_hex_token_is_skipped(v2_client):
    assert contains_pan("a" * 47 + VISA) is True                  # 63 chars: not a digest
    assert contains_pan(FINGERPRINT + "0") is True                # 65 chars: not a digest
    assert contains_pan(FINGERPRINT.upper()) is True              # card-vault writes lowercase
    assert contains_pan(f"{FINGERPRINT} {VISA}") is True          # a PAN next to it
    assert post(v2_client, facts(card_fingerprint=FINGERPRINT[:40] + "x" + FINGERPRINT[41:])).status_code == 422


def test_a_card_number_after_another_number_is_found_and_masked(v2_client):
    # before Day 22 only the whole digit run was checked: "12 4111111111111111" = 18 digits, fails Luhn -> missed
    for text in (f"order 12 {VISA}", f"order-12-{VISA}", f"{VISA} {MASTERCARD}", f"ref 383 4111 1111 1111 1111"):
        assert contains_pan(text) is True, text
        assert "1111111111" not in mask(text).replace(" ", "").replace("-", ""), mask(text)
    assert mask(f"{VISA} {MASTERCARD}") == "[PAN ****1111] [PAN ****4444]"
    assert post(v2_client, facts(device_id=f"dev-12-{VISA}")).status_code == 422


def test_a_card_number_next_to_a_uuid_is_still_found(v2_client):
    assert contains_pan(f"{REAL_UUID} {VISA}") is True
    assert post(v2_client, facts(device_id=f"{REAL_UUID}-{VISA}")).status_code == 422


def test_long_numbers_that_are_not_card_numbers_are_fine(v2_client):
    assert post(v2_client, facts(customer_id="cust-4111111111111112")).status_code == 200   # fails Luhn
    assert post(v2_client, facts(customer_id="cust-411111111111")).status_code == 200       # 12 digits


def test_v1_rejects_pans_too_without_echo(v2_client, sample_transaction):
    response = v2_client.post("/assess", json={**sample_transaction, "customer_id": VISA})
    assert response.status_code == 422 and VISA not in response.text


def test_no_422_echoes_the_sent_value(v2_client):
    response = post(v2_client, facts(currency="inr", card_type="platinum"))
    assert response.status_code == 422
    assert "platinum" not in response.text and '"input"' not in response.text


def test_audit_lookup_404_does_not_echo_a_pan(v2_client):
    response = v2_client.get(f"/audit/{VISA}")
    assert response.status_code == 404 and VISA not in response.text and "[PAN ****1111]" in response.text


def test_audit_masks_pans_and_emails(tmp_path):
    path = str(tmp_path / "a.jsonl")
    audit.write_entry({"transaction_id": "t1", "llm_raw_output": f"card {VISA} from john.doe@example.com",
                       "nested": [{"note": "4111 1111 1111 1111"}]}, path=path)
    raw = open(path).read()
    assert VISA not in raw and "4111 1111 1111 1111" not in raw and "john.doe" not in raw
    entry = json.loads(raw)
    assert entry["llm_raw_output"] == "card [PAN ****1111] from j***@example.com"
    assert entry["nested"][0]["note"] == "[PAN ****1111]"


def test_log_lines_are_masked_including_uvicorns_access_log(caplog):
    import api.main  # noqa: F401 - installs the masking record factory
    with caplog.at_level(logging.INFO):
        logging.getLogger("uvicorn.access").info('%s - "%s %s HTTP/%s" %d', "10.0.0.1", "GET",
                                                 f"/audit/{VISA}", "1.1", 404)
        logging.getLogger("risk_v2.anything").warning(f"odd value {VISA}")
    assert VISA not in caplog.text and caplog.text.count("[PAN ****1111]") == 2
    access = [r for r in caplog.records if r.name == "uvicorn.access"][0]
    assert isinstance(access.args, tuple) and access.args[2] == "/audit/[PAN ****1111]"   # formatter still works


def test_mask_leaves_normal_text_alone():
    assert mask("amount 150000 at 14:00, order 12345") == "amount 150000 at 14:00, order 12345"
    assert not contains_pan("2026-10-31T08:30:00+00:00")


# ---- versions on every decision ----

def test_every_v2_decision_names_model_rules_and_prompt_versions(v2_client):
    body = post(v2_client, facts(timestamp=AFTERNOON_IST)).json()
    assert body["model_version"] == load_scorer().model_version
    assert len(body["rules_version"]) == 12 and body["prompt_version"].startswith("v2-")
    entry = v2_client.get(f"/audit/{body['transaction_id']}").json()
    assert (entry["model_version"], entry["rules_version"], entry["prompt_version"]) == \
           (body["model_version"], body["rules_version"], body["prompt_version"])


def test_v1_verdicts_carry_versions_too(v2_client, sample_transaction):
    body = v2_client.post("/assess", json=sample_transaction).json()
    assert body["model_version"] == load_scorer().model_version and body["prompt_version"].startswith("v1-")


# ---- shadow model ----

class FakeShadow:
    model_version = "fake-shadow-1"
    threshold = 0.5

    def __init__(self, score=0.99, fail=False):
        self._score, self._fail = score, fail

    def score_batch(self, rows):
        if self._fail:
            raise RuntimeError("corrupt shadow")
        return [self._score] * len(rows)


@pytest.fixture
def shadow(monkeypatch):
    def use(fake):
        monkeypatch.setattr(decision_mod, "load_shadow", lambda: fake)
        monkeypatch.setattr("api.main.load_shadow", lambda: fake)
    return use


def test_the_real_shadow_model_scores_but_does_not_decide(v2_client):
    body = post(v2_client, facts(timestamp=AFTERNOON_IST)).json()
    s = body["shadow"]
    assert s["model_version"] != body["model_version"] and 0 <= s["score"] <= 1
    assert s["model_action"] in ("allow", "hold_for_review", "block")


def test_a_shadow_that_would_block_changes_nothing_and_is_reported(v2_client, shadow):
    shadow(FakeShadow(score=0.99))
    body = post(v2_client, facts(timestamp=AFTERNOON_IST)).json()
    assert body["action"] == "allow"                               # the champion decided
    assert body["shadow"]["final_action"] == "block"               # the shadow would have blocked
    post(v2_client, facts(timestamp=AFTERNOON_IST))
    report = v2_client.get("/governance/shadow").json()
    assert report["shadow"] == "fake-shadow-1" and report["n"] == 2
    assert report["final_action_matrix"]["allow"]["block"] == 2 and report["final_action_agreement"] == 0
    assert report["shadow_scores_higher_share"] == 1


def test_replays_are_not_counted_twice(v2_client, shadow):
    shadow(FakeShadow(score=0.1))
    body = facts(timestamp=AFTERNOON_IST)
    for _ in range(3):
        post(v2_client, body)
    assert v2_client.get("/governance/shadow").json()["n"] == 1


def test_a_broken_shadow_never_breaks_a_decision(v2_client, shadow):
    shadow(FakeShadow(fail=True))
    response = post(v2_client, facts(timestamp=AFTERNOON_IST))
    assert response.status_code == 200 and "shadow failed" in response.json()["shadow"]["skipped"]


def test_no_budget_left_skips_the_shadow(v2_client, shadow, monkeypatch):
    shadow(FakeShadow())
    monkeypatch.setattr(budget, "BUDGET_S", 0.0)
    body = post(v2_client, facts(timestamp=AFTERNOON_IST)).json()
    assert body["shadow"]["skipped"] == "no budget left"
    assert v2_client.get("/governance/shadow").json()["n"] == 0


def test_without_a_shadow_file_there_is_no_shadow(v2_client, shadow):
    shadow(None)
    assert post(v2_client, facts(timestamp=AFTERNOON_IST)).json()["shadow"] is None
    assert v2_client.get("/governance/shadow").json()["note"] == "no shadow model loaded"


# ---- PSI drift ----

def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    ref = reference(list(range(1000)))
    same = [100] * len(ref["expected"])                       # 10% per decile, like the reference
    assert psi(ref["expected"], same) < 0.01 and level(psi(ref["expected"], same)) == "stable"
    shifted = [1000] + [0] * (len(ref["expected"]) - 1)       # everything in the lowest bin
    assert level(psi(ref["expected"], shifted)) == "significant"


def test_drift_report_needs_enough_data_then_flags_brand_new_customers(v2_client):
    for i in range(20):
        post(v2_client, facts(customer_id=f"new-{i}", timestamp=AFTERNOON_IST))
    early = v2_client.get("/governance/drift").json()["features"]["account_age_days"]
    assert early["n"] == 20 and early["level"].startswith("insufficient")

    for i in range(20, 120):
        post(v2_client, facts(customer_id=f"new-{i}", timestamp=AFTERNOON_IST))
    report = v2_client.get("/governance/drift").json()
    age = report["features"]["account_age_days"]
    assert report["model_version"] == load_scorer().model_version and age["n"] == 120
    # the engine has no history yet: every customer is 0 days old, unlike the training data
    assert age["level"] == "significant" and age["actual"][0] == 1.0
    assert "risk_drift_psi{" in v2_client.get("/metrics/prometheus").text


def test_governance_endpoints_need_a_key(store):
    from fastapi.testclient import TestClient

    from api.main import app
    client = TestClient(app)
    assert client.get("/governance/shadow").status_code == 401
    assert client.get("/governance/drift").status_code == 401
