"""
Day 18 contract test for `chargeback.created` (PLAN section 3 item 9): billing only becomes the real producer on
Day 19, so the event is produced here by scripts/produce_chargeback.py - the same JSON billing must send - into a
real Kafka (Testcontainers, apache/kafka like billing's compose) and consumed by the real consumer thread.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

import pytest
from confluent_kafka import Consumer

from risk_v2.chargeback_consumer import ChargebackConsumer
from risk_v2.feedback import ChargebackEvent, FeedbackLoop
from scripts.produce_chargeback import chargeback, produce
from v2_support import facts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(ROOT, "docs", "contracts", "chargeback.created.v1.schema.json")
VISA = "4111111111111111"


def wait_until(check, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def read_topic(bootstrap: str, topic: str, n: int, timeout=30.0) -> list:
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": f"reader-{uuid.uuid4()}",
                  "auto.offset.reset": "earliest"})
    c.subscribe([topic])
    out, end = [], time.monotonic() + timeout
    while len(out) < n and time.monotonic() < end:
        msg = c.poll(0.5)
        if msg is not None and not msg.error():
            out.append(msg)
    c.close()
    return out


@pytest.fixture
def topic():
    return f"chargeback.created.test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def run_consumer(kafka_bootstrap, topic):
    started = []

    def _start(loop, group="risk-engine-feedback-test", **kw):
        consumer = ChargebackConsumer(loop, kafka_bootstrap, group_id=f"{group}-{topic}", topic=topic,
                                      dlt_topic=topic + ".DLT", **kw)
        consumer.start()
        started.append(consumer)
        return consumer
    yield _start
    for consumer in started:
        consumer.stop()


# ---- the contract itself: the JSON schema billing codes against == what the consumer accepts ----

def test_the_schema_file_matches_the_consumer_model():
    with open(SCHEMA_PATH) as fh:
        schema = json.load(fh)
    model = ChargebackEvent.model_json_schema()
    assert schema["required"] == model["required"]
    assert set(schema["properties"]) == set(model["properties"])
    for name, spec in schema["properties"].items():
        ours = {k: v for k, v in model["properties"][name].items() if k != "title"}
        theirs = {k: v for k, v in spec.items() if k != "description"}
        if "const" in theirs:
            ours.pop("type", None)
        assert theirs == ours, name
    assert schema["additionalProperties"] is True        # tolerant reader: new optional fields don't break us


def test_the_schema_examples_and_the_producer_script_follow_the_contract():
    with open(SCHEMA_PATH) as fh:
        for example in json.load(fh)["examples"]:
            ChargebackEvent.model_validate(example)
    event = ChargebackEvent.model_validate(json.loads(json.dumps(chargeback("pay-1"))))
    assert event.schema_version == 1 and event.created_at.tzinfo is not None


# ---- through Kafka ----

def test_a_chargeback_through_kafka_labels_the_decision(v2_client, outcomes, redis_client, kafka_bootstrap, topic,
                                                       run_consumer):
    txn = v2_client.post("/v2/assess", json=facts(customer_id="cust-kafka")).json()["transaction_id"]
    event = chargeback(txn, reason="fraud")
    produce(kafka_bootstrap, [event, event, chargeback(txn, reason="duplicate"), chargeback("pay-other-system")],
            topic=topic)
    consumer = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: sum(consumer.counts.values()) == 4)
    assert consumer.counts == {"stored": 2, "duplicate": 1, "unmatched": 1, "dead_lettered": 0, "retried": 0}

    body = v2_client.get(f"/feedback/{txn}").json()
    assert body["label"] == 1 and sorted(c["reason"] for c in body["chargebacks"]) == ["duplicate", "fraud"]
    assert redis_client.hget("risk:cust:cust-kafka:profile", "chargebacks") == "1"     # once per payment
    later = v2_client.post("/v2/assess", json=facts(customer_id="cust-kafka")).json()
    assert later["model_input"]["past_chargebacks"] == 1


def test_bad_events_go_to_the_dlt_masked_and_do_not_block_the_partition(outcomes, redis_client, kafka_bootstrap,
                                                                       topic, run_consumer):
    produce(kafka_bootstrap, [
        b"not json",
        {**chargeback("pay-1"), "schema_version": 2},
        {**chargeback("pay-2"), "reason": VISA},
        chargeback("pay-3"),                       # a good one after the bad ones
    ], topic=topic)
    consumer = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: consumer.counts["dead_lettered"] == 3 and consumer.counts["unmatched"] == 1)
    assert outcomes.chargebacks("pay-3") and not outcomes.chargebacks("pay-1")

    dead = read_topic(kafka_bootstrap, topic + ".DLT", 3)
    assert len(dead) == 3
    errors = [dict(m.headers())["error"].decode() for m in dead]
    assert "not JSON" in errors[0] and "schema_version" in errors[1] and errors[2].startswith("event: ")
    assert "card number" in errors[2]
    assert all(VISA not in m.value().decode() for m in dead)                  # no PAN copied into the DLT
    assert dict(dead[2].headers())["source.topic"].decode() == topic


def test_store_trouble_is_retried_not_skipped(v2_client, outcomes, redis_client, kafka_bootstrap, topic,
                                              run_consumer):
    txn = v2_client.post("/v2/assess", json=facts(customer_id="cust-flaky")).json()["transaction_id"]

    class Flaky(FeedbackLoop):
        failures = 2

        def apply(self, event):
            if self.failures:
                self.failures -= 1
                raise sqlite3.OperationalError("database is locked")
            return super().apply(event)

    produce(kafka_bootstrap, [chargeback(txn)], topic=topic)
    consumer = run_consumer(Flaky(outcomes, redis_client), retry_pause_s=0.1)
    wait_until(lambda: consumer.counts["stored"] == 1)
    assert consumer.counts["retried"] == 2 and consumer.last_error is None          # cleared once it worked
    assert redis_client.hget("risk:cust:cust-flaky:profile", "chargebacks") == "1"


def test_committed_events_are_not_read_again_after_a_restart(outcomes, redis_client, kafka_bootstrap, topic,
                                                             run_consumer):
    produce(kafka_bootstrap, [chargeback("pay-a"), chargeback("pay-b")], topic=topic)
    first = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: first.counts["unmatched"] == 2)
    first.stop()
    produce(kafka_bootstrap, [chargeback("pay-c")], topic=topic)
    second = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: second.counts["unmatched"] == 1)
    time.sleep(1.0)                                     # anything re-read would have been counted again by now
    assert second.counts == {"stored": 0, "duplicate": 0, "unmatched": 1, "dead_lettered": 0, "retried": 0}
