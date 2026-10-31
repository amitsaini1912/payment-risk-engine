"""
Contract tests for the feedback topics (Day 18 chargeback.created, Day 19b payment.outcome): events are produced by
scripts/produce_feedback.py - the same JSON billing-service sends - into a real Kafka (Testcontainers, apache/kafka
like billing's compose) and consumed by the real consumer thread.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

import pytest
from confluent_kafka import Consumer

from risk_v2.feedback import ChargebackEvent, FeedbackLoop, PaymentOutcomeEvent
from risk_v2.feedback_consumer import FeedbackConsumer
from scripts.produce_feedback import chargeback, outcome, produce
from v2_support import facts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(ROOT, "docs", "contracts", "chargeback.created.v1.schema.json")
OUTCOME_SCHEMA_PATH = os.path.join(ROOT, "docs", "contracts", "payment.outcome.v1.schema.json")
ZERO = {"stored": 0, "duplicate": 0, "unmatched": 0, "dead_lettered": 0, "retried": 0}
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
def outcome_topic(topic):
    return topic + "-outcome"


@pytest.fixture
def run_consumer(kafka_bootstrap, topic, outcome_topic):
    started = []

    def _start(loop, group="risk-engine-feedback-test", **kw):
        consumer = FeedbackConsumer(loop, kafka_bootstrap, group_id=f"{group}-{topic}", chargeback_topic=topic,
                                    outcome_topic=outcome_topic, **kw)
        consumer.start()
        started.append(consumer)
        return consumer
    yield _start
    for consumer in started:
        consumer.stop()


# ---- the contract itself: the JSON schema billing codes against == what the consumer accepts ----

@pytest.mark.parametrize("path, event_model", [(SCHEMA_PATH, ChargebackEvent),
                                               (OUTCOME_SCHEMA_PATH, PaymentOutcomeEvent)])
def test_the_schema_file_matches_the_consumer_model(path, event_model):
    with open(path) as fh:
        schema = json.load(fh)
    model = event_model.model_json_schema()
    assert schema["required"] == model["required"]
    assert set(schema["properties"]) == set(model["properties"])
    for name, spec in schema["properties"].items():
        ours = {k: v for k, v in model["properties"][name].items() if k != "title"}
        theirs = {k: v for k, v in spec.items() if k != "description"}
        if "const" in theirs or "enum" in theirs:
            ours.pop("type", None)
        assert theirs == ours, name
    assert schema["additionalProperties"] is True        # tolerant reader: new optional fields don't break us


def test_the_schema_examples_and_the_producer_script_follow_the_contract():
    for path, event_model in ((SCHEMA_PATH, ChargebackEvent), (OUTCOME_SCHEMA_PATH, PaymentOutcomeEvent)):
        with open(path) as fh:
            for example in json.load(fh)["examples"]:
                event_model.model_validate(example)
    event = ChargebackEvent.model_validate(json.loads(json.dumps(chargeback("pay-1"))))
    assert event.schema_version == 1 and event.created_at.tzinfo is not None
    for kind in ("captured", "review_approved", "review_rejected"):
        assert PaymentOutcomeEvent.model_validate(json.loads(json.dumps(outcome("pay-1", kind)))).outcome == kind


def test_what_billing_sends_is_accepted():
    """The exact JSON billing-service's PaymentEvents writes (its ChargebackIntegrationTest checks the same shape):
    Java Instant.toString() at millisecond precision, 'Z' for UTC."""
    ChargebackEvent.model_validate({"event_id": "5b0d7c1e-8a40-4c4e-9a8e-0e6a3f1d2c11", "transaction_id":
                                    "a220ae17-ac26-3387-874d-fc6d10c84ead", "amount_minor": 100000, "currency": "INR",
                                    "reason": "product_not_received", "created_at": "2026-10-31T05:38:02.376Z",
                                    "schema_version": 1})
    PaymentOutcomeEvent.model_validate({"event_id": "9c3e1f0a-5d2b-4a7e-8f61-2b9d0c4e7a15", "transaction_id":
                                        "a220ae17-ac26-3387-874d-fc6d10c84ead", "outcome": "review_approved",
                                        "amount_minor": 6000000, "currency": "INR",
                                        "occurred_at": "2026-10-31T05:38:02Z", "schema_version": 1})


# ---- through Kafka ----

def test_a_chargeback_through_kafka_labels_the_decision(v2_client, outcomes, redis_client, kafka_bootstrap, topic,
                                                       run_consumer):
    txn = v2_client.post("/v2/assess", json=facts(customer_id="cust-kafka")).json()["transaction_id"]
    event = chargeback(txn, reason="fraud")
    produce(kafka_bootstrap, [event, event, chargeback(txn, reason="duplicate"), chargeback("pay-other-system")],
            topic=topic)
    consumer = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: sum(consumer.counts[topic].values()) == 4)
    assert consumer.counts[topic] == {**ZERO, "stored": 2, "duplicate": 1, "unmatched": 1}

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
    wait_until(lambda: consumer.counts[topic]["dead_lettered"] == 3 and consumer.counts[topic]["unmatched"] == 1)
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
    wait_until(lambda: consumer.counts[topic]["stored"] == 1)
    assert consumer.counts[topic]["retried"] == 2 and consumer.last_error is None   # cleared once it worked
    assert redis_client.hget("risk:cust:cust-flaky:profile", "chargebacks") == "1"


def test_committed_events_are_not_read_again_after_a_restart(outcomes, redis_client, kafka_bootstrap, topic,
                                                             run_consumer):
    produce(kafka_bootstrap, [chargeback("pay-a"), chargeback("pay-b")], topic=topic)
    first = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: first.counts[topic]["unmatched"] == 2)
    first.stop()
    produce(kafka_bootstrap, [chargeback("pay-c")], topic=topic)
    second = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: second.counts[topic]["unmatched"] == 1)
    time.sleep(1.0)                                     # anything re-read would have been counted again by now
    assert second.counts[topic] == {**ZERO, "unmatched": 1}


def test_payment_outcomes_through_kafka(v2_client, outcomes, redis_client, kafka_bootstrap, topic, outcome_topic,
                                        run_consumer):
    held = v2_client.post("/v2/assess", json=facts()).json()["transaction_id"]
    captured_event = outcome(held, "captured")
    produce(kafka_bootstrap, [outcome(held, "review_approved"), captured_event, captured_event,
                              outcome("pay-billing-fallback", "captured"), {**outcome(held), "outcome": "settled"}],
            topic=outcome_topic)
    produce(kafka_bootstrap, [chargeback(held)], topic=topic)
    consumer = run_consumer(FeedbackLoop(outcomes, redis_client))
    wait_until(lambda: sum(consumer.counts[outcome_topic].values()) == 5 and consumer.counts[topic]["stored"] == 1)

    assert consumer.counts[outcome_topic] == {**ZERO, "stored": 2, "duplicate": 1, "unmatched": 1, "dead_lettered": 1}
    assert sorted(o["outcome"] for o in outcomes.outcomes(held)) == ["captured", "review_approved"]
    dead = read_topic(kafka_bootstrap, outcome_topic + ".DLT", 1)                     # its own DLT
    assert "outcome" in dict(dead[0].headers())["error"].decode()
    assert consumer.status()["topics"] == [topic, outcome_topic]


def test_topics_created_after_the_consumer_started_are_read(outcomes, redis_client, kafka_bootstrap, topic,
                                                            outcome_topic, run_consumer):
    """The usual order in compose: the engine starts before billing creates its topics. Found by hand on Day 19b -
    with librdkafka's 5-minute metadata refresh billing's first events sat unread."""
    consumer = run_consumer(FeedbackLoop(outcomes, redis_client), metadata_refresh_ms=2000)
    time.sleep(3)                                            # subscribed, topics don't exist yet
    produce(kafka_bootstrap, [chargeback("pay-late")], topic=topic)
    produce(kafka_bootstrap, [outcome("pay-late")], topic=outcome_topic)
    wait_until(lambda: consumer.counts[topic]["unmatched"] == 1 and consumer.counts[outcome_topic]["unmatched"] == 1,
               timeout=30)
