"""
Day 20a: tracing (risk_v2/tracing.py, docs/adr/0007). An in-memory exporter (conftest.py) stands in for Jaeger.
billing-service sends a W3C traceparent with /v2/assess and with every Kafka event (its outbox keeps the request's
context): both must continue THAT trace here.
"""

from __future__ import annotations

import json
import time
import uuid

import audit
from opentelemetry.trace import SpanKind

from risk_v2 import tracing
from risk_v2.feedback import FeedbackLoop
from risk_v2.feedback_consumer import FeedbackConsumer
from scripts.produce_feedback import chargeback
from v2_support import facts

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
BILLING_SPAN = "00f067aa0ba902b7"
TRACEPARENT = f"00-{TRACE_ID}-{BILLING_SPAN}-01"


def by_name(spans):
    return {s.name: s for s in spans.get_finished_spans()}


def test_assess_continues_billings_trace(v2_client, spans):
    body = facts()
    response = v2_client.post("/v2/assess", json=body, headers={"traceparent": TRACEPARENT})
    assert response.status_code == 200

    named = by_name(spans)
    server = named["POST /v2/assess"]
    assert server.kind == SpanKind.SERVER
    assert format(server.context.trace_id, "032x") == TRACE_ID
    assert format(server.parent.span_id, "016x") == BILLING_SPAN                   # billing's span is the parent
    for name in ("risk.features", "risk.decide", "risk.store"):
        assert format(named[name].context.trace_id, "032x") == TRACE_ID, name
    assert named["risk.decide"].attributes["risk.action"] == response.json()["action"]
    entry = audit.get_by_transaction_id(body["transaction_id"])
    assert entry["trace_id"] == TRACE_ID                                           # the audit record points at it


def test_without_a_traceparent_a_new_trace_starts(v2_client, spans):
    v2_client.post("/v2/assess", json=facts())
    server = by_name(spans)["POST /v2/assess"]
    assert server.parent is None and server.context.is_valid


def test_probes_are_not_traced(v2_client, spans):
    for path in ("/health", "/ready", "/metrics/prometheus"):
        v2_client.get(path)
    assert not [s for s in spans.get_finished_spans() if s.kind == SpanKind.SERVER]


def test_no_endpoint_means_no_exporter(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert tracing.setup() is False


def test_a_consumed_event_continues_the_producers_trace(outcomes, redis_client, kafka_bootstrap, spans):
    from confluent_kafka import Producer

    topic = f"chargeback.created.trace-{uuid.uuid4().hex[:8]}"
    producer = Producer({"bootstrap.servers": kafka_bootstrap})
    producer.produce(topic, key=b"pay-traced", value=json.dumps(chargeback("pay-traced")).encode(),
                     headers={"traceparent": TRACEPARENT})
    producer.produce(topic, key=b"pay-untraced", value=json.dumps(chargeback("pay-untraced")).encode())
    producer.flush(15)
    consumer = FeedbackConsumer(FeedbackLoop(outcomes, redis_client), kafka_bootstrap,
                                group_id=f"trace-{topic}", chargeback_topic=topic, outcome_topic=topic + "-outcome",
                                metadata_refresh_ms=2000)
    consumer.start()
    try:
        end = time.monotonic() + 30
        while consumer.counts[topic]["unmatched"] < 2 and time.monotonic() < end:
            time.sleep(0.1)
    finally:
        consumer.stop()
    processed = [s for s in spans.get_finished_spans() if s.name == f"{topic} process"]
    assert len(processed) == 2
    traced, untraced = sorted(processed, key=lambda s: s.attributes["messaging.kafka.message.offset"])
    assert traced.kind == SpanKind.CONSUMER
    assert format(traced.context.trace_id, "032x") == TRACE_ID
    assert format(traced.parent.span_id, "016x") == BILLING_SPAN
    assert untraced.parent is None                                                  # no header: its own trace
    assert traced.attributes["messaging.destination.name"] == topic
