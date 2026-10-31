"""
Distributed tracing (Portfolio Day 20a, docs/adr/0007): OpenTelemetry, W3C trace context.

  OTEL_EXPORTER_OTLP_ENDPOINT set (compose: http://jaeger:4318) -> spans exported over OTLP/HTTP
  unset                                                            -> the API stays a no-op (nothing exported)

What gets a span:
  - every request (FastAPI instrumentation; /health, /ready and /metrics are left out - probes would bury the rest);
    billing-service's traceparent header makes /v2/assess part of the payment's trace;
  - inside /v2/assess: features (Redis), decide (rules + model + explanation), store (Redis + SQLite);
  - every consumed feedback event, continuing the trace billing put in the Kafka headers (its outbox stores the
    request's context, billing ADR 0019): payment request -> outbox -> Kafka -> this consumer, one trace id.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from opentelemetry import propagate, trace
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer("risk-engine")
EXCLUDED_URLS = "health,ready,metrics"


def setup() -> bool:
    """Install the SDK + OTLP exporter when an endpoint is configured. Call once, at startup."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "risk-engine")}))
    # the exporter reads OTEL_EXPORTER_OTLP_ENDPOINT itself and appends /v1/traces; batching keeps it off the hot path
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return True


def instrument(app) -> None:
    """Server spans for every request. Uses the global provider lazily: works whether setup() ran before or not."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app, excluded_urls=EXCLUDED_URLS)


def current_trace_id() -> Optional[str]:
    """32-hex trace id of the current span, for the audit trail; None when nothing is traced."""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


@contextmanager
def consumer_span(msg) -> Iterator[trace.Span]:
    """A CONSUMER span for one Kafka message, child of the producer's context from its headers (if any)."""
    headers = {}
    for key, value in msg.headers() or []:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                continue
        headers[key] = value
    parent = propagate.extract(headers)
    with tracer.start_as_current_span(f"{msg.topic()} process", context=parent, kind=SpanKind.CONSUMER, attributes={
        "messaging.system": "kafka", "messaging.destination.name": msg.topic(),
        "messaging.kafka.destination.partition": msg.partition(), "messaging.kafka.message.offset": msg.offset(),
        "messaging.operation": "process",
    }) as span:
        yield span
