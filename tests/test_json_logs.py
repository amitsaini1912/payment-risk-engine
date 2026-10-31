"""Day 20b: JSON log lines (risk_v2/json_logs.py) - fields, the current trace id, masking, exceptions."""

from __future__ import annotations

import json
import logging

from opentelemetry import trace

from risk_v2 import json_logs

VISA = "4111111111111111"


def record(msg, *args, exc_info=None, name="risk_v2.test"):
    return logging.LogRecord(name, logging.WARNING, __file__, 1, msg, args, exc_info)


def test_fields_and_the_current_trace_id():
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("POST /v2/assess") as span:
        line = json.loads(json_logs.JsonFormatter().format(record("decided %s", "pay-1")))
    ctx = span.get_span_context()
    assert line["message"] == "decided pay-1" and line["log.level"] == "WARNING"
    assert line["log.logger"] == "risk_v2.test" and line["service.name"] == "risk-engine"
    assert line["trace_id"] == format(ctx.trace_id, "032x") and line["span_id"] == format(ctx.span_id, "016x")
    assert line["@timestamp"].endswith("+00:00")


def test_outside_a_span_there_is_no_trace_id():
    line = json.loads(json_logs.JsonFormatter().format(record("startup")))
    assert "trace_id" not in line


def test_card_numbers_are_masked_in_message_and_exception():
    try:
        raise ValueError(f"bad card {VISA}")
    except ValueError:
        import sys
        out = json_logs.JsonFormatter().format(record("card %s", VISA, exc_info=sys.exc_info()))
    assert VISA not in out
    line = json.loads(out)
    assert line["message"] == "card [PAN ****1111]" and line["error.type"] == "ValueError"
    assert "[PAN ****1111]" in line["error.stack_trace"]


def test_only_when_asked(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    assert json_logs.install() is False
    logger = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before = {h: h.formatter for name in json_logs.LOGGERS for h in logging.getLogger(name).handlers}
    try:
        assert json_logs.install("json") is True
        assert isinstance(handler.formatter, json_logs.JsonFormatter)
    finally:   # don't leave JSON formatters on pytest's own capture handlers
        logger.removeHandler(handler)
        for h, formatter in before.items():
            h.setFormatter(formatter)
        for h in list(root.handlers):
            if h not in before_handlers:
                root.removeHandler(h)
