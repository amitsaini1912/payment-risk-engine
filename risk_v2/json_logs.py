"""
JSON logs (Portfolio Day 20b, docs/adr/0008): LOG_FORMAT=json -> one JSON object per line on stdout, for the risk
engine's own loggers AND uvicorn's (error + access). Same field names as the Java services' ECS logs
(@timestamp, log.level, log.logger, message, service.name, error.*), plus trace_id / span_id of the current span -
the same ids Jaeger shows (Day 20a), so a log line leads to its trace.

PANs/emails are already masked by the log-record factory (agent/pci.py, Day 17); the formatter masks once more.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone

from opentelemetry import trace
from pci import mask

SERVICE = os.environ.get("OTEL_SERVICE_NAME", "risk-engine")
LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {
            "@timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "log.level": record.levelname,
            "log.logger": record.name,
            "service.name": SERVICE,
            "message": mask(record.getMessage()),
        }
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            line["trace_id"] = format(ctx.trace_id, "032x")
            line["span_id"] = format(ctx.span_id, "016x")
        if record.exc_info:
            line["error.type"] = record.exc_info[0].__name__
            line["error.message"] = mask(str(record.exc_info[1]))
            line["error.stack_trace"] = mask("".join(traceback.format_exception(*record.exc_info)))
        return json.dumps(line, ensure_ascii=False)


def install(fmt: str | None = None) -> bool:
    """Switch stdout logging to JSON when LOG_FORMAT=json. Uvicorn has set up its handlers before importing the app,
    so their formatters are replaced here; the root logger gets a stdout handler if it has none."""
    if (fmt or os.environ.get("LOG_FORMAT", "")).lower() != "json":
        return False
    formatter = JsonFormatter()
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
        root.setLevel(logging.INFO)
    for name in LOGGERS:
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(formatter)
    return True
