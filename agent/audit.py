"""
Append-only audit trail.

Every risk decision - whether it came from the LLM or the deterministic fallback,
and whatever failure (if any) triggered the fallback - is written here as one
JSON object per line. This is what a compliance reviewer, a disputing customer's
support agent, or an incident post-mortem reads later.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from pci import scrub

_LOCK = threading.Lock()

DEFAULT_PATH = os.environ.get(
    "AUDIT_LOG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "logs", "audit_trail.jsonl"),
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(path: str | None) -> str:
    # resolved at call time, not def time: tests (and callers) that set DEFAULT_PATH must actually redirect
    # the writes - before Day 15 the test suite appended to the real logs/audit_trail.jsonl
    return path or DEFAULT_PATH


def write_entry(entry: dict, path: str | None = None) -> dict:
    """Append one entry. Never raises to the caller - a logging failure must not
    take down the decision path - but it does record that logging failed."""
    path = _path(path)
    # PAN-looking text and emails are masked before anything is written (Day 17): the audit trail must never
    # become a store of card data, whatever an LLM echoed or an error message contained
    record = scrub({"logged_at": _utcnow(), **entry})
    line = json.dumps(record, default=str)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] WARNING: could not write audit entry: {exc}")
    return record


def read_all(path: str | None = None) -> list[dict]:
    path = _path(path)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def get_by_transaction_id(transaction_id: str, path: str | None = None) -> dict | None:
    """Most recent entry for a transaction id, or None."""
    matches = [e for e in read_all(path) if e.get("transaction_id") == transaction_id]
    return matches[-1] if matches else None


def recent(n: int = 25, path: str | None = None) -> list[dict]:
    return read_all(path)[-n:][::-1]
