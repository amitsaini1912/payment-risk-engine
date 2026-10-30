"""
Keep card numbers (PANs) out of the risk engine (Portfolio Day 17). The engine only ever needs card-vault tokens and
fingerprints (Day 10), so a PAN arriving here is always a caller bug - and anything we store or log would put this
service in PCI DSS scope.

A PAN here = 13-19 digits (optionally separated by single spaces or dashes) that pass the Luhn check.
  - requests: any text field that contains one -> 422, and the error never repeats the value;
  - audit trail + logs: any PAN-looking text is replaced by "[PAN ****1234]" (defence in depth).
False positives: a long number that happens to pass Luhn (1 in 10), e.g. a 13-digit millisecond timestamp inside an
id. Ids from billing are UUIDs, so this is acceptable; callers must not embed long numbers in ids.
"""

from __future__ import annotations

import re

_CANDIDATE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _pans(text: str):
    for m in _CANDIDATE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            yield m, digits


def contains_pan(text: str) -> bool:
    return any(True for _ in _pans(text))


def mask(text: str) -> str:
    """PANs -> "[PAN ****1234]", emails -> "a***@domain"."""
    if not isinstance(text, str):
        return text
    out, last = [], 0
    for m, digits in _pans(text):
        out += [text[last:m.start()], f"[PAN ****{digits[-4:]}]"]
        last = m.end()
    text = "".join(out) + text[last:]
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


def scrub(obj):
    """mask() every string inside dicts/lists/tuples (keys too)."""
    if isinstance(obj, str):
        return mask(obj)
    if isinstance(obj, dict):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(scrub(v) for v in obj)
    return obj


def reject_pans(values: dict) -> None:
    """For a model_validator(mode="before"): raise if any text field holds a PAN. The message names the field only."""
    if not isinstance(values, dict):
        return
    for name, value in values.items():
        if isinstance(value, str) and contains_pan(value):
            raise ValueError(f"field '{name}' looks like a card number (PAN): send the card-vault token, "
                             f"never the card number")


_installed = False


def install_log_masking() -> None:
    """Mask PANs/emails in EVERY log record (all loggers, incl. uvicorn's access log), once per process.

    Done in the record factory, so it works whatever handlers/formatters are configured later. Arguments are masked
    one by one (not merged into the message): uvicorn's access formatter reads record.args as a tuple.
    """
    import logging
    global _installed
    if _installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if isinstance(record.msg, str):
            record.msg = mask(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(mask(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: mask(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
