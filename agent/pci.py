"""
Keep card numbers (PANs) out of the risk engine (Portfolio Day 17). The engine only ever needs card-vault tokens and
fingerprints (Day 10), so a PAN arriving here is always a caller bug - and anything we store or log would put this
service in PCI DSS scope.

A PAN here = 13-19 digits (optionally separated by single spaces or dashes) that pass the Luhn check.
  - requests: any text field that contains one -> 422, and the error never repeats the value;
  - audit trail + logs: any PAN-looking text is replaced by "[PAN ****1234]" (defence in depth).
False positives: a long number that happens to pass Luhn (1 in 10), e.g. a 13-digit millisecond timestamp inside an
id; callers must not embed long numbers in ids.

UUIDs (8-4-4-4-12 hex) are never read as PANs (Day 20a): digit groups joined by dashes inside a UUID, e.g.
f2fdd154-9879-3051-96c0-... -> "154-9879-3051-96", passed Luhn for 0.22% of random UUIDs. Billing's payment and
billing account ids are UUIDs, so ~1 payment in 225 was refused (422) and fell back to billing's rules - found
in the Day 20a traces. Day 22 (k6 load test): also with a prefix - device id "dev-<uuid>" hit 1 payment in 100.
A PAN glued to other text (pay_5555..., M3782...) is still found.
"""

from __future__ import annotations

import re

_CANDIDATE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
# A dash before/after the UUID is fine (Day 22: device id "dev-<uuid>" was refused): only digit runs ENTIRELY inside
# the UUID are skipped, so a PAN joined to it by a dash ("<uuid>-4111...") starts outside and is still found.
_UUID = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}(?![0-9A-Fa-f])")
# Day 22 (k6 load test): card-vault's card fingerprint is HMAC-SHA256 hex (64 lowercase hex chars); 0.44% of them hold
# a Luhn-valid digit run (a6bc59a6535789919780188... -> 65357899197801886), and a fingerprint never changes, so that
# card's payments were ALWAYS refused (422) and decided by billing's fallback rules. A whole 64-hex token is a
# digest, never read as a PAN. Blind spot (accepted): a PAN hidden inside an exactly-64-lowercase-hex token.
_DIGEST = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{64}(?![0-9A-Za-z])")
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


_GROUP = re.compile(r"\d+")


def _pans(text: str):
    """(start, end, digits) of each PAN. Inside a run of digit groups ("order 12 4111 1111 1111 1111") every window of
    whole groups with 13-19 digits is tried, longest first: before Day 22 only the WHOLE run was checked, so a PAN
    after a short number ("12 4111111111111111") or two PANs in a row were never found or masked."""
    skip = [m.span() for m in _UUID.finditer(text)] + [m.span() for m in _DIGEST.finditer(text)]
    for run in _CANDIDATE.finditer(text):
        groups = [(run.start() + g.start(), run.start() + g.end()) for g in _GROUP.finditer(run.group(0))]
        i = 0
        while i < len(groups):
            for j in range(len(groups) - 1, i - 1, -1):
                start, end = groups[i][0], groups[j][1]
                digits = re.sub(r"[ -]", "", text[start:end])
                if not 13 <= len(digits) <= 19 or not luhn_ok(digits):
                    continue
                if any(s <= start and end <= e for s, e in skip):
                    continue   # digits inside a UUID or a SHA-256 hex digest
                yield start, end, digits
                i = j
                break
            i += 1


def contains_pan(text: str) -> bool:
    return any(True for _ in _pans(text))


def mask(text: str) -> str:
    """PANs -> "[PAN ****1234]", emails -> "a***@domain"."""
    if not isinstance(text, str):
        return text
    out, last = [], 0
    for start, end, digits in _pans(text):
        out += [text[last:start], f"[PAN ****{digits[-4:]}]"]
        last = end
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
