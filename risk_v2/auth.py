"""
API keys for the risk engine (Day 16). Header `X-API-Key`.

RISK_API_KEYS="billing:<secret>,dashboard:<secret>"  - one key per caller, so the audit trail says who asked and a
leaked key can be rotated for one caller. Compared in constant time. No keys configured -> every protected call is
refused (503): the engine never runs open by accident.
"""

from __future__ import annotations

import hmac
import os
from typing import Optional


class AuthNotConfigured(RuntimeError):
    pass


def parse_keys(raw: Optional[str]) -> dict[str, str]:
    """'name:key,name2:key2' -> {key: name}. Raises ValueError on a malformed or too-short entry."""
    keys: dict[str, str] = {}
    for entry in filter(None, (e.strip() for e in (raw or "").split(","))):
        name, sep, key = entry.partition(":")
        if not sep or not name or len(key) < 16:
            raise ValueError(f"RISK_API_KEYS entry for {name or '?'!r} must be name:key with a key of 16+ chars")
        keys[key] = name
    return keys


class ApiKeys:
    def __init__(self, raw: Optional[str] = None):
        self._keys = parse_keys(os.environ.get("RISK_API_KEYS") if raw is None else raw)

    def caller(self, presented: Optional[str]) -> Optional[str]:
        """The caller's name for a valid key, else None."""
        if not self._keys:
            raise AuthNotConfigured("RISK_API_KEYS is not set")
        if not presented:
            return None
        match = None
        for key, name in self._keys.items():   # check every key: timing doesn't reveal which one matched
            if hmac.compare_digest(key.encode(), presented.encode()):
                match = name
        return match
