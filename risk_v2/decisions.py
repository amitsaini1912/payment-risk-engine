"""
Stored /v2 decisions: the same transaction_id always gets the same answer (Day 16, docs/adr/0003).

Day 14 already made the FEATURES idempotent (velocity.lua snapshot). This stores the whole DECISION, so a retry
also gets the same action, reason codes, rules_version and explanation - even if rules.yaml or the model changed
in between, or the LLM would word it differently.

- Kept with the request's hash: the same id with different facts is a client bug -> 409, never a silent reuse.
- SET NX: when two copies of one request race, the first stored decision wins and both get it.
- Same TTL as the feature snapshot (25 h): billing retries within minutes; outcomes for the feedback loop are
  kept elsewhere (Day 18).
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import redis

from risk_v2.store import FeaturesUnavailable

KEEP_S = 25 * 3600


def request_hash(facts) -> str:
    canonical = json.dumps(facts.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class DecisionConflict(ValueError):
    """The transaction_id was already decided for DIFFERENT facts."""


class DecisionStore:
    def __init__(self, client: redis.Redis):
        self._client = client

    @staticmethod
    def _key(transaction_id: str) -> str:
        return f"risk:decision:{transaction_id}"

    def get(self, transaction_id: str, req_hash: str) -> Optional[dict]:
        try:
            raw = self._client.get(self._key(transaction_id))
        except redis.RedisError as exc:
            raise FeaturesUnavailable(f"decision store unavailable: {exc}") from exc
        return self._check(raw, req_hash)

    def put_first(self, transaction_id: str, req_hash: str, decision: dict) -> dict:
        """Store the decision unless one exists; return the one that counts (ours, or the earlier racer's)."""
        record = json.dumps({"request_hash": req_hash, "decision": decision}, default=str)
        try:
            if self._client.set(self._key(transaction_id), record, nx=True, ex=KEEP_S):
                return decision
            raw = self._client.get(self._key(transaction_id))
        except redis.RedisError as exc:
            raise FeaturesUnavailable(f"decision store unavailable: {exc}") from exc
        return self._check(raw, req_hash) or decision

    @staticmethod
    def _check(raw, req_hash: str) -> Optional[dict]:
        if raw is None:
            return None
        record = json.loads(raw)
        if record["request_hash"] != req_hash:
            raise DecisionConflict("transaction_id was already assessed with different facts")
        return record["decision"]
