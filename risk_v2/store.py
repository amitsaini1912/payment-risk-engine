"""
Velocity + profile signals for /v2/assess, kept in Redis by the engine itself (Day 14, docs/adr/0001).

`FeatureStore.observe()` runs velocity.lua: one atomic step that reads the history BEFORE this transaction and
then records it. The same transaction_id again returns the first snapshot and counts nothing twice.

Windows use the engine's clock (when the request arrives), not the caller's timestamp: a caller can't push a payment
out of the 1h window by back-dating it. The caller's timestamp is only used for the hour-of-day feature.

Card signals are keyed by the card FINGERPRINT, not the token: card-vault issues a new token per tokenization, so a
per-token count would reset every time the same card is tokenized again (same reasoning as billing, ADR 0015).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import redis

_LUA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "velocity.lua")

# A slow Redis must not stall a payment: fail fast (Day 12 lesson: the default waits forever).
REDIS_TIMEOUT_S = float(os.environ.get("REDIS_TIMEOUT_S", "0.2"))


class FeaturesUnavailable(RuntimeError):
    """Redis is down or too slow. The caller gets 503 and must use its own fallback (billing, Day 19)."""


@dataclass(frozen=True)
class Signals:
    replayed: bool
    observed_at_ms: int
    customer_txn_count_1h: int
    customer_txn_count_24h: int
    customer_device_known: bool
    customer_first_seen_ms: int
    customer_prior_orders: int
    customer_prior_amount_sum_minor: int
    customer_chargebacks: int
    card_txn_count_1h: int
    card_txn_count_24h: int
    card_amount_sum_24h_minor: int
    card_distinct_devices_24h: int
    card_distinct_ips_24h: int
    card_is_new: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class FeatureStore:
    def __init__(self, client: redis.Redis, clock: Callable[[], float] = time.time):
        self._client = client
        self._clock = clock
        with open(_LUA) as fh:
            self._script = client.register_script(fh.read())

    @property
    def client(self) -> redis.Redis:
        return self._client

    @classmethod
    def from_url(cls, url: str) -> "FeatureStore":
        return cls(redis.Redis.from_url(url, socket_timeout=REDIS_TIMEOUT_S,
                                        socket_connect_timeout=REDIS_TIMEOUT_S, decode_responses=True))

    def observe(self, *, transaction_id: str, customer_id: str, card_fingerprint: str, device_id: str,
                ip: str, amount_inr_minor: int) -> Signals:
        cust, card = f"risk:cust:{customer_id}", f"risk:card:{card_fingerprint}"
        keys = [f"risk:txn:{transaction_id}", f"{cust}:txns", f"{cust}:devices", f"{cust}:profile",
                f"{card}:txns", f"{card}:amounts", f"{card}:devices", f"{card}:ips", f"{card}:first_seen"]
        now_ms = int(self._clock() * 1000)
        try:
            replayed, snapshot = self._script(keys=keys, args=[now_ms, transaction_id, amount_inr_minor,
                                                               device_id, ip])
        except redis.RedisError as exc:
            raise FeaturesUnavailable(f"feature store unavailable: {exc}") from exc
        data = json.loads(snapshot)
        return Signals(
            replayed=bool(int(replayed)),
            observed_at_ms=int(data["observed_at_ms"]),
            customer_txn_count_1h=int(data["customer_txn_count_1h"]),
            customer_txn_count_24h=int(data["customer_txn_count_24h"]),
            customer_device_known=bool(data["customer_device_known"]),
            customer_first_seen_ms=int(data["customer_first_seen_ms"]),
            customer_prior_orders=int(data["customer_prior_orders"]),
            customer_prior_amount_sum_minor=int(data["customer_prior_amount_sum_minor"]),
            customer_chargebacks=int(data["customer_chargebacks"]),
            card_txn_count_1h=int(data["card_txn_count_1h"]),
            card_txn_count_24h=int(data["card_txn_count_24h"]),
            card_amount_sum_24h_minor=int(data["card_amount_sum_24h_minor"]),
            card_distinct_devices_24h=int(data["card_distinct_devices_24h"]),
            card_distinct_ips_24h=int(data["card_distinct_ips_24h"]),
            card_is_new=bool(data["card_is_new"]),
        )
