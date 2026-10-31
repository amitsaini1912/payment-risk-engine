"""
Send feedback events the way billing-service does (Day 19b): JSON values per docs/contracts/ (chargeback.created v1,
payment.outcome v1), key = transaction_id. Stood in for billing before Day 19b (PLAN section 3 item 9); still what
the contract tests (tests/test_chargeback_kafka.py) produce with, and handy by hand.

  python scripts/produce_feedback.py pay-123                          # one chargeback for payment pay-123
  python scripts/produce_feedback.py pay-123 --amount-minor 50000 --reason product_not_received
  python scripts/produce_feedback.py pay-123 --event-id evt-1          # run twice: stored once
  python scripts/produce_feedback.py pay-123 --outcome captured        # payment.outcome instead of a chargeback
  python scripts/produce_feedback.py --raw '{"oops": 1}'               # a bad event -> chargeback.created.DLT

  --bootstrap (default $KAFKA_BOOTSTRAP_SERVERS or localhost:9092)
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from confluent_kafka import Producer

TOPIC = "chargeback.created"
OUTCOME_TOPIC = "payment.outcome"


def chargeback(transaction_id: str, *, amount_minor: int = 150_000, currency: str = "INR", reason: str = "fraud",
               event_id: Optional[str] = None, created_at: Optional[datetime] = None) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "transaction_id": transaction_id,
        "amount_minor": amount_minor,
        "currency": currency,
        "reason": reason,
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "schema_version": 1,
    }


def outcome(transaction_id: str, kind: str = "captured", *, amount_minor: int = 150_000, currency: str = "INR",
            event_id: Optional[str] = None, occurred_at: Optional[datetime] = None) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "transaction_id": transaction_id,
        "outcome": kind,
        "amount_minor": amount_minor,
        "currency": currency,
        "occurred_at": (occurred_at or datetime.now(timezone.utc)).isoformat(),
        "schema_version": 1,
    }


def produce(bootstrap: str, messages: list, topic: str = TOPIC) -> None:
    """messages: event dicts (sent as JSON, keyed by transaction_id) or raw bytes (sent as they are, no key)."""
    producer = Producer({"bootstrap.servers": bootstrap, "acks": "all", "enable.idempotence": True})
    for message in messages:
        if isinstance(message, (bytes, bytearray)):
            producer.produce(topic, value=bytes(message))
        else:
            producer.produce(topic, key=str(message.get("transaction_id", "")).encode(),
                             value=json.dumps(message).encode())
    left = producer.flush(15)
    if left:
        raise RuntimeError(f"{left} message(s) not delivered to {bootstrap}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("transaction_id", nargs="?")
    p.add_argument("--amount-minor", type=int, default=150_000)
    p.add_argument("--currency", default="INR")
    p.add_argument("--reason", default="fraud")
    p.add_argument("--event-id")
    p.add_argument("--outcome", choices=["captured", "review_approved", "review_rejected"],
                   help="send a payment.outcome event instead of a chargeback")
    p.add_argument("--raw", help="send this text as the message value instead of a valid event")
    p.add_argument("--bootstrap", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    args = p.parse_args()
    if args.raw is not None:
        produce(args.bootstrap, [args.raw.encode()])
        print(f"sent raw message to {TOPIC}")
        return
    if not args.transaction_id:
        p.error("transaction_id is required (or --raw)")
    if args.outcome:
        event = outcome(args.transaction_id, args.outcome, amount_minor=args.amount_minor, currency=args.currency,
                        event_id=args.event_id)
        produce(args.bootstrap, [event], topic=OUTCOME_TOPIC)
        print(json.dumps(event))
        return
    event = chargeback(args.transaction_id, amount_minor=args.amount_minor, currency=args.currency,
                       reason=args.reason, event_id=args.event_id)
    produce(args.bootstrap, [event])
    print(json.dumps(event))


if __name__ == "__main__":
    main()
