"""
Kafka consumer for `chargeback.created` (Portfolio Day 18, docs/adr/0005). Optional: KAFKA_ENABLED=true starts it
in a background thread of the API process (the same process writes the decisions it labels).

  KAFKA_BOOTSTRAP_SERVERS  default localhost:9092 (billing's compose: kafka:19092 from containers)
  KAFKA_GROUP_ID           default risk-engine-feedback

At-least-once: the offset is committed only after the event is stored (or dead-lettered). Storing is idempotent
(event_id primary key, profile counter once per transaction), so a redelivery changes nothing.

  bad event (not JSON, fails the schema, unknown schema_version, a PAN in it)
      -> `chargeback.created.DLT` (value masked, error in a header), then committed: retrying can't fix it
  SQLite / Redis trouble
      -> NOT committed: seek back to the same offset and retry after a pause; later events wait (per partition
         order is kept)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Optional

import redis
from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition
from pci import mask
from pydantic import ValidationError

from risk_v2 import metrics as m
from risk_v2.feedback import ChargebackEvent, FeedbackLoop

log = logging.getLogger(__name__)

TOPIC = "chargeback.created"
DLT_TOPIC = TOPIC + ".DLT"


class BadEvent(ValueError):
    """The message can never be processed as it is."""


def parse(value: Optional[bytes]) -> ChargebackEvent:
    if value is None:
        raise BadEvent("empty message")
    try:
        body = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadEvent(f"not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise BadEvent("not a JSON object")
    try:
        return ChargebackEvent.model_validate(body)
    except ValidationError as exc:
        # field + reason only, never the value (it might be a PAN)
        raise BadEvent("; ".join(f"{'.'.join(map(str, e['loc'])) or 'event'}: {e['msg']}"
                                 for e in exc.errors())) from exc


class ChargebackConsumer:
    def __init__(self, loop: FeedbackLoop, bootstrap: str, *, group_id: str = "risk-engine-feedback",
                 topic: str = TOPIC, dlt_topic: str = DLT_TOPIC, retry_pause_s: float = 2.0):
        self._loop = loop
        self._topic, self._dlt_topic = topic, dlt_topic
        self._retry_pause_s = retry_pause_s
        self._consumer_conf = {
            "bootstrap.servers": bootstrap, "group.id": group_id,
            "enable.auto.commit": False, "auto.offset.reset": "earliest",   # a new group reads what billing sent
        }
        self._producer_conf = {"bootstrap.servers": bootstrap, "acks": "all"}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None
        self.counts = {"stored": 0, "duplicate": 0, "unmatched": 0, "dead_lettered": 0, "retried": 0}

    @classmethod
    def from_env(cls, loop: FeedbackLoop) -> "ChargebackConsumer":
        return cls(loop, os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                   group_id=os.environ.get("KAFKA_GROUP_ID", "risk-engine-feedback"))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="chargeback-consumer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    def status(self) -> dict:
        return {"running": bool(self._thread and self._thread.is_alive()), "topic": self._topic,
                "counts": dict(self.counts), "last_error": self.last_error}

    def _run(self) -> None:
        consumer, producer = Consumer(self._consumer_conf), Producer(self._producer_conf)
        consumer.subscribe([self._topic])
        try:
            while not self._stop.is_set():
                msg = consumer.poll(0.5)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        self.last_error = str(msg.error())
                        log.warning("chargeback consumer: %s", msg.error())
                    continue
                if self._handle(msg, producer):
                    consumer.commit(message=msg, asynchronous=False)
                else:   # not stored: read the same message again after a pause
                    consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))
                    self._stop.wait(self._retry_pause_s)
        finally:
            consumer.close()   # leaves the group now (no wait for the session timeout)
            producer.flush(5)

    def _handle(self, msg, producer: Producer) -> bool:
        """True = done with this message (commit); False = try again later."""
        try:
            event = parse(msg.value())
        except BadEvent as exc:
            return self._dead_letter(msg, producer, str(exc))
        try:
            applied = self._loop.apply(event)
        except (sqlite3.Error, redis.RedisError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.counts["retried"] += 1
            m.CHARGEBACK_EVENTS.labels("retried").inc()
            log.warning("chargeback %s not stored, will retry: %s", event.event_id, exc)
            return False
        self.counts[applied.outcome] += 1
        self.last_error = None   # it worked again: an old error would only mislead (/health shows this)
        m.CHARGEBACK_EVENTS.labels(applied.outcome).inc()
        return True

    def _dead_letter(self, msg, producer: Producer, error: str) -> bool:
        value = msg.value() or b""
        try:
            value = mask(value.decode("utf-8")).encode("utf-8")   # don't copy a PAN into one more topic
        except UnicodeDecodeError:
            pass
        headers = {"error": mask(error)[:1000], "source.topic": msg.topic(),
                   "source.partition": str(msg.partition()), "source.offset": str(msg.offset())}
        producer.produce(self._dlt_topic, value=value, key=msg.key(), headers=headers)
        if producer.flush(5) > 0:   # DLT not reachable: keep the message, try again
            self.last_error = "dead-letter topic not reachable"
            return False
        self.counts["dead_lettered"] += 1
        m.CHARGEBACK_EVENTS.labels("dead_lettered").inc()
        log.warning("chargeback event dead-lettered (%s offset %s): %s", msg.topic(), msg.offset(), mask(error))
        return True
