"""
Feedback loop (Portfolio Day 18, docs/adr/0005): chargebacks become labels on the decisions that let them through.

Redis keeps a /v2 decision for 25 h (risk_v2/decisions.py); a chargeback arrives days or months later. So every new
/v2 decision is also written here, to SQLite (FEEDBACK_DB_PATH), and the `chargeback.created` consumer
(risk_v2/feedback_consumer.py) adds the chargeback next to it:

  decisions    one row per transaction_id (first decision wins, like Redis): action, model score + versions, and
               the exact model_input - what scripts/retrain.py trains on
  chargebacks  one row per event_id: a redelivered event is stored once. The label is "the transaction has at least
               one chargeback" (a second dispute on the same payment doesn't make it more fraudulent)

  payment_outcomes  (Day 19b, docs/adr/0006) one row per event_id from `payment.outcome`: billing says a payment was
               captured (money moved: only then CAN it charge back) or a person approved / rejected a held one.
               The honest label population is the CAPTURED payments - allowed ones and held-then-approved ones.

A chargeback also feeds the engine's own `past_chargebacks` profile counter in Redis (until Day 18 it was always 0):
counted once per TRANSACTION, atomically, so a redelivery or a second dispute on the same payment can't add 1 again.

Why SQLite: one file, no new server, standard library. It is per container - with replicas it moves to Postgres
(same SQL); see the ADR.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

import redis
from pci import reject_pans
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, model_validator

from risk_v2.contract import ID

log = logging.getLogger(__name__)

DAY_MS = 86_400_000
# How long after a payment a chargeback can still arrive. Card networks allow ~120 days for most reasons: until then
# "no chargeback yet" is not a label.
LABEL_WINDOW_DAYS = int(os.environ.get("FEEDBACK_LABEL_WINDOW_DAYS", "120"))
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feedback.db")
FLAGGED = ("hold_for_review", "block")


class PaymentOutcomeEvent(BaseModel):
    """docs/contracts/payment.outcome.v1.schema.json. Unknown fields are ignored (tolerant reader)."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    event_id: str = Field(min_length=1, max_length=64, pattern=ID)
    transaction_id: str = Field(min_length=1, max_length=64, pattern=ID)
    outcome: Literal["captured", "review_approved", "review_rejected"]
    amount_minor: StrictInt = Field(gt=0, le=10**12)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    occurred_at: AwareDatetime
    schema_version: Literal[1]

    @model_validator(mode="before")
    @classmethod
    def _no_card_numbers(cls, values):
        reject_pans(values)
        return values


class ChargebackEvent(BaseModel):
    """docs/contracts/chargeback.created.v1.schema.json. Unknown fields are ignored (tolerant reader)."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    event_id: str = Field(min_length=1, max_length=64, pattern=ID)
    transaction_id: str = Field(min_length=1, max_length=64, pattern=ID)
    amount_minor: StrictInt = Field(gt=0, le=10**12)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    reason: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    created_at: AwareDatetime
    schema_version: Literal[1]

    @model_validator(mode="before")
    @classmethod
    def _no_card_numbers(cls, values):
        reject_pans(values)
        return values


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    transaction_id  TEXT PRIMARY KEY,
    decided_at_ms   INTEGER NOT NULL,      -- engine clock (signals.observed_at_ms)
    customer_id     TEXT NOT NULL,
    action          TEXT NOT NULL,         -- final: allow | hold_for_review | block
    model_action    TEXT,
    model_score     REAL,                  -- NULL when the model could not score (rules_only)
    threshold       REAL,
    model_version   TEXT,
    rules_version   TEXT NOT NULL,
    amount_inr      REAL NOT NULL,
    model_input     TEXT NOT NULL          -- JSON: the raw transaction the model scored
);
CREATE INDEX IF NOT EXISTS decisions_by_time ON decisions (decided_at_ms);
CREATE TABLE IF NOT EXISTS chargebacks (
    event_id        TEXT PRIMARY KEY,
    transaction_id  TEXT NOT NULL,
    amount_minor    INTEGER NOT NULL,
    currency        TEXT NOT NULL,
    reason          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    received_at_ms  INTEGER NOT NULL,
    schema_version  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chargebacks_by_txn ON chargebacks (transaction_id);
CREATE TABLE IF NOT EXISTS payment_outcomes (
    event_id        TEXT PRIMARY KEY,
    transaction_id  TEXT NOT NULL,
    outcome         TEXT NOT NULL,         -- captured | review_approved | review_rejected
    amount_minor    INTEGER NOT NULL,
    currency        TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    received_at_ms  INTEGER NOT NULL,
    schema_version  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS outcomes_by_txn ON payment_outcomes (transaction_id, outcome);
"""


class OutcomeStore:
    def __init__(self, path: str, clock: Callable[[], float] = time.time):
        self.path = path
        self._clock = clock
        self._local = threading.local()   # sqlite3 connections are per thread (FastAPI runs handlers in a pool)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn().executescript(_SCHEMA)

    @classmethod
    def from_env(cls) -> "OutcomeStore":
        return cls(os.environ.get("FEEDBACK_DB_PATH", DEFAULT_DB_PATH))

    def now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # short busy timeout: the decision path writes here and must not wait long for the consumer's write
            conn = sqlite3.connect(self.path, timeout=0.25, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")      # readers don't block the writer
            conn.execute("PRAGMA synchronous=NORMAL")    # no fsync per commit (WAL keeps the file consistent)
            self._local.conn = conn
        return conn

    # ---- written by /v2/assess ----

    def record_decision(self, decision: dict) -> bool:
        """Keep a NEW /v2 decision. False = it was already kept (the first one wins, like the Redis copy)."""
        model_input = decision["model_input"]
        cur = self._conn().execute(
            "INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (decision["transaction_id"], int(decision["signals"]["observed_at_ms"]), model_input["customer_id"],
             decision["action"], decision.get("model_action"), decision.get("model_score"),
             decision.get("threshold"), decision.get("model_version"), decision["rules_version"],
             float(model_input["amount"]), json.dumps(model_input, sort_keys=True)))
        return cur.rowcount == 1

    # ---- written by the chargeback consumer ----

    def record_chargeback(self, event: ChargebackEvent) -> bool:
        """True = new event; False = this event_id was already stored (redelivery)."""
        cur = self._conn().execute(
            "INSERT OR IGNORE INTO chargebacks VALUES (?,?,?,?,?,?,?,?)",
            (event.event_id, event.transaction_id, event.amount_minor, event.currency, event.reason,
             event.created_at.isoformat(), self.now_ms(), event.schema_version))
        return cur.rowcount == 1

    def record_outcome(self, event: PaymentOutcomeEvent) -> bool:
        """True = new event; False = this event_id was already stored (redelivery)."""
        cur = self._conn().execute(
            "INSERT OR IGNORE INTO payment_outcomes VALUES (?,?,?,?,?,?,?,?)",
            (event.event_id, event.transaction_id, event.outcome, event.amount_minor, event.currency,
             event.occurred_at.isoformat(), self.now_ms(), event.schema_version))
        return cur.rowcount == 1

    # ---- reads ----

    def decision(self, transaction_id: str) -> Optional[dict]:
        row = self._conn().execute("SELECT * FROM decisions WHERE transaction_id = ?", (transaction_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["model_input"] = json.loads(out["model_input"])
        return out

    def outcomes(self, transaction_id: str) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM payment_outcomes WHERE transaction_id = ? ORDER BY received_at_ms",
                                    (transaction_id,)).fetchall()
        return [dict(r) for r in rows]

    def chargebacks(self, transaction_id: str) -> list[dict]:
        rows = self._conn().execute("SELECT * FROM chargebacks WHERE transaction_id = ? ORDER BY received_at_ms",
                                    (transaction_id,)).fetchall()
        return [dict(r) for r in rows]

    def performance(self, *, bucket: Literal["day", "week"] = "day", days: int = 30,
                    model_version: Optional[str] = None, label_window_days: int = LABEL_WINDOW_DAYS) -> dict:
        """Precision/recall per decision-date bucket (UTC), from the chargebacks received so far.

        Final action: flagged = hold_for_review or block. Model band: flagged = score >= its threshold (rows the
        model could not score are left out). Over ALL decisions both are LOWER BOUNDS: a blocked fraud is never
        captured, so it never charges back - it counts as a false positive and is missing from recall's true positives.
        `captured` (Day 19b) = the same numbers over CAPTURED payments only (billing's payment.outcome): every row
        there could charge back, so its labels are real. Flagged-and-captured = held, approved by a person, captured.
        A bucket is `mature` once all its payments are older than the label window; before that, chargebacks are
        still arriving and recall looks better than it is.
        """
        bucket_ms = DAY_MS * (7 if bucket == "week" else 1)
        now = self.now_ms()
        since = (now - days * DAY_MS) // bucket_ms * bucket_ms
        where, args = "d.decided_at_ms >= ?", [since]
        if model_version:
            where += " AND d.model_version = ?"
            args.append(model_version)
        rows = self._conn().execute(f"""
            SELECT b, COUNT(*) AS n, SUM(cb) AS chargebacks,
                   SUM(action IN ('hold_for_review', 'block')) AS flagged,
                   SUM(action IN ('hold_for_review', 'block') AND cb) AS tp,
                   SUM(action = 'allow' AND cb) AS fn,
                   SUM(action = 'hold_for_review') AS held, SUM(action = 'hold_for_review' AND cb) AS held_cb,
                   SUM(action = 'block') AS blocked, SUM(action = 'block' AND cb) AS blocked_cb,
                   SUM(model_score IS NOT NULL) AS scored,
                   SUM(IFNULL(model_score >= threshold, 0)) AS m_flagged,
                   SUM(IFNULL(model_score >= threshold AND cb, 0)) AS m_tp,
                   SUM(IFNULL(model_score < threshold AND cb, 0)) AS m_fn,
                   SUM(cap) AS captured, SUM(appr) AS approved, SUM(rej) AS rejected,
                   SUM(cap AND action IN ('hold_for_review', 'block')) AS c_flagged,
                   SUM(cap AND action IN ('hold_for_review', 'block') AND cb) AS c_tp,
                   SUM(cap AND action = 'allow' AND cb) AS c_fn,
                   SUM(IFNULL(cap AND model_score >= threshold, 0)) AS cm_flagged,
                   SUM(IFNULL(cap AND model_score >= threshold AND cb, 0)) AS cm_tp,
                   SUM(IFNULL(cap AND model_score < threshold AND cb, 0)) AS cm_fn,
                   GROUP_CONCAT(DISTINCT model_version) AS versions
            FROM (SELECT d.*, d.decided_at_ms / {bucket_ms} AS b,
                         EXISTS (SELECT 1 FROM chargebacks c WHERE c.transaction_id = d.transaction_id) AS cb,
                         EXISTS (SELECT 1 FROM payment_outcomes o WHERE o.transaction_id = d.transaction_id
                                 AND o.outcome = 'captured') AS cap,
                         EXISTS (SELECT 1 FROM payment_outcomes o WHERE o.transaction_id = d.transaction_id
                                 AND o.outcome = 'review_approved') AS appr,
                         EXISTS (SELECT 1 FROM payment_outcomes o WHERE o.transaction_id = d.transaction_id
                                 AND o.outcome = 'review_rejected') AS rej
                  FROM decisions d WHERE {where})
            GROUP BY b ORDER BY b""", args).fetchall()

        def ratio(a, b):
            return round(a / b, 4) if b else None

        buckets = []
        for r in rows:
            start_ms, end_ms = r["b"] * bucket_ms, (r["b"] + 1) * bucket_ms
            buckets.append({
                "start": _iso(start_ms), "end": _iso(end_ms),
                "mature": end_ms + label_window_days * DAY_MS <= now,
                "model_versions": sorted((r["versions"] or "").split(",")) if r["versions"] else [],
                "decisions": r["n"], "chargebacks": r["chargebacks"],
                "final": {"flagged": r["flagged"], "tp": r["tp"], "fp": r["flagged"] - r["tp"], "fn": r["fn"],
                          "precision": ratio(r["tp"], r["flagged"]), "recall": ratio(r["tp"], r["tp"] + r["fn"]),
                          "held": r["held"], "held_with_chargeback": r["held_cb"],
                          "blocked": r["blocked"], "blocked_with_chargeback": r["blocked_cb"]},
                "model": {"scored": r["scored"], "flagged": r["m_flagged"], "tp": r["m_tp"],
                          "fp": r["m_flagged"] - r["m_tp"], "fn": r["m_fn"],
                          "precision": ratio(r["m_tp"], r["m_flagged"]),
                          "recall": ratio(r["m_tp"], r["m_tp"] + r["m_fn"])},
                "outcomes": {"captured": r["captured"], "review_approved": r["approved"],
                             "review_rejected": r["rejected"]},
                "captured": {"final": {"flagged": r["c_flagged"], "tp": r["c_tp"], "fp": r["c_flagged"] - r["c_tp"],
                                       "fn": r["c_fn"], "precision": ratio(r["c_tp"], r["c_flagged"]),
                                       "recall": ratio(r["c_tp"], r["c_tp"] + r["c_fn"])},
                             "model": {"flagged": r["cm_flagged"], "tp": r["cm_tp"], "fp": r["cm_flagged"] - r["cm_tp"],
                                       "fn": r["cm_fn"], "precision": ratio(r["cm_tp"], r["cm_flagged"]),
                                       "recall": ratio(r["cm_tp"], r["cm_tp"] + r["cm_fn"])}},
            })
        unmatched = self._conn().execute("""
            SELECT COUNT(DISTINCT transaction_id) FROM chargebacks c
            WHERE NOT EXISTS (SELECT 1 FROM decisions d WHERE d.transaction_id = c.transaction_id)""").fetchone()[0]
        return {
            "bucket": bucket, "since": _iso(since), "model_version": model_version,
            "label_window_days": label_window_days, "buckets": buckets,
            "unmatched_chargebacks": unmatched,   # chargebacks for payments this engine has no /v2 decision for
            "note": "final/model over all decisions are lower bounds: blocked payments are never captured, so they "
                    "can't charge back (docs/adr/0005). `captured` counts only payments billing reported captured "
                    "(docs/adr/0006): real labels, still blind to what was blocked or rejected",
        }

    def labeled_decisions(self, *, decided_before_ms: int,
                          population: Literal["captured", "allow"] = "captured") -> list[dict]:
        """Matured decisions with their label, oldest first: model_input + is_chargeback + decided_at_ms.

        captured (default, Day 19b): payments billing reported CAPTURED - allowed ones and held-then-approved ones.
          Only those could charge back, so "no chargeback" is a real label.
        allow (Day 18, when billing sends no outcomes): final action allow, assumed captured.
        Never blocked or rejected payments: they are labeled by what WE did to them (a blocked fraud never charges
        back), so as 'no chargeback' rows they would teach the model that what we stop is safe.
        """
        population_sql = {
            "captured": "EXISTS (SELECT 1 FROM payment_outcomes o WHERE o.transaction_id = d.transaction_id "
                        "AND o.outcome = 'captured')",
            "allow": "d.action = 'allow'",
        }[population]
        rows = self._conn().execute(f"""
            SELECT d.model_input, d.decided_at_ms, d.action,
                   EXISTS (SELECT 1 FROM chargebacks c WHERE c.transaction_id = d.transaction_id) AS cb
            FROM decisions d WHERE d.decided_at_ms < ? AND {population_sql}
            ORDER BY d.decided_at_ms, d.transaction_id""", (decided_before_ms,)).fetchall()
        return [{**json.loads(r["model_input"]), "is_chargeback": int(r["cb"]), "decided_at_ms": r["decided_at_ms"],
                 "action": r["action"]} for r in rows]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# Count a chargeback in the customer's Redis profile once per transaction (SADD says whether it is new). Same
# 400-day expiry as the profile itself (velocity.lua).
_COUNT_LUA = """
if redis.call('SADD', KEYS[1], ARGV[1]) == 1 then
  redis.call('HINCRBY', KEYS[2], 'chargebacks', 1)
end
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[2])
return redis.call('HGET', KEYS[2], 'chargebacks')
"""
KEEP_PROFILE_S = 400 * 86400


@dataclass(frozen=True)
class Applied:
    outcome: str                          # stored | duplicate | unmatched
    customer_chargebacks: Optional[int]   # the profile counter after this event (None if unmatched)


class FeedbackLoop:
    """One chargeback event -> label + profile counter; one payment.outcome event -> stored next to the decision. Raises sqlite3.Error / redis.RedisError on trouble: the
    consumer then does NOT commit the offset and tries again (both steps are safe to repeat)."""

    def __init__(self, outcomes: OutcomeStore, client: redis.Redis):
        self.outcomes = outcomes
        self._count = client.register_script(_COUNT_LUA)

    def apply(self, event: ChargebackEvent) -> Applied:
        new = self.outcomes.record_chargeback(event)
        decision = self.outcomes.decision(event.transaction_id)
        if decision is None:
            # no /v2 decision here (a v1 payment, or the decision write failed): the label is kept and reported as
            # unmatched; there is no customer to count it against
            return Applied("unmatched", None)
        cust = f"risk:cust:{decision['customer_id']}"
        count = self._count(keys=[f"{cust}:chargeback_txns", f"{cust}:profile"],
                            args=[event.transaction_id, KEEP_PROFILE_S])
        return Applied("stored" if new else "duplicate", int(count))

    def apply_outcome(self, event: PaymentOutcomeEvent) -> Applied:
        new = self.outcomes.record_outcome(event)
        if self.outcomes.decision(event.transaction_id) is None:
            return Applied("unmatched", None)   # e.g. billing's fallback decided it: the engine never saw it
        return Applied("stored" if new else "duplicate", None)
