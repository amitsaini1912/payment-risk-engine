"""
The rules engine for /v2/assess (docs/adr/0002). Rules live in rules.yaml, not in code, so risk ops can change a
limit without a deploy.

- Hot reload: every request stats the file; when its mtime/size/inode changes it is parsed and validated again.
- A broken file (bad YAML, unknown field, typo in an operator, unknown reason code, duplicate id) is REJECTED as a
  whole and the last good version keeps running. Its error shows in /health. No good version at all -> 503.
- yaml.safe_load only: a `!!python/object` tag in the file is an error, never code that runs.
- Rules can only make a decision stricter (hold_for_review / block); the model and the rules are combined by taking
  the strictest action.

A rule is a condition tree over the CONTEXT_FIELDS below:
    when: {all: [{field: card_txn_count_1h, op: ">=", value: 3}, {any: [...]}]}
"""

from __future__ import annotations

import hashlib
import logging
import operator
import os
import threading
from dataclasses import dataclass
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import (BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationError, field_validator,
                      model_validator)

from risk_v2.reason_codes import REASON_CODES

log = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rules.yaml")

# What a rule can look at. Counts and sums are of EARLIER payments (before this one); distinct devices/IPs include
# this one. Flags are 0/1.
NUMERIC_FIELDS = frozenset({
    "amount_inr", "avg_order_value_inr", "amount_to_avg_ratio",
    "customer_txn_count_1h", "customer_txn_count_24h",
    "card_txn_count_1h", "card_txn_count_24h", "card_amount_sum_24h_inr",
    "card_distinct_devices_24h", "card_distinct_ips_24h", "card_is_new",
    "is_new_device", "account_age_days", "past_chargebacks", "past_orders",
    "hour", "is_odd_hour", "ip_billing_country_mismatch", "is_international_card", "email_is_disposable",
})
STRING_FIELDS = frozenset({"currency", "card_type", "merchant_category"})
CONTEXT_FIELDS = NUMERIC_FIELDS | STRING_FIELDS

_COMPARE = {"==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
            "<": operator.lt, "<=": operator.le}
Scalar = Union[bool, int, float, str]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Leaf(_Strict):
    field: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in"]
    value: Union[Scalar, list[Scalar]]

    @model_validator(mode="after")
    def _types_match(self) -> "Leaf":
        if self.field not in CONTEXT_FIELDS:
            raise ValueError(f"unknown field {self.field!r} (known: {sorted(CONTEXT_FIELDS)})")
        values = self.value if isinstance(self.value, list) else [self.value]
        if self.op in ("in", "not_in") and not isinstance(self.value, list):
            raise ValueError(f"op {self.op!r} needs a list value")
        if self.op not in ("in", "not_in") and isinstance(self.value, list):
            raise ValueError(f"op {self.op!r} needs a single value")
        if self.field in NUMERIC_FIELDS:
            if any(isinstance(v, str) for v in values):
                raise ValueError(f"{self.field} is numeric, got {self.value!r}")
        else:
            if self.op not in ("==", "!=", "in", "not_in"):
                raise ValueError(f"{self.field} is text: only ==, !=, in, not_in")
            if any(not isinstance(v, str) for v in values):
                raise ValueError(f"{self.field} is text, got {self.value!r}")
        return self

    def test(self, ctx: dict) -> Optional[str]:
        """The trace ("card_txn_count_1h=5 >= 3") when true, else None."""
        actual = ctx[self.field]
        if self.op == "in":
            ok = actual in self.value
        elif self.op == "not_in":
            ok = actual not in self.value
        else:
            ok = _COMPARE[self.op](actual, self.value)
        return f"{self.field}={_fmt(actual)} {self.op} {_fmt(self.value)}" if ok else None


class All(_Strict):
    all: list["Condition"] = Field(min_length=1)

    def test(self, ctx: dict) -> Optional[str]:
        traces = []
        for c in self.all:
            t = c.test(ctx)
            if t is None:
                return None
            traces.append(t)
        return " and ".join(traces)


class AnyOf(_Strict):
    any: list["Condition"] = Field(min_length=1)

    def test(self, ctx: dict) -> Optional[str]:
        traces = [t for t in (c.test(ctx) for c in self.any) if t is not None]
        return " or ".join(traces) if traces else None


def _kind(v) -> str:
    """Pick the condition type by its key, so a typo reports ONE error, not one per type pydantic tried."""
    if isinstance(v, dict):
        return "all" if "all" in v else "any" if "any" in v else "leaf"
    return "all" if isinstance(v, All) else "any" if isinstance(v, AnyOf) else "leaf"


Condition = Annotated[Union[Annotated[Leaf, Tag("leaf")], Annotated[All, Tag("all")], Annotated[AnyOf, Tag("any")]],
                      Discriminator(_kind)]
All.model_rebuild()
AnyOf.model_rebuild()


class Rule(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)
    description: str = ""
    enabled: bool = True
    action: Literal["hold_for_review", "block"]
    reasons: list[str] = Field(min_length=1)
    when: Condition

    @field_validator("reasons")
    @classmethod
    def _known_codes(cls, v: list[str]) -> list[str]:
        unknown = [c for c in v if c not in REASON_CODES]
        if unknown:
            raise ValueError(f"unknown reason code(s) {unknown} (known: {sorted(REASON_CODES)})")
        return v


class ModelReasons(_Strict):
    min_contribution: float = Field(default=0.02, ge=0, le=1)
    max_reasons: int = Field(default=3, ge=0, le=len(REASON_CODES))


class ModelPolicy(_Strict):
    # review_at is NOT here: it is the threshold shipped in the model bundle (one source of truth, weakness #19)
    block_at: float = Field(default=0.80, gt=0, le=1)
    reasons: ModelReasons = ModelReasons()


class RuleFile(_Strict):
    model: ModelPolicy = ModelPolicy()
    rules: list[Rule]

    @model_validator(mode="after")
    def _unique_ids(self) -> "RuleFile":
        ids = [r.id for r in self.rules]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate rule id(s) {dupes}")
        return self


@dataclass(frozen=True)
class FiredRule:
    id: str
    action: str
    reasons: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class RuleBook:
    version: str          # sha256 of the file's bytes (first 12 hex) - goes on every decision
    policy: ModelPolicy
    rules: tuple[Rule, ...]

    def evaluate(self, ctx: dict) -> list[FiredRule]:
        missing = CONTEXT_FIELDS - ctx.keys()
        if missing:
            raise KeyError(f"context is missing {sorted(missing)}")
        fired = []
        for rule in self.rules:
            if rule.enabled and (trace := rule.when.test(ctx)) is not None:
                fired.append(FiredRule(rule.id, rule.action, tuple(rule.reasons), trace))
        return fired


class RulesInvalid(ValueError):
    pass


class RulesUnavailable(RuntimeError):
    """No valid rules file has ever loaded. /v2/assess answers 503."""


def parse(raw: bytes) -> RuleBook:
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RulesInvalid(f"not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RulesInvalid("top level must be a mapping with a `rules:` list")
    try:
        parsed = RuleFile.model_validate(data)
    except ValidationError as exc:
        raise RulesInvalid(str(exc)) from exc
    return RuleBook(hashlib.sha256(raw).hexdigest()[:12], parsed.model, tuple(parsed.rules))


class RuleLoader:
    """Serves the last good RuleBook; reloads when the file changes (checked on every get())."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._book: Optional[RuleBook] = None
        self._seen: Optional[tuple] = None    # (mtime_ns, size, inode) of the last file we tried, good or bad
        self.last_error: Optional[str] = None

    def get(self) -> RuleBook:
        try:
            st = os.stat(self.path)
            key = (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError as exc:
            key = None
            self._remember_error(f"cannot read {self.path}: {exc}")
        if key is not None and key != self._seen:
            with self._lock:
                if key != self._seen:
                    self._reload(key)
        if self._book is None:
            raise RulesUnavailable(self.last_error or "no rules loaded")
        return self._book

    def _reload(self, key: tuple) -> None:
        self._seen = key
        try:
            with open(self.path, "rb") as fh:
                book = parse(fh.read())
        except (OSError, RulesInvalid) as exc:
            self._remember_error(str(exc))
            return
        if self._book is None or book.version != self._book.version:
            log.info("rules %s loaded from %s (%d rules)", book.version, self.path, len(book.rules))
        self._book = book
        self.last_error = None

    def _remember_error(self, error: str) -> None:
        if error != self.last_error:
            kept = f"keeping {self._book.version}" if self._book else "no good version to keep"
            log.error("rules file rejected (%s): %s", kept, error)
        self.last_error = error


def build_context(facts, signals, model_input: dict) -> dict:
    """What rules see: the facts, the engine's signals and the model's input, under the names in CONTEXT_FIELDS."""
    return {
        "amount_inr": model_input["amount"],
        "avg_order_value_inr": model_input["avg_order_value"],
        "amount_to_avg_ratio": round(model_input["amount"] / model_input["avg_order_value"], 4),
        "currency": facts.currency,
        "card_type": model_input["card_type"],
        "merchant_category": model_input["merchant_category"],
        "customer_txn_count_1h": signals.customer_txn_count_1h,
        "customer_txn_count_24h": signals.customer_txn_count_24h,
        "card_txn_count_1h": signals.card_txn_count_1h,
        "card_txn_count_24h": signals.card_txn_count_24h,
        "card_amount_sum_24h_inr": signals.card_amount_sum_24h_minor / 100,
        "card_distinct_devices_24h": signals.card_distinct_devices_24h,
        "card_distinct_ips_24h": signals.card_distinct_ips_24h,
        "card_is_new": int(signals.card_is_new),
        "is_new_device": model_input["is_new_device"],
        "account_age_days": model_input["account_age_days"],
        "past_chargebacks": model_input["past_chargebacks"],
        "past_orders": model_input["past_orders"],
        "hour": model_input["hour"],
        "is_odd_hour": int(model_input["hour"] >= 23 or model_input["hour"] <= 5),
        "ip_billing_country_mismatch": model_input["ip_billing_country_mismatch"],
        "is_international_card": model_input["is_international_card"],
        "email_is_disposable": model_input["email_is_disposable"],
    }


def _fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)
