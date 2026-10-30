"""
Explanation for a /v2 decision that is ALREADY made (rules + model). The LLM only writes the sentence; it cannot
change the action, and any LLM failure falls back to a template built from the reason codes (docs/adr/0002).

Time: the LLM gets only what is left of the request's 300 ms budget (Day 16), at most LLM_EXPLAIN_TIMEOUT_S, and runs
on the bounded llm_pool (no orphan threads). Less than MIN_LLM_S left -> not even tried ("llm_no_budget").
In practice a ~1 s Gemini call rarely fits: the template is the normal explanation on /v2.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

from llm_client import LLMError, LLMUnavailable, call_gemini, is_configured
from llm_pool import LLMBusy, LLMTimeout, run as run_llm
from reasoning_agent import _extract_json

from risk_v2.reason_codes import REASON_CODES

EXPLAIN_TIMEOUT_S = float(os.environ.get("LLM_EXPLAIN_TIMEOUT_S", "5"))
MIN_LLM_S = 0.05
MAX_EXPLANATION_CHARS = 400

_PROMPT = """You explain payment-risk decisions to a fraud analyst. The decision below is FINAL - do not \
question or change it. Write one or two plain sentences (max 40 words) explaining it, using only the reasons given.

Decision: {action}
Model score: {score} (review at {review_at}, block at {block_at})
Reasons (code, where it came from, evidence):
{reasons}

Respond with ONLY a JSON object: {{"explanation": "<text>"}}"""


def template(action: str, reasons: list[dict], model_score: Optional[float]) -> str:
    """One clause per fired rule (its evidence once, then its codes), then the model's top factors."""
    parts, by_rule, model = [], {}, []
    for r in reasons:
        if r["source"] == "rule":
            by_rule.setdefault((r["rule"], r["detail"]), []).append(r["code"])
        elif "contribution" in r:
            model.append(f"{r['code']} +{r['contribution']:.3f}")
        else:
            parts.append(f"{r['code']} ({r['detail']})")
    rules = [f"rule {rule} [{', '.join(codes)}]: {detail}" for (rule, detail), codes in by_rule.items()]
    score = f"model score {model_score:.3f}" if model_score is not None else "not scored"
    factors = f"{score}" + (f", top factors {', '.join(model)}" if model else "")
    return f"{action.upper()}: " + "; ".join(rules + parts + [factors]) + "."


def explain(action: str, reasons: list[dict], model_score: Optional[float], review_at: Optional[float],
            block_at: float, llm_fn: Optional[Callable[[str], str]] = None,
            time_left_s: Optional[float] = None) -> tuple[str, str, Optional[str]]:
    """Return (explanation, source "llm"|"template", failure_mode or None)."""
    fallback = template(action, reasons, model_score)
    if llm_fn is None:
        if not is_configured():
            return fallback, "template", None   # no key: not a failure, just no LLM
        llm_fn = call_gemini
    timeout_s = EXPLAIN_TIMEOUT_S if time_left_s is None else min(EXPLAIN_TIMEOUT_S, time_left_s)
    if timeout_s < MIN_LLM_S:
        return fallback, "template", "llm_no_budget"
    lines = "\n".join(f"- {r['code']} ({r['source']}): {r.get('detail') or REASON_CODES[r['code']]}"
                      for r in reasons) or "- none"
    prompt = _PROMPT.format(action=action, score=model_score, review_at=review_at, block_at=block_at,
                            reasons=lines)
    try:
        raw = run_llm(llm_fn, prompt, timeout_s)
        text = _extract_json(raw or "").get("explanation")
    except LLMTimeout:
        return fallback, "template", "llm_timeout"
    except LLMBusy:
        return fallback, "template", "llm_busy"
    except LLMUnavailable:
        return fallback, "template", "llm_unavailable"
    except (LLMError, ValueError, json.JSONDecodeError, AttributeError):
        return fallback, "template", "llm_malformed"
    except Exception:  # noqa: BLE001 - the explanation must never break a decision
        return fallback, "template", "llm_error"
    if not isinstance(text, str) or not text.strip():
        return fallback, "template", "llm_malformed"
    return text.strip()[:MAX_EXPLANATION_CHARS], "llm", None
