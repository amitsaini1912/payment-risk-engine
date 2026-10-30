"""
/v2 decision pipeline (docs/adr/0002):  rules  ->  model (+ per-reason contributions)  ->  explanation.

- Rules and the model DECIDE; the final action is the strictest of the fired rules and the model's band.
  Bands: score >= bundle threshold -> hold_for_review, >= rules.yaml block_at -> block.
- The model scores even when a rule already blocks: the score is recorded for later analysis (Day 17/18).
- Model can't score (no model file / bad row) -> at least hold_for_review with R99_NOT_SCORED, never allow.
- Every hold or block carries at least one reason code.
- The LLM (if configured) only words the explanation (risk_v2/explain.py), with whatever is left of the 300 ms
  budget (risk_v2/budget.py).
"""

from __future__ import annotations

from typing import Callable, Optional

from scorer import ScoringError, load_scorer

from risk_v2.budget import Deadline
from risk_v2.contributions import score_with_contributions
from risk_v2.explain import explain
from risk_v2.rules import RuleBook, build_context

_STRICTNESS = {"allow": 0, "hold_for_review": 1, "block": 2}


def _strictest(*actions: str) -> str:
    return max(actions, key=_STRICTNESS.__getitem__)


def decide(facts, signals, model_input: dict, book: RuleBook,
           llm_fn: Optional[Callable[[str], str]] = None, deadline: Optional[Deadline] = None) -> dict:
    fired = book.evaluate(build_context(facts, signals, model_input))
    reasons = [{"code": code, "source": "rule", "rule": f.id, "detail": f.detail}
               for f in fired for code in f.reasons]

    model_score = review_at = None
    contributions: dict[str, float] = {}
    features: dict = {}
    failure_mode = None
    try:
        scorer = load_scorer()
        review_at = scorer.threshold
        model_score, contributions = score_with_contributions(scorer.score_batch, model_input)
        features = scorer.feature_snapshot(model_input)
    except (ScoringError, FileNotFoundError):
        failure_mode = "model_unavailable"
    block_at = max(book.policy.block_at, review_at or 0)

    if model_score is None:
        band = "hold_for_review"
        reasons.append({"code": "R99_NOT_SCORED", "source": "engine", "detail": "model unavailable"})
    else:
        band = "block" if model_score >= block_at else "hold_for_review" if model_score >= review_at else "allow"
        top = sorted(((c, v) for c, v in contributions.items() if v >= book.policy.reasons.min_contribution),
                     key=lambda cv: -cv[1])[:book.policy.reasons.max_reasons]
        reasons += [{"code": c, "source": "model", "contribution": v} for c, v in top]
    action = _strictest(band, *(f.action for f in fired))
    if action != "allow" and not reasons:
        reasons.append({"code": "R90_MODEL_SCORE", "source": "model",
                        "detail": f"score {model_score:.3f} >= {review_at}"})

    codes = list(dict.fromkeys(r["code"] for r in reasons))
    try:
        explanation, explanation_source, llm_failure = explain(
            action, reasons, model_score, review_at, block_at, llm_fn=llm_fn,
            time_left_s=deadline.left_for_optional_work() if deadline else None)
    except Exception:  # noqa: BLE001 - the decision is made; wording it must never turn it into a 500
        explanation, explanation_source, llm_failure = f"{action.upper()}: {', '.join(codes)}", "codes", \
            "explanation_error"
    result = {
        "transaction_id": facts.transaction_id,
        "action": action,
        "reason_codes": codes,
        "reasons": reasons,
        "rules_fired": [f.id for f in fired],
        "rules_version": book.version,
        "model_score": round(model_score, 4) if model_score is not None else None,
        "threshold": review_at,
        "block_at": block_at,
        "model_contributions": contributions,
        "path_used": "rules+model" if model_score is not None else "rules_only",
        "failure_mode": failure_mode or llm_failure,
        "explanation": explanation,
        "explanation_source": explanation_source,
        "confidence": _confidence(model_score, review_at, block_at, bool(fired)),
        "feature_snapshot": features,
    }
    if deadline:
        result["decided_in_ms"] = round(deadline.elapsed_s() * 1000, 1)
    return result


def _confidence(score: Optional[float], review_at: Optional[float], block_at: float, rule_fired: bool) -> float:
    """How far the score is from a band edge (same idea as the v1 fallback); a fired rule is a definite signal."""
    if score is None:
        return 0.4
    nearest = min(abs(score - edge) for edge in (0.0, review_at, block_at, 1.0))
    return round(min(0.95, 0.55 + nearest + (0.2 if rule_fired else 0.0)), 2)
