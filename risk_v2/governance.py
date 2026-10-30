"""
Model governance for /v2 (Portfolio Day 17, docs/adr/0004): shadow comparison + PSI drift, kept in Redis so every
replica adds to the same numbers.

After each NEW decision (not replays) one pipelined round trip adds:
  risk:shadow:<champion>:<shadow>   HASH  n, model:<champ band>><shadow band>, final:<champ>><shadow>,
                                          abs_diff_micro (sum |score diff| x 1e6), shadow_higher
  risk:drift:<model>:<feature>      HASH  <bin> -> count       (bins from the model bundle's training deciles)
Keys are per model version, so a retrain starts clean numbers. Failures here never touch the decision.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis

from drift import DRIFT_FEATURES, MIN_SAMPLES, bin_of, feature_value, level, psi

log = logging.getLogger(__name__)
ACTIONS = ("allow", "hold_for_review", "block")


class Governance:
    def __init__(self, client: redis.Redis):
        self._client = client

    @staticmethod
    def _shadow_key(champion: str, shadow: str) -> str:
        return f"risk:shadow:{champion}:{shadow}"

    @staticmethod
    def _drift_key(model: str, feature: str) -> str:
        return f"risk:drift:{model}:{feature}"

    def record(self, decision: dict, model_input: dict, drift_reference: dict) -> bool:
        """Count one new decision. Returns False (and logs) if Redis refused - the decision is unaffected."""
        try:
            pipe = self._client.pipeline(transaction=False)
            shadow = decision.get("shadow") or {}
            if shadow.get("score") is not None and decision.get("model_score") is not None:
                key = self._shadow_key(decision["model_version"], shadow["model_version"])
                pipe.hincrby(key, "n", 1)
                pipe.hincrby(key, f"model:{decision['model_action']}>{shadow['model_action']}", 1)
                pipe.hincrby(key, f"final:{decision['action']}>{shadow['final_action']}", 1)
                pipe.hincrby(key, "abs_diff_micro", int(abs(shadow["score"] - decision["model_score"]) * 1e6))
                if shadow["score"] > decision["model_score"]:
                    pipe.hincrby(key, "shadow_higher", 1)
            if decision.get("model_score") is not None:
                for feature, ref in drift_reference.items():
                    pipe.hincrby(self._drift_key(decision["model_version"], feature),
                                 str(bin_of(feature_value(model_input, feature), ref["edges"])), 1)
            pipe.execute()
            return True
        except (redis.RedisError, KeyError, TypeError, ValueError) as exc:
            log.warning("governance counters not updated: %s", exc)
            return False

    def shadow_report(self, champion: str, shadow: Optional[str]) -> dict:
        if shadow is None:
            return {"champion": champion, "shadow": None, "n": 0, "note": "no shadow model loaded"}
        raw = {k: int(v) for k, v in self._client.hgetall(self._shadow_key(champion, shadow)).items()}
        n = raw.get("n", 0)

        def matrix(prefix: str) -> dict:
            return {a: {b: raw.get(f"{prefix}:{a}>{b}", 0) for b in ACTIONS} for a in ACTIONS}

        model_m, final_m = matrix("model"), matrix("final")
        agree = lambda m: sum(m[a][a] for a in ACTIONS) / n if n else None  # noqa: E731
        return {
            "champion": champion, "shadow": shadow, "n": n,
            "model_band_agreement": agree(model_m),        # champion band vs shadow band (model only)
            "final_action_agreement": agree(final_m),      # with the same rules applied to both
            "mean_abs_score_diff": raw.get("abs_diff_micro", 0) / 1e6 / n if n else None,
            "shadow_scores_higher_share": raw.get("shadow_higher", 0) / n if n else None,
            "model_band_matrix": model_m,                  # rows: champion, columns: shadow
            "final_action_matrix": final_m,
        }

    def drift_report(self, model_version: str, reference: dict) -> dict:
        features = {}
        for feature in DRIFT_FEATURES:
            ref = reference.get(feature)
            if ref is None:
                continue
            counts_raw = self._client.hgetall(self._drift_key(model_version, feature))
            counts = [int(counts_raw.get(str(i), 0)) for i in range(len(ref["expected"]))]
            n = sum(counts)
            value = psi(ref["expected"], counts) if n else None
            features[feature] = {
                "n": n, "psi": round(value, 4) if value is not None else None,
                "level": level(value) if n >= MIN_SAMPLES else f"insufficient data (< {MIN_SAMPLES})",
                "expected": [round(e, 4) for e in ref["expected"]],
                "actual": [round(c / n, 4) for c in counts] if n else None,
                "edges": ref["edges"],
            }
        return {"model_version": model_version, "features": features}
