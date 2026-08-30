"""
Cost-based threshold selection.

A risk model outputs a probability in [0, 1]. Turning that into an
allow/review/block decision needs a cutoff. Picking 0.5 (or eyeballing a
precision-recall curve) ignores the fact that the two kinds of mistake do not
cost the same:

  * A MISSED chargeback (false negative) costs us the disputed amount back to
    the issuer, plus a fixed dispute fee, plus ops handling.
  * A FALSE ALARM (false positive) costs an analyst's time for a manual review,
    plus some risk of annoying a good customer.

Missed fraud is roughly an order of magnitude more expensive per event, so the
cost-minimising threshold sits well below 0.5. This module sweeps every
threshold and picks the one with the lowest expected total cost, so the number
we ship is defensible ("0.23 because it minimises cost under these assumptions"),
not arbitrary.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

# --- Cost assumptions (INR). Documented, deliberately conservative. ---
DISPUTE_FEE = 1500.0          # issuer/network fee charged on every chargeback
FN_OPS_COST = 500.0           # internal handling of a chargeback that slipped through
REVIEW_COST = 400.0           # analyst time + friction cost of one manual review
# A missed fraud additionally costs the transaction `amount` (goods/refund lost).


@dataclass
class ThresholdResult:
    threshold: float
    total_cost: float
    cost_per_txn: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    # cost if we did nothing (approve everything) and if we reviewed everything
    baseline_cost_approve_all: float
    baseline_cost_review_all: float


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def total_cost(y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray) -> float:
    """Expected rupee cost of a set of predictions."""
    fn_mask = (y_pred == 0) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_cost = np.sum(amounts[fn_mask] + DISPUTE_FEE + FN_OPS_COST)
    fp_cost = np.sum(fp_mask) * REVIEW_COST
    return float(fn_cost + fp_cost)


# Manual-review teams have finite capacity, and flagging transactions whose
# hit-rate is below ~30% burns reviewer time and adds customer friction for
# little gain. So the threshold we actually ship is the cost-minimiser SUBJECT
# TO a precision floor - not the unconstrained cost minimum.
DEFAULT_PRECISION_FLOOR = 0.30


def _result_at(t, y_true, y_scores, amounts, approve_all, review_all) -> ThresholdResult:
    y_pred = (y_scores >= t).astype(int)
    c = total_cost(y_true, y_pred, amounts)
    tp, fp, fn, tn = _confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ThresholdResult(
        threshold=round(float(t), 4),
        total_cost=round(c, 2),
        cost_per_txn=round(c / len(y_true), 2),
        true_positives=tp, false_positives=fp,
        false_negatives=fn, true_negatives=tn,
        precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4),
        baseline_cost_approve_all=round(approve_all, 2),
        baseline_cost_review_all=round(review_all, 2),
    )


def sweep(y_true, y_scores, amounts, steps: int = 200,
          precision_floor: float = DEFAULT_PRECISION_FLOOR):
    """Return (chosen, unconstrained, curve).

    chosen        - cost minimiser with precision >= precision_floor (what we ship)
    unconstrained - cost minimiser with no precision constraint (for comparison)
    curve         - list of (threshold, cost) points
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    thresholds = np.linspace(0.01, 0.99, steps)
    approve_all = total_cost(y_true, np.zeros_like(y_true), amounts)
    review_all = total_cost(y_true, np.ones_like(y_true), amounts)

    results = [
        _result_at(t, y_true, y_scores, amounts, approve_all, review_all)
        for t in thresholds
    ]
    curve = [(r.threshold, r.total_cost) for r in results]

    unconstrained = min(results, key=lambda r: r.total_cost)
    eligible = [r for r in results if r.precision >= precision_floor]
    chosen = min(eligible, key=lambda r: r.total_cost) if eligible else unconstrained
    return chosen, unconstrained, curve


def print_report(chosen: ThresholdResult, unconstrained: ThresholdResult, curve,
                 precision_floor: float = DEFAULT_PRECISION_FLOOR,
                 sample_every: int = 20) -> None:
    print("=" * 68)
    print("COST-BASED THRESHOLD ANALYSIS")
    print("-" * 68)
    print("  Cost assumptions (INR):")
    print(f"    missed fraud (FN) = transaction amount + Rs{DISPUTE_FEE:.0f} dispute fee"
          f" + Rs{FN_OPS_COST:.0f} ops")
    print(f"    false alarm  (FP) = Rs{REVIEW_COST:.0f} manual review + friction")
    print("-" * 68)
    print(f"  {'threshold':>10} {'total cost (Rs)':>18}")
    for i, (t, c) in enumerate(curve):
        if i % sample_every == 0:
            marker = ""
            if abs(t - unconstrained.threshold) < 1e-6:
                marker = "  <-- unconstrained min"
            elif abs(t - chosen.threshold) < 1e-6:
                marker = "  <-- chosen"
            print(f"  {t:>10.3f} {c:>18,.0f}{marker}")
    print("-" * 68)
    print(f"  unconstrained cost min : thr={unconstrained.threshold}  "
          f"cost=Rs {unconstrained.total_cost:,.0f}  "
          f"precision={unconstrained.precision:.3f}  recall={unconstrained.recall:.3f}")
    print(f"  precision floor imposed : {precision_floor:.2f}  "
          f"(manual-review capacity / customer-friction constraint)")
    print("-" * 68)
    print(f"  CHOSEN threshold        : {chosen.threshold}")
    print(f"  total cost @ threshold  : Rs {chosen.total_cost:,.0f}"
          f"  ({chosen.cost_per_txn:.2f} / txn)")
    print(f"  cost if approve-all     : Rs {chosen.baseline_cost_approve_all:,.0f}")
    print(f"  cost if review-all      : Rs {chosen.baseline_cost_review_all:,.0f}")
    saved = chosen.baseline_cost_approve_all - chosen.total_cost
    print(f"  cost avoided vs approve-all: Rs {saved:,.0f}"
          f"  ({100 * saved / chosen.baseline_cost_approve_all:.1f}%)")
    premium = chosen.total_cost - unconstrained.total_cost
    print(f"  cost premium vs unconstrained: Rs {premium:,.0f}"
          f"  ({100 * premium / unconstrained.total_cost:.1f}%)")
    print(f"  confusion @ threshold   : TP={chosen.true_positives} FP={chosen.false_positives}"
          f" FN={chosen.false_negatives} TN={chosen.true_negatives}")
    print(f"  precision / recall / F1 : {chosen.precision:.3f} / {chosen.recall:.3f}"
          f" / {chosen.f1:.3f}")
    print("=" * 68)


def save_plot(curve, chosen_threshold: float, path: str,
              unconstrained_threshold: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [t for t, _ in curve]
    cs = [c for _, c in curve]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, cs, lw=2)
    if unconstrained_threshold is not None:
        ax.axvline(unconstrained_threshold, color="gray", ls=":",
                   label=f"unconstrained min = {unconstrained_threshold:.3f}")
    ax.axvline(chosen_threshold, color="crimson", ls="--",
               label=f"chosen = {chosen_threshold:.3f}")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("total expected cost (INR)")
    ax.set_title("Cost vs. decision threshold (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _standalone() -> None:
    """Run against the saved model + a fresh stratified test split."""
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from features import to_model_frame

    bundle = joblib.load("model/model.pkl")
    pipe = bundle["pipeline"]

    df = pd.read_csv("data/transactions.csv")
    _, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_chargeback"]
    )
    X_test = to_model_frame(test)
    y_test = test["is_chargeback"].to_numpy()
    scores = pipe.predict_proba(X_test)[:, 1]

    chosen, unconstrained, curve = sweep(y_test, scores, test["amount"].to_numpy())
    print_report(chosen, unconstrained, curve)
    save_plot(curve, chosen.threshold, "model/threshold_cost_curve.png",
              unconstrained.threshold)
    print("\nsaved plot -> model/threshold_cost_curve.png")
    print("\nchosen result as JSON:")
    print(json.dumps(asdict(chosen), indent=2))


if __name__ == "__main__":
    _standalone()
