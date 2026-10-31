"""
Retrain from labeled /v2 decisions (Portfolio Day 18, docs/adr/0005). Writes a CANDIDATE model, never the live one.

  python scripts/retrain.py                              # from the repo root
  python scripts/retrain.py --label-window-days 0        # demo: every decision counts as labeled now

1. Labeled data = /v2 decisions in the feedback store (FEEDBACK_DB_PATH) that are
   - older than the label window (FEEDBACK_LABEL_WINDOW_DAYS, 120): before that "no chargeback" isn't known yet;
   - --population captured (default, Day 19b): billing reported the payment CAPTURED (payment.outcome) - allowed
     ones and held-then-approved ones. --population allow (Day 18 behaviour, billing sends no outcomes): final
     action allow. Never blocked / rejected payments: they can't charge back, so as "no chargeback" rows they
     would teach the model that what the engine stops is safe (the loop would erase its own knowledge).
   Label = the payment has at least one chargeback.
2. The newest --holdout share (by decision time) is kept out of training: it is the "future" both models are
   compared on.
3. Training data = the synthetic base set (data/transactions.csv) + the older labeled rows. The base set stays in
   because live labels only cover what we let through - without it the model forgets the patterns we block.
4. model/train.py runs unchanged (same split, cost-based selection, threshold on validation) into --out.
5. Live champion vs candidate on the holdout -> <out>/comparison.json.

Promotion stays manual (ADR 0004): copy <out>/model.pkl to model/shadow.pkl first, compare on live traffic
(GET /governance/shadow), then make it the champion.
Exit codes: 0 candidate written, 2 not enough labeled data (overall, or chargebacks in the holdout).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
for sub in ("agent", "model"):
    sys.path.insert(0, os.path.join(_ROOT, sub))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import train  # noqa: E402
from features import REQUIRED_RAW_FIELDS, to_model_frame  # noqa: E402
from risk_v2.feedback import DAY_MS, DEFAULT_DB_PATH, LABEL_WINDOW_DAYS, OutcomeStore  # noqa: E402
from threshold_analysis import total_cost  # noqa: E402

COLUMNS = REQUIRED_RAW_FIELDS + ["is_chargeback"]


def evaluate(bundle: dict, holdout: pd.DataFrame) -> dict:
    y = holdout["is_chargeback"].to_numpy()
    scores = bundle["pipeline"].predict_proba(to_model_frame(holdout))[:, 1]
    pred = (scores >= bundle["threshold"]).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "model_version": bundle["model_version"], "model_name": bundle["model_name"],
        "threshold": round(float(bundle["threshold"]), 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "avg_precision": round(float(average_precision_score(y, scores)), 4) if 0 < y.sum() < len(y) else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": int(len(y) - tp - fp - fn)},
        "total_cost_inr": round(float(total_cost(y, pred, holdout["amount"].to_numpy(dtype=float))), 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=os.environ.get("FEEDBACK_DB_PATH", DEFAULT_DB_PATH))
    p.add_argument("--base", default=train.DATA_PATH, help="synthetic base training set ('' = none)")
    p.add_argument("--champion", default=os.environ.get("MODEL_PATH", train.MODEL_PATH))
    p.add_argument("--out", default="model/candidate")
    p.add_argument("--label-window-days", type=int, default=LABEL_WINDOW_DAYS)
    p.add_argument("--population", choices=["captured", "allow"], default="captured")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--min-labeled", type=int, default=200)
    p.add_argument("--min-chargebacks", type=int, default=10)
    p.add_argument("--min-holdout-chargebacks", type=int, default=5,
                   help="chargebacks needed in the holdout - with fewer, the two models can't be compared")
    args = p.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"no feedback store at {args.db} - nothing to retrain from")
        return 2
    store = OutcomeStore(args.db)
    rows = store.labeled_decisions(decided_before_ms=store.now_ms() - args.label_window_days * DAY_MS,
                                   population=args.population)
    positives = sum(r["is_chargeback"] for r in rows)
    print(f"labeled decisions ({args.population}, older than {args.label_window_days} days): {len(rows)}, "
          f"with chargeback: {positives}")
    if len(rows) < args.min_labeled or positives < args.min_chargebacks:
        print(f"not enough labeled data (need >= {args.min_labeled} decisions and >= {args.min_chargebacks} "
              f"chargebacks) - no candidate written")
        return 2

    live = pd.DataFrame(rows)                                  # oldest first
    cut = int(len(live) * (1 - args.holdout))
    live_train, holdout = live.iloc[:cut], live.iloc[cut:]
    if holdout["is_chargeback"].sum() < args.min_holdout_chargebacks:
        # found by hand on Day 18: chargebacks only on older payments -> a holdout with 0 chargebacks, a comparison
        # that says nothing about catching fraud, and still "exit 0"
        print(f"only {int(holdout['is_chargeback'].sum())} chargebacks in the newest {len(holdout)} payments (need >= "
              f"{args.min_holdout_chargebacks} to compare the models) - no candidate written")
        return 2
    parts = [live_train[COLUMNS].assign(source="live")]
    if args.base:
        parts.insert(0, pd.read_csv(args.base)[COLUMNS].assign(source="synthetic"))
    data = pd.concat(parts, ignore_index=True)

    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(args.out, "training_data.csv")
    data.to_csv(data_path, index=False)
    print(f"training data: {len(data):,} rows ({len(live_train):,} live, {len(data) - len(live_train):,} synthetic)"
          f" | live holdout: {len(holdout):,} ({int(holdout['is_chargeback'].sum())} chargebacks)\n")
    out = {name: os.path.join(args.out, name) for name in
           ("model.pkl", "shadow.pkl", "metrics.json", "threshold_cost_curve.png")}
    train.main(data_path=data_path, model_path=out["model.pkl"], shadow_path=out["shadow.pkl"],
               metrics_path=out["metrics.json"], plot_path=out["threshold_cost_curve.png"])

    comparison = {
        "holdout": {"rows": len(holdout), "chargebacks": int(holdout["is_chargeback"].sum()),
                    "from_ms": int(holdout["decided_at_ms"].iloc[0]), "to_ms": int(holdout["decided_at_ms"].iloc[-1])},
        "training": {"live_rows": len(live_train), "synthetic_rows": len(data) - len(live_train)},
        "label_window_days": args.label_window_days, "population": args.population,
        "champion": evaluate(joblib.load(args.champion), holdout) if os.path.exists(args.champion) else None,
        "candidate": evaluate(joblib.load(out["model.pkl"]), holdout),
    }
    with open(os.path.join(args.out, "comparison.json"), "w") as fh:
        json.dump(comparison, fh, indent=2)

    print(f"\nlive holdout ({len(holdout)} payments, {comparison['holdout']['chargebacks']} chargebacks):")
    for role in ("champion", "candidate"):
        r = comparison[role]
        if r:
            print(f"  {role:9s} {r['model_version']:16s} precision {r['precision']}  recall {r['recall']}  "
                  f"AP {r['avg_precision']}  cost Rs {r['total_cost_inr']:,.0f}")
    print(f"\nwrote {args.out}/ (model.pkl, shadow.pkl, metrics.json, comparison.json). Live model unchanged.\n"
          f"Next: cp {out['model.pkl']} model/shadow.pkl -> watch GET /governance/shadow -> promote by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
