"""
Train the chargeback-risk classifier.

Methodology (baseline-first):
  1. Logistic Regression with balanced class weights  -> interpretable baseline.
  2. Random Forest (balanced)                          -> non-linear, gives
     feature importances.
  3. Histogram Gradient Boosting (balanced)            -> usually the strongest
     on tabular data; still a tree model, so still explainable.

Split (Portfolio Day 17, fixes weakness #15): 60% train / 20% validation / 20% test, stratified.
  - models are fitted on TRAIN;
  - the model is SELECTED and its threshold TUNED on VALIDATION;
  - TEST is touched once, at the end, for the numbers we report. (Before Day 17 the threshold was tuned on the
    same test set the metrics came from - slightly optimistic numbers.)

Selection (Day 17): each candidate gets its cost-minimising threshold on VALIDATION (with the precision floor),
and the candidate with the LOWEST VALIDATION COST wins - we deploy at a threshold and pay in rupees, so that is
what we compare. Average precision (PR-AUC, threshold-free) is the tie-break and is still reported. (Days 2-16
selected by AP; on Day 17's data that picked logistic regression at Rs 317,645 validation cost vs random forest's
Rs 265,274.) The runner-up is saved as the SHADOW
model: it scores live traffic next to the real model, with no effect on decisions,
so we can compare them before ever switching (risk_v2/governance.py).

The operating threshold is then chosen by `threshold_analysis.sweep` to minimise
expected business cost - not left at 0.5.

Output:
  model/model.pkl      - dict: {pipeline, threshold, model_version, drift_reference, metrics, ...}
  model/shadow.pkl     - the runner-up, same format
  model/metrics.json   - validation metrics for every candidate + test metrics of the selected model
  model/threshold_cost_curve.png

Run:
  python model/train.py            # from the repo root
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from drift import DRIFT_FEATURES, feature_value, reference  # noqa: E402
from features import MODEL_FEATURES, build_pipeline, to_model_frame  # noqa: E402
from threshold_analysis import (  # noqa: E402
    DEFAULT_PRECISION_FLOOR,
    total_cost,
    DISPUTE_FEE,
    FN_OPS_COST,
    REVIEW_COST,
    print_report,
    save_plot,
    sweep,
)

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "model/model.pkl"
SHADOW_PATH = "model/shadow.pkl"
METRICS_PATH = "model/metrics.json"
PLOT_PATH = "model/threshold_cost_curve.png"
RANDOM_STATE = 42


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=400,
            l2_regularization=1.0, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
    }


def evaluate_at_half(y_true, scores) -> dict:
    pred = (scores >= 0.5).astype(int)
    return {
        "avg_precision": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "precision@0.5": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall@0.5": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1@0.5": round(float(f1_score(y_true, pred, zero_division=0)), 4),
    }


def top_feature_importances(pipeline, X_test, y_test, n: int = 12) -> dict | None:
    """Permutation importance on the raw input columns.

    Works for any estimator (HistGradientBoosting has no `feature_importances_`),
    and stays in terms of the columns a fraud reviewer actually sees - which is
    the point: a flagged transaction can be explained by naming the inputs that
    moved the score.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE,
        scoring="average_precision", n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1][:n]
    return {
        str(X_test.columns[i]): round(float(result.importances_mean[i]), 4)
        for i in order if result.importances_mean[i] > 0
    }


def model_version(name: str, data_bytes: bytes) -> str:
    """Same data + same code settings -> same version; a retrain on new data -> a new version."""
    short = {"hist_gradient_boosting": "hgb", "random_forest": "rf", "logistic_regression": "lr"}.get(name, name)
    digest = hashlib.sha256(data_bytes + f"{name}|{RANDOM_STATE}|{MODEL_FEATURES}".encode()).hexdigest()
    return f"{short}-{digest[:10]}"


def drift_reference(train_df: pd.DataFrame) -> dict:
    rows = train_df.to_dict("records")
    return {f: reference([feature_value(r, f) for r in rows]) for f in DRIFT_FEATURES}


def at_threshold(y_true, scores, amounts, t: float) -> dict:
    pred = (scores >= t).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "avg_precision": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "total_cost_inr": round(float(total_cost(y_true, pred, amounts)), 2),
    }


def main(data_path: str = DATA_PATH, model_path: str = MODEL_PATH, shadow_path: str = SHADOW_PATH,
         metrics_path: str = METRICS_PATH, plot_path: str = PLOT_PATH) -> dict:
    """Train on `data_path`, write the bundles + metrics. scripts/retrain.py (Day 18) calls this with its own data
    and a candidate folder, so a retrain never overwrites the live model. Returns the metrics."""
    if not os.path.exists(data_path):
        sys.exit(f"missing {data_path} - run: python data/generate_data.py")

    with open(data_path, "rb") as fh:
        data_bytes = fh.read()
    df = pd.read_csv(data_path)
    rest_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["is_chargeback"])
    train_df, val_df = train_test_split(rest_df, test_size=0.25, random_state=RANDOM_STATE,
                                        stratify=rest_df["is_chargeback"])      # 0.25 x 0.8 = 20% of all
    X_train, y_train = to_model_frame(train_df), train_df["is_chargeback"].to_numpy()
    X_val, y_val = to_model_frame(val_df), val_df["is_chargeback"].to_numpy()
    X_test, y_test = to_model_frame(test_df), test_df["is_chargeback"].to_numpy()

    print(f"train {len(train_df):,} ({int(y_train.sum())} cb) | validation {len(val_df):,} "
          f"({int(y_val.sum())} cb) | test {len(test_df):,} ({int(y_test.sum())} cb)")

    results, fitted, sweeps = {}, {}, {}
    for name, clf in candidate_models().items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        val_scores = pipe.predict_proba(X_val)[:, 1]
        results[name] = evaluate_at_half(y_val, val_scores)
        sweeps[name] = sweep(y_val, val_scores, val_df["amount"].to_numpy())
        results[name]["validation_cost_at_threshold_inr"] = sweeps[name][0].total_cost
        fitted[name] = pipe
        print(f"  {name:24s} val cost Rs {sweeps[name][0].total_cost:>9,.0f} at {sweeps[name][0].threshold:.3f}  "
              f"AP={results[name]['avg_precision']:.3f} ROC-AUC={results[name]['roc_auc']:.3f}")

    ranked = sorted(results, key=lambda n: (sweeps[n][0].total_cost, -results[n]["avg_precision"]))
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    drift_ref = drift_reference(train_df)
    bundles = {}
    for role, name in (("champion", ranked[0]), ("shadow", ranked[1])):
        pipe = fitted[name]
        thr, thr_unconstrained, curve = sweeps[name]
        test_scores = pipe.predict_proba(X_test)[:, 1]
        test = at_threshold(y_test, test_scores, test_df["amount"].to_numpy(), thr.threshold)
        bundles[role] = {
            "pipeline": pipe, "model_name": name, "role": role, "threshold": thr.threshold,
            "model_version": model_version(name, data_bytes), "trained_at": trained_at,
            "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "raw_feature_order": MODEL_FEATURES, "drift_reference": drift_ref,
            "costs_inr": {"fn_dispute_fee": DISPUTE_FEE, "fn_ops_cost": FN_OPS_COST, "fp_review_cost": REVIEW_COST},
            "metrics": test, "sklearn_random_state": RANDOM_STATE,
            "_curve": curve, "_thr": thr, "_thr_unconstrained": thr_unconstrained,
        }
        print(f"\n{role}: {name} {bundles[role]['model_version']} threshold {thr.threshold:.4f} (validation) -> "
              f"TEST precision {test['precision']:.3f} recall {test['recall']:.3f} AP {test['avg_precision']:.3f} "
              f"cost Rs {test['total_cost_inr']:,.0f}")

    champ = bundles["champion"]
    print_report(champ["_thr"], champ["_thr_unconstrained"], champ["_curve"])
    save_plot(champ["_curve"], champ["threshold"], plot_path, champ["_thr_unconstrained"].threshold)
    thr, thr_u, test = champ["_thr"], champ["_thr_unconstrained"], champ["metrics"]
    selected_metrics = {
        "threshold": champ["threshold"],
        "threshold_rationale": (
            "cost-minimising threshold on the VALIDATION set, subject to a precision floor of "
            f"{DEFAULT_PRECISION_FLOOR} (manual-review capacity constraint); precision/recall/cost below are on "
            "the untouched TEST set. See threshold_analysis.py."
        ),
        "threshold_unconstrained": {"threshold": thr_u.threshold, "precision_val": thr_u.precision,
                                    "recall_val": thr_u.recall, "total_cost_val_inr": thr_u.total_cost},
        "precision_floor": DEFAULT_PRECISION_FLOOR,
        **{k: test[k] for k in ("precision", "recall", "f1", "avg_precision", "roc_auc", "confusion_matrix")},
        "validation": {"precision": thr.precision, "recall": thr.recall, "total_cost_inr": thr.total_cost},
        "cost": {
            "assumptions_inr": {"fn_dispute_fee": DISPUTE_FEE, "fn_ops_cost": FN_OPS_COST,
                                "fn_amount": "actual transaction amount", "fp_review_cost": REVIEW_COST},
            "total_cost_at_threshold_inr": test["total_cost_inr"],
            "cost_per_txn_inr": round(test["total_cost_inr"] / len(test_df), 2),
            "baseline_cost_approve_all_inr": round(float(total_cost(y_test, np.zeros_like(y_test),
                                                                   test_df["amount"].to_numpy())), 2),
            "baseline_cost_review_all_inr": round(float(total_cost(y_test, np.ones_like(y_test),
                                                                  test_df["amount"].to_numpy())), 2),
        },
    }
    metrics = {
        "dataset": {"rows": len(df), "chargeback_rate": round(float(df["is_chargeback"].mean()), 4),
                    "split": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
                    "test_rows": len(test_df), "test_chargebacks": int(y_test.sum()),
                    "data_sha256": champ["data_sha256"]},
        "selection": "lowest validation cost at the validation-tuned threshold (precision floor), tie-break AP",
        "candidates_validation": results,
        "selected_model": champ["model_name"],
        "model_version": champ["model_version"],
        "trained_at": trained_at,
        "selected": selected_metrics,
        "shadow": {"model_name": bundles["shadow"]["model_name"],
                   "model_version": bundles["shadow"]["model_version"],
                   "threshold": bundles["shadow"]["threshold"], "test": bundles["shadow"]["metrics"]},
        "top_feature_importances": top_feature_importances(champ["pipeline"], X_test, y_test),
    }
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nwrote {metrics_path}")
    for role, path in (("champion", model_path), ("shadow", shadow_path)):
        joblib.dump({k: v for k, v in bundles[role].items() if not k.startswith("_")}, path)
        print(f"wrote {path} ({role})")

    if test["recall"] < 0.75:
        print("\nWARNING: test recall below the 0.75 acceptance bar - iterate on the model.")
    else:
        print(f"\ntest recall {test['recall']:.3f} meets the >=0.75 bar at precision {test['precision']:.3f}")
    return metrics


if __name__ == "__main__":
    main()
