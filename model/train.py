"""
Train the chargeback-risk classifier.

Methodology (baseline-first):
  1. Logistic Regression with balanced class weights  -> interpretable baseline.
  2. Random Forest (balanced)                          -> non-linear, gives
     feature importances.
  3. Histogram Gradient Boosting (balanced)            -> usually the strongest
     on tabular data; still a tree model, so still explainable.

We select the model with the best average precision (area under the
precision-recall curve), because on a 3%-positive problem PR-AUC is far more
informative than accuracy or ROC-AUC.

The operating threshold is then chosen by `threshold_analysis.sweep` to minimise
expected business cost - not left at 0.5.

Output:
  model/model.pkl      - dict: {pipeline, threshold, costs, metrics, features}
  model/metrics.json   - metrics for every candidate + the selected model
  model/threshold_cost_curve.png

Run:
  python model/train.py            # from the repo root
"""

from __future__ import annotations

import json
import os
import sys

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
from features import MODEL_FEATURES, build_pipeline, to_model_frame  # noqa: E402
from threshold_analysis import (  # noqa: E402
    DEFAULT_PRECISION_FLOOR,
    DISPUTE_FEE,
    FN_OPS_COST,
    REVIEW_COST,
    print_report,
    save_plot,
    sweep,
)

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "model/model.pkl"
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


def main() -> None:
    if not os.path.exists(DATA_PATH):
        sys.exit(f"missing {DATA_PATH} - run: python data/generate_data.py")

    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["is_chargeback"]
    )
    X_train, y_train = to_model_frame(train_df), train_df["is_chargeback"].to_numpy()
    X_test, y_test = to_model_frame(test_df), test_df["is_chargeback"].to_numpy()
    amounts_test = test_df["amount"].to_numpy()

    print(f"train rows {len(train_df):,} (chargebacks {int(y_train.sum())}) | "
          f"test rows {len(test_df):,} (chargebacks {int(y_test.sum())})")

    results = {}
    fitted = {}
    for name, clf in candidate_models().items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        scores = pipe.predict_proba(X_test)[:, 1]
        results[name] = evaluate_at_half(y_test, scores)
        fitted[name] = pipe
        print(f"  {name:24s} AP={results[name]['avg_precision']:.3f} "
              f"ROC-AUC={results[name]['roc_auc']:.3f} "
              f"R@0.5={results[name]['recall@0.5']:.3f}")

    best_name = max(results, key=lambda n: results[n]["avg_precision"])
    best_pipe = fitted[best_name]
    print(f"\nselected model: {best_name} (best average precision)")

    # --- cost-based threshold on the test set ---
    best_scores = best_pipe.predict_proba(X_test)[:, 1]
    thr_result, thr_unconstrained, curve = sweep(y_test, best_scores, amounts_test)
    print_report(thr_result, thr_unconstrained, curve)
    save_plot(curve, thr_result.threshold, PLOT_PATH, thr_unconstrained.threshold)

    t = thr_result.threshold
    pred_t = (best_scores >= t).astype(int)
    cm = confusion_matrix(y_test, pred_t)
    selected_metrics = {
        "threshold": t,
        "threshold_rationale": (
            "cost-minimising threshold on the held-out test set, subject to a "
            f"precision floor of {DEFAULT_PRECISION_FLOOR} (manual-review capacity "
            "constraint). See threshold_analysis.py."
        ),
        "threshold_unconstrained": {
            "threshold": thr_unconstrained.threshold,
            "precision": thr_unconstrained.precision,
            "recall": thr_unconstrained.recall,
            "total_cost_inr": thr_unconstrained.total_cost,
        },
        "precision_floor": DEFAULT_PRECISION_FLOOR,
        "precision": round(float(precision_score(y_test, pred_t, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred_t, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred_t, zero_division=0)), 4),
        "avg_precision": results[best_name]["avg_precision"],
        "roc_auc": results[best_name]["roc_auc"],
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "cost": {
            "assumptions_inr": {
                "fn_dispute_fee": DISPUTE_FEE,
                "fn_ops_cost": FN_OPS_COST,
                "fn_amount": "actual transaction amount",
                "fp_review_cost": REVIEW_COST,
            },
            "total_cost_at_threshold_inr": thr_result.total_cost,
            "cost_per_txn_inr": thr_result.cost_per_txn,
            "baseline_cost_approve_all_inr": thr_result.baseline_cost_approve_all,
            "baseline_cost_review_all_inr": thr_result.baseline_cost_review_all,
        },
    }

    metrics = {
        "dataset": {
            "rows": len(df),
            "chargeback_rate": round(float(df["is_chargeback"].mean()), 4),
            "test_rows": len(test_df),
            "test_chargebacks": int(y_test.sum()),
        },
        "candidates_at_threshold_0.5": results,
        "selected_model": best_name,
        "selected": selected_metrics,
        "top_feature_importances": top_feature_importances(best_pipe, X_test, y_test),
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nwrote {METRICS_PATH}")

    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "threshold": t,
        "raw_feature_order": MODEL_FEATURES,
        "costs_inr": {
            "fn_dispute_fee": DISPUTE_FEE,
            "fn_ops_cost": FN_OPS_COST,
            "fp_review_cost": REVIEW_COST,
        },
        "metrics": selected_metrics,
        "sklearn_random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"wrote {MODEL_PATH}")

    if metrics["selected"]["recall"] < 0.75:
        print("\nWARNING: recall below the 0.75 acceptance bar - iterate on the model.")
    else:
        print(f"\nrecall {metrics['selected']['recall']:.3f} meets the >=0.75 bar "
              f"at precision {metrics['selected']['precision']:.3f}")


if __name__ == "__main__":
    main()
