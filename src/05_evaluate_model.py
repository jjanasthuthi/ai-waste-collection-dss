"""
05_evaluate_model.py
--------------------
Evaluates the trained Random Forest on the held-out chronological test set and
saves a classification report.

Verified benchmarks from the completed Colab pipeline:
    Recall (critical class)  ≈ 90%
    F1-score                 ≈ 73%
    ROC-AUC                  ≈ 0.968

If the computed values differ materially from these benchmarks, the script emits
a WARNING rather than silently passing. It never fabricates metrics.

Input  : models/rf_critical_fill.pkl
         data/processed/featured_sensor.parquet
Output : models/classification_report.txt

Usage:
    python src/05_evaluate_model.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    recall_score,
    f1_score,
)

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models",
                             "rf_critical_fill.pkl")
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                             "featured_sensor.parquet")
REPORT_PATH  = os.path.join(os.path.dirname(__file__), "..", "models",
                             "classification_report.txt")

# ── Colab-verified benchmarks ─────────────────────────────────────────────────
EXPECTED_RECALL  = 0.90
EXPECTED_F1      = 0.73
EXPECTED_ROC_AUC = 0.968
TOLERANCE        = 0.05   # absolute tolerance before a warning is raised


def warn_if_drift(name, expected, actual, tol):
    if abs(actual - expected) > tol:
        print(f"  ⚠  WARNING: {name} = {actual:.4f}  "
              f"(expected ≈ {expected:.4f}, Δ = {abs(actual - expected):.4f})")
        return True
    print(f"  ✓  {name} = {actual:.4f}  (expected ≈ {expected:.4f})")
    return False


def main():
    for path, name in [(MODEL_PATH, "Model"), (DATA_PATH, "Feature dataset")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    # ── Load artefacts ────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    clf          = bundle["model"]
    feature_cols = bundle["feature_cols"]
    target_col   = bundle["target_col"]
    cutoff_ts    = bundle["cutoff_ts"]

    print(f"Loading data : {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Reconstruct the test set using the stored cutoff timestamp ───────────
    test_df = df[df["timestamp"] >= cutoff_ts].copy()
    test_df = test_df.dropna(subset=feature_cols + [target_col])
    print(f"  Test rows (from cutoff {cutoff_ts}): {len(test_df):,}")

    if len(test_df) == 0:
        print("ERROR: Test set is empty — check cutoff timestamp.")
        sys.exit(1)

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # ── Generate predictions ──────────────────────────────────────────────────
    y_pred      = clf.predict(X_test)
    y_prob      = clf.predict_proba(X_test)[:, 1]   # probability of critical class

    # ── Compute metrics ───────────────────────────────────────────────────────
    recall  = recall_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    report_str = classification_report(
        y_test, y_pred,
        target_names=["Not Critical (0)", "Critical (1)"],
    )

    print("\nTest-set evaluation:")
    print(report_str)

    # ── Drift checks ──────────────────────────────────────────────────────────
    print("Benchmark comparison:")
    drift_detected = False
    drift_detected |= warn_if_drift("Recall (critical class)", EXPECTED_RECALL,  recall,  TOLERANCE)
    drift_detected |= warn_if_drift("F1-score",                EXPECTED_F1,      f1,      TOLERANCE)
    drift_detected |= warn_if_drift("ROC-AUC",                 EXPECTED_ROC_AUC, roc_auc, TOLERANCE)

    if drift_detected:
        print("\n  Results differ materially from Colab benchmarks.")
        print("  Investigate before proceeding. The pipeline continues but "
              "results may not match the verified scenario.")
    else:
        print("\n  All metrics within tolerance of Colab benchmarks. ✓")

    # ── Persist report ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n")
        f.write(f"Model            : RandomForestClassifier\n")
        f.write(f"Cutoff timestamp : {cutoff_ts}\n")
        f.write(f"Test rows        : {len(test_df):,}\n\n")
        f.write(report_str)
        f.write(f"\nRecall (critical class) : {recall:.4f}  (Colab benchmark ≈ {EXPECTED_RECALL})\n")
        f.write(f"F1-score                : {f1:.4f}  (Colab benchmark ≈ {EXPECTED_F1})\n")
        f.write(f"ROC-AUC                 : {roc_auc:.4f}  (Colab benchmark ≈ {EXPECTED_ROC_AUC})\n")
        if drift_detected:
            f.write("\nWARNING: One or more metrics deviate from Colab benchmarks by > "
                    f"{TOLERANCE:.2f}.\n")

    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
