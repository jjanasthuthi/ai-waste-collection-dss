"""
06_score_containers.py
-----------------------
Scores every container using the trained Random Forest and the exact priority
formula from the Colab pipeline. Selects 20 containers for the predictive
collection scenario and writes results.csv.

Priority formula (exact):
    Priority_Score = 0.50 * Critical_Fill_Probability
                   + 0.35 * (Fill / 100)
                   + 0.15 * Fill_Change_Score

Container selection (exact Colab rule):
    Urgent (10)       : containers where latest Fill >= 80,
                        ranked descending by Priority_Score
    Early-warning (10): containers where latest Fill <  80,
                        ranked descending by Critical_Fill_Probability

Fill_Change_Score is Fill_Change clipped to [0, max_observed_fill_change]
and normalised to [0, 1].

Inputs : models/rf_critical_fill.pkl
         data/processed/featured_sensor.parquet
Output : data/processed/results.csv

Usage:
    python src/06_score_containers.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models",
                            "rf_critical_fill.pkl")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                            "featured_sensor.parquet")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                            "results.csv")

N_URGENT        = 10
N_EARLY_WARNING = 10
URGENT_FILL_THRESHOLD = 80  # Fill >= this → eligible for urgent group


def main():
    for path, name in [(MODEL_PATH, "Model"), (DATA_PATH, "Feature dataset")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_PATH}")
    bundle       = joblib.load(MODEL_PATH)
    clf          = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # ── Load full featured dataset ────────────────────────────────────────────
    print(f"Loading data : {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["container_id", "timestamp"]).reset_index(drop=True)

    # ── Extract the LATEST reading per container ──────────────────────────────
    latest = df.groupby("container_id").last().reset_index()
    print(f"  Containers with latest reading: {len(latest)}")

    # Drop rows missing required features
    before = len(latest)
    latest = latest.dropna(subset=feature_cols).reset_index(drop=True)
    if len(latest) < before:
        print(f"  Dropped {before - len(latest)} containers due to missing features.")

    X = latest[feature_cols].values

    # ── Predict probability of next fill being critical ───────────────────────
    proba = clf.predict_proba(X)
    critical_class_idx = list(clf.classes_).index(1)
    latest["Critical_Fill_Probability"] = proba[:, critical_class_idx]

    # ── Compute Fill_Change_Score (normalised to [0, 1]) ─────────────────────
    fc = latest["Fill_Change"].clip(lower=0)
    fc_max = fc.max()
    if fc_max > 0:
        latest["Fill_Change_Score"] = fc / fc_max
    else:
        latest["Fill_Change_Score"] = 0.0

    # ── Exact priority formula ────────────────────────────────────────────────
    latest["Priority_Score"] = (
        0.50 * latest["Critical_Fill_Probability"]
        + 0.35 * (latest["fill_level"] / 100.0)
        + 0.15 * latest["Fill_Change_Score"]
    )

    # ── Container selection (exact Colab rule) ────────────────────────────────
    urgent_pool  = latest[latest["fill_level"] >= URGENT_FILL_THRESHOLD]
    early_pool   = latest[latest["fill_level"] <  URGENT_FILL_THRESHOLD]

    urgent       = urgent_pool.nlargest(N_URGENT,        "Priority_Score")
    early_warn   = early_pool.nlargest(N_EARLY_WARNING,  "Critical_Fill_Probability")

    print(f"\nUrgent containers (fill >= {URGENT_FILL_THRESHOLD}): {len(urgent_pool)} eligible → "
          f"top {len(urgent)} selected")
    print(f"Early-warning containers (fill < {URGENT_FILL_THRESHOLD}): {len(early_pool)} eligible → "
          f"top {len(early_warn)} selected")

    # ── Assign collection_priority labels ────────────────────────────────────
    latest["collection_priority"] = "NORMAL"
    latest.loc[latest["container_id"].isin(early_warn["container_id"]), "collection_priority"] = "EARLY_WARNING"
    latest.loc[latest["container_id"].isin(urgent["container_id"]),     "collection_priority"] = "URGENT"

    # ── Route order placeholder (filled by 07_route_optimiser.py) ────────────
    latest["route_order"] = -1

    # ── Build results DataFrame ───────────────────────────────────────────────
    # Carry latitude/longitude through if present in the combined dataset
    lat_col = "latitude"  if "latitude"  in latest.columns else None
    lon_col = "longitude" if "longitude" in latest.columns else None

    out_cols = ["container_id"]
    if lat_col:
        out_cols.append(lat_col)
    if lon_col:
        out_cols.append(lon_col)
    out_cols += [
        "fill_level", "Fill_Change",
        "Critical_Fill_Probability", "Priority_Score",
        "collection_priority", "route_order",
    ]

    results = latest[out_cols].copy()
    results = results.rename(columns={
        "fill_level":   "fill_pct",
        "Fill_Change":  "fill_change",
        "Critical_Fill_Probability": "critical_fill_prob",
        "Priority_Score":            "priority_score",
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved: {OUTPUT_PATH}  ({len(results)} containers)")

    # ── Summary printout for the selected 20 ─────────────────────────────────
    selected_ids = list(urgent["container_id"]) + list(early_warn["container_id"])
    selected     = results[results["container_id"].isin(selected_ids)]
    print(f"\nSelected 20 containers for predictive collection scenario:")
    print(selected[["container_id", "fill_pct", "critical_fill_prob",
                     "priority_score", "collection_priority"]].to_string(index=False))


if __name__ == "__main__":
    main()
