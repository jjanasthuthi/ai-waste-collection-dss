"""
03_feature_engineering.py
--------------------------
Constructs the full feature set required by the Random Forest classifier
and the priority-scoring formula from the Colab pipeline.

Input  : data/processed/combined_sensor.parquet
Output : data/processed/featured_sensor.parquet

Features created (per container, sorted chronologically):
  Previous_Fill   – fill level at the preceding reading (shift +1)
  Fill_Change     – current fill minus Previous_Fill
  Hour            – hour of day extracted from timestamp
  Day_of_Week     – integer 0 (Monday) … 6 (Sunday)
  Month           – integer 1 … 12
  Rolling_Mean_3  – mean of the 3 PREVIOUS readings per container (no current-row leakage)
                    Computed as shift(1) then rolling(3, min_periods=1).mean()
                    This means row i sees the mean of rows i-3 … i-1 only.
  Next_Fill       – fill level at the following reading (shift -1) [label source]
  Critical_Next   – 1 if Next_Fill >= 80, else 0  [target variable]

The first row per container (no Previous_Fill) and the last row per container
(no Next_Fill) are dropped. Rolling_Mean_3 is NaN for the first row per
container (shift produces NaN) but that row is already dropped by the
Previous_Fill NaN filter, so no extra rows are lost.

Usage:
    python src/03_feature_engineering.py
"""

import os
import sys
import pandas as pd
import numpy as np

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                           "combined_sensor.parquet")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                           "featured_sensor.parquet")

CRITICAL_FILL_THRESHOLD = 80  # fill level (%) above which Next_Fill is critical


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering steps in-place on a per-container basis.

    The DataFrame must be sorted by [container_id, timestamp] before calling.
    """
    grp = df.groupby("container_id", sort=False)["fill_level"]

    # ── Lag / lead features ───────────────────────────────────────────────────
    df["Previous_Fill"]  = grp.shift(1)          # prior reading
    df["Next_Fill"]      = grp.shift(-1)          # next reading  ← label source

    # ── Derived numeric features ──────────────────────────────────────────────
    df["Fill_Change"]    = df["fill_level"] - df["Previous_Fill"]

    # ── Temporal features ─────────────────────────────────────────────────────
    df["Hour"]           = df["timestamp"].dt.hour
    df["Day_of_Week"]    = df["timestamp"].dt.dayofweek   # 0 = Monday
    df["Month"]          = df["timestamp"].dt.month

    # ── Rolling mean (window = 3, per container) — PRIOR READINGS ONLY ───────
    # shift(1) moves the series back by one position so the rolling window
    # at row i covers rows i-3 … i-1, not i-2 … i. This prevents the current
    # fill_level from appearing in its own Rolling_Mean_3 value.
    df["Rolling_Mean_3"] = (
        grp.transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )

    # ── Target variable ───────────────────────────────────────────────────────
    df["Critical_Next"] = (df["Next_Fill"] >= CRITICAL_FILL_THRESHOLD).astype(int)

    return df


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("       Run src/02_combine_sensors.py first.")
        sys.exit(1)

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows loaded: {len(df):,}")

    # Ensure required columns are present
    for col in ["container_id", "timestamp", "fill_level"]:
        if col not in df.columns:
            print(f"ERROR: Required column '{col}' missing from combined dataset.")
            sys.exit(1)

    # Ensure chronological order
    df = df.sort_values(["container_id", "timestamp"]).reset_index(drop=True)

    # ── Engineer features ─────────────────────────────────────────────────────
    print("Engineering features ...")
    df = engineer_features(df)

    # ── Drop rows without a known Next_Fill (last row per container) ──────────
    before = len(df)
    df = df.dropna(subset=["Next_Fill", "Previous_Fill"]).reset_index(drop=True)
    dropped = before - len(df)
    print(f"  Rows dropped (no Next_Fill or no Previous_Fill): {dropped}")
    print(f"  Final row count: {len(df):,}")

    # ── Class balance report ──────────────────────────────────────────────────
    target_counts = df["Critical_Next"].value_counts().sort_index()
    total = len(df)
    print(f"\nTarget distribution (Critical_Next):")
    for label, count in target_counts.items():
        print(f"  {label}: {count:,}  ({count/total*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PATH, index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"\nSaved: {OUTPUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
