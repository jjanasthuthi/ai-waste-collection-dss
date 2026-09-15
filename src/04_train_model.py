"""
04_train_model.py
-----------------
Trains a Random Forest classifier on the engineered sensor dataset using an
exact chronological 80/20 train-test split — matching the completed Colab pipeline.

Model parameters (exact):
    RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

Features : Previous_Fill, Fill_Change, Hour, Day_of_Week, Month, Rolling_Mean_3
Target   : Critical_Next (1 = next fill reading ≥ 80, else 0)
Split    : first 80% of rows (chronological order) → train
           last  20% of rows                        → test

IMPORTANT: The dataset is globally sorted by timestamp before splitting.
No random shuffling is applied. This replicates the Colab pipeline split strategy.

Training sample: The RF is fitted on a random 200,000-row sample drawn from
the chronological training set using random_state=42, matching the original
Colab experiment. The full training set is kept for the cutoff timestamp record;
only the sample is passed to clf.fit().

Input  : data/processed/featured_sensor.parquet
Outputs: models/rf_critical_fill.pkl      — trained model
         models/train_test_split_info.txt — cutoff timestamp and row counts

Usage:
    python src/04_train_model.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

INPUT_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                             "featured_sensor.parquet")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "rf_critical_fill.pkl")
SPLIT_INFO   = os.path.join(MODELS_DIR, "train_test_split_info.txt")

FEATURE_COLS = [
    "Previous_Fill",
    "Fill_Change",
    "Hour",
    "Day_of_Week",
    "Month",
    "Rolling_Mean_3",
]
TARGET_COL = "Critical_Next"

# ── Exact model parameters from the Colab pipeline ───────────────────────────
MODEL_PARAMS = dict(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

TRAIN_RATIO   = 0.80
TRAIN_SAMPLE  = 200_000   # rows drawn from train set for fitting (Colab experiment)
SAMPLE_SEED   = 42        # random_state for the sample draw


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("       Run src/03_feature_engineering.py first.")
        sys.exit(1)

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Rows: {len(df):,}")

    # ── Validate features are present ────────────────────────────────────────
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Drop any remaining NaNs in feature/target columns
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaN features/target.")

    # ── Chronological sort (global) ───────────────────────────────────────────
    # Sort by timestamp across all containers, then apply the 80/20 cutoff.
    # This matches the Colab pipeline's split strategy.
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_RATIO)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    cutoff_ts = df["timestamp"].iloc[split_idx]

    print(f"\nChronological 80/20 split:")
    print(f"  Train rows : {len(train_df):,}  "
          f"(up to {train_df['timestamp'].max()})")
    print(f"  Test rows  : {len(test_df):,}  "
          f"(from {cutoff_ts})")
    print(f"  Cutoff     : {cutoff_ts}")

    # ── Sample the training set ───────────────────────────────────────────────
    # Draw a random 200,000-row sample from the chronological training set,
    # matching the original Colab experiment. If the train set is smaller than
    # TRAIN_SAMPLE, use it in full and emit a warning.
    if len(train_df) > TRAIN_SAMPLE:
        train_sample = train_df.sample(n=TRAIN_SAMPLE, random_state=SAMPLE_SEED)
        print(f"\nSampled {TRAIN_SAMPLE:,} rows from {len(train_df):,} train rows "
              f"(random_state={SAMPLE_SEED}).")
    else:
        train_sample = train_df
        print(f"\nWARNING: Train set ({len(train_df):,} rows) is smaller than "
              f"TRAIN_SAMPLE={TRAIN_SAMPLE:,}. Using full train set.")

    X_train = train_sample[FEATURE_COLS].values
    y_train = train_sample[TARGET_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].values

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining RandomForestClassifier({MODEL_PARAMS}) ...")
    clf = RandomForestClassifier(**MODEL_PARAMS)
    clf.fit(X_train, y_train)
    print("  Training complete.")

    # ── Persist model ─────────────────────────────────────────────────────────
    joblib.dump({
        "model":         clf,
        "feature_cols":  FEATURE_COLS,
        "target_col":    TARGET_COL,
        "cutoff_ts":     cutoff_ts,
        "train_rows":    len(train_df),
        "train_sample":  len(train_sample),
        "test_rows":     len(test_df),
    }, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")

    # ── Save split info for reproducibility ──────────────────────────────────
    with open(SPLIT_INFO, "w") as f:
        f.write("Train/Test Split Information\n")
        f.write("============================\n")
        f.write(f"Total rows         : {len(df):,}\n")
        f.write(f"Train rows (80%)   : {len(train_df):,}\n")
        f.write(f"Train sample used  : {len(train_sample):,}  (random_state={SAMPLE_SEED})\n")
        f.write(f"Test rows  (20%)   : {len(test_df):,}\n")
        f.write(f"Cutoff timestamp   : {cutoff_ts}\n")
        f.write(f"Train end          : {train_df['timestamp'].max()}\n")
        f.write(f"Test start         : {test_df['timestamp'].min()}\n")
        f.write(f"\nModel parameters   : {MODEL_PARAMS}\n")
        f.write(f"Feature columns    : {FEATURE_COLS}\n")
        f.write(f"Target column      : {TARGET_COL}\n")
    print(f"Split info saved: {SPLIT_INFO}")

    # ── Quick in-sample sanity check (on the training sample) ────────────────
    from sklearn.metrics import classification_report
    y_pred_train = clf.predict(X_train)
    print("\nTrain-sample classification report (sanity check):")
    print(classification_report(y_train, y_pred_train, target_names=["Not Critical", "Critical"]))


if __name__ == "__main__":
    main()
