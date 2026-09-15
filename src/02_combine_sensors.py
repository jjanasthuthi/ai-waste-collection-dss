"""
02_combine_sensors.py
---------------------
Merges all corrected fill-sensor CSV files from data/raw/ into a single
consolidated Parquet file for efficient downstream processing.

Expected inputs : CSV files inside data/raw/ (including subdirectories)
                  Each file must contain at least a container-ID column,
                  a timestamp column, and a fill-level column.
Expected output : data/processed/combined_sensor.parquet
                  ~788,595 rows, 484 unique containers

Usage:
    python src/02_combine_sensors.py
"""

import os
import sys
import glob
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "combined_sensor.parquet")

# ── Expected approximate totals (from completed Colab pipeline) ────────────────
EXPECTED_ROWS = 788_595
EXPECTED_CONTAINERS = 484
ROW_TOLERANCE = 0.05          # 5 % tolerance before emitting a warning

# ── Column-name candidates (case-insensitive search) ─────────────────────────
# The dataset may use different capitalisation or spellings across files.
# We normalise to lowercase internal names immediately after loading.
TIMESTAMP_CANDIDATES = ["timestamp", "datetime", "date_time", "time", "date"]
CONTAINER_CANDIDATES = ["container_id", "containerid", "container", "bin_id", "binid", "id"]
FILL_CANDIDATES      = ["fill_level", "fill", "filllevel", "fill_pct", "percentage", "level"]


def find_column(df_cols, candidates):
    """Return the first column name (case-insensitive) that matches a candidate list."""
    lower_cols = {c.lower(): c for c in df_cols}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    return None


def load_csv(path):
    """Load a corrected fill CSV and extract container ID from its filename."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"  WARNING: Could not read {path}: {exc}")
        return None

    if df.empty:
        return None

    # Only use corrected fill-sensor files.
    filename = os.path.basename(path).lower()
    if "_fill_corrected_with_metrics" not in filename:
        return None

    # Extract container ID from filenames such as:
    # Container_10085_fill_Corrected_with_metrics.csv
    import re
    match = re.search(r"container_(\d+)_fill", filename)
    if not match:
        return None

    container_id = int(match.group(1))

    # Actual dataset columns: Date, Fill, Rec, Cidx, Max, Min, Mean
    if "Date" not in df.columns or "Fill" not in df.columns:
        return None

    df = df.rename(columns={
        "Date": "timestamp",
        "Fill": "fill_level"
    })

    df["container_id"] = container_id

    return df
def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Discover all CSV files ────────────────────────────────────────────────
    csv_paths = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True))
    if not csv_paths:
        print(f"ERROR: No CSV files found under {RAW_DIR}")
        print("       Run src/01_download_data.py first.")
        sys.exit(1)

    print(f"Found {len(csv_paths)} CSV file(s) in data/raw/")

    # ── Load and combine ──────────────────────────────────────────────────────
    frames = []
    skipped = 0
    for path in csv_paths:
        df = load_csv(path)
        if df is None:
            skipped += 1
            continue
        frames.append(df)

    if not frames:
        print("ERROR: No valid sensor CSV files could be loaded.")
        sys.exit(1)

    print(f"  Loaded: {len(frames)} file(s)   Skipped: {skipped}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined shape (before dedup): {combined.shape}")

    # ── Parse timestamps ──────────────────────────────────────────────────────
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    n_bad_ts = combined["timestamp"].isna().sum()
    if n_bad_ts > 0:
        print(f"  WARNING: {n_bad_ts} rows had unparseable timestamps — dropping them.")
        combined = combined.dropna(subset=["timestamp"])

    # ── Deduplicate ───────────────────────────────────────────────────────────
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["container_id", "timestamp"])
    after_dedup = len(combined)
    print(f"  Rows removed by deduplication: {before_dedup - after_dedup}")

    # ── Sort chronologically ──────────────────────────────────────────────────
    combined = combined.sort_values(["container_id", "timestamp"]).reset_index(drop=True)

    # ── Validation ───────────────────────────────────────────────────────────
    n_rows = len(combined)
    n_containers = combined["container_id"].nunique()
    print(f"\nValidation:")
    print(f"  Rows:       {n_rows:,}  (expected ≈ {EXPECTED_ROWS:,})")
    print(f"  Containers: {n_containers}  (expected {EXPECTED_CONTAINERS})")

    if abs(n_rows - EXPECTED_ROWS) / EXPECTED_ROWS > ROW_TOLERANCE:
        print(f"  WARNING: Row count {n_rows:,} differs from expected {EXPECTED_ROWS:,} "
              f"by more than {ROW_TOLERANCE*100:.0f}%. "
              "Check that all source files were downloaded correctly.")

    if n_containers != EXPECTED_CONTAINERS:
        print(f"  WARNING: Container count {n_containers} differs from expected "
              f"{EXPECTED_CONTAINERS}.")

    # ── Save ──────────────────────────────────────────────────────────────────
    combined.to_parquet(OUTPUT_PATH, index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"\nSaved: {OUTPUT_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
