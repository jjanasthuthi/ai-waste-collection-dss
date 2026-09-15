"""
_validate_features.py  — internal validation, not part of the pipeline.
Tests feature engineering logic against known expected values.

Rolling_Mean_3 uses shift(1) then rolling(3), so row i sees the mean of
PREVIOUS readings only (rows i-3 … i-1). This matches 03_feature_engineering.py.
"""
import pandas as pd
import numpy as np

# ── Build a tiny 2-container, 4-reading mock dataset ─────────────────────────
data = {
    "container_id": ["A","A","A","A","B","B","B","B"],
    "timestamp": pd.to_datetime([
        "2023-01-01 08:00","2023-01-01 10:00","2023-01-01 12:00","2023-01-01 14:00",
        "2023-01-01 09:00","2023-01-01 11:00","2023-01-01 13:00","2023-01-01 15:00",
    ]),
    "fill_level": [20, 35, 55, 82, 10, 50, 70, 85],
}
df = pd.DataFrame(data).sort_values(["container_id","timestamp"]).reset_index(drop=True)

# ── Apply feature engineering exactly as in 03_feature_engineering.py ─────────
grp = df.groupby("container_id", sort=False)["fill_level"]

df["Previous_Fill"] = grp.shift(1)
df["Next_Fill"]     = grp.shift(-1)
df["Fill_Change"]   = df["fill_level"] - df["Previous_Fill"]
df["Hour"]          = df["timestamp"].dt.hour
df["Day_of_Week"]   = df["timestamp"].dt.dayofweek
df["Month"]         = df["timestamp"].dt.month

# CORRECTED: shift(1) before rolling so row i uses rows i-3…i-1 only (no current row)
df["Rolling_Mean_3"] = grp.transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
df["Critical_Next"] = (df["Next_Fill"] >= 80).astype(int)

# Drop rows without Next_Fill or Previous_Fill (first/last per container)
df_feat = df.dropna(subset=["Next_Fill","Previous_Fill"]).reset_index(drop=True)

# ── Check container A (fill_level sequence: 20, 35, 55, 82) ───────────────────
a = df_feat[df_feat["container_id"]=="A"].reset_index(drop=True)
# After drop: row 0 (fill=20) dropped (no Previous_Fill),
#             row 3 (fill=82) dropped (no Next_Fill).
# Remaining: row 1 (fill=35) and row 2 (fill=55).
assert len(a) == 2, f"Container A: expected 2 rows, got {len(a)}"

# Row 0 of df_feat for A: original fill=35, prev=20, next=55
assert a.loc[0,"fill_level"]    == 35, "A row 0 fill wrong"
assert a.loc[0,"Previous_Fill"] == 20, "A row 0 Previous_Fill wrong"
assert a.loc[0,"Next_Fill"]     == 55, "A row 0 Next_Fill wrong"
assert a.loc[0,"Fill_Change"]   == 15, "A row 0 Fill_Change wrong"
assert a.loc[0,"Critical_Next"] == 0,  "A row 0 Critical_Next wrong (55 < 80)"

# Row 1 of df_feat for A: original fill=55, prev=35, next=82 => Critical_Next=1
assert a.loc[1,"fill_level"]    == 55, "A row 1 fill wrong"
assert a.loc[1,"Critical_Next"] == 1,  "A row 1 Critical_Next wrong (82 >= 80)"
assert a.loc[1,"Hour"]          == 12, "A row 1 Hour wrong"
assert a.loc[1,"Month"]         == 1,  "A row 1 Month wrong"
assert a.loc[1,"Day_of_Week"]   == 6,  "A row 1 Day_of_Week wrong (Sunday=6)"
print("  Container A feature checks: OK")

# ── Rolling_Mean_3 check (corrected: prior readings only) ────────────────────
# Container B fill_level sequence: 10, 50, 70, 85
#
#   row 0 (fill=10): shift(1) → NaN, rolling of [NaN]      → NaN
#   row 1 (fill=50): shift(1) → 10,  rolling of [10]        → 10.0
#   row 2 (fill=70): shift(1) → 50,  rolling of [10, 50]    → 30.0
#   row 3 (fill=85): shift(1) → 70,  rolling of [10, 50, 70]→ 43.33
#
# row 0 is NaN (dropped by Previous_Fill filter) so we only check rows 1–3.
b_all = df[df["container_id"]=="B"].reset_index(drop=True)

# Row 0: NaN (no prior value after shift)
assert pd.isna(b_all.loc[0,"Rolling_Mean_3"]), \
    f"B row 0 Rolling_Mean_3 should be NaN, got {b_all.loc[0,'Rolling_Mean_3']}"

# Row 1: mean of [10] = 10.0
assert abs(b_all.loc[1,"Rolling_Mean_3"] - 10.0) < 0.001, \
    f"B row 1 Rolling_Mean_3 wrong: {b_all.loc[1,'Rolling_Mean_3']}"

# Row 2: mean of [10, 50] = 30.0
assert abs(b_all.loc[2,"Rolling_Mean_3"] - 30.0) < 0.001, \
    f"B row 2 Rolling_Mean_3 wrong: {b_all.loc[2,'Rolling_Mean_3']}"

# Row 3: mean of [10, 50, 70] = 43.333…
assert abs(b_all.loc[3,"Rolling_Mean_3"] - 130/3) < 0.001, \
    f"B row 3 Rolling_Mean_3 wrong: {b_all.loc[3,'Rolling_Mean_3']}"

print("  Rolling_Mean_3 checks: OK  (shift+roll — prior readings only, no current-row leakage)")

# ── Confirm current fill_level does NOT appear in Rolling_Mean_3 ──────────────
# For row 3 of B (fill=85): if current row were included, window [50,70,85] → 68.33
# Correct value is 43.33 (window [10,50,70]). Verify it is NOT 68.33.
assert abs(b_all.loc[3,"Rolling_Mean_3"] - 68.33) > 0.1, \
    "LEAKAGE DETECTED: current fill_level (85) appears to be included in Rolling_Mean_3"
print("  No current-row leakage in Rolling_Mean_3: confirmed")

# ── Critical_Next threshold ───────────────────────────────────────────────────
assert (df_feat["Critical_Next"] == (df_feat["Next_Fill"] >= 80).astype(int)).all(), \
    "Critical_Next does not match threshold rule"
print("  Critical_Next threshold (>= 80): OK")

# ── No NaN in feature columns after drop ─────────────────────────────────────
feature_cols = ["Previous_Fill","Fill_Change","Hour","Day_of_Week","Month","Rolling_Mean_3"]
assert not df_feat[feature_cols].isna().any().any(), \
    "NaN found in feature columns after drop — Rolling_Mean_3 NaN not fully removed"
print("  No NaN in features after drop: OK")

# ── Row count unchanged by rolling correction ─────────────────────────────────
# The shift NaN on row 0 per container is already removed by the Previous_Fill
# NaN filter, so the corrected rolling should not drop any additional rows.
df_old = df.copy()
df_old["Rolling_Mean_3_old"] = grp.transform(
    lambda s: s.rolling(window=3, min_periods=1).mean()
)
df_old_feat = df_old.dropna(subset=["Next_Fill","Previous_Fill"]).reset_index(drop=True)
assert len(df_feat) == len(df_old_feat), \
    (f"Corrected rolling drops different number of rows ({len(df_feat)}) "
     f"than old formula ({len(df_old_feat)})")
print(f"  Row count unchanged by rolling correction: {len(df_feat)} rows: OK")

print()
print("Feature engineering checks passed.")
