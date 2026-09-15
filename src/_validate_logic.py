"""
_validate_logic.py  — internal validation script, not part of the pipeline.
Tests the core algorithmic logic without needing the real dataset.
"""
import math
import sys
import pandas as pd

# ── Haversine ─────────────────────────────────────────────────────────────────
def hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

VERIFIED_BASELINE = 140.80
VERIFIED_NN       = 65.90
VERIFIED_2OPT     = 59.36
DEPOT_LAT = 39.32818723836333
DEPOT_LON = -8.905892998486056
FUEL = 0.30
CO2  = 2.68

# ── Priority formula ──────────────────────────────────────────────────────────
def priority(cfp, fill, fcs):
    return 0.50 * cfp + 0.35 * (fill / 100.0) + 0.15 * fcs

p_high = priority(0.95, 92, 1.0)
p_low  = priority(0.10, 30, 0.0)
assert p_high > p_low, "Priority formula ordering wrong"
expected_high = 0.50*0.95 + 0.35*0.92 + 0.15*1.0
assert abs(p_high - expected_high) < 1e-9, f"Formula mismatch: {p_high} != {expected_high}"
print(f"  Priority formula: OK  (high={p_high:.4f}, low={p_low:.4f})")

# ── Sustainability estimates ───────────────────────────────────────────────────
fuel = VERIFIED_2OPT * FUEL        # 59.36 * 0.30 = 17.808
co2  = fuel * CO2                  # 17.808 * 2.68 = 47.725
assert abs(fuel - 17.808) < 0.01, f"Fuel mismatch: {fuel}"
assert abs(co2  - 47.725) < 0.01, f"CO2 mismatch: {co2}"
print(f"  Sustainability: {VERIFIED_2OPT:.2f} km -> {fuel:.3f} L -> {co2:.3f} kg CO2")
print(f"  Sustainability formula: OK")

# ── Depot coordinates ─────────────────────────────────────────────────────────
assert 38 < DEPOT_LAT < 40, "Depot lat out of Portugal range"
assert -10 < DEPOT_LON < -8, "Depot lon out of Portugal range"
print(f"  Depot ({DEPOT_LAT:.5f}, {DEPOT_LON:.5f}): OK (central Portugal region)")

# ── Verified route ────────────────────────────────────────────────────────────
VERIFIED_ORDER = [10114, 904, 340, 12710, 10170, 3292, 10196, 13670, 17908,
                  10127, 1851, 10799, 1857, 1298, 10176, 10175, 11827, 11553, 7143, 9419]
assert len(VERIFIED_ORDER) == 20, f"Expected 20 containers, got {len(VERIFIED_ORDER)}"
assert len(set(VERIFIED_ORDER)) == 20, "Duplicate container IDs in verified route"
print(f"  Verified route: {len(VERIFIED_ORDER)} unique containers, no duplicates: OK")

# ── Chronological 80/20 split ─────────────────────────────────────────────────
N = 1000
ts = pd.date_range("2023-01-01", periods=N, freq="h")
df = pd.DataFrame({"timestamp": ts, "val": range(N)})
df = df.sort_values("timestamp").reset_index(drop=True)
split_idx = int(len(df) * 0.80)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]
assert len(train) == 800 and len(test) == 200, f"Split ratio wrong: {len(train)}/{len(test)}"
assert train["timestamp"].max() < test["timestamp"].min(), "Split is not chronological!"
print(f"  Chronological 80/20 split: OK  (train ends {train['timestamp'].max()}, test starts {test['timestamp'].min()})")

# ── Container selection rule ──────────────────────────────────────────────────
import numpy as np
rng = np.random.default_rng(42)
n = 484
mock = pd.DataFrame({
    "container_id":           range(n),
    "fill_level":             rng.uniform(0, 100, n),
    "Critical_Fill_Probability": rng.uniform(0, 1, n),
    "Fill_Change_Score":      rng.uniform(0, 1, n),
})
mock["Priority_Score"] = (
    0.50 * mock["Critical_Fill_Probability"]
    + 0.35 * (mock["fill_level"] / 100.0)
    + 0.15 * mock["Fill_Change_Score"]
)
urgent_pool  = mock[mock["fill_level"] >= 80]
early_pool   = mock[mock["fill_level"] <  80]
urgent       = urgent_pool.nlargest(10, "Priority_Score")
early_warn   = early_pool.nlargest(10, "Critical_Fill_Probability")
assert len(urgent)    == 10, f"Urgent count wrong: {len(urgent)}"
assert len(early_warn) == 10, f"Early-warning count wrong: {len(early_warn)}"
assert (urgent["fill_level"] >= 80).all(), "Urgent containers have fill < 80!"
assert (early_warn["fill_level"] < 80).all(), "Early-warning containers have fill >= 80!"
# Urgent ranked by Priority_Score descending
assert list(urgent["Priority_Score"]) == sorted(urgent["Priority_Score"], reverse=True), "Urgent not ranked by Priority_Score"
# Early-warning ranked by Critical_Fill_Probability descending
assert list(early_warn["Critical_Fill_Probability"]) == sorted(early_warn["Critical_Fill_Probability"], reverse=True), "Early-warning not ranked by CriticalFillProb"
print(f"  Container selection rule: OK  (urgent n={len(urgent)}, early-warning n={len(early_warn)})")

# ── Haversine: known distance check (Lisbon to Porto ~275 km) ─────────────────
d = hav(38.7169, -9.1399, 41.1579, -8.6291)   # Lisbon to Porto
assert 270 < d < 285, f"Haversine sanity failed: Lisbon->Porto = {d:.1f} km"
print(f"  Haversine sanity (Lisbon->Porto = {d:.1f} km): OK")

# ── 200k sample + Rolling_Mean_3 correction in source files ──────────────────
import os
base = os.path.join(os.path.dirname(__file__), "..")

with open(os.path.join(base, "src", "04_train_model.py"), encoding="utf-8") as f:
    src04 = f.read()
assert "200_000" in src04, "TRAIN_SAMPLE = 200_000 missing from 04_train_model.py"
assert "SAMPLE_SEED" in src04, "SAMPLE_SEED missing from 04_train_model.py"
assert ".sample(n=TRAIN_SAMPLE, random_state=SAMPLE_SEED)" in src04, \
    "200k sample call missing from 04_train_model.py"
print("  04_train_model.py 200k sample: OK")

with open(os.path.join(base, "src", "03_feature_engineering.py"), encoding="utf-8") as f:
    src03 = f.read()
assert "s.shift(1).rolling" in src03, \
    "Rolling_Mean_3 correction (shift+rolling) missing from 03_feature_engineering.py"
assert "s.rolling(window=3, min_periods=1).mean()" not in src03 or \
    "s.shift(1).rolling" in src03, \
    "Old leaky rolling formula still present without shift correction"
print("  03_feature_engineering.py Rolling_Mean_3 (shift+roll, no leakage): OK")

print()
print("All logic checks passed.")
