"""
_validate_e2e.py  — end-to-end integration test.

Runs the full pipeline (combine → features → train → evaluate → score → route)
on a synthetic dataset shaped identically to the real one:
  - 484 containers
  - ~1,600 readings (small for speed)
  - Timestamps, fill levels, coordinates in the same geographic region

Does NOT use real data. Verifies that every pipeline stage runs without error,
produces the correct output shapes, and that the container selection and route
logic behave exactly as specified.
"""
import os, sys, math, tempfile, joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score

DEPOT_LAT = 39.32818723836333
DEPOT_LON = -8.905892998486056
FUEL = 0.30
CO2  = 2.68
FEATURE_COLS = ["Previous_Fill","Fill_Change","Hour","Day_of_Week","Month","Rolling_Mean_3"]
TARGET_COL   = "Critical_Next"

# ── 1. Synthesise raw sensor data ────────────────────────────────────────────
print("Step 1: Synthesise raw sensor data")
rng = np.random.default_rng(7)
N_CONTAINERS = 484
READINGS_PER = 32   # small for speed; real dataset has ~1,630 per container

records = []
for cid in range(1, N_CONTAINERS + 1):
    fill = 10.0
    # Stagger each container's start by a unique offset to avoid shared timestamps
    # at the train/test split boundary (real dataset has irregular per-container timing)
    ts = pd.Timestamp("2022-01-01") + pd.Timedelta(minutes=int(cid) * 37)
    lat = DEPOT_LAT + rng.uniform(-0.5, 0.5)
    lon = DEPOT_LON + rng.uniform(-0.7, 0.7)
    for _ in range(READINGS_PER):
        ts += pd.Timedelta(hours=rng.integers(2, 6))
        fill = min(100, max(0, fill + rng.uniform(-2, 8)))
        records.append({
            "container_id": cid,
            "timestamp":    ts,
            "fill_level":   round(fill, 1),
            "latitude":     round(lat, 6),
            "longitude":    round(lon, 6),
        })

df_raw = pd.DataFrame(records)
df_raw = df_raw.sort_values(["container_id","timestamp"]).reset_index(drop=True)
assert df_raw["container_id"].nunique() == N_CONTAINERS
assert len(df_raw) == N_CONTAINERS * READINGS_PER
print(f"  {len(df_raw):,} rows, {df_raw['container_id'].nunique()} containers: OK")

# ── 2. Feature engineering ───────────────────────────────────────────────────
print("Step 2: Feature engineering")
grp = df_raw.groupby("container_id", sort=False)["fill_level"]
df_raw["Previous_Fill"]  = grp.shift(1)
df_raw["Next_Fill"]      = grp.shift(-1)
df_raw["Fill_Change"]    = df_raw["fill_level"] - df_raw["Previous_Fill"]
df_raw["Hour"]           = df_raw["timestamp"].dt.hour
df_raw["Day_of_Week"]    = df_raw["timestamp"].dt.dayofweek
df_raw["Month"]          = df_raw["timestamp"].dt.month
# CORRECTED: shift(1) before rolling — row i sees mean of rows i-3…i-1 only
df_raw["Rolling_Mean_3"] = grp.transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
df_raw["Critical_Next"]  = (df_raw["Next_Fill"] >= 80).astype(int)
df_feat = df_raw.dropna(subset=["Next_Fill","Previous_Fill"]).reset_index(drop=True)
assert not df_feat[FEATURE_COLS + [TARGET_COL]].isna().any().any(), "NaN in features"
n_critical = df_feat[TARGET_COL].sum()
print(f"  {len(df_feat):,} rows after drop, {n_critical} critical ({n_critical/len(df_feat)*100:.1f}%): OK")

# ── 3. Train/test split (chronological) ─────────────────────────────────────
print("Step 3: Chronological 80/20 split")
df_sorted  = df_feat.sort_values("timestamp").reset_index(drop=True)
split_idx  = int(len(df_sorted) * 0.80)
train_df   = df_sorted.iloc[:split_idx]
test_df    = df_sorted.iloc[split_idx:]
cutoff_ts  = df_sorted["timestamp"].iloc[split_idx]
# The split is positional (iloc), not timestamp-based, so test rows may share
# the exact cutoff timestamp with the last train row. The real invariant is:
# no test row appears before the last train row in global time order.
assert train_df["timestamp"].max() <= test_df["timestamp"].min(), "Split not chronological!"
print(f"  Train {len(train_df):,} / Test {len(test_df):,}, cutoff {cutoff_ts}: OK")

# ── 4. Train model ───────────────────────────────────────────────────────────
print("Step 4: Train RandomForestClassifier (exact params + 200k sample)")
TRAIN_SAMPLE = 200_000
SAMPLE_SEED  = 42

# Sample 200,000 rows from the chronological training set (Colab experiment match).
# The synthetic train set here is ~11k rows, so the full set is used with a warning.
if len(train_df) > TRAIN_SAMPLE:
    train_sample = train_df.sample(n=TRAIN_SAMPLE, random_state=SAMPLE_SEED)
    print(f"  Sampled {TRAIN_SAMPLE:,} rows from {len(train_df):,} train rows.")
else:
    train_sample = train_df
    print(f"  WARNING: Train set ({len(train_df):,}) < {TRAIN_SAMPLE:,}. Using full set.")

X_train = train_sample[FEATURE_COLS].values
y_train = train_sample[TARGET_COL].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET_COL].values

clf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)
print(f"  Trained on {len(X_train):,} samples: OK")

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
print("Step 5: Evaluate on test set")
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
recall  = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"  Recall={recall:.4f}  F1={f1:.4f}  ROC-AUC={roc_auc:.4f}")
assert 0 <= recall <= 1 and 0 <= f1 <= 1 and 0 <= roc_auc <= 1, "Metrics out of range"
print(f"  Metrics in range [0,1]: OK")

# ── 6. Score containers (latest reading per container) ──────────────────────
print("Step 6: Score all 484 containers")
# df_raw has all columns (incl. latitude/longitude) and the engineered features
# because feature engineering was done in-place on df_raw above.
# df_feat is the NaN-dropped view; we take the latest reading from df_raw
# that is also present in df_feat (i.e. has valid features).
latest = df_raw.dropna(subset=FEATURE_COLS + [TARGET_COL]).groupby("container_id").last().reset_index()

X_latest = latest[FEATURE_COLS].values
proba = clf.predict_proba(X_latest)
latest["Critical_Fill_Probability"] = proba[:, list(clf.classes_).index(1)]

fc = latest["Fill_Change"].clip(lower=0)
fc_max = fc.max()
latest["Fill_Change_Score"] = fc / fc_max if fc_max > 0 else 0.0

latest["Priority_Score"] = (
    0.50 * latest["Critical_Fill_Probability"]
    + 0.35 * (latest["fill_level"] / 100.0)
    + 0.15 * latest["Fill_Change_Score"]
)

# Exact Colab selection rule
urgent_pool = latest[latest["fill_level"] >= 80]
early_pool  = latest[latest["fill_level"] <  80]
urgent      = urgent_pool.nlargest(10, "Priority_Score")
early_warn  = early_pool.nlargest(10, "Critical_Fill_Probability")

print(f"  Urgent pool: {len(urgent_pool)}, selected: {len(urgent)}")
print(f"  Early-warning pool: {len(early_pool)}, selected: {len(early_warn)}")
assert len(urgent)    == 10, f"Expected 10 urgent, got {len(urgent)}"
assert len(early_warn) == 10, f"Expected 10 early-warning, got {len(early_warn)}"
assert (urgent["fill_level"] >= 80).all(),    "Urgent container has fill < 80"
assert (early_warn["fill_level"] < 80).all(), "Early-warning container has fill >= 80"
# Ranking checks
ps = list(urgent["Priority_Score"])
assert ps == sorted(ps, reverse=True), "Urgent not ranked by Priority_Score desc"
cfp = list(early_warn["Critical_Fill_Probability"])
assert cfp == sorted(cfp, reverse=True), "Early-warning not ranked by CritFillProb desc"
print(f"  Selection rule and ranking: OK")

# ── 7. Route optimisation ────────────────────────────────────────────────────
print("Step 7: Route optimisation (NN + 2-opt) for 20 containers")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def build_dist(lats, lons):
    n = len(lats)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            v = haversine_km(lats[i], lons[i], lats[j], lons[j])
            d[i,j] = d[j,i] = v
    return d

def route_dist(order, dm):
    return sum(dm[order[k], order[k+1]] for k in range(len(order)-1))

def nn(dm, start=0):
    n = dm.shape[0]; vis = [False]*n; r = [start]; vis[start] = True
    for _ in range(n-1):
        last = r[-1]; best, bd = None, float("inf")
        for j in range(n):
            if not vis[j] and dm[last,j] < bd: bd = dm[last,j]; best = j
        r.append(best); vis[best] = True
    r.append(start); return r

def two_opt(route, dm):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                nr = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_dist(nr, dm) < route_dist(best, dm):
                    best = nr; improved = True
    return best

selected_20 = pd.concat([urgent, early_warn]).reset_index(drop=True)
node_lats = [DEPOT_LAT] + list(selected_20["latitude"])
node_lons = [DEPOT_LON] + list(selected_20["longitude"])
dm = build_dist(node_lats, node_lons)

baseline_route = [0] + list(range(1, 21)) + [0]
baseline_km = route_dist(baseline_route, dm)
nn_route = nn(dm)
nn_km    = route_dist(nn_route, dm)
opt_route = two_opt(nn_route, dm)
opt_km    = route_dist(opt_route, dm)

print(f"  Baseline (priority order) : {baseline_km:.2f} km")
print(f"  After nearest-neighbour   : {nn_km:.2f} km")
print(f"  After 2-opt               : {opt_km:.2f} km")

assert opt_km <= nn_km, "2-opt worsened NN route"
assert nn_km  <  baseline_km, "NN did not improve on baseline"
assert opt_route[0] == 0 and opt_route[-1] == 0, "Route does not start/end at depot"
assert set(opt_route[1:-1]) == set(range(1, 21)), "Route misses or repeats a container"
print(f"  Route invariants: OK")

# Sustainability
fuel_l = opt_km * FUEL
co2_kg = fuel_l * CO2
print(f"  Fuel: {fuel_l:.2f} L  |  CO2: {co2_kg:.2f} kg  (prototype estimates)")

# ── 8. Persist and reload (joblib round-trip) ────────────────────────────────
print("Step 8: Model joblib serialise/reload")
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "m.pkl")
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS,
                 "target_col": TARGET_COL, "cutoff_ts": cutoff_ts,
                 "train_rows": len(train_df), "test_rows": len(test_df)}, p)
    b2 = joblib.load(p)
    assert (b2["model"].predict(X_test[:5]) == clf.predict(X_test[:5])).all()
    print("  Serialise/reload: OK")

print()
print("END-TO-END INTEGRATION TEST PASSED")
print(f"  Pipeline stages   : 8")
print(f"  Containers        : {N_CONTAINERS}")
print(f"  Total readings    : {len(df_raw):,}")
print(f"  Features rows     : {len(df_feat):,}")
print(f"  RF params verified: n_estimators=100, max_depth=15, min_samples_leaf=5,")
print(f"                      class_weight=balanced, random_state=42, n_jobs=-1")
print(f"  Selected          : 10 urgent (fill>=80) + 10 early-warning (fill<80)")
print(f"  Route nodes       : 1 depot + 20 containers")
print(f"  Optimised route   : {opt_km:.2f} km  ({nn_km:.2f} km after NN, {baseline_km:.2f} km baseline)")
print(f"  CO2 estimate      : {co2_kg:.2f} kg  (prototype, Haversine-based)")
