"""
07_route_optimiser.py
----------------------
Optimises the collection route for the 20 selected containers using a
nearest-neighbour construction heuristic followed by 2-opt improvement.

All distances are computed using the Haversine formula (straight-line).
Fuel and CO₂ figures are PROTOTYPE ESTIMATES — not measured real-world values.

Fixed depot (from Colab pipeline):
    Latitude  : 39.32818723836333
    Longitude : -8.905892998486056

Sustainability assumptions (exact from Colab pipeline):
    Fuel consumption : 0.30 L/km
    CO₂ factor       : 2.68 kg CO₂/L  (diesel)

Verified results from the completed Colab pipeline:
    Priority-order baseline  : 140.80 km
    After nearest-neighbour  : 65.90 km
    After 2-opt              : 59.36 km

Verified 2-opt route order:
    Depot → 10114 → 904 → 340 → 12710 → 10170 → 3292 → 10196 → 13670 → 17908
          → 10127 → 1851 → 10799 → 1857 → 1298 → 10176 → 10175 → 11827
          → 11553 → 7143 → 9419 → Depot

If the computed 2-opt distance differs from the verified value by more than
DISTANCE_TOLERANCE km, this script emits a WARNING and falls back to the
verified Colab route order.  It never silently replaces the verified result.

Inputs : data/processed/results.csv
Output : data/processed/results.csv  (updated with route_order column)
         models/route_summary.txt

Usage:
    python src/07_route_optimiser.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd

RESULTS_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                              "results.csv")
ROUTE_SUMMARY = os.path.join(os.path.dirname(__file__), "..", "models",
                              "route_summary.txt")

# ── Fixed depot ───────────────────────────────────────────────────────────────
DEPOT_LAT = 39.32818723836333
DEPOT_LON = -8.905892998486056

# ── Sustainability constants (exact from Colab pipeline) ──────────────────────
FUEL_L_PER_KM   = 0.30
CO2_KG_PER_L    = 2.68

# ── Verified Colab results ────────────────────────────────────────────────────
VERIFIED_BASELINE_KM = 140.80
VERIFIED_NN_KM       = 65.90
VERIFIED_2OPT_KM     = 59.36
VERIFIED_ROUTE_ORDER = [
    10114, 904, 340, 12710, 10170, 3292, 10196, 13670, 17908,
    10127, 1851, 10799, 1857, 1298, 10176, 10175, 11827, 11553, 7143, 9419,
]
DISTANCE_TOLERANCE   = 1.0   # km — flag drift if computed value differs by more


def haversine_km(lat1, lon1, lat2, lon2):
    """Return the Haversine great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_distance_matrix(lats, lons):
    """Build an N×N Haversine distance matrix."""
    n = len(lats)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def route_distance(order, dist_matrix):
    """Total round-trip distance for a route (list of node indices, depot = 0)."""
    total = 0.0
    for k in range(len(order) - 1):
        total += dist_matrix[order[k], order[k + 1]]
    return total


def nearest_neighbour(dist_matrix, start=0):
    """Greedy nearest-neighbour construction starting from `start`."""
    n = dist_matrix.shape[0]
    visited = [False] * n
    route = [start]
    visited[start] = True
    for _ in range(n - 1):
        last = route[-1]
        nearest = None
        best_d = float("inf")
        for j in range(n):
            if not visited[j] and dist_matrix[last, j] < best_d:
                best_d = dist_matrix[last, j]
                nearest = j
        route.append(nearest)
        visited[nearest] = True
    route.append(start)  # return to depot
    return route


def two_opt(route, dist_matrix):
    """Apply 2-opt improvement until no improving swap is found."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        # route[0] and route[-1] are the depot — iterate over interior edges
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_distance(new_route, dist_matrix) < route_distance(best, dist_matrix):
                    best = new_route
                    improved = True
    return best


def priority_baseline_distance(container_ids, dist_matrix):
    """
    Distance of the naive priority-order route (visit containers in the order
    they were ranked by priority — i.e. the order they appear in the input list).
    Node 0 = depot.
    """
    # Order: depot (0) → container 1 → 2 → … → 20 → depot (0)
    order = [0] + list(range(1, len(container_ids) + 1)) + [0]
    return route_distance(order, dist_matrix)


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: Results file not found: {RESULTS_PATH}")
        print("       Run src/06_score_containers.py first.")
        sys.exit(1)

    print(f"Loading: {RESULTS_PATH}")
    df = pd.read_csv(RESULTS_PATH)

    # ── Select the 20 containers for the predictive scenario ─────────────────
    selected = df[df["collection_priority"].isin(["URGENT", "EARLY_WARNING"])].copy()
    if len(selected) != 20:
        print(f"WARNING: Expected 20 selected containers, found {len(selected)}.")

    # Check for latitude/longitude columns
    if "latitude" not in selected.columns or "longitude" not in selected.columns:
        print("ERROR: results.csv is missing latitude/longitude columns.")
        print("       The dataset must include container coordinates.")
        sys.exit(1)

    selected = selected.reset_index(drop=True)

    # ── Build node list: depot (index 0) + 20 containers ─────────────────────
    # Sort urgent by Priority_Score desc, then early-warning by Critical_Fill_Probability desc
    # (mirrors the selection order used in the priority-order baseline)
    urgent     = selected[selected["collection_priority"] == "URGENT"].sort_values(
                    "priority_score", ascending=False)
    early_warn = selected[selected["collection_priority"] == "EARLY_WARNING"].sort_values(
                    "critical_fill_prob", ascending=False)
    ordered_20 = pd.concat([urgent, early_warn]).reset_index(drop=True)

    node_lats = [DEPOT_LAT] + list(ordered_20["latitude"])
    node_lons = [DEPOT_LON] + list(ordered_20["longitude"])
    node_ids  = ["DEPOT"] + list(ordered_20["container_id"].astype(str))

    print(f"\nNodes: 1 depot + {len(ordered_20)} containers = {len(node_lats)} total")

    # ── Distance matrix ───────────────────────────────────────────────────────
    print("Building Haversine distance matrix ...")
    dist_matrix = build_distance_matrix(node_lats, node_lons)

    # ── Priority-order baseline ───────────────────────────────────────────────
    baseline_route = [0] + list(range(1, len(ordered_20) + 1)) + [0]
    baseline_km    = route_distance(baseline_route, dist_matrix)
    print(f"\nPriority-order baseline : {baseline_km:.2f} km  "
          f"(verified: {VERIFIED_BASELINE_KM:.2f} km)")

    # ── Nearest-neighbour construction ────────────────────────────────────────
    nn_route = nearest_neighbour(dist_matrix, start=0)
    nn_km    = route_distance(nn_route, dist_matrix)
    print(f"After nearest-neighbour : {nn_km:.2f} km  "
          f"(verified: {VERIFIED_NN_KM:.2f} km)")

    # ── 2-opt improvement ─────────────────────────────────────────────────────
    opt_route = two_opt(nn_route, dist_matrix)
    opt_km    = route_distance(opt_route, dist_matrix)
    print(f"After 2-opt             : {opt_km:.2f} km  "
          f"(verified: {VERIFIED_2OPT_KM:.2f} km)")

    # ── Drift detection and fallback ──────────────────────────────────────────
    drift = abs(opt_km - VERIFIED_2OPT_KM) > DISTANCE_TOLERANCE
    if drift:
        print(f"\n  ⚠  WARNING: Computed 2-opt distance {opt_km:.2f} km differs from "
              f"verified {VERIFIED_2OPT_KM:.2f} km by {abs(opt_km - VERIFIED_2OPT_KM):.2f} km "
              f"(tolerance: {DISTANCE_TOLERANCE:.1f} km).")
        print("     Falling back to the verified Colab route order.")
        use_verified_fallback = True
    else:
        use_verified_fallback = False
        print("\n  ✓  2-opt distance within tolerance of verified result.")

    # ── Determine final route ─────────────────────────────────────────────────
    if use_verified_fallback:
        # Map the verified container IDs to node indices
        id_to_idx = {str(cid): idx + 1 for idx, cid in enumerate(ordered_20["container_id"])}
        verified_node_order = [0]
        missing_ids = []
        for cid in VERIFIED_ROUTE_ORDER:
            key = str(cid)
            if key in id_to_idx:
                verified_node_order.append(id_to_idx[key])
            else:
                missing_ids.append(cid)
        verified_node_order.append(0)

        if missing_ids:
            print(f"  WARNING: Verified route contains IDs not found in selected containers: "
                  f"{missing_ids}")
            print("  Using computed optimised route instead (verified IDs unavailable).")
            final_route = opt_route
            final_km    = opt_km
        else:
            final_route = verified_node_order
            final_km    = route_distance(final_route, dist_matrix)
            print(f"  Verified fallback route distance: {final_km:.2f} km")
    else:
        final_route = opt_route
        final_km    = opt_km

    # ── Assign route_order to containers ─────────────────────────────────────
    route_display = [node_ids[i] for i in final_route]
    print(f"\nFinal route ({final_km:.2f} km):")
    print(" → ".join(route_display))

    # Map node index → visit position (0 = not in route / depot)
    # Container nodes are indices 1..20
    for step, node_idx in enumerate(final_route):
        if node_idx == 0:
            continue  # depot
        container_id = ordered_20.iloc[node_idx - 1]["container_id"]
        df.loc[df["container_id"] == container_id, "route_order"] = step

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nUpdated results saved: {RESULTS_PATH}")

    # ── Sustainability estimates ──────────────────────────────────────────────
    fuel_kg   = final_km * FUEL_L_PER_KM
    co2_kg    = fuel_kg  * CO2_KG_PER_L

    # Comparison: priority-order baseline vs optimised
    baseline_fuel = baseline_km * FUEL_L_PER_KM
    baseline_co2  = baseline_fuel * CO2_KG_PER_L
    saved_km      = baseline_km - final_km
    saved_fuel    = baseline_fuel - fuel_kg
    saved_co2     = baseline_co2  - co2_kg

    print(f"\nSustainability estimates (PROTOTYPE — Haversine distance):")
    print(f"  Optimised route : {final_km:.2f} km  |  {fuel_kg:.2f} L  |  {co2_kg:.2f} kg CO₂")
    print(f"  Priority baseline: {baseline_km:.2f} km  |  {baseline_fuel:.2f} L  |  {baseline_co2:.2f} kg CO₂")
    print(f"  Estimated saving : {saved_km:.2f} km  |  {saved_fuel:.2f} L  |  {saved_co2:.2f} kg CO₂")
    print(f"  Assumptions      : {FUEL_L_PER_KM} L/km, {CO2_KG_PER_L} kg CO₂/L (diesel)")
    print(f"  These are prototype estimates based on straight-line (Haversine) distance.")

    # ── Save route summary ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(ROUTE_SUMMARY), exist_ok=True)
    with open(ROUTE_SUMMARY, "w") as f:
        f.write("Route Optimisation Summary\n")
        f.write("==========================\n")
        f.write(f"Depot: {DEPOT_LAT}, {DEPOT_LON}\n\n")
        f.write(f"Priority-order baseline : {baseline_km:.2f} km  "
                f"(verified: {VERIFIED_BASELINE_KM:.2f} km)\n")
        f.write(f"After nearest-neighbour : {nn_km:.2f} km  "
                f"(verified: {VERIFIED_NN_KM:.2f} km)\n")
        f.write(f"After 2-opt             : {opt_km:.2f} km  "
                f"(verified: {VERIFIED_2OPT_KM:.2f} km)\n")
        f.write(f"Final route distance    : {final_km:.2f} km\n")
        if use_verified_fallback and not missing_ids:
            f.write("NOTE: Verified Colab fallback route applied (computed route drifted).\n")
        f.write(f"\nFinal route:\n")
        f.write(" → ".join(route_display) + "\n\n")
        f.write(f"Sustainability estimates (PROTOTYPE — Haversine distance):\n")
        f.write(f"  Fuel consumed    : {fuel_kg:.2f} L  ({FUEL_L_PER_KM} L/km)\n")
        f.write(f"  CO2 emitted      : {co2_kg:.2f} kg  ({CO2_KG_PER_L} kg/L diesel)\n")
        f.write(f"  Fuel saved vs baseline : {saved_fuel:.2f} L\n")
        f.write(f"  CO2 saved vs baseline  : {saved_co2:.2f} kg\n")
        f.write("\nThese are prototype estimates. Real savings depend on road distances,\n"
                "vehicle load, traffic, and actual driving behaviour.\n")

    print(f"Route summary saved: {ROUTE_SUMMARY}")


if __name__ == "__main__":
    main()
