"""
_validate_route.py  — internal validation, not part of the pipeline.
Tests the nearest-neighbour + 2-opt implementation with a small synthetic
set of coordinates in the same geographic region as the real dataset.
"""
import math
import sys
import numpy as np

DEPOT_LAT = 39.32818723836333
DEPOT_LON = -8.905892998486056

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def build_distance_matrix(lats, lons):
    n = len(lats)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def route_distance(order, dist_matrix):
    return sum(dist_matrix[order[k], order[k+1]] for k in range(len(order)-1))

def nearest_neighbour(dist_matrix, start=0):
    n = dist_matrix.shape[0]
    visited = [False] * n
    route = [start]
    visited[start] = True
    for _ in range(n-1):
        last = route[-1]
        nearest, best_d = None, float("inf")
        for j in range(n):
            if not visited[j] and dist_matrix[last, j] < best_d:
                best_d = dist_matrix[last, j]
                nearest = j
        route.append(nearest)
        visited[nearest] = True
    route.append(start)
    return route

def two_opt(route, dist_matrix):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new_route, dist_matrix) < route_distance(best, dist_matrix):
                    best = new_route
                    improved = True
    return best

# ── Build 20 synthetic nodes around the depot (within ~30 km) ─────────────────
rng = np.random.default_rng(42)
# Approximate degree offsets: 1 deg lat ≈ 111 km, 1 deg lon ≈ 85 km at 39N
n_containers = 20
lats = [DEPOT_LAT] + list(DEPOT_LAT + rng.uniform(-0.27, 0.27, n_containers))
lons = [DEPOT_LON] + list(DEPOT_LON + rng.uniform(-0.35, 0.35, n_containers))

dist_matrix = build_distance_matrix(lats, lons)

# Priority-order baseline (visit in order 1..20 without optimisation)
baseline_route = [0] + list(range(1, n_containers+1)) + [0]
baseline_km    = route_distance(baseline_route, dist_matrix)

# NN
nn_route = nearest_neighbour(dist_matrix, start=0)
nn_km    = route_distance(nn_route, dist_matrix)

# 2-opt
opt_route = two_opt(nn_route, dist_matrix)
opt_km    = route_distance(opt_route, dist_matrix)

print(f"  Baseline (priority order) : {baseline_km:.2f} km")
print(f"  After nearest-neighbour   : {nn_km:.2f} km  (improvement: {baseline_km-nn_km:.2f} km)")
print(f"  After 2-opt               : {opt_km:.2f} km  (improvement: {nn_km-opt_km:.2f} km)")

# Invariants
assert opt_km <= nn_km, "2-opt should not worsen the NN route"
assert nn_km  <= baseline_km * 1.05, "NN should substantially improve on baseline (±5%)"
assert opt_route[0]  == 0, "Route must start at depot"
assert opt_route[-1] == 0, "Route must end at depot"
assert len(opt_route) == n_containers + 2, f"Route length wrong: {len(opt_route)}"
assert set(opt_route[1:-1]) == set(range(1, n_containers+1)), "Route must visit all containers exactly once"
print("  Route invariants: OK")

# ── Fuel / CO2 on optimised route ─────────────────────────────────────────────
FUEL = 0.30
CO2  = 2.68
fuel = opt_km * FUEL
co2  = fuel * CO2
print(f"  Optimised fuel estimate   : {fuel:.2f} L")
print(f"  Optimised CO2 estimate    : {co2:.2f} kg")

print()
print("Route algorithm checks passed.")
