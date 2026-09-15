"""_validate_dashboard_parsing.py — internal validation."""
import re, sys

route_text = """Route Optimisation Summary
==========================
Depot: 39.32818723836333, -8.905892998486056

Priority-order baseline : 140.80 km  (verified: 140.80 km)
After nearest-neighbour : 65.90 km  (verified: 65.90 km)
After 2-opt             : 59.36 km  (verified: 59.36 km)
Final route distance    : 59.36 km

Final route:
Depot -> 10114 -> 904 -> 340 -> 12710 -> 10170 -> 3292 -> 10196 -> 13670 -> 17908 -> 10127 -> 1851 -> 10799 -> 1857 -> 1298 -> 10176 -> 10175 -> 11827 -> 11553 -> 7143 -> 9419 -> Depot

Sustainability estimates (PROTOTYPE):
  Fuel consumed    : 17.81 L  (0.30 L/km)
  CO2 emitted      : 47.73 kg  (2.68 kg/L diesel)
"""

summary = {}
for line in route_text.splitlines():
    if "Priority-order baseline" in line:
        m = re.search(r"([\d.]+)\s*km", line)
        if m: summary["baseline_km"] = float(m.group(1))
    elif "After nearest-neighbour" in line:
        m = re.search(r"([\d.]+)\s*km", line)
        if m: summary["nn_km"] = float(m.group(1))
    elif "Final route distance" in line:
        m = re.search(r"([\d.]+)\s*km", line)
        if m: summary["opt_km"] = float(m.group(1))
    elif "Fuel consumed" in line:
        m = re.search(r"([\d.]+)\s*L", line)
        if m: summary["fuel_l"] = float(m.group(1))
    elif "CO2 emitted" in line:
        m = re.search(r"([\d.]+)\s*kg", line)
        if m: summary["co2_kg"] = float(m.group(1))
route_match = re.search(r"Final route:\n(.+)", route_text)
if route_match:
    summary["route_str"] = route_match.group(1).strip()

print(f"  baseline_km = {summary.get('baseline_km')}")
print(f"  nn_km       = {summary.get('nn_km')}")
print(f"  opt_km      = {summary.get('opt_km')}")
print(f"  fuel_l      = {summary.get('fuel_l')}")
print(f"  co2_kg      = {summary.get('co2_kg')}")
print(f"  route_str   = {summary.get('route_str','')[:60]}...")

assert summary["baseline_km"] == 140.80
assert summary["nn_km"]       == 65.90
assert summary["opt_km"]      == 59.36
assert summary["fuel_l"]      == 17.81
assert summary["co2_kg"]      == 47.73
print("  Dashboard route summary parsing: OK")
