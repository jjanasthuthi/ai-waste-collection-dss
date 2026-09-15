import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Smart Waste Collection DSS",
    page_icon="♻️",
    layout="wide"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "data", "processed", "results.csv")

st.title("♻️ AI-Assisted Smart Waste Collection")
st.subheader("Sustainable Waste Collection Decision Support System")

st.markdown("""
**SDG 11 — Sustainable Cities and Communities**

An AI-assisted decision-support prototype that predicts containers
approaching critical fill levels, prioritizes collection needs, and
supports efficient collection planning.
""")

# Load results
df = pd.read_csv(RESULTS)

# Basic metrics
total = len(df)
urgent = (df["collection_priority"] == "URGENT").sum()
early = (df["collection_priority"] == "EARLY_WARNING").sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Containers Analysed", f"{total:,}")
col2.metric("Urgent Containers", urgent)
col3.metric("Early Warning", early)
col4.metric("Model Recall", "90.23%")

st.divider()

# Selected containers
st.header("🚛 Collection Priority")

selected = df[
    df["collection_priority"].isin(["URGENT", "EARLY_WARNING"])
].copy()

selected = selected.sort_values(
    "priority_score",
    ascending=False
)

display_cols = [
    "container_id",
    "fill_pct",
    "critical_fill_prob",
    "priority_score",
    "collection_priority"
]

st.dataframe(
    selected[display_cols],
    use_container_width=True,
    hide_index=True
)

st.divider()

# Model performance
st.header("🤖 AI Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", "93%")
c2.metric("Critical Recall", "90.23%")
c3.metric("Critical F1", "72.52%")
c4.metric("ROC-AUC", "0.9662")

st.info(
    "The Random Forest model predicts whether the next observed "
    "reading is likely to reach the critical fill threshold. "
    "The prediction supports collection prioritization."
)

st.divider()

# Route scenario
st.header("🗺️ Route Optimization Scenario")

route = [
    "Depot",
    "10114",
    "904",
    "340",
    "12710",
    "10170",
    "3292",
    "10196",
    "13670",
    "17908",
    "10127",
    "1851",
    "10799",
    "1857",
    "1298",
    "10176",
    "10175",
    "11827",
    "11553",
    "7143",
    "9419",
    "Depot"
]

st.write(" → ".join(route))

r1, r2, r3 = st.columns(3)

r1.metric("Priority-order baseline", "140.80 km")
r2.metric("Optimized route", "59.36 km")
r3.metric("Distance reduction", "57.84%")

st.divider()

# Sustainability
st.header("🌱 Sustainability Impact — Prototype Estimate")

s1, s2, s3 = st.columns(3)

s1.metric("Estimated Fuel Saved", "24.43 L")
s2.metric("Estimated CO₂ Saved", "65.48 kg")
s3.metric("Estimated CO₂ Reduction", "57.84%")

st.caption(
    "Prototype estimates based on Haversine distance and assumed "
    "fuel consumption of 0.30 L/km and diesel emissions of "
    "2.68 kg CO₂/L. These are not measured real-world savings."
)

st.divider()

# Responsible AI
st.header("🔐 Responsible AI")

st.markdown("""
- **Transparency:** Model performance and decision rules are displayed.
- **Human oversight:** The system supports collection decisions rather than
  replacing operational staff.
- **Fairness:** The model should be monitored across locations and
  container groups for systematic differences.
- **Privacy:** The prototype uses sensor and geographic information rather
  than personal information.
- **Limitations:** Route distances are geographic estimates and do not
  account for traffic, road restrictions, truck capacity, or multiple
  collection vehicles.
""")

st.success(
    "Prototype successfully demonstrates AI-assisted prediction, "
    "prioritization, route planning, and sustainability estimation."
)