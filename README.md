# AI-Assisted Sustainable Smart Waste Collection Decision Support System

An AI-assisted decision-support system for smarter and more sustainable urban waste collection. The system uses IoT smart-bin sensor data to predict critical waste-fill conditions, prioritize containers for collection, and support optimized collection routing.

## SDG Alignment

**Primary SDG: SDG 11 – Sustainable Cities and Communities**

The project supports sustainable urban waste management by helping waste-management teams make data-driven collection decisions and reduce unnecessary travel.

## Problem Statement

Urban waste collection is often based on fixed schedules or manual decisions. This can result in collection vehicles visiting containers that do not require immediate service while other containers approach overflow conditions.

This project addresses the problem using historical IoT smart-bin sensor readings to predict whether a container is likely to reach a critical fill level in its next observation. The predictions are then used to prioritize containers and support more efficient collection route planning.

## Solution Overview

The project follows an end-to-end AI-assisted workflow:

1. Collect and combine historical IoT smart-bin sensor readings.
2. Clean and preprocess the sensor data.
3. Perform feature engineering using historical fill behaviour.
4. Train a Random Forest classification model.
5. Predict the probability of a container reaching a critical fill level.
6. Calculate collection priority using current fill level, predicted risk and recent fill behaviour.
7. Identify urgent and early-warning containers.
8. Optimize the collection route using Nearest Neighbour and 2-opt optimization.
9. Estimate potential fuel and CO₂ savings.
10. Present the results through an interactive Streamlit dashboard.

## Dataset

The project uses the **Ultra Sound Garbage Bin Sensor Reading Records** dataset containing IoT sensor readings from smart waste containers.

The processed dataset contains:

- **484 waste containers**
- **788K+ sensor observations**
- Fill-level measurements
- Container information
- Geographic coordinates
- Historical sensor behaviour

Dataset source:

https://zenodo.org/records/14988663

## AI / Machine Learning

A **Random Forest Classifier** is used to predict whether the next fill-level observation will reach the critical threshold.

### Important Features

The model uses features including:

- Previous fill level
- Fill-level change
- Time of day
- Day of week
- Month
- Leakage-safe rolling statistics

The critical fill threshold used in the prototype is **80%**.

### Model Performance

The model was evaluated using a chronological 80/20 train-test split.

| Metric | Result |
|---|---:|
| Accuracy | 93% |
| Critical-class Recall | 90.23% |
| Critical-class F1-score | 72.52% |
| ROC-AUC | 0.9662 |

The high recall for the critical class is particularly important because missing a container that is approaching overflow is more undesirable than unnecessarily flagging a container for attention.

## Collection Priority

The system classifies containers into:

- **URGENT** – containers requiring immediate attention
- **EARLY WARNING** – containers showing a high likelihood of reaching the critical level

A priority score combines the predicted critical-fill probability with the current fill level and recent fill behaviour.

## Route Optimization

The system supports collection route optimization using:

- Haversine geographic distance
- Nearest Neighbour route construction
- 2-opt route improvement

In a demonstrated 20-container prototype scenario:

| Route | Distance |
|---|---:|
| Priority-order baseline | 140.80 km |
| Nearest Neighbour | 65.90 km |
| 2-opt optimized route | 59.36 km |

The optimized route represents a **57.84% reduction in geographic route distance** compared with the priority-order baseline.

> Note: These are geographic prototype estimates based on Haversine distance, not real road-network distances. Traffic, road restrictions, vehicle capacity and real-world routing constraints are not included.

## Sustainability Estimate

Using prototype assumptions of:

- Fuel consumption: 0.30 L/km
- Diesel emissions factor: 2.68 kg CO₂/L

The demonstrated routing scenario estimates:

- **24.43 L of fuel saved**
- **65.48 kg CO₂ avoided**
- **57.84% estimated reduction**

These values are **prototype estimates**, not measured real-world savings.

## Dashboard

The project includes an interactive **Streamlit dashboard** showing:

- Number of containers
- Urgent containers
- Early-warning containers
- Model performance
- Collection priorities
- Route optimization results
- Sustainability estimates
- Responsible AI considerations

### Dashboard Preview

![Dashboard Overview](screenshots/dashboard_overview.png)

![Route and Sustainability](screenshots/route_sustainability.png)

## Responsible AI

Responsible AI considerations were included throughout the project:

### Transparency
The dashboard communicates model performance, prediction outputs and limitations.

### Human Oversight
The system is designed as a decision-support tool. Final collection decisions remain with human waste-management teams.

### Fairness
The system should be evaluated across different locations, container types and operating conditions before real-world deployment.

### Privacy
The project uses sensor and container-level operational data rather than personal user information.

### Limitations
The prototype uses historical data and geographic distance calculations. Real-world deployment would require live sensor integration, road-network routing, traffic information, vehicle constraints and continuous model monitoring.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- IBM BOB
- IoT Sensor Data
- Haversine Distance
- Nearest Neighbour
- 2-opt Route Optimization

## Project Structure

```text
ai-waste-collection-dss/
│
├── src/
│   ├── 01_download_data.py
│   ├── 02_combine_sensors.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_model.py
│   ├── 05_evaluate_model.py
│   ├── 06_score_containers.py
│   ├── 07_route_optimiser.py
│   └── 08_dashboard.py
│
├── models/
│   └── rf_critical_fill.pkl
│
├── notebooks/
│   └── waste_collection_analysis.ipynb
│
├── screenshots/
│   ├── dashboard_overview.png
│   └── route_sustainability.png
│
├── requirements.txt
└── README.md
