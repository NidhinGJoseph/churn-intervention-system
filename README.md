[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-intervention-system.streamlit.app)

# Churn Intervention Optimization System

An end-to-end machine learning and decision optimization system for telecom customer retention.

This project goes beyond churn prediction by combining:
- Churn probability modeling
- Probabilistic customer lifetime value (LTV)
- Intervention uplift estimation
- Budget-constrained optimization
- ROI-based customer targeting

The system identifies which customers should receive retention offers in order to maximize expected business value under real-world budget constraints.

---

## Problem Statement

A telecom company has a limited monthly retention budget and cannot intervene on all at-risk customers. This system identifies *which* customers to target to maximize retained revenue under a fixed budget constraint.

---

## Results (at ₹5,000 monthly budget)

Metric - Value 

Customers analyzed - 7,043

Customers targeted - 272

Budget used - ₹4,994

Revenue saved - ₹34,803

Expected profit - ₹29,809

ROI - 5.97x

Lift over random targeting - 2.57x 

---

## Project Structure
src/
├── data_loader.py          # Data ingestion
├── data_preprocessing.py   # Cleaning and encoding
├── features.py             # Feature engineering
├── model.py                # Churn model training
├── intervention.py         # LTV and ROI estimation
├── optimizer.py            # Budget-constrained targeting
├── predict.py              # Inference pipeline
├── train.py                # Training entry point
└── config.py               # Configuration
dashboard/
└── dashboard.py            # Streamlit app
data/raw/
└── telco_churn.csv         # IBM Telco dataset (7,043 customers)
models/
├── churn_model.pkl
└── encoder.pkl
test/
├── test_model.py
├── test_optimizer.py
└── test_preprocessing.py

---

## Tech Stack

Python · Scikit-learn · Pandas · NumPy · Streamlit · Plotly · XGBoost · LightGBM

---

## Setup

```bash
pip install -e .
streamlit run dashboard/dashboard.py
```
