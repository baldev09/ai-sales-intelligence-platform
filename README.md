# SalesIQ Pro — Enterprise Sales Intelligence Platform

> Production-grade sales forecasting and analytics dashboard built with Python, Streamlit, Statsmodels, and Plotly.

---

## Overview

SalesIQ Pro is an end-to-end sales intelligence platform that ingests daily transaction data and delivers interactive forecasting, anomaly detection, segment analysis, and AI-generated strategic insights — all within a fully custom, enterprise-quality Streamlit interface.

Designed to demonstrate production-level data science engineering: rigorous statistical methodology, clean modular code architecture, and a polished user experience that eliminates all default Streamlit aesthetics.

---

## Features

| Module | Capability |
|---|---|
| **Time-Series Forecasting** | SARIMA(1,1,1)(1,1,1,7), Holt-Winters Triple Exponential Smoothing, Ensemble average |
| **Forecast Evaluation** | 80/20 back-test with MAE, RMSE, MAPE, R² comparison across models |
| **Monte Carlo Simulation** | 1,000 Geometric Brownian Motion paths — P10 / P50 / P90 scenarios |
| **Anomaly Detection** | Rolling Z-score (2.8σ) with 21-day window, anomaly log export |
| **Time-Series Decomposition** | Additive decomposition into Trend, Seasonal, and Residual components |
| **Stationarity Testing** | Augmented Dickey-Fuller test with p-value and interpretation |
| **Segment Analysis** | Region, channel, and product drill-down with revenue heatmaps |
| **Insight Engine** | Rule-based engine generating contextual strategic recommendations |
| **Data Upload** | CSV and Excel ingestion with automatic column normalisation |
| **Export** | One-click CSV export for filtered data and forecast output |

---

## Tech Stack

| Layer | Library |
|---|---|
| Application | Streamlit 1.32+ |
| Data | Pandas 2.x, NumPy 1.26+ |
| Visualisation | Plotly (Graph Objects + Express) |
| Time-Series | Statsmodels (SARIMAX, ExponentialSmoothing, seasonal_decompose) |
| Statistics | SciPy (ADF, Z-score) |
| Simulation | NumPy (Geometric Brownian Motion) |

---

## Quick Start

```bash
# 1. Clone or extract the project
cd salesiq_v3

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## Using Your Own Data

Upload a CSV or Excel file via the sidebar. Required column schema:

| Column | Type | Example |
|---|---|---|
| `date` | Date (any standard format) | `2024-01-15` |
| `product` | String | `Enterprise Suite` |
| `region` | String | `North` |
| `channel` | String | `Online` |
| `revenue` | Float | `2450.75` |
| `units` | Integer | `42` |
| `cost` | Float | `1100.00` |

`profit` is computed automatically if not present.

---

## Dashboard Structure

| Tab | Contents |
|---|---|
| Overview | Revenue trend with forecast overlay, channel trend, day-of-week analysis |
| Revenue Intelligence | Monthly heatmap, decomposition, ADF test, descriptive statistics |
| Forecasting Engine | Forecast chart, back-test comparison, Monte Carlo, export |
| Segment Analysis | Regional ranking, regional trend, product-region heatmap, summary table |
| Product Analytics | Revenue ranking, margin waterfall, product trend, SKU table |
| Risk Monitor | Anomaly detection chart, rolling volatility, anomaly log |
| Strategic Insights | AI-generated insights, prioritised action checklist |
| Data Explorer | Correlation matrix, raw data viewer, CSV export |

---

## Project Structure

```
salesiq_v3/
├── app.py                  # Complete application — single-file architecture
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Theme and server configuration
```

---

## Notes

- The app ships with a 540-day synthetic dataset (25,920 rows) generated on first load — no setup required.
- The `add_vline` Plotly/Pandas Timestamp bug is resolved by using `add_shape` + `add_annotation` throughout.
- All SARIMA and Holt-Winters calls are wrapped in exception handlers with deterministic fallbacks.
- The time series index is reindexed with `asfreq('D')` before model fitting to ensure a regular frequency.

---

*Python · Streamlit · Plotly · Statsmodels · SciPy*
