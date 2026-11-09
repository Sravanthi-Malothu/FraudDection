# FraudLens — Fraud Intelligence Platform

A professional fraud detection dashboard powered by machine learning (Random Forest + XGBoost), built with Flask and a dark-themed fintech UI.

## Stack
- **Backend:** Python 3, Flask, scikit-learn, XGBoost, pandas, numpy
- **Frontend:** Single HTML + Tailwind CSS + Chart.js (dark glass-morphism UI)
- **Models:** Random Forest & XGBoost ensemble (trained on credit card transaction data)

## Quick Start

### 1. Install Dependencies
```bash
cd /Users/sravanthimalothu/Desktop/FraudDecetion
pip3 install flask scikit-learn xgboost pandas numpy joblib flask-cors
```

### 2. Train the Models
```bash
python3 train_model.py
```
Training uses the first 100,000 rows of `fraudTest.csv` for speed (~2–5 min).

### 3. Start the App
```bash
python3 app.py
```
Open your browser at **http://localhost:5000**

## Project Structure
```
FraudDecetion/
├── app.py                  Flask backend (API + server)
├── train_model.py          Model training script
├── requirements.txt
├── templates/
│   └── index.html          Dashboard UI (single-page)
├── static/
│   ├── css/style.css
│   └── js/dashboard.js
├── models/                 Saved model artifacts (after training)
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── le_category.joblib
│   ├── feature_cols.joblib
│   └── metrics.joblib
└── fraudTest.csv           Dataset (not included in repo)
```

## API Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/` | Serve dashboard |
| GET  | `/api/stats` | Dataset statistics |
| POST | `/api/predict` | Predict fraud for a transaction |
| GET  | `/api/model-metrics` | Model performance metrics |
| GET  | `/api/fraud-by-category` | Fraud count per merchant category |
| GET  | `/api/fraud-by-hour` | Fraud count per hour |

### Predict Request Example
```json
POST /api/predict
{
  "amt": 299.99,
  "category": "shopping_net",
  "city_pop": 5000,
  "lat": 36.16,
  "long": -86.78,
  "hour": 2,
  "gender": "M",
  "age": 45
}
```

### Predict Response
```json
{
  "is_fraud": 1,
  "confidence": 87.3,
  "rf_score": 84.2,
  "xgb_score": 90.4,
  "risk_level": "HIGH",
  "risk_factors": [
    "High transaction amount ($299.99)",
    "Unusual transaction hour (2:00)"
  ]
}
```

## Features
- **Glass morphism dark UI** with animated charts (Chart.js)
- **Real-time analysis** via POST form → prediction result
- **Ensemble model** (Random Forest + XGBoost averaged confidence)
- **Leakage-safe per-card behavioural features** (txn counts over 1h/24h, card amount mean/std/zscore, time/distance since last txn, new merchant & category flags) — past transactions only
- **KPI cards** with dataset-level fraud statistics
- **Transaction history table** (last 20 analyzed)
- **30-second polling** for simulated live updates
- **Model performance panel** (Precision, Recall, F1, ROC-AUC, PR-AUC; accuracy is not a headline metric)

## Evaluation (time-based, leakage-safe)

`train_model.py` evaluates with a rigorous, leakage-safe methodology:

- **Train:** `fraudTrain.csv` (full dataset, no subsampling), sorted by `trans_date_trans_time`.
- **Validation:** last 15% of `fraudTrain.csv` by time (85/15 time split — never random). Used for XGBoost early stopping and fraud-class F1 threshold selection.
- **Test:** `fraudTest.csv` held out strictly as the final test set. Test features are computed on `train + test` history so per-card history carries over, using only past transactions (`shift` / `rolling(closed='left')` — the current row is never included).

### Ablation: baseline features vs baseline + behavioural features (test set)

Run `python3 train_model.py` (requires `fraudTrain.csv`) to populate:

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|-------|--------|---------|-----------|--------|----|
| XGBoost — baseline features | — | — | — | — | — |
| XGBoost — baseline + behavioural | — | — | — | — | — |

Metrics and plots are saved to `models/metrics.json` and `docs/` (`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`).
