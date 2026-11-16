# Sentra — Fraud & AML Alert Investigation Platform

A complete fraud and anti-money laundering (AML) detection platform powered by an ensemble of **XGBoost** and **Random Forest** models, wrapped in the **Sentra Alert Investigation** dashboard.

---

## 📁 Project Structure

```
fraud-detection/
├── data/                       # Datasets & data documentation
│   ├── README.md               # Dataset source link and synthetic data note
│   └── fraudTest.csv           # Transaction dataset (downloaded locally)
├── notebooks/                  # Exploratory & research notebooks
│   └── README.md
├── src/                        # Core ML pipeline
│   ├── __init__.py
│   ├── features.py             # Feature engineering & Haversine distance
│   ├── train.py                # Model training (RF + XGBoost)
│   ├── evaluate.py             # Classification evaluation & threshold selection
│   └── explain.py              # Rule-based triggers & SHAP attribution
├── app/                        # Web application & dashboard
│   ├── app.py                  # Flask REST API server
│   ├── templates/
│   │   └── index.html          # Sentra investigation UI
│   └── static/
│       ├── css/style.css       # Custom styles
│       └── js/dashboard.js     # Frontend logic & API integration
├── tests/                      # Automated test suite (pytest)
│   ├── __init__.py
│   ├── test_features.py        # Feature extraction tests
│   ├── test_models.py          # Metric & model tests
│   └── test_api.py             # Flask endpoint tests
├── docs/                       # System documentation
│   ├── architecture.md         # System design & execution flow
│   └── api_spec.md             # REST API endpoint specification
├── models/                     # Serialized model artifacts (git-ignored)
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── le_category.joblib
│   ├── feature_cols.joblib
│   └── metrics.joblib
├── requirements.txt            # Pinned dependency versions
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🎨 UI & Investigation Features

The dashboard mirrors the modern **Sentra** financial crime investigation workspace:

- **Deep Navy Sidebar Navigation (`#0d2055`)**:
  - `Workspace`: Dashboard, **Alerts (47)**, **Cases (18)**, Customers, Transactions.
  - `Management`: Rules & Models, Network, Monitoring, Reports, Logs.
  - Fraud Analyst Profile: `Anna Reyes (Fraud Analyst • L2)`.
- **Top Investigation Header**:
  - Alert Title: `Alert Investigation` (`ALT-10482 • Assigned to You`).
  - Priority Queue Switcher: Jump between `ALT-10482` (€9,800 rapid onward transfer), `ALT-10483` ($1,077.69 outlier), `ALT-10484` (rapid velocity), and `ALT-10485` (cleared baseline).
  - Quick Actions Dropdown & **"Review decision →"** CTA.
- **Card 1 — Key Metrics & 3-Step Velocity Timeline**:
  - `FLAGGED TRANSFER`: **€9,800** (SEPA Instant • Account 4821)
  - `RISK SCORE`: **94/100** (Crimson red • High risk)
  - `DETECTED`: **12:48 CET** (Rapid onward transfers)
  - `SLA LEFT`: **12 min** (Due 14:12 CET)
  - Status Indicator: `• In review` (live pulsing green dot)
  - Timeline visualization connecting:
    1. `12:40` (Green): `€9,800 transfer completed` (Marta Kowalska • Account 4821 → •9134 • TX-7814)
    2. `12:48` (Amber): `€9,600 sent onward` (David Oister • Account 9134 → •7712 • TX-7819)
    3. `12:48` (Red): `Alert triggered` (External account •7712 • Rapid onward transfers)
- **Card 2 — Risk Triggers & Signals**:
  - Rapid onward transfer (`Triggered rule` pill + action link)
  - No prior transfers in 90 days (`Risk signal` pill + action link)
  - Shared device fingerprint D-074 (`Risk signal` pill + action link)
- **Card 3 — Explore Connections**:
  - Pathway pill: `•4821 → •9134 → •7712`
  - Connected alerts & devices linkage (`ALT-90228 • Lucas Martin • D-074`).
- **Right Column Cards**:
  - **Customer / Beneficiary details tabs**: Marta Kowalska (ID 4821 • Premium account • since 2021), outgoing transfer statistics, device session status (`New`), and interactive `[📞 Call Customer]` modal with voice verification simulator.
  - **AI Insight**: Sparkle badge, diagnostic rationale, and `[Review Sources]` button.
  - **Evidences**: Item tracker (4 saved • 1 verified • 3 to verify).
- **Interactive Multi-Tab Investigation**:
  - `Overview`: The full Sentra investigation screen.
  - `Transactions`: Real transaction ledger from `data/fraudTest.csv` with search filter, amounts, and flags.
  - `Connections`: Interactive SVG network graph displaying nodes, intermediary transfers, and device linkages.
  - `Evidence`: ML model benchmarks and feature importance weights.
  - `Activity log`: Timestamped investigation audit log.

---

## 🧠 Machine Learning Models

Trained on credit card transactions (`data/fraudTest.csv`) with class imbalance handling:

| Metric | ⚡ XGBoost Classifier | 🌲 Random Forest |
|---|---|---|
| **ROC-AUC** | **0.9986** | 0.9911 |
| **Accuracy** | **99.90%** | 99.56% |
| **Precision (Fraud)** | **88.31%** | 47.14% |
| **Recall (Fraud)** | **85.00%** | 82.50% |
| **F1 Score** | **0.8662** | 0.6000 |

Engineered features:
1. `amt` — Raw transaction amount
2. `amt_log` — Log1p scaled amount
3. `amt_per_city_pop` — Relative local spend intensity
4. `distance` — Haversine distance between customer and terminal (km)
5. `hour`, `day_of_week`, `month` — Temporal velocity features
6. `age`, `gender_enc` — Demographic features
7. `category_enc` — Encoded merchant sector

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. (Optional) Re-Train Models
```bash
python src/train.py
```

### 4. Start the Application
```bash
python app/app.py
```

Open your browser and navigate to:
👉 **http://localhost:8080**
