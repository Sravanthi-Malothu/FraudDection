# Architecture & System Design

## Overview
The Fraud & AML Detection Platform is structured into modular layers:
- **`data/`**: Stores synthetic datasets (`fraudTest.csv`) and dataset documentation.
- **`src/`**: Core machine learning modules:
  - `features.py`: Haversine geographic distance calculation, log amount scaling, demographic features, and categorical encoding.
  - `train.py`: Model training pipeline for Random Forest and XGBoost with class imbalance handling.
  - `evaluate.py`: Evaluation metrics (ROC-AUC, PR-AUC, Precision, Recall, F1 score, Confusion Matrix) and threshold selection.
  - `explain.py`: Rule-based risk triggers and model explainability attribution.
- **`app/`**: Flask web server, Sentra Alert Investigation templates, and static assets.
- **`tests/`**: Automated unit and integration tests using pytest.
- **`docs/`**: Project and API specifications.
- **`notebooks/`**: Research and exploratory data analysis.

## Flow of Execution
```
[Raw Transaction / Request]
           │
           ▼
[src/features.py] ──> Extracts 15 Feature Columns
           │
           ▼
[Ensemble Inference] ──> Average(Random Forest, XGBoost)
           │
           ▼
[src/explain.py] ──> Attributed Risk Triggers & Signals
           │
           ▼
[app/app.py] ──> REST Response to Dashboard UI
```
