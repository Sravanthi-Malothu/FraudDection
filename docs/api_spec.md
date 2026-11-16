# API Specification

## Endpoints

### 1. Root Dashboard
- **Route:** `GET /`
- **Description:** Renders the Sentra Alert Investigation dashboard.

### 2. Alert Queue
- **Route:** `GET /api/alerts`
- **Description:** Returns a summary list of priority investigation alerts.
- **Response:**
  ```json
  {
    "alerts": [
      {
        "id": "ALT-10482",
        "customer_name": "Marta Kowalska",
        "flagged_amount": "€9,800",
        "risk_score": 94,
        "risk_level": "High risk",
        "status": "In review"
      }
    ],
    "total_alerts": 47,
    "cases_count": 18
  }
  ```

### 3. Alert Details
- **Route:** `GET /api/alerts/<alert_id>`
- **Description:** Returns the full investigation payload including timeline nodes, risk signals, entity connections, and customer details.

### 4. Record Analyst Decision
- **Route:** `POST /api/alerts/<alert_id>/decision`
- **Payload:**
  ```json
  {
    "decision": "Approved" | "Rejected" | "Escalated",
    "notes": "Optional analyst rationale"
  }
  ```

### 5. Real-Time Transaction Scoring
- **Route:** `POST /api/predict`
- **Payload:**
  ```json
  {
    "amt": 850.0,
    "category": "shopping_net",
    "city_pop": 500,
    "lat": 36.0,
    "long": -86.0,
    "hour": 2,
    "gender": "F",
    "age": 28
  }
  ```
- **Response:**
  ```json
  {
    "is_fraud": 1,
    "risk_score": 93,
    "risk_level": "High risk",
    "confidence": 93.28,
    "rf_score": 63.75,
    "xgb_score": 93.28,
    "triggers": [...],
    "timeline": [...]
  }
  ```

### 6. Transaction Ledger
- **Route:** `GET /api/transactions`
- **Description:** Retrieves real transaction records from the dataset for manual inspection.

### 7. Model Performance Metrics
- **Route:** `GET /api/model-metrics`
- **Description:** Returns test benchmark metrics for Random Forest and XGBoost.
