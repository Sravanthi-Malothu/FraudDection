"""
test_api.py — Comprehensive Integration Tests for Sentra Flask Backend
Validates Health check, Alert queue, SQLite decision persistence,
Evidence benchmarks from docs/results.json, and /api/predict input validation.
"""

import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["status"] in ("healthy", "degraded")
    assert "models_loaded" in data
    assert "database" in data
    assert "model_version" in data
    assert "total_alerts_queued" in data


def test_index_route(client):
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"Sentra" in rv.data or b"Alert Investigation" in rv.data


def test_api_alerts_route(client):
    rv = client.get("/api/alerts")
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert "alerts" in json_data
    assert len(json_data["alerts"]) > 0

    # Ensure sorted by risk_score descending
    scores = [a["risk_score"] for a in json_data["alerts"]]
    assert scores == sorted(scores, reverse=True)

    # Check alert structure and USD currency formatting
    first_alert = json_data["alerts"][0]
    assert first_alert["flagged_amount"].startswith("$")
    assert 0 <= first_alert["risk_score"] <= 100
    assert "customer_name" in first_alert
    assert "risk_level" in first_alert
    assert "status" in first_alert
    assert "sla_left" in first_alert


def test_api_alert_detail_real_data(client):
    rv_list = client.get("/api/alerts")
    alerts = rv_list.get_json()["alerts"]
    alert_id = alerts[0]["id"]

    rv = client.get(f"/api/alerts/{alert_id}")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["id"] == alert_id
    assert "flagged_amount" in data
    assert data["flagged_amount"].startswith("$")
    assert "triggers" in data
    assert len(data["triggers"]) > 0
    assert "explore_connections" in data
    assert "merchant" in data["explore_connections"]
    assert "customer" in data
    assert "ai_insight" in data


def test_api_alert_detail_not_found(client):
    rv = client.get("/api/alerts/NONEXISTENT_ALERT_999")
    assert rv.status_code == 404
    data = rv.get_json()
    assert "error" in data


def test_api_alert_decision_sqlite_persistence(client):
    rv_list = client.get("/api/alerts")
    alerts = rv_list.get_json()["alerts"]
    alert_id = alerts[0]["id"]

    # 1. Test Approved
    payload_approve = {
        "decision": "Approved",
        "notes": "Verified via test suite with cardholder",
        "analyst": "Anna Reyes",
    }
    rv = client.post(f"/api/alerts/{alert_id}/decision", json=payload_approve)
    assert rv.status_code == 200
    res = rv.get_json()
    assert res["success"] is True
    assert res["alert"]["status"] == "Approved"

    # Query decision history endpoint
    rv_hist = client.get(f"/api/alerts/{alert_id}/decisions")
    assert rv_hist.status_code == 200
    decisions = rv_hist.get_json()["decisions"]
    assert len(decisions) >= 1
    assert decisions[0]["decision"] == "Approved"

    # 2. Test Rejected (Blocked / Frozen)
    payload_reject = {
        "decision": "Rejected",
        "notes": "Confirmed compromised card",
        "analyst": "Security Admin",
    }
    rv_rej = client.post(f"/api/alerts/{alert_id}/decision", json=payload_reject)
    assert rv_rej.status_code == 200
    assert rv_rej.get_json()["alert"]["status"] == "Blocked / Frozen"

    # 3. Test Escalated
    payload_esc = {
        "decision": "Escalated",
        "notes": "Requires Tier 3 fraud analysis",
        "analyst": "Anna Reyes",
    }
    rv_esc = client.post(f"/api/alerts/{alert_id}/decision", json=payload_esc)
    assert rv_esc.status_code == 200
    assert rv_esc.get_json()["alert"]["status"] == "Escalated (Tier 3)"


def test_api_alert_decision_validation_and_not_found(client):
    rv_list = client.get("/api/alerts")
    alerts = rv_list.get_json()["alerts"]
    alert_id = alerts[0]["id"]

    # Invalid decision value
    rv_bad = client.post(f"/api/alerts/{alert_id}/decision", json={"decision": "UnknownStatus"})
    assert rv_bad.status_code == 400
    assert "Invalid decision value" in rv_bad.get_json()["error"]

    # Decision on nonexistent alert
    rv_404 = client.post("/api/alerts/NONEXISTENT_ALERT/decision", json={"decision": "Approved"})
    assert rv_404.status_code == 404

    # Decision history on nonexistent alert
    rv_hist_404 = client.get("/api/alerts/NONEXISTENT_ALERT/decisions")
    assert rv_hist_404.status_code == 404


def test_api_evidence_route(client):
    rv = client.get("/api/evidence")
    assert rv.status_code == 200
    data = rv.get_json()
    assert "test_set_summary" in data
    assert "models" in data
    assert "xgboost" in data["models"]
    assert "ensemble" in data["models"]
    assert "pr_auc" in data["models"]["xgboost"]
    assert "roc_auc" in data["models"]["xgboost"]


def test_api_predict_valid(client):
    payload = {
        "amt": 850.0,
        "category": "shopping_net",
        "city_pop": 50000,
        "lat": 36.0,
        "long": -86.0,
        "hour": 23,
        "gender": "F",
        "age": 28,
    }
    rv = client.post("/api/predict", json=payload)
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert "model_version" in json_data
    assert len(json_data["model_version"]) > 0
    assert "risk_score" in json_data
    assert "risk_level" in json_data
    assert 0 <= json_data["risk_score"] <= 100
    assert "triggers" in json_data
    assert "ai_insight" in json_data
    assert "top_features" in json_data


def test_api_predict_missing_input(client):
    # 1. Missing body or non-JSON
    rv_empty = client.post("/api/predict", data="")
    assert rv_empty.status_code == 400
    assert "Validation failed" in rv_empty.get_json()["error"]

    # 2. Empty JSON object (missing required 'amt')
    rv_empty_dict = client.post("/api/predict", json={})
    assert rv_empty_dict.status_code == 400
    details = rv_empty_dict.get_json()["details"]
    assert any("amt" in d for d in details)


def test_api_predict_validation_errors(client):
    # Invalid: negative amount
    rv_bad_amt = client.post("/api/predict", json={"amt": -50.0})
    assert rv_bad_amt.status_code == 400
    assert "Validation failed" in rv_bad_amt.get_json()["error"]

    # Invalid: non-numeric amount
    rv_bad_type = client.post("/api/predict", json={"amt": "invalid"})
    assert rv_bad_type.status_code == 400

    # Invalid: hour out of range (25)
    rv_bad_hour = client.post("/api/predict", json={"amt": 100.0, "hour": 25})
    assert rv_bad_hour.status_code == 400

    # Invalid: negative hour (-1)
    rv_neg_hour = client.post("/api/predict", json={"amt": 100.0, "hour": -1})
    assert rv_neg_hour.status_code == 400

    # Invalid: age out of range (< 10 or > 120)
    rv_bad_age_low = client.post("/api/predict", json={"amt": 100.0, "age": 5})
    assert rv_bad_age_low.status_code == 400

    rv_bad_age_high = client.post("/api/predict", json={"amt": 100.0, "age": 130})
    assert rv_bad_age_high.status_code == 400

    # Invalid: city_pop negative
    rv_bad_pop = client.post("/api/predict", json={"amt": 100.0, "city_pop": -500})
    assert rv_bad_pop.status_code == 400

    # Invalid: lat out of range (95)
    rv_bad_lat = client.post("/api/predict", json={"amt": 100.0, "lat": 95.0})
    assert rv_bad_lat.status_code == 400

    # Invalid: long out of range (-200)
    rv_bad_long = client.post("/api/predict", json={"amt": 100.0, "long": -200.0})
    assert rv_bad_long.status_code == 400
