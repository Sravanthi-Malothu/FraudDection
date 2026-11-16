import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


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


def test_api_predict_route(client):
    payload = {
        "amt": 850.0,
        "category": "shopping_net",
        "city_pop": 500,
        "lat": 36.0,
        "long": -86.0,
        "hour": 2,
        "gender": "F",
        "age": 28,
    }
    rv = client.post("/api/predict", json=payload)
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert "risk_score" in json_data
    assert "risk_level" in json_data
    assert 0 <= json_data["risk_score"] <= 100
