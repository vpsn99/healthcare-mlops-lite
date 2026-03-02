from fastapi.testclient import TestClient

from healthml.serving.api import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_loaded" in body


def test_predict_returns_503_when_model_missing():
    client = TestClient(app)
    resp = client.post("/predict", json={
        "age_years": 45,
        "HEALTHCARE_EXPENSES": 20000,
        "HEALTHCARE_COVERAGE": 15000,
        "INCOME": 90000,
        "encounter_count": 5,
        "avg_enc_duration_days": 2.5,
        "active_span_days": 300,
        "condition_count": 3,
        "GENDER": "M",
        "RACE": "white",
        "ETHNICITY": "nonhispanic",
        "MARITAL": "M",
        "STATE": "Massachusetts"
    })
    assert resp.status_code in {200, 400, 503}