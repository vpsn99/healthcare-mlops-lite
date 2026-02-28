from fastapi.testclient import TestClient

from healthml.serving.api import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    # In tests, startup may not run predictor if model files missing
    # so just verify endpoint exists and returns JSON
    assert resp.status_code in (200, 503)
    assert resp.headers["content-type"].startswith("application/json")