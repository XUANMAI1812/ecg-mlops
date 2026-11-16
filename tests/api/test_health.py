"""Smoke test cho API: GET /health trả 200 và đúng schema."""

from fastapi.testclient import TestClient

from ecg.serving.app import app


def test_health_returns_ok():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
