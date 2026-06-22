"""Tests for the FastAPI application surface."""

from __future__ import annotations

from datetime import date, timedelta


def _payload() -> dict:
    return {
        "hotel_id": "HOTEL_A",
        "room_type": "DELUXE_KING",
        "stay_date": (date.today() + timedelta(days=21)).isoformat(),
        "inventory_total": 50,
        "rooms_sold": 20,
        "previous_price": 180.0,
        "competitor_rates": [175, 189, 205, 210],
        "booking_velocity": 3.0,
        "include_explanation": True,
    }


def test_health_ok(client) -> None:
    """Health endpoint returns status, version, and model availability."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_available" in body
    assert body["environment"] == "test"


def test_prediction_heuristic_fallback(client) -> None:
    """Prediction works without a trained model via the heuristic fallback."""
    response = client.post("/api/v1/recommendations", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["recommended_price"] > 0
    assert body["effective_floor"] <= body["recommended_price"] <= body["effective_ceiling"]
    assert body["explanation"] is not None


def test_prediction_validation_error(client) -> None:
    """An invalid request yields a uniform 422 error body."""
    bad = _payload()
    bad["inventory_total"] = 0  # violates ge=1
    response = client.post("/api/v1/recommendations", json=bad)
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_explanation_endpoint(client) -> None:
    """The explanation endpoint returns a rationale payload."""
    response = client.post("/api/v1/explanations", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body.get("summary")
    assert body["generated_by"] == "rule-based"


def test_metrics_endpoint(client) -> None:
    """The metrics endpoint responds in Prometheus text format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_request_id_echoed(client) -> None:
    """A supplied X-Request-ID is echoed back on the response."""
    response = client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers.get("X-Request-ID") == "abc-123"


def test_auth_enforced_in_production(monkeypatch) -> None:
    """In production, a missing API key is rejected with 401."""
    from app.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    from app.api.main import create_app

    app = create_app()
    with TestClient(app) as production_client:
        response = production_client.post("/api/v1/recommendations", json=_payload())
    assert response.status_code == 401
    assert response.json()["code"] == "authentication_error"

    monkeypatch.setenv("ENVIRONMENT", "test")
    get_settings.cache_clear()
