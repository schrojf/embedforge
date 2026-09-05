"""Health, readiness, and service metadata."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_reports_service_and_model(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["service"] == "embedforge"
    assert body["model"] == "dev-hash"


def test_healthz_needs_no_token(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_reports_engine_stats(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["engine"]["ready"] is True
    assert body["engine"]["max_batch_size"] >= 1


def test_readyz_is_503_before_startup(app: object) -> None:
    """Without the lifespan, no model is loaded and readiness must say so."""
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)
    client = TestClient(app)  # Not used as a context manager: lifespan never runs.
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    assert client.get("/healthz").headers["x-request-id"]


def test_incoming_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


def test_metrics_endpoint_exposes_counters(client: TestClient) -> None:
    client.get("/healthz")
    body = client.get("/metrics").text
    assert "embedforge_http_requests_total" in body
