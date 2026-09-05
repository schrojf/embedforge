"""The /v1/embed and /v1/query endpoints."""

import math

from fastapi.testclient import TestClient


def test_embed_single_text(client: TestClient, auth: dict[str, str]) -> None:
    body = client.post("/v1/embed", json={"input": "hello"}, headers=auth).json()
    assert body["model"] == "dev-hash"
    assert body["task"] == "document"
    assert body["dimension"] == 384
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert len(body["data"][0]["embedding"]) == 384
    assert body["usage"] == {"items": 1, "characters": 5}


def test_embed_list_keeps_input_order(client: TestClient, auth: dict[str, str]) -> None:
    texts = ["alpha", "beta", "gamma"]
    body = client.post("/v1/embed", json={"input": texts}, headers=auth).json()
    assert [item["index"] for item in body["data"]] == [0, 1, 2]

    singles = [
        client.post("/v1/embed", json={"input": text}, headers=auth).json()["data"][0]["embedding"]
        for text in texts
    ]
    assert [item["embedding"] for item in body["data"]] == singles


def test_vectors_are_normalized(client: TestClient, auth: dict[str, str]) -> None:
    body = client.post("/v1/embed", json={"input": "hello"}, headers=auth).json()
    assert body["normalized"] is True
    assert math.isclose(
        math.sqrt(sum(value * value for value in body["data"][0]["embedding"])), 1.0, rel_tol=1e-5
    )


def test_query_matches_embed_for_a_symmetric_model(
    client: TestClient, auth: dict[str, str]
) -> None:
    """dev-hash is symmetric, so both endpoints must agree and say so."""
    embed = client.post("/v1/embed", json={"input": "hello"}, headers=auth).json()
    query = client.post("/v1/query", json={"input": "hello"}, headers=auth).json()
    assert embed["symmetric"] is True
    assert query["task"] == "query"
    assert query["data"][0]["embedding"] == embed["data"][0]["embedding"]


def test_same_text_always_gives_the_same_vector(client: TestClient, auth: dict[str, str]) -> None:
    first = client.post("/v1/embed", json={"input": "stable"}, headers=auth).json()
    second = client.post("/v1/embed", json={"input": "stable"}, headers=auth).json()
    assert first["data"][0]["embedding"] == second["data"][0]["embedding"]


def test_model_guard_rejects_a_mismatch(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post("/v1/embed", json={"input": "x", "model": "other"}, headers=auth)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_model_guard_accepts_the_active_model(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post("/v1/embed", json={"input": "x", "model": "dev-hash"}, headers=auth)
    assert response.status_code == 200


def test_rejects_empty_and_oversized_input(client: TestClient, auth: dict[str, str]) -> None:
    assert client.post("/v1/embed", json={"input": ""}, headers=auth).status_code == 400
    assert client.post("/v1/embed", json={"input": "   "}, headers=auth).status_code == 400
    assert client.post("/v1/embed", json={"input": []}, headers=auth).status_code == 400

    # settings fixture caps items at 8 and characters at 64.
    too_many = client.post("/v1/embed", json={"input": ["x"] * 9}, headers=auth)
    assert too_many.status_code == 413
    too_long = client.post("/v1/embed", json={"input": "x" * 65}, headers=auth)
    assert too_long.status_code == 413
    assert too_long.json()["error"]["type"] == "payload_too_large"


def test_unknown_fields_are_rejected(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post("/v1/embed", json={"input": "x", "typo": 1}, headers=auth)
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_models_endpoint_describes_the_catalog(client: TestClient, auth: dict[str, str]) -> None:
    body = client.get("/v1/models", headers=auth).json()
    assert body["active"] == "dev-hash"
    active = [model for model in body["models"] if model["active"]]
    assert len(active) == 1
    assert active[0]["pros"] and active[0]["cons"], "models must document their trade-offs"

    single = client.get("/v1/models/dev-hash", headers=auth).json()
    assert single["dimension"] == 384
    assert client.get("/v1/models/nope", headers=auth).status_code == 400
