"""The /v1/user endpoint."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from embedforge.auth.models import Scope
from embedforge.auth.store import TokenStore


def test_user_describes_the_calling_token(
    client: TestClient, store: TokenStore, auth: dict[str, str]
) -> None:
    body = client.get("/v1/user", headers=auth).json()
    record = store.list_tokens()[0]

    assert body["id"] == record.id
    assert body["name"] == "test-client"
    assert body["authenticated"] is True
    assert set(body["scopes"]) == {Scope.EMBED, Scope.QUERY, Scope.MODELS_READ}
    assert body["expires_at"] is None
    assert set(body["endpoints"]) == {"POST /v1/embed", "POST /v1/query", "GET /v1/models"}


def test_user_reports_remaining_lifetime(client: TestClient, store: TokenStore) -> None:
    expires = datetime.now(UTC) + timedelta(hours=1)
    _, plaintext = store.create("expiring", expires_at=expires)
    body = client.get("/v1/user", headers={"Authorization": f"Bearer {plaintext}"}).json()
    assert 3500 <= body["expires_in_seconds"] <= 3600


def test_user_lists_only_permitted_endpoints(client: TestClient, store: TokenStore) -> None:
    _, plaintext = store.create("narrow", scopes=[Scope.QUERY])
    body = client.get("/v1/user", headers={"Authorization": f"Bearer {plaintext}"}).json()
    assert body["endpoints"] == ["POST /v1/query"]
