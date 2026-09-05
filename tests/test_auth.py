"""Token authentication and scope enforcement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from embedforge.api.app import create_app
from embedforge.auth.models import Scope, hash_token, parse_token_id
from embedforge.auth.store import TokenStore
from embedforge.config import Settings


def test_token_roundtrip(store: TokenStore) -> None:
    record, plaintext = store.create("ci")
    assert parse_token_id(plaintext) == record.id
    assert record.token_hash == hash_token(plaintext)
    assert plaintext not in store.path.read_text("utf-8"), "plaintext must never be persisted"

    principal = store.authenticate(plaintext)
    assert principal is not None
    assert principal.name == "ci"


def test_secret_with_underscores_still_parses(store: TokenStore) -> None:
    """url-safe secrets contain '_', so the id split must be bounded."""
    for _ in range(50):
        _, plaintext = store.create("many")
        assert store.authenticate(plaintext) is not None


def test_rejects_unknown_wrong_and_malformed_tokens(store: TokenStore) -> None:
    record, plaintext = store.create("ci")
    assert store.authenticate("garbage") is None
    assert store.authenticate("ef_unknown_secret") is None
    assert store.authenticate(f"ef_{record.id}_wrong") is None
    assert store.authenticate(plaintext.upper()) is None


def test_disabled_and_expired_tokens_stop_working(store: TokenStore) -> None:
    record, plaintext = store.create("ci")
    store.set_disabled(record.id, True)
    assert store.authenticate(plaintext) is None
    store.set_disabled(record.id, False)
    assert store.authenticate(plaintext) is not None

    _, expired = store.create("old", expires_at=datetime.now(UTC) - timedelta(seconds=1))
    assert store.authenticate(expired) is None


def test_deleting_a_token_revokes_it(store: TokenStore) -> None:
    record, plaintext = store.create("ci")
    assert store.delete(record.id) is True
    assert store.authenticate(plaintext) is None
    assert store.delete(record.id) is False


def test_store_picks_up_changes_from_another_process(settings: Settings) -> None:
    """The CLI writes the file; a running server must notice without a restart."""
    writer = TokenStore(settings.tokens_file, reload_interval=0)
    reader = TokenStore(settings.tokens_file, reload_interval=0)
    record, plaintext = writer.create("late")
    assert reader.authenticate(plaintext) is not None
    writer.delete(record.id)
    assert reader.authenticate(plaintext) is None


def test_corrupt_file_keeps_last_good_state(settings: Settings) -> None:
    store = TokenStore(settings.tokens_file, reload_interval=0)
    _, plaintext = store.create("ci")
    settings.tokens_file.write_text("{not json", encoding="utf-8")
    assert store.authenticate(plaintext) is not None, "must not lock everyone out"


def test_endpoints_require_a_token(client: TestClient) -> None:
    responses = [
        client.get("/v1/user"),
        client.get("/v1/models"),
        client.post("/v1/embed", json={"input": "x"}),
        client.post("/v1/query", json={"input": "x"}),
    ]
    for response in responses:
        assert response.status_code == 401, response.request.url
        assert response.json()["error"]["type"] == "authentication_error"
        assert response.headers["www-authenticate"] == "Bearer"


def test_bad_token_is_rejected(client: TestClient) -> None:
    response = client.get("/v1/user", headers={"Authorization": "Bearer ef_00_bad"})
    assert response.status_code == 401


def test_scopes_are_enforced_per_endpoint(settings: Settings, store: TokenStore) -> None:
    _, query_only = store.create("query-only", scopes=[Scope.QUERY])
    headers = {"Authorization": f"Bearer {query_only}"}
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/query", json={"input": "x"}, headers=headers).status_code == 200
        forbidden = client.post("/v1/embed", json={"input": "x"}, headers=headers)
        assert forbidden.status_code == 403
        assert forbidden.json()["error"]["type"] == "permission_error"


def test_admin_scope_implies_the_rest(settings: Settings, store: TokenStore) -> None:
    _, admin = store.create("admin", scopes=[Scope.ADMIN])
    headers = {"Authorization": f"Bearer {admin}"}
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/embed", json={"input": "x"}, headers=headers).status_code == 200
        assert client.get("/v1/models", headers=headers).status_code == 200


def test_auth_can_be_disabled_for_local_development(settings: Settings) -> None:
    with TestClient(create_app(settings.model_copy(update={"auth_enabled": False}))) as client:
        body = client.get("/v1/user").json()
        assert body["authenticated"] is False
        assert body["id"] == "anonymous"
