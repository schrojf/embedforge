"""Shared fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from embedforge.api.app import create_app
from embedforge.auth.models import Scope
from embedforge.auth.store import TokenStore
from embedforge.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Server settings isolated to a temp directory."""
    return Settings(
        env="dev",
        log_format="console",
        tokens_file=tmp_path / "tokens.json",
        model_dir=tmp_path / "models",
        model_id="dev-hash",
        tokens_reload_interval=0,
        max_items_per_request=8,
        max_input_chars=64,
        batch_wait_ms=1,
    )


@pytest.fixture
def store(settings: Settings) -> TokenStore:
    return TokenStore(settings.tokens_file, reload_interval=0)


@pytest.fixture
def token(store: TokenStore) -> str:
    """A token with the default scopes."""
    _, plaintext = store.create("test-client")
    return plaintext


@pytest.fixture
def app(settings: Settings, store: TokenStore) -> FastAPI:
    del store  # Ensures the token file exists before the app reads it.
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def all_scopes() -> list[Scope]:
    return list(Scope)
