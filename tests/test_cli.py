"""The token and model command-line interfaces."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from embedforge.api.app import create_app
from embedforge.auth.store import TokenStore
from embedforge.cli.main import app as cli
from embedforge.cli.tokens import parse_expiry, parse_scopes
from embedforge.config import Settings, get_settings

runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the CLI at a temp token file, as an operator would with env vars."""
    tokens_file = tmp_path / "tokens.json"
    monkeypatch.setenv("EMBEDFORGE_TOKENS_FILE", str(tokens_file))
    get_settings.cache_clear()
    yield tokens_file
    get_settings.cache_clear()


def run(*args: str) -> str:
    result = runner.invoke(cli, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_parse_expiry_accepts_durations_and_dates() -> None:
    assert parse_expiry(None) is None

    in_30_days = parse_expiry("30d")
    assert in_30_days is not None
    assert (in_30_days - datetime.now(UTC)).days == 29

    iso = parse_expiry("2030-01-01")
    assert iso is not None
    assert iso.year == 2030
    assert iso.tzinfo is not None, "naive input must be treated as UTC"
    with pytest.raises(Exception, match="not a duration"):
        parse_expiry("soon")


def test_parse_scopes_defaults_and_validates() -> None:
    assert parse_scopes(None) == parse_scopes("embed,query,models:read")
    with pytest.raises(Exception, match="Unknown scope"):
        parse_scopes("embed,wat")


def test_create_prints_the_token_once_and_stores_only_its_hash(cli_env: Path) -> None:
    output = run("token", "create", "indexer", "--scopes", "embed")
    token = next(word for word in output.split() if word.startswith("ef_"))

    stored = cli_env.read_text("utf-8")
    assert token not in stored
    assert TokenStore(cli_env, reload_interval=0).authenticate(token) is not None


def test_list_show_disable_enable_and_revoke(cli_env: Path) -> None:
    output = run("token", "create", "bot")
    token = next(word for word in output.split() if word.startswith("ef_"))
    store = TokenStore(cli_env, reload_interval=0)
    token_id = store.list_tokens()[0].id

    assert "bot" in run("token", "list")
    assert token_id in run("token", "show", token_id)

    run("token", "disable", token_id)
    assert store.authenticate(token) is None
    run("token", "enable", token_id)
    assert store.authenticate(token) is not None

    run("token", "revoke", token_id, "--yes")
    assert store.authenticate(token) is None
    assert "No tokens yet" in run("token", "list")


def test_missing_token_is_an_error(cli_env: Path) -> None:
    del cli_env
    for args in (("token", "show", "nope"), ("token", "disable", "nope")):
        assert runner.invoke(cli, list(args)).exit_code == 1


def test_model_commands_describe_the_catalog(cli_env: Path) -> None:
    del cli_env
    assert "dev-hash" in run("model", "list")
    info = run("model", "info", "dev-hash")
    assert "Pros" in info and "Cons" in info
    assert runner.invoke(cli, ["model", "info", "nope"]).exit_code == 1


def test_config_command_shows_effective_settings(cli_env: Path) -> None:
    del cli_env
    output = run("config")
    assert "model_id" in output
    assert "intra_op_threads (resolved)" in output


def test_a_cli_created_token_works_against_the_server(cli_env: Path, tmp_path: Path) -> None:
    """The end-to-end contract: create a token, call the API with it."""
    output = run("token", "create", "e2e")
    token = next(word for word in output.split() if word.startswith("ef_"))

    settings = Settings(tokens_file=cli_env, model_dir=tmp_path, tokens_reload_interval=0)
    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/user", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["name"] == "e2e"
