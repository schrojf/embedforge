"""`embedforge token` - manage API client tokens.

The plaintext token is shown exactly once, at creation. Only its SHA-256 digest is
stored, so a lost token is replaced rather than recovered.
"""

import re
from datetime import UTC, datetime, timedelta
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from embedforge.auth.models import DEFAULT_SCOPES, Scope, TokenRecord
from embedforge.auth.store import TokenStore
from embedforge.config import get_settings

app = typer.Typer(no_args_is_help=True, help="Manage API client tokens.")
console = Console()
err_console = Console(stderr=True)

_DURATION = re.compile(r"^(\d+)([smhdw])$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def get_store() -> TokenStore:
    settings = get_settings()
    return TokenStore(settings.tokens_file, reload_interval=0)


def parse_expiry(value: str | None) -> datetime | None:
    """Parse `30d`, `12h`, or an ISO 8601 date/datetime."""
    if value is None:
        return None
    match = _DURATION.match(value.strip())
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        return datetime.now(UTC) + timedelta(**{_UNITS[unit]: amount})
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter(
            f"{value!r} is not a duration like '30d' or an ISO date like '2027-01-01'."
        ) from None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_scopes(value: str | None) -> list[Scope]:
    if not value:
        return list(DEFAULT_SCOPES)
    scopes: list[Scope] = []
    for name in value.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            scopes.append(Scope(name))
        except ValueError:
            raise typer.BadParameter(
                f"Unknown scope {name!r}. Valid scopes: {', '.join(s.value for s in Scope)}."
            ) from None
    if not scopes:
        raise typer.BadParameter("At least one scope is required.")
    return scopes


def _table(records: list[TokenRecord]) -> Table:
    table = Table(title="API tokens", title_justify="left")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Scopes")
    table.add_column("Created")
    table.add_column("Expires")
    styles = {"active": "green", "disabled": "yellow", "expired": "red"}
    for record in records:
        status = record.status()
        table.add_row(
            record.id,
            record.name,
            f"[{styles[status]}]{status}[/{styles[status]}]",
            ",".join(scope.value for scope in record.scopes),
            record.created_at.strftime("%Y-%m-%d"),
            record.expires_at.strftime("%Y-%m-%d") if record.expires_at else "never",
        )
    return table


@app.command("create")
def create_token(
    name: Annotated[str, typer.Argument(help="Who this token is for, e.g. 'search-indexer'.")],
    scopes: Annotated[
        str | None,
        typer.Option("--scopes", "-s", help="Comma-separated: embed,query,models:read,admin."),
    ] = None,
    expires_in: Annotated[
        str | None,
        typer.Option("--expires-in", "-e", help="Lifetime, e.g. '90d', '12h', or an ISO date."),
    ] = None,
    note: Annotated[str | None, typer.Option("--note", help="Free-form reminder.")] = None,
) -> None:
    """Create a token and print it once."""
    store = get_store()
    record, plaintext = store.create(
        name, scopes=parse_scopes(scopes), expires_at=parse_expiry(expires_in), note=note
    )
    console.print(f"[bold green]Created token[/bold green] [cyan]{record.id}[/cyan] for {name!r}")
    console.print(f"  scopes:  {', '.join(scope.value for scope in record.scopes)}")
    console.print(f"  expires: {record.expires_at.isoformat() if record.expires_at else 'never'}")
    console.print(f"  file:    {store.path}", soft_wrap=True)
    console.print()
    console.print("[bold]Token (shown once, store it now):[/bold]")
    # soft_wrap keeps the token on one line, so it can be copied intact.
    console.print(f"[bold yellow]{plaintext}[/bold yellow]", soft_wrap=True)
    console.print()
    console.print(
        f'Try it: curl -H "Authorization: Bearer {plaintext}" http://localhost:8000/v1/user',
        soft_wrap=True,
    )


@app.command("list")
def list_tokens() -> None:
    """List every token, without revealing any of them."""
    store = get_store()
    records = store.list_tokens()
    if not records:
        console.print(
            f"No tokens yet in {store.path}. Create one with 'embedforge token create'.",
            soft_wrap=True,
        )
        raise typer.Exit(0)
    console.print(_table(records))


@app.command("show")
def show_token(token_id: Annotated[str, typer.Argument(help="Token id.")]) -> None:
    """Show one token's details."""
    store = get_store()
    record = store.get(token_id)
    if record is None:
        err_console.print(f"[red]No token with id {token_id!r}.[/red]")
        raise typer.Exit(1)
    console.print(_table([record]))
    if record.note:
        console.print(f"note: {record.note}")


@app.command("disable")
def disable_token(token_id: Annotated[str, typer.Argument(help="Token id.")]) -> None:
    """Disable a token, keeping the record for the audit trail."""
    _set_disabled(token_id, True)


@app.command("enable")
def enable_token(token_id: Annotated[str, typer.Argument(help="Token id.")]) -> None:
    """Re-enable a disabled token."""
    _set_disabled(token_id, False)


def _set_disabled(token_id: str, disabled: bool) -> None:
    store = get_store()
    record = store.set_disabled(token_id, disabled)
    if record is None:
        err_console.print(f"[red]No token with id {token_id!r}.[/red]")
        raise typer.Exit(1)
    word = "Disabled" if disabled else "Enabled"
    console.print(f"[green]{word}[/green] token [cyan]{token_id}[/cyan] ({record.name}).")


@app.command("revoke")
def revoke_token(
    token_id: Annotated[str, typer.Argument(help="Token id.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete a token permanently. Prefer 'disable' if you may want it back."""
    store = get_store()
    record = store.get(token_id)
    if record is None:
        err_console.print(f"[red]No token with id {token_id!r}.[/red]")
        raise typer.Exit(1)
    if not yes:
        typer.confirm(f"Permanently delete token {token_id} ({record.name})?", abort=True)
    store.delete(token_id)
    console.print(f"[green]Revoked[/green] token [cyan]{token_id}[/cyan].")
