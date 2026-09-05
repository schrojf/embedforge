"""`embedforge model` - inspect and prepare embedding models.

Model files are downloaded and verified here rather than at server startup, so a
container starts fast, starts offline, and cannot half-download a model under load.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Inspect and prepare embedding models.")
console = Console()
err_console = Console(stderr=True)


@app.command("list")
def list_models() -> None:
    """List every model this build can serve, and which one is configured."""
    from embedforge.config import get_settings
    from embedforge.engine import registry

    active = get_settings().model_id
    table = Table(title="Available models", title_justify="left")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Dim", justify="right")
    table.add_column("Max tokens", justify="right")
    table.add_column("Symmetric")
    table.add_column("Size", justify="right")
    table.add_column("Description")
    for spec in registry.list_specs():
        info = spec.info
        marker = " [green](active)[/green]" if info.id == active else ""
        table.add_row(
            f"{info.id}{marker}",
            str(info.dimension),
            str(info.max_input_tokens),
            "yes" if info.symmetric else "no",
            f"{info.size_mb} MB" if info.size_mb is not None else "-",
            info.description.split(". ")[0],
        )
    console.print(table)
    console.print(f"\nActive model: [cyan]{active}[/cyan]  (set EMBEDFORGE_MODEL_ID to change)")


@app.command("info")
def model_info(
    model_id: Annotated[str, typer.Argument(help="Model id, as shown by 'model list'.")],
) -> None:
    """Show one model in full, including its trade-offs."""
    from embedforge.engine import registry
    from embedforge.errors import InvalidRequestError

    try:
        spec = registry.get_spec(model_id)
    except InvalidRequestError as error:
        err_console.print(f"[red]{error.message}[/red]")
        raise typer.Exit(1) from None

    info = spec.info
    console.print(f"[bold cyan]{info.id}[/bold cyan]  ({info.name})")
    console.print(f"  {info.description}\n")
    console.print(f"  dimension:     {info.dimension}")
    console.print(f"  max tokens:    {info.max_input_tokens}")
    console.print(f"  modalities:    {', '.join(m.value for m in info.modalities)}")
    console.print(f"  symmetric:     {'yes' if info.symmetric else 'no'}")
    console.print(f"  license:       {info.license}")
    if info.size_mb is not None:
        console.print(f"  size on disk:  {info.size_mb} MB")
    if info.pros:
        console.print("\n[green]Pros[/green]")
        for item in info.pros:
            console.print(f"  + {item}")
    if info.cons:
        console.print("\n[yellow]Cons[/yellow]")
        for item in info.cons:
            console.print(f"  - {item}")
