"""`embedforge model` - inspect, compare, and prepare embedding models.

Model files are downloaded and verified here rather than at server startup, so a
container starts fast, starts offline, and cannot half-download a model under load.

Model comparison lives here too, rather than behind a development API endpoint: the
server loads exactly one model on purpose, and a code path that loads arbitrary models
per request is one that must never be reachable in production.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from embedforge.comparison import ComparisonResult

app = typer.Typer(no_args_is_help=True, help="Inspect, compare, and prepare embedding models.")
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


def _shorten(text: str, width: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "\u2026"


@app.command("compare")
def compare(
    models: Annotated[
        list[str] | None,
        typer.Argument(help="Model ids to compare. Defaults to every registered model."),
    ] = None,
    query: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="A search query. Repeatable."),
    ] = None,
    text: Annotated[
        list[str] | None,
        typer.Option("--text", "-t", help="A document to rank. Repeatable."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", "-f", help="File of documents, one per line."),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Results to show per query.")] = 5,
    rounds: Annotated[int, typer.Option("--rounds", help="Timed passes, averaged.")] = 1,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Run the same inputs through several models and compare what they return.

    Each model is loaded, used, and released in turn, so peak memory is one model
    rather than all of them.

        embedforge model compare -q "reset my password" -f docs.txt

    With queries, each model ranks the documents and the rankings are compared.
    Without queries, you get each model's document-to-document similarity instead.
    """
    from embedforge.comparison import compare_models
    from embedforge.config import get_settings
    from embedforge.engine import registry

    documents = list(text or [])
    if file is not None:
        if not file.exists():
            err_console.print(f"[red]No such file: {file}[/red]")
            raise typer.Exit(1)
        documents += [line.strip() for line in file.read_text("utf-8").splitlines() if line.strip()]
    if not documents:
        err_console.print(
            "[red]Nothing to embed.[/red] Pass documents with --text/-t or --file/-f, "
            'e.g. embedforge model compare -q "cats" -t "a cat" -t "a boat"'
        )
        raise typer.Exit(1)

    model_ids = list(models) if models else registry.known_ids()
    queries = list(query or [])

    result = compare_models(get_settings(), model_ids, documents, queries, rounds=rounds)
    if not result.runs:
        err_console.print("[red]No model could be loaded.[/red]")
        for model_id, reason in result.failures.items():
            err_console.print(f"  {model_id}: {reason}")
        raise typer.Exit(1)

    if as_json:
        _print_json(result, top_k)
    else:
        _print_report(result, top_k)

    if result.failures:
        for model_id, reason in result.failures.items():
            err_console.print(f"[yellow]Skipped {model_id}:[/yellow] {reason}")


def _print_report(result: "ComparisonResult", top_k: int) -> None:
    table = Table(title="Models compared", title_justify="left")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Dim", justify="right")
    table.add_column("Symmetric")
    table.add_column("Load", justify="right")
    table.add_column("ms/item", justify="right")
    table.add_column("items/s", justify="right")
    for run in result.runs:
        table.add_row(
            run.model_id,
            str(run.dimension),
            "yes" if run.symmetric else "no",
            f"{run.load_seconds:.2f}s",
            f"{run.ms_per_item:.2f}",
            f"{run.items_per_second:,.0f}",
        )
    console.print(table)

    if result.queries:
        for index, text in enumerate(result.queries):
            console.print(f"\n[bold]Query:[/bold] {text}")
            for run in result.runs:
                console.print(f"  [cyan]{run.model_id}[/cyan]")
                for rank, (document_index, score) in enumerate(run.ranking(index, top_k), start=1):
                    console.print(
                        f"    {rank}. {score:+.3f}  {_shorten(result.documents[document_index])}"
                    )
        _print_agreement(result)
    else:
        _print_document_similarity(result)


def _print_agreement(result: "ComparisonResult") -> None:
    if len(result.runs) < 2 or len(result.documents) < 2:
        return
    console.print("\n[bold]Ranking agreement[/bold] (Spearman, 1.0 = identical order)")
    table = Table(show_header=True)
    table.add_column("")
    for run in result.runs:
        table.add_column(run.model_id, justify="right")
    for row in result.runs:
        cells = [f"[cyan]{row.model_id}[/cyan]"]
        for column in result.runs:
            value = result.agreement(row, column)
            cells.append("-" if value != value else f"{value:+.2f}")
        table.add_row(*cells)
    console.print(table)
    console.print(
        "[dim]High agreement means the models retrieve the same things, so prefer the "
        "cheaper one. Low agreement means the choice matters: evaluate on real queries."
        "[/dim]"
    )


def _print_document_similarity(result: "ComparisonResult") -> None:
    limit = 8
    shown = result.documents[:limit]
    if len(shown) < 2:
        console.print(
            "\n[dim]Pass --query to rank documents, or more documents to compare them.[/dim]"
        )
        return
    for run in result.runs:
        console.print(f"\n[bold]Document similarity[/bold] - [cyan]{run.model_id}[/cyan]")
        matrix = run.document_similarity()
        table = Table(show_header=True)
        table.add_column("")
        for index in range(len(shown)):
            table.add_column(str(index), justify="right")
        for row in range(len(shown)):
            table.add_row(
                f"[cyan]{row}[/cyan] {_shorten(shown[row], 30)}",
                *[f"{matrix[row][column]:+.2f}" for column in range(len(shown))],
            )
        console.print(table)
    if len(result.documents) > limit:
        console.print(f"[dim]Showing the first {limit} of {len(result.documents)} documents.[/dim]")


def _print_json(result: "ComparisonResult", top_k: int) -> None:
    payload = {
        "documents": result.documents,
        "queries": result.queries,
        "failures": result.failures,
        "models": [
            {
                "id": run.model_id,
                "name": run.name,
                "dimension": run.dimension,
                "symmetric": run.symmetric,
                "load_seconds": round(run.load_seconds, 4),
                "ms_per_item": round(run.ms_per_item, 4),
                "items_per_second": round(run.items_per_second, 2),
                "rankings": [
                    [
                        {"document": index, "score": round(score, 6)}
                        for index, score in run.ranking(query_index, top_k)
                    ]
                    for query_index in range(len(result.queries))
                ],
            }
            for run in result.runs
        ],
        "agreement": {
            left.model_id: {
                right.model_id: (
                    None
                    if result.agreement(left, right) != result.agreement(left, right)
                    else round(result.agreement(left, right), 4)
                )
                for right in result.runs
            }
            for left in result.runs
        },
    }
    console.print_json(json.dumps(payload))
