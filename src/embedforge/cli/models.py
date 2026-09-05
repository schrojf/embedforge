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
    """List every model this build can serve, and whether its files are present."""
    from embedforge.config import get_settings
    from embedforge.engine import registry
    from embedforge.engine.download import is_downloaded

    settings = get_settings()
    active = settings.model_id
    table = Table(title="Available models", title_justify="left")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Dim", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Sym")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    for spec in registry.list_specs():
        info = spec.info
        if registry.is_onnx_model(info.id):
            config = registry.onnx_config(info.id)
            if is_downloaded(settings, info.id, config):
                status = "[green]ready[/green]"
            elif config.needs_export:
                status = "[yellow]needs export[/yellow]"
            else:
                status = "[yellow]not downloaded[/yellow]"
        else:
            status = "[green]built in[/green]"
        marker = " [green]*[/green]" if info.id == active else ""
        table.add_row(
            f"{info.id}{marker}",
            str(info.dimension),
            f"{info.max_input_tokens:,}",
            "yes" if info.symmetric else "no",
            f"{info.size_mb:,} MB" if info.size_mb else "-",
            status,
        )
    console.print(table)
    console.print(
        f"\n[green]*[/green] active model: [cyan]{active}[/cyan]"
        "   (set EMBEDFORGE_MODEL_ID to change, then restart)"
    )
    console.print("[dim]embedforge model info ID  for the trade-offs of one model.[/dim]")


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


def _usable_models(settings: object) -> list[str]:
    """Models that can be loaded right now: built-in ones plus downloaded ones.

    Comparing everything in the catalog would mostly report models that were never
    downloaded, which is noise rather than a result.
    """
    from embedforge.engine import registry
    from embedforge.engine.download import is_downloaded

    ready: list[str] = []
    for model_id in registry.known_ids():
        if not registry.is_onnx_model(model_id):
            ready.append(model_id)
        elif is_downloaded(settings, model_id, registry.onnx_config(model_id)):  # pyright: ignore[reportArgumentType]
            ready.append(model_id)
    return ready


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

    model_ids = list(models) if models else _usable_models(get_settings())
    if not model_ids:
        err_console.print("[red]No models available.[/red] Download one first.")
        raise typer.Exit(1)
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


@app.command("download")
def download(
    model_id: Annotated[str, typer.Argument(help="Model id, as shown by 'model list'.")],
    force: Annotated[
        bool, typer.Option("--force", help="Re-fetch even if the files are already present.")
    ] = False,
) -> None:
    """Fetch a model's files and record their digests.

    Files are pinned to a commit sha, so this is reproducible: the same command gives
    the same bytes tomorrow.
    """
    from embedforge.config import get_settings
    from embedforge.engine import registry
    from embedforge.engine.download import download_model, is_downloaded
    from embedforge.engine.onnx_backend import model_directory
    from embedforge.errors import InvalidRequestError

    settings = get_settings()
    try:
        config = registry.onnx_config(model_id)
    except InvalidRequestError as error:
        err_console.print(f"[red]{error.message}[/red]")
        raise typer.Exit(1) from None

    spec = registry.get_spec(model_id)
    if not force and is_downloaded(settings, model_id, config):
        console.print(
            f"[green]{model_id}[/green] is already downloaded. Use --force to fetch it again."
        )
        raise typer.Exit(0)

    if config.needs_export:
        console.print(
            f"[cyan]{model_id}[/cyan] publishes no ONNX build, so one is exported locally "
            f"from {config.export_from} at revision {config.revision[:12]}."
        )
        console.print(
            "[dim]This downloads the source model and runs the exporter through uvx: "
            "several minutes, and roughly twice the final size in disk while it runs.[/dim]"
        )
    else:
        console.print(
            f"Downloading [cyan]{model_id}[/cyan] from {config.repo_id}"
            f" ({spec.info.size_mb:,} MB) at revision {config.revision[:12]}"
        )
    try:
        manifest = download_model(settings, model_id, config, force=force)
    except Exception as error:
        err_console.print(f"[red]Download failed:[/red] {error}")
        raise typer.Exit(1) from None

    total = sum(record.size for record in manifest.files.values())
    verb = "Exported" if config.needs_export else "Downloaded"
    console.print(
        f"[green]{verb}[/green] {len(manifest.files)} files "
        f"({total / 1e6:,.0f} MB) to {model_directory(settings, model_id)}"
    )
    console.print(f"Serve it with: EMBEDFORGE_MODEL_ID={model_id} embedforge serve")


@app.command("verify")
def verify(
    model_id: Annotated[str, typer.Argument(help="Model id, as shown by 'model list'.")],
    quick: Annotated[
        bool, typer.Option("--quick", help="Check presence and size only, skipping hashes.")
    ] = False,
) -> None:
    """Check a model's files against the digests recorded when it was downloaded.

    This is what distinguishes a complete model from a truncated download, which
    otherwise fails much later and much less clearly.
    """
    from embedforge.config import get_settings
    from embedforge.engine import registry
    from embedforge.engine.download import verify_model
    from embedforge.errors import InvalidRequestError

    try:
        config = registry.onnx_config(model_id)
    except InvalidRequestError as error:
        err_console.print(f"[red]{error.message}[/red]")
        raise typer.Exit(1) from None

    report = verify_model(get_settings(), model_id, config, deep=not quick)
    styles = {"ok": "green", "missing": "red", "corrupt": "red", "unverified": "yellow"}
    console.print(f"[bold]{model_id}[/bold]  {report.directory}")
    for check in report.checks:
        style = styles[check.status.value]
        detail = f"  [dim]{check.detail}[/dim]" if check.detail else ""
        console.print(f"  [{style}]{check.status.value:<10}[/{style}] {check.name}{detail}")

    if not report.has_manifest:
        console.print(
            "[yellow]No manifest.[/yellow] Re-download to record digests: "
            f"embedforge model download {model_id} --force"
        )
    if report.ok:
        console.print("[green]All files verified.[/green]")
    else:
        err_console.print(
            f"[red]{len(report.problems)} problem(s).[/red] Fix with: "
            f"embedforge model download {model_id} --force"
        )
        raise typer.Exit(1)


@app.command("remove")
def remove(
    model_id: Annotated[str, typer.Argument(help="Model id, as shown by 'model list'.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
) -> None:
    """Delete a model's downloaded files."""
    from embedforge.config import get_settings
    from embedforge.engine.download import remove_model
    from embedforge.engine.onnx_backend import model_directory

    settings = get_settings()
    directory = model_directory(settings, model_id)
    if not directory.exists():
        console.print(f"{model_id} is not downloaded.")
        raise typer.Exit(0)
    if not yes:
        typer.confirm(f"Delete {directory}?", abort=True)
    remove_model(settings, model_id)
    console.print(f"[green]Removed[/green] {directory}")
