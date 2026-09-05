"""`embedforge` - the command-line entry point.

Two independent CLIs live under one binary: `embedforge token ...` manages API
clients, `embedforge model ...` manages model files. They share nothing but the
configuration, so either can be used without the server running.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console

from embedforge.cli import models as models_cli
from embedforge.cli import tokens as tokens_cli
from embedforge.logging import configure_logging
from embedforge.version import __version__

app = typer.Typer(
    no_args_is_help=True,
    add_completion=True,
    help="EmbedForge: an embedding server with one active model.",
)
app.add_typer(tokens_cli.app, name="token")
app.add_typer(models_cli.app, name="model")

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"embedforge {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """EmbedForge command line."""
    del version
    # Quiet, human-readable logs for CLI use; `serve` reconfigures from settings.
    configure_logging("WARNING", "console")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address. Overrides config.")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port. Overrides config.")] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes. Overrides config.")] = None,
    reload: Annotated[
        bool, typer.Option(help="Reload on code changes (development only).")
    ] = False,
) -> None:
    """Run the HTTP server."""
    # Overrides go through the environment so reloader and worker subprocesses,
    # which re-read the configuration on their own, see the same values.
    if host is not None:
        os.environ["EMBEDFORGE_HOST"] = host
    if port is not None:
        os.environ["EMBEDFORGE_PORT"] = str(port)
    if workers is not None:
        os.environ["EMBEDFORGE_WORKERS"] = str(workers)

    from embedforge.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    from embedforge.runtime import apply_thread_limits

    # Must happen before the app (and therefore numpy and the model runtime) is imported.
    apply_thread_limits(settings.effective_intra_op_threads())

    import uvicorn

    configure_logging(settings.log_level, settings.log_format)

    uvicorn.run(
        "embedforge.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=None if reload else settings.workers,
        reload=reload,
        # Logging and access logs are ours; uvicorn's defaults would fight them.
        log_config=None,
        access_log=False,
        proxy_headers=settings.proxy_headers,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        timeout_keep_alive=settings.timeout_keep_alive,
        timeout_graceful_shutdown=settings.timeout_graceful_shutdown,
        limit_concurrency=settings.limit_concurrency,
        server_header=False,
    )


@app.command("config")
def show_config() -> None:
    """Print the effective configuration and where it came from."""
    from embedforge.config import get_settings

    settings = get_settings()
    console.print("[bold]Effective configuration[/bold]  (env prefix: EMBEDFORGE_)\n")
    for name, value in settings.model_dump().items():
        console.print(f"  {name:<28} {value}")
    console.print(
        f"\n  {'intra_op_threads (resolved)':<28} {settings.effective_intra_op_threads()}"
    )


if __name__ == "__main__":
    app()
