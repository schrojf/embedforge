# Command line

One binary, three independent command groups. All of them read the same configuration
as the server ([configuration.md](configuration.md)), so they act on the same token file
and model directory.

```bash
embedforge --help
embedforge --version
```

In a container, the CLI is the entrypoint:
`docker run --rm -v embedforge-data:/var/lib/embedforge embedforge:local token list`.

## `embedforge serve`

Runs the HTTP server.

```bash
embedforge serve
embedforge serve --host 0.0.0.0 --port 8080
embedforge serve --reload            # development only
```

| Option | Effect |
| --- | --- |
| `--host`, `--port`, `--workers` | Override the configured values. |
| `--reload` | Restart on code changes. Forces a single process. |

Overrides are exported into the environment before uvicorn starts, so reloader and
worker subprocesses — which load their own configuration — see the same values.

`serve` also caps the BLAS and OpenMP thread pools before the model runtime is imported;
[performance.md](performance.md) explains why that matters.

## `embedforge token`

Manages API clients. See [authentication.md](authentication.md) for the concepts.

| Command | Purpose |
| --- | --- |
| `token create NAME [--scopes …] [--expires-in …] [--note …]` | Create a token and print it once. |
| `token list` | All tokens with status, scopes, and expiry. Never shows a secret. |
| `token show ID` | One token in detail. |
| `token disable ID` / `token enable ID` | Reversible suspension. |
| `token revoke ID [--yes]` | Permanent deletion. Prompts unless `--yes`. |

```bash
embedforge token create search-frontend --scopes query --expires-in 90d
embedforge token create indexer --scopes embed --note "nightly reindex job"
```

## `embedforge model`

Inspects the model catalog. Downloading and verifying model files land here alongside
the ONNX backends; see [models.md](models.md).

| Command | Purpose |
| --- | --- |
| `model list` | Every model this build can serve, marking the configured one. |
| `model info ID` | One model in full, including its pros and cons. |

## `embedforge config`

Prints the effective configuration — every setting, its resolved value, and what it
does — including the thread count derived from the available cores. Use it to confirm
what a container actually picked up:

```bash
docker compose run --rm embedforge config
```
