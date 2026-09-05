# EmbedForge documentation

An embedding server with two endpoints — `/v1/embed` for content you store and
`/v1/query` for search queries — serving one model per process.

## Start here

| Document | What it covers |
| --- | --- |
| [quickstart.md](quickstart.md) | Run the server and get your first vector, in five minutes. |
| [api.md](api.md) | Every endpoint, with request and response shapes. |
| [authentication.md](authentication.md) | API tokens, scopes, and how to manage them. |
| [cli.md](cli.md) | The `embedforge` command line. |
| [configuration.md](configuration.md) | Every environment variable. |

## Running it for real

| Document | What it covers |
| --- | --- |
| [deployment.md](deployment.md) | Docker, Compose, reverse proxy, systemd, upgrades. |
| [operations.md](operations.md) | Health checks, metrics, logs, backup, troubleshooting. |
| [performance.md](performance.md) | How concurrency works here, and what to tune. |

## Understanding and extending it

| Document | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | How a request flows, and where the seams are. |
| [models.md](models.md) | The model catalog and how a model is selected. |
| [development.md](development.md) | Local workflow, tests, linting. |
| [installation.md](installation.md) | Installing uv and Python. |
