# EmbedForge

An embedding server: `POST /v1/embed` for content you store, `POST /v1/query` for search
queries, one model loaded per process.

- **ONNX-first, CPU-first.** Built for CPU inference, with GPU as an option.
- **Built for concurrency.** Requests from different callers are merged into batches,
  run on a thread pool that releases the GIL, and shed with 503 rather than queued
  without bound. See [performance.md](docs/performance.md).
- **Token auth.** Scoped bearer tokens managed by a CLI, hashed at rest, revocable
  without a restart.
- **Deployable.** A production container image, health and readiness probes, Prometheus
  metrics, and structured JSON logs.

## Quick start

```bash
make install
embedforge token create my-client    # prints the token once
embedforge serve
```

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"input": ["hello world"]}' http://127.0.0.1:8000/v1/embed
```

With Docker:

```bash
docker compose build
docker compose run --rm embedforge token create my-client
docker compose up -d
```

The default model is `dev-hash`, which produces deterministic but **meaningless**
vectors so the server runs with no downloads. Real ONNX models are the next step; see
[models.md](docs/models.md).

## Documentation

Everything is in [docs/](docs/README.md):
[quickstart](docs/quickstart.md) ·
[API](docs/api.md) ·
[authentication](docs/authentication.md) ·
[CLI](docs/cli.md) ·
[configuration](docs/configuration.md) ·
[deployment](docs/deployment.md) ·
[operations](docs/operations.md) ·
[performance](docs/performance.md) ·
[architecture](docs/architecture.md) ·
[models](docs/models.md)

## Development

```bash
make install   # sync the environment
make lint      # ruff, codespell, basedpyright
make test      # pytest
make           # all three
```

See [development.md](docs/development.md).

* * *

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
