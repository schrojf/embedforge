# EmbedForge

An embedding server: `POST /v1/embed` for content you store, `POST /v1/query` for search
queries, one model loaded per process.

- **ONNX-first, CPU-first.** Eight multilingual models pre-configured, from a 135 MB
  quantized model to a 2.3 GB one, each documented with its trade-offs. GPU optional.
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
vectors so the server starts with no downloads. For real work, pick one:

```bash
embedforge model list                  # the catalog and what is downloaded
embedforge model download e5-base      # recommended starting point
EMBEDFORGE_MODEL_ID=e5-base embedforge serve
```

Model choice for Slovak and English is backed by [SkMTEB](https://arxiv.org/html/2606.13647v1)
rather than the global leaderboard, which ranks these models quite differently. Compare
them on your own data with `embedforge model compare`. See [models.md](docs/models.md).

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
