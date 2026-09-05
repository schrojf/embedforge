# Configuration

Every setting is an environment variable prefixed with `EMBEDFORGE_`. A `.env` file in
the working directory is read too, which is convenient locally; production should set
real environment variables. `.env.example` is a starting point.

Print what a process actually resolved, including defaults:

```bash
embedforge config
```

The reference below is generated from the settings model, so it cannot drift from the
code.


## Runtime

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDFORGE_ENV` | `prod` | Deployment environment. `dev` enables friendlier defaults, never use in production. |
| `EMBEDFORGE_LOG_LEVEL` | `INFO` | Minimum level to log: DEBUG, INFO, WARNING, ERROR. |
| `EMBEDFORGE_LOG_FORMAT` | `json` | `json` for log collectors, `console` for human-readable local output. |

## HTTP server

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDFORGE_HOST` | `127.0.0.1` | Bind address. The container image sets 0.0.0.0; do not expose that directly. |
| `EMBEDFORGE_PORT` | `8000` | Bind port. |
| `EMBEDFORGE_WORKERS` | `1` | OS processes serving HTTP. Keep this at 1 for CPU inference: every worker loads its own copy of the model and starts its own inference threads, so more workers means more memory and CPU oversubscription, not more throughput. Scale with `inference_workers`, `max_batch_size`, and horizontally with containers instead. |
| `EMBEDFORGE_ROOT_PATH` | *(empty)* | ASGI root path, when mounted behind a reverse proxy on a subpath. |
| `EMBEDFORGE_PROXY_HEADERS` | `True` | Trust X-Forwarded-* headers. Only meaningful behind a reverse proxy. |
| `EMBEDFORGE_FORWARDED_ALLOW_IPS` | `127.0.0.1` | Which peers may set X-Forwarded-*. Set this to your proxy; never to `*` in public. |
| `EMBEDFORGE_TIMEOUT_KEEP_ALIVE` | `5` | Seconds an idle keep-alive connection is held open. |
| `EMBEDFORGE_TIMEOUT_GRACEFUL_SHUTDOWN` | `30` | Seconds to let in-flight requests finish after SIGTERM. |
| `EMBEDFORGE_LIMIT_CONCURRENCY` | `512` | Maximum concurrent connections uvicorn accepts before returning 503. |
| `EMBEDFORGE_CORS_ORIGINS` | *(empty)* | Browser origins allowed to call the API, comma-separated. Empty disables CORS. |
| `EMBEDFORGE_DOCS_ENABLED` | `True` | Serve the OpenAPI schema and interactive docs at /docs and /redoc. |

## Data locations

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDFORGE_TOKENS_FILE` | `data/tokens.json` | Where API tokens live. Written by the CLI, read by the server. Back this up. |
| `EMBEDFORGE_MODEL_DIR` | `data/models` | Where downloaded model files live. |

## Authentication

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDFORGE_AUTH_ENABLED` | `True` | Require a bearer token on API endpoints. Only disable for local development. |
| `EMBEDFORGE_TOKENS_RELOAD_INTERVAL` | `5.0` | Seconds between checks for changes to the token file, so the CLI can add or revoke tokens without restarting the server. Set to 0 to reload on every request. |

## Model and inference

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDFORGE_MODEL_ID` | `dev-hash` | The single model loaded at startup. See `embedforge model list`. |
| `EMBEDFORGE_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` to prefer GPU when its runtime is available. |
| `EMBEDFORGE_INFERENCE_WORKERS` | `1` | Threads running model inference concurrently. Inference releases the GIL, so these are real parallel workers. One is right for most CPU deployments because a single session already uses every core; raise it only alongside a matching drop in `intra_op_threads`. |
| `EMBEDFORGE_INTRA_OP_THREADS` | `0` | Threads used *inside* one inference call. 0 means auto (cores // inference_workers). |
| `EMBEDFORGE_MAX_BATCH_SIZE` | `32` | Largest batch handed to the model in one call. |
| `EMBEDFORGE_BATCH_WAIT_MS` | `5.0` | How long the dispatcher waits to merge concurrent requests into one batch. This is the main latency/throughput dial: a few milliseconds buys a large throughput win under concurrency and is invisible next to inference time. Set to 0 to disable waiting (batches still form from already-queued work). |
| `EMBEDFORGE_QUEUE_MAX_SIZE` | `512` | Items allowed to wait for inference. Beyond this the server sheds load with 503. |
| `EMBEDFORGE_REQUEST_TIMEOUT` | `30.0` | Seconds a request may wait for its embeddings before returning 504. |
| `EMBEDFORGE_MAX_ITEMS_PER_REQUEST` | `256` | Largest list a single request may submit. |
| `EMBEDFORGE_MAX_INPUT_CHARS` | `32768` | Per-item input limit, checked before tokenization to bound the work a caller can buy. |
| `EMBEDFORGE_NORMALIZE_EMBEDDINGS` | `True` | L2-normalize output vectors, so cosine similarity is a plain dot product. |

## Observability

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDFORGE_METRICS_ENABLED` | `True` | Expose Prometheus metrics at /metrics. Keep this endpoint off the public internet. |

## Notes

**Booleans** accept `true`/`false`, `1`/`0`, `yes`/`no`.

**`EMBEDFORGE_CORS_ORIGINS`** takes a comma-separated list:
`EMBEDFORGE_CORS_ORIGINS=https://app.example.com,https://admin.example.com`.

**Changing a setting requires a restart.** The one exception is the token file, which
the server re-reads when it changes (see
[authentication.md](authentication.md)).

**Which settings actually matter for throughput** is covered in
[performance.md](performance.md); do not tune `inference_workers`, `max_batch_size`, or
`batch_wait_ms` without reading it.
