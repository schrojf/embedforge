# Architecture

## Layout

```
src/embedforge/
  config.py          Settings: one flat, env-first surface (EMBEDFORGE_*)
  logging.py         structlog + stdlib unified into one JSON stream
  errors.py          Error taxonomy; each error knows its HTTP status
  metrics.py         Prometheus metric definitions
  runtime.py         Thread-limit env vars, applied before numpy loads
  comparison.py      Offline multi-model comparison, used by the CLI
  auth/
    models.py        Token, principal, scopes, hashing
    store.py         Atomic, mtime-reloading JSON token file
  engine/
    base.py          Backend interface, ModelInfo, inputs, task types
    batching.py      The dispatcher: merging, backpressure, thread pool
    registry.py      Model catalog: id -> (ModelInfo, factory)
    catalog.py       The ONNX models this build can serve, with their trade-offs
    onnx_backend.py  ONNX Runtime session, tokenizer, pooling, task prefixes
    download.py      Fetching model files and verifying them against a manifest
    dev_hash.py      Deterministic no-download backend
  schemas/           Pydantic request/response models
  api/
    app.py           Application factory and lifespan
    deps.py          Dependency injection: settings, engine, principal, scopes
    middleware.py    Request id, access log, HTTP metrics (raw ASGI)
    errors.py        Exception handlers -> one response envelope
    responses.py     orjson responses with native numpy support
    routers/         health, user, embeddings, models
  cli/               serve, token, model, config
```

## Request path

```
uvicorn
  └─ RequestContextMiddleware      request id, access log, metrics
      └─ FastAPI router
          └─ auth dependency        token -> principal -> scope check
              └─ endpoint           validate limits, build inputs
                  └─ InferenceEngine.embed()
                       ├─ admission control (bounded by items in flight)
                       ├─ dispatcher merges across requests into a batch
                       └─ ThreadPoolExecutor -> backend.embed()
```

The layers are deliberately independent: the HTTP layer knows nothing about models, and
the engine knows nothing about HTTP. That is what lets the model change without the API
changing, and vice versa.

## Design decisions

**One model per process.** Loading several models multiplies memory and makes latency
depend on which model a request happens to want. A process serves one model; run more
containers to serve more models.

**The engine owns concurrency, not the web layer.** All queueing, batching, and
backpressure live in one class, so the behavior under load is in one file rather than
spread across endpoints. See [performance.md](performance.md).

**Errors carry their own status.** Each error class knows its HTTP status and stable
`type` string, so handlers do not translate exceptions into status codes at the edge and
every failure comes back in the same envelope.

**A file, not a database, for tokens.** Tokens are few and change rarely. A file is
trivially backed up, mounted, and inspected, and the server picks up changes without a
restart. Adding a database would buy nothing today.

**Responses skip pydantic on the way out.** Embedding responses are large float arrays;
validating them back through pydantic costs milliseconds. The endpoint returns a
response built straight from the numpy matrix, serialized by orjson. The response model
still declares the shape for OpenAPI.

## Extension points

**Adding a model.** For another ONNX text model, add an entry to `CATALOG` in
`engine/catalog.py` — the shared `OnnxTextBackend` runs it. For a different runtime,
implement `EmbeddingBackend` (`embed(inputs, task) -> ndarray`, plus optional `load` and
`close`) and register a `ModelSpec`. Nothing else changes: the CLI, the `/v1/models`
endpoint, and the docs all read the registry. See [models.md](models.md).

**Pooling and prefixes live with the model.** How token vectors are reduced to one
vector, and what prefix each task gets, are properties of how a model was trained.
Guessing them at load time produces vectors that look correct and retrieve badly, so
they are declared per model and asserted by tests.

**Adding an input modality.** Text is the only implemented modality, but the pipeline is
already modality-aware rather than string-typed:

- `engine/base.py` defines `Modality` and `TextInput`; `EmbedInput` is the union point
  where `ImageInput` joins.
- `ModelInfo.modalities` declares what a model accepts, and `/v1/models` reports it.
- The engine queues, batches, and returns `EmbedInput` objects without inspecting them.

Adding images means: a new input type, a backend that accepts it, a request field on
`EmbedRequest` (a new field, so existing clients are unaffected), and validation for
size and content type. The queueing, batching, backpressure, auth, and metrics layers
are untouched.

**Asymmetric models.** `TaskType.DOCUMENT` and `TaskType.QUERY` already flow from the
endpoint to the backend, and batches are never mixed between them. A backend that
prepends different instructions per task needs no engine changes.

## What is not here yet

- Multi-model serving. One process serves one model by design; compare models offline
  with `embedforge model compare` (see [models.md](models.md)).
- Per-token rate limiting. Overload protection is global; use the reverse proxy for
  per-client quotas.
- Multi-process metric aggregation. Scrape each replica separately.
