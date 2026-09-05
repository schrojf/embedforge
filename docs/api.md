# HTTP API

Base URL is the server root. Versioned endpoints live under `/v1`; probes and metrics
do not, because they belong to the deployment rather than the API.

All requests and responses are JSON. Authenticate with
`Authorization: Bearer ef_<id>_<secret>` — see [authentication.md](authentication.md).

## `POST /v1/embed`

Vectorize content you intend to store and search over. Requires the `embed` scope.

```json
{
  "input": ["first document", "second document"],
  "model": "dev-hash"
}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input` | string or array of strings | yes | The text to embed. |
| `model` | string | no | A guard: the request fails if the server has a different model loaded. |

Unknown fields are rejected, so a typo in a field name is an error rather than silence.

Response:

```json
{
  "model": "dev-hash",
  "task": "document",
  "dimension": 384,
  "normalized": true,
  "symmetric": true,
  "data": [
    {"index": 0, "embedding": [0.021, -0.045, "..."]},
    {"index": 1, "embedding": [-0.013, 0.067, "..."]}
  ],
  "usage": {"items": 2, "characters": 30}
}
```

`data` is in request order, and `index` states that explicitly. When `normalized` is
true the vectors are unit length, so cosine similarity is a plain dot product.

## `POST /v1/query`

Identical request and response shape, for search queries. Requires the `query` scope.

**When the two endpoints return the same vector.** Some models use one representation
for everything (`symmetric: true` in the response); for those, `/v1/embed` and
`/v1/query` genuinely return identical vectors, and the response says so. Others encode
a query differently from a passage — usually by prepending an instruction — and the
results differ.

Use the endpoint that matches your intent, not the one that happens to be equivalent
today. Then switching the server to an asymmetric model is a configuration change
rather than a client rewrite.

## `GET /v1/user`

Reports the token making the call. Requires any valid token. This is the credential
check: a 200 means the token is valid, live, and reaching this server.

```json
{
  "id": "de5ba8d6d33e",
  "name": "my-client",
  "scopes": ["embed", "models:read", "query"],
  "authenticated": true,
  "created_at": "2026-09-05T14:52:48.029165Z",
  "expires_at": null,
  "expires_in_seconds": null,
  "endpoints": ["POST /v1/embed", "POST /v1/query", "GET /v1/models"]
}
```

`authenticated` is false only when the server runs with `EMBEDFORGE_AUTH_ENABLED=false`.

## `GET /v1/models` and `GET /v1/models/{id}`

Every model this build can serve, which one is loaded, and each model's dimension,
maximum input, license, and documented trade-offs. Requires the `models:read` scope.

One model is active per process; switching means changing `EMBEDFORGE_MODEL_ID` and
restarting.

## Probes and metrics

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /` | none | Service name, version, active model. |
| `GET /healthz` | none | Liveness. Never touches the model, so a busy model cannot fail it. |
| `GET /readyz` | none | Readiness: 200 once the model is loaded and warm, 503 before. Includes engine stats. |
| `GET /metrics` | none | Prometheus exposition. Keep it off the public internet. |
| `GET /docs`, `/redoc`, `/openapi.json` | none | Interactive docs, when `EMBEDFORGE_DOCS_ENABLED` is true. |

## Errors

Every failure uses one envelope:

```json
{
  "error": {"type": "payload_too_large", "message": "300 inputs exceeds the limit of 256 per request."},
  "request_id": "b8d0568df77f448d83e6e36c9234c875"
}
```

`request_id` also comes back in the `X-Request-ID` response header and appears in the
server logs, so a user-reported failure can be found directly. If the caller sends
`X-Request-ID`, that value is used instead.

| Status | `error.type` | Cause and what to do |
| --- | --- | --- |
| 400 | `invalid_request_error` | Empty input, or a `model` that is not the loaded one. Fix the request. |
| 401 | `authentication_error` | Missing, malformed, unknown, expired, or disabled token. |
| 403 | `permission_error` | Valid token without the scope this endpoint needs. |
| 413 | `payload_too_large` | Over `max_items_per_request` or `max_input_chars`. Split the work. |
| 422 | `invalid_request_error` | Body failed validation; `error.details` says which field. |
| 503 | `overloaded` | The inference queue is full. Retry with backoff; `Retry-After` is set. |
| 503 | `model_not_ready` | Still starting, or shutting down. Wait for `/readyz`. |
| 504 | `timeout` | Work did not finish within `EMBEDFORGE_REQUEST_TIMEOUT`. |

A 503 with `overloaded` is deliberate load shedding, not a bug: see
[performance.md](performance.md).

## Future input types

`input` is text today. Other modalities (images first) will arrive as an additional
request field rather than a change to `input`, so existing clients keep working. The
internals are already modality-aware; see [architecture.md](architecture.md).
