# Operations

## Health checks

| Endpoint | Use it for | Fails when |
| --- | --- | --- |
| `/healthz` | Liveness. Restart the process if this fails. | The process is wedged. It never touches the model, so a busy model cannot fail it. |
| `/readyz` | Readiness. Route traffic only when this passes. | The model is still loading, or the server is shutting down. |

Keeping them separate matters: a slow model must not look like a dead process, or your
orchestrator will restart a container that was only busy.

`/readyz` also reports live engine state:

```json
{
  "status": "ready",
  "engine": {
    "ready": true, "inflight_items": 12, "queued_items": 4,
    "queue_max_size": 512, "max_batch_size": 32,
    "batch_wait_ms": 5.0, "inference_workers": 1
  }
}
```

## Logs

Structured JSON on stdout, one object per line, with uvicorn's own records in the same
format. `EMBEDFORGE_LOG_FORMAT=console` gives human-readable output for local work.

Every request logs one access line:

```json
{"event": "request", "status": 200, "duration_ms": 12.4, "method": "POST",
 "path": "/v1/embed", "request_id": "b8d0568d...", "principal": "de5ba8d6d33e",
 "level": "info", "timestamp": "2026-09-05T14:52:48.029165Z"}
```

`request_id` is echoed to the client in `X-Request-ID`, so a user's report of a failure
leads straight to the log line. `principal` is the token id, which is how you trace one
client's traffic. Tokens themselves are never logged.

Events worth alerting on:

| Event | Meaning |
| --- | --- |
| `inference_failed` | A model call raised. Every caller in that batch got a 500. |
| `token_file_invalid` | The token file failed to parse; the last good copy is still serving. |
| `token_file_missing` | No token file, so every request will 401. |
| `unhandled_exception` | A bug. The client got a generic 500; the detail is here. |

## Metrics

Prometheus exposition at `/metrics` (disable with `EMBEDFORGE_METRICS_ENABLED=false`).

| Metric | Type | Use |
| --- | --- | --- |
| `embedforge_http_requests_total` | counter | Traffic and error rate, by endpoint and status. |
| `embedforge_http_request_duration_seconds` | histogram | End-to-end latency. |
| `embedforge_inference_duration_seconds` | histogram | Model time, excluding queue wait. |
| `embedforge_queue_wait_seconds` | histogram | Time waiting for a free worker. |
| `embedforge_inference_batch_size` | histogram | Whether batching is working. |
| `embedforge_items_total` | counter | Inputs embedded, by task. |
| `embedforge_queue_items` | gauge | Items in flight right now. |
| `embedforge_inflight_batches` | gauge | Model calls executing right now. |
| `embedforge_rejected_total` | counter | Load shed, by reason (`overloaded`, `timeout`). |
| `embedforge_inference_errors_total` | counter | Failed model calls. |
| `embedforge_model` | info | The loaded model's id, name, and dimension. |

Endpoint labels use the route template, not the raw path, so label cardinality stays
bounded.

Suggested alerts:

```promql
# Sustained load shedding.
rate(embedforge_rejected_total[5m]) > 0.1

# Any model failure.
rate(embedforge_inference_errors_total[5m]) > 0

# Server errors.
rate(embedforge_http_requests_total{status=~"5.."}[5m]) > 0.05

# Latency regression.
histogram_quantile(0.95, rate(embedforge_http_request_duration_seconds_bucket[5m])) > 1
```

With more than one process, metrics are per-process; scrape each replica separately
rather than aggregating at the load balancer.

## Backup

One thing is irreplaceable: **the token file** (`EMBEDFORGE_TOKENS_FILE`). It holds only
digests, so a lost file means every client needs a new token.

Model files under `EMBEDFORGE_MODEL_DIR` are re-downloadable and pinned to commit shas,
so they need no backup — but do budget the disk: between 135 MB and 2.3 GB per model,
and `model compare` runs want more than one present at a time.

```bash
docker run --rm -v embedforge-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/embedforge-tokens-"$(date +%F)".tar.gz -C /data tokens.json
```

Treat the backup as a secret: the digests are not reversible, but the file is still a
list of who can reach your service. Model files under `EMBEDFORGE_MODEL_DIR` are
re-downloadable and need no backup.

## Capacity

Watch three numbers, in this order:

1. `embedforge_rejected_total` — anything sustained means you are over capacity now.
2. `queue_wait_seconds` p95 versus `inference_duration_seconds` p95 — if waiting
   dominates, you need more capacity, not a faster model.
3. Average batch size — if it stays far below `max_batch_size` under load, raise
   `EMBEDFORGE_BATCH_WAIT_MS` before adding hardware.

[performance.md](performance.md) has the queries and what each answer implies.

## Troubleshooting

**Every request returns 401.** Check the server can see the token file:
`docker compose run --rm embedforge config` prints the resolved path, and
`token list` prints its contents. In the container the file must be inside the mounted
volume. Note that a newly created token takes up to
`EMBEDFORGE_TOKENS_RELOAD_INTERVAL` seconds (default 5) to be accepted.

**`/readyz` stays 503.** The model has not loaded. Check the logs for `engine_started`;
if it never appears, the load failed and the process should have exited. `docker logs`
has the traceback.

**Startup fails with a missing model file.** The model was never downloaded on this
host, or the volume is not mounted where the server expects. The error names the path
and the command to fix it:
`docker compose run --rm embedforge model download <id>`.

**Startup fails with a corrupt or unreadable model graph.** Usually a truncated
download. Confirm with `embedforge model verify <id>`, then re-fetch with
`embedforge model download <id> --force`.

**Retrieval quality is bad but nothing is broken.** Check that clients use `/v1/query`
for queries and `/v1/embed` for stored content. With an asymmetric model the prefixes
differ, and using the wrong endpoint degrades ranking without any error. Also confirm
your stored vectors came from the model currently loaded — send the `model` field on
requests so a mismatch is a 400 rather than silently wrong results.

**Requests return 503 `overloaded`.** Working as designed: the queue is full. Either the
load is genuinely above capacity, or the client is sending many tiny requests instead of
batching. See [performance.md](performance.md).

**Requests return 504.** Work was accepted but exceeded `EMBEDFORGE_REQUEST_TIMEOUT`.
Usually very large batches, or a queue that is deep but not yet full.

**Latency is fine alone and terrible under load.** Look at batch size first. If the
average is near 1 while concurrency is high, `batch_wait_ms` is too small for your
traffic pattern.

**The container is killed during deploys.** `stop_grace_period` (Compose) or
`TimeoutStopSec` (systemd) must exceed `EMBEDFORGE_TIMEOUT_GRACEFUL_SHUTDOWN`.

**The version reads `0.0.0`.** The image was built without
`--build-arg VERSION=...`. See [deployment.md](deployment.md).
