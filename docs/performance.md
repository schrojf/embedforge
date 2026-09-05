# Performance and concurrency

The short version: **one process, an async HTTP layer, a small pool of inference
threads, and dynamic batching in front of them.** The rest of this document explains why
that shape, and which dial to turn when it is not fast enough.

## Why not the usual answers

**Why not many worker processes?** The reflex for a Python web service is
`--workers $(nproc)`. It is wrong for CPU inference. Each worker loads its own copy of
the model (memory multiplied by N) and starts its own thread pool, so N workers on M
cores create N×M threads competing for M cores. Throughput drops and tail latency gets
much worse. `EMBEDFORGE_WORKERS` defaults to 1 for that reason.

**Why threads work here at all.** The GIL usually makes threads useless for CPU-bound
Python. It does not apply: ONNX Runtime and the Hugging Face tokenizers release the GIL
for the duration of their work. A thread running inference is genuinely parallel, so a
thread pool is the right primitive and it shares one copy of the model.

**Why batching is the real win.** An embedding model is a matrix multiply. Feeding it 32
texts at once costs far less than 32 separate calls: fixed per-call overhead is paid
once, and the arithmetic is what CPUs are best at. Batching, not parallelism, is where
most of the throughput comes from.

## How a request is served

```
HTTP request ──┐
HTTP request ──┼──► dispatcher (merges across requests) ──► batch ──► inference thread
HTTP request ──┘         ▲                                                    │
                         └──────────── results scattered back ◄───────────────┘
```

1. The endpoint validates the request and submits its inputs to the engine, then awaits
   a future. The event loop is never blocked, so the server keeps accepting connections
   while the model works.
2. A single dispatcher coroutine collects inputs. It takes everything already queued,
   then waits up to `EMBEDFORGE_BATCH_WAIT_MS` for more — so requests that arrive within
   a few milliseconds of each other travel together.
3. Full batches (up to `EMBEDFORGE_MAX_BATCH_SIZE`) run on the inference thread pool.
4. Results are scattered back to the requests they came from, in the original order.

Two details that matter in production:

- **A large request is split.** 200 inputs with `max_batch_size=32` becomes several
  batches, reassembled in order. One big request cannot monopolize the model.
- **Abandoned work is dropped.** If a caller times out or disconnects, its items are
  skipped when the next batch forms. Under load you do not want to spend cores
  computing answers nobody is waiting for.
- **Document and query work never share a batch**, since asymmetric models encode them
  differently. The dispatcher alternates between the two so neither starves.

## Backpressure: the queue is bounded on purpose

Admission control counts **items**, not requests. When more than
`EMBEDFORGE_QUEUE_MAX_SIZE` items are in flight, new requests are rejected immediately
with `503 overloaded` and a `Retry-After` header.

This is deliberate. An unbounded queue under sustained overload does not serve more
traffic; it just makes every client wait past its own timeout, so the server burns
cores producing results that are thrown away. Shedding load early keeps the requests it
does accept fast, and tells clients to back off.

Clients should retry a 503 with exponential backoff and jitter. A 504 (`timeout`) means
work was accepted but did not finish within `EMBEDFORGE_REQUEST_TIMEOUT`; treat it the
same way.

## The dials

| Setting | Default | Raise it when | Cost of raising |
| --- | --- | --- | --- |
| `EMBEDFORGE_MAX_BATCH_SIZE` | 32 | Throughput matters more than single-request latency. | Slower worst-case latency; more memory per call. |
| `EMBEDFORGE_BATCH_WAIT_MS` | 5 | Batches are small under load (see metrics below). | Adds directly to every request's latency. |
| `EMBEDFORGE_INFERENCE_WORKERS` | 1 | Cores sit idle with one worker (rare on CPU). | Only useful with a matching drop in `intra_op_threads`. |
| `EMBEDFORGE_INTRA_OP_THREADS` | auto | You need explicit control. | Auto is `cores / inference_workers`, which is usually right. |
| `EMBEDFORGE_QUEUE_MAX_SIZE` | 512 | You would rather queue than shed, and clients wait patiently. | Longer waits before the 503 that was going to happen anyway. |
| `EMBEDFORGE_REQUEST_TIMEOUT` | 30s | Large batches legitimately take longer. | Failing requests occupy capacity for longer. |

### Threads, and why `serve` sets environment variables

`intra_op_threads` is the thread count *inside* one model call. Left at `0`, it resolves
to `cores / inference_workers`, where cores comes from the process's CPU affinity — so a
container limited to 4 CPUs computes 4, not the host's 64.

`embedforge serve` exports `OMP_NUM_THREADS` and its siblings before numpy or the model
runtime is imported, because those libraries read the environment once at load time and
otherwise start one thread per host core. If you set those variables yourself, they are
respected.

The rule of thumb: **`inference_workers × intra_op_threads` should equal your CPU
budget.** One worker with all the cores is the best default for CPU inference; several
workers each with a few cores can help when requests are small and latency-sensitive.

## Reading the metrics

The numbers that tell you what to change, all at `/metrics`:

```promql
# Average batch size. Well under max_batch_size under load means batching is not
# paying off: raise batch_wait_ms.
rate(embedforge_inference_batch_size_sum[5m]) / rate(embedforge_inference_batch_size_count[5m])

# Queue wait vs. inference time. Wait dominating means you are capacity-bound, not
# model-bound: add replicas or a smaller model.
histogram_quantile(0.95, rate(embedforge_queue_wait_seconds_bucket[5m]))
histogram_quantile(0.95, rate(embedforge_inference_duration_seconds_bucket[5m]))

# Load shedding. Anything sustained here means you are over capacity.
rate(embedforge_rejected_total[5m])

# Items actually embedded per second.
rate(embedforge_items_total[5m])
```

`GET /readyz` shows the same state without Prometheus: in-flight items, queued items,
and the configured limits.

## Scaling beyond one box

When one process on one machine is not enough, add replicas behind a load balancer
rather than workers inside one container. Each replica is independent — no shared state
beyond the token file — so round-robin is enough.

Before adding hardware, check the cheaper wins in order:

1. **Batch on the client.** One request with 64 texts beats 64 requests with one. It is
   usually the largest single improvement available.
2. **Cache.** Embeddings are a pure function of (model, text). If your corpus repeats,
   cache in the client and never send the request.
3. **Pick a smaller model.** Dimension and layer count drive latency directly; see
   [models.md](models.md).
