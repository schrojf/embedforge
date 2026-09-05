# Benchmarks

Every model in the catalog, measured on one real machine. Numbers here are not a
substitute for [SkMTEB](https://arxiv.org/html/2606.13647v1) or MTEB — those measure
retrieval quality on large datasets, this measures what the models cost to run *here*.

Reproduce with:

```bash
uv run python devtools/benchmark.py --out results.json
uv run python devtools/benchmark.py --markdown results.json
```

Measurements come in two kinds. The `ms/item` columns call the backend **directly**, so
they are the model's own cost with nothing else in the way. The `Idle` columns go
through `InferenceEngine`, the same dispatcher a real HTTP request goes through, so they
include queueing and the batch window.

The harness downloads one model, measures it, **deletes it, and moves on**, so peak disk
use is one model rather than the whole catalog (about 15 GB). Deletion happens in a
`finally` block, so a failed measurement still reclaims the space. Raw results are in
[benchmark-results.json](benchmark-results.json).

## The machine

| | |
| --- | --- |
| CPU | Intel Core i7-7700HQ @ 2.80 GHz (Kaby Lake, 2017) |
| Cores | 4 physical, 8 logical |
| RAM | 34 GB |
| Instruction sets | AVX2, FMA, F16C — **no AVX-512, no VNNI, no AMX** |
| Governor | `powersave` |
| OS | Ubuntu 24.04.4, kernel 6.8 |
| Runtime | Python 3.12.14, onnxruntime 1.29.0, CPUExecutionProvider |

**The missing instruction sets matter.** The quantized E5 builds are published as
`model_qint8_avx512_vnni.onnx`: they are tuned for AVX-512 VNNI, which this CPU does not
have. They still run — onnxruntime falls back to AVX2 — but they do not get the speedup
they were built for. That is measured below.

A `powersave` governor and a laptop CPU also mean these figures are a floor. A server
part with AVX-512 would change the quantized numbers substantially and the
full-precision ones less.

## Results

Latency is the best of three runs after a warmup, embedding documents of roughly 10–20
tokens. The `ms/item` columns call the model directly; the two `Idle` columns go through
the full serving path, dispatcher included. RSS is peak resident memory with the model
loaded once.

| Model | Disk | RSS | Load | ms/item b=1 | ms/item b=32 | items/s b=32 | Idle p50 | Idle @200 ms | Batch-stable | R@1 | Margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dev-hash` *(baseline)* | 0 MB | 76 MB | 0.0s | 0.0 | 0.0 | 35,837 | 0.3 ms | 0.3 ms | 1.0000 | 0.12 | −0.075 |
| `e5-small-int8` | 135 MB | 602 MB | 0.9s | 5.0 | 3.3 | **299** | 5.6 ms | 5.4 ms | 0.9959 | 1.00 | +0.058 |
| `e5-base-int8` | 296 MB | 859 MB | 1.2s | 11.4 | 10.3 | 98 | 15.0 ms | 12.1 ms | 0.9761 | 1.00 | +0.055 |
| `gte-base-int8` | 357 MB | 954 MB | 1.2s | 13.8 | 13.5 | 74 | 18.4 ms | 14.4 ms | 0.9733 | 1.00 | +0.245 |
| `e5-small` | 487 MB | 1,242 MB | 1.5s | 7.5 | 4.4 | 228 | 7.5 ms | 9.2 ms | 1.0000 | 1.00 | +0.058 |
| `e5-base` | 1,127 MB | 2,266 MB | 2.6s | 20.0 | 17.0 | 59 | 21.1 ms | 22.1 ms | 1.0000 | 1.00 | +0.072 |
| `gte-base` | 1,273 MB | 2,411 MB | 2.8s | 24.0 | 20.3 | 49 | 28.2 ms | 24.9 ms | 1.0000 | 1.00 | +0.261 |
| `e5-sk-large` | 1,462 MB | 1,988 MB | 3.2s | 70.8 | 58.3 | 17 | 82.5 ms | 80.7 ms | 1.0000 | 1.00 | +0.094 |
| `e5-large-instruct` | 2,253 MB | 1,715 MB | 2.1s | 72.3 | 52.5 | 19 | 73.0 ms | 72.4 ms | 1.0000 | 1.00 | +0.098 |
| `bge-m3` | 2,285 MB | 2,906 MB | 2.0s | 58.8 | 51.1 | 20 | 68.8 ms | 69.4 ms | 1.0000 | 1.00 | +0.293 |
| `jina-v3` | 2,310 MB | **6,909 MB** | 1.2s | 854.5 | 82.1 | 12 | 858.4 ms | 844.2 ms | 1.0000 | 1.00 | +0.578 |
| `qwen3-0.6b` | 2,412 MB | 2,906 MB | 2.4s | 116.6 | 111.1 | 9 | 121.0 ms | 127.7 ms | 1.0000 | 1.00 | +0.357 |

## The baseline

`dev-hash` is the empty run: it does no real inference, so its figures are the cost of
everything that is not the model.

- **76 MB resident** before any model is loaded — Python, onnxruntime, FastAPI, and this
  project's own code.
- **0.028 ms/item** of pipeline overhead: tokenization dispatch, pooling, normalization,
  batching machinery.

Both are negligible. At 3 ms/item, the cheapest real model spends **99% of its time
inside the model** and 1% in everything this project does. That is the right answer for
a serving layer, and it means the only performance decision that matters here is which
model you pick.

Its `R@1` of 0.12 (one query out of eight, by chance) and negative margin confirm the
evaluation set is measuring something real.

## What the numbers say

### Quantization still pays here, but only about 1.4×, not 2×

| Pair | items/s | Speedup | Disk |
| --- | --- | --- | --- |
| `e5-small-int8` vs `e5-small` | 328 vs 235 | 1.39× | 0.28× |
| `e5-base-int8` vs `e5-base` | 91 vs 61 | 1.50× | 0.26× |
| `gte-base-int8` vs `gte-base` | 74 vs 48 | 1.55× | 0.28× |

The catalog claims quantized builds are "typically around twice as fast on CPU". On this
machine they are not — because the E5 int8 builds target AVX-512 VNNI and this CPU has
AVX2 only. The **size** win is the full 3.5×; the **speed** win is roughly 1.5×.

That is still worth having. But if you are choosing a server to run this on, a CPU with
VNNI is where the rest of that speedup is.

### Memory costs two to five times the download

Peak RSS versus file size: `e5-small-int8` 4.4×, `e5-base` 2.0×, `bge-m3` 1.4×,
`jina-v3` 3.0×. The absolute figure is what matters for sizing a container:

- **`jina-v3` peaks at 6.9 GB.** [deployment.md](deployment.md) previously suggested 4 GB
  for it, which would have been killed by the OOM reaper. It is now corrected.
- `bge-m3` and `qwen3-0.6b` both peak at 3.2 GB.
- `e5-small-int8` fits comfortably in 1 GB.

Set container limits from this column, not from the download size.

### Batching is worth up to 10×, and it is not uniform

| Model | b=1 | b=32 | Gain |
| --- | ---: | ---: | ---: |
| `jina-v3` | 892.9 ms | 91.1 ms | **9.8×** |
| `e5-small-int8` | 5.1 ms | 3.0 ms | 1.7× |
| `e5-small` | 6.5 ms | 4.2 ms | 1.5× |
| `e5-base` | 20.0 ms | 16.4 ms | 1.2× |
| `qwen3-0.6b` | 119.0 ms | 120.7 ms | 1.0× |

This is the clearest vindication of the dynamic batching in
[performance.md](performance.md): `jina-v3` served one request at a time is unusable at
893 ms, and merely slow at 91 ms once requests are merged. The server does that merging
automatically for concurrent callers.

The other end is instructive too. `qwen3-0.6b` gains **nothing** from batching — a
28-layer decoder with a key/value cache is already saturating the cores on a single
item, so there is nothing left to parallelize. For that model, `max_batch_size` is not
the dial to turn; a GPU is.

### The batch window costs nothing when the server is idle

The two `Idle` columns are the same measurement at two settings of
`EMBEDFORGE_BATCH_WAIT_MS`: the 5 ms default, and a deliberately absurd 200 ms. Both go
through the dispatcher, one request at a time, with nothing else in flight.

**They are the same number.** Differences run from −4 ms to +7 ms, which is measurement
noise on a `powersave` laptop, and never the ~195 ms that a window actually being
waited on would add.

That is the point of the rule introduced in v0.5.1: the engine only holds a batch open
while every worker is busy. With a worker free there is no company coming that starting
now would miss, so a lone request is dispatched immediately. Before that rule, a single
request on an idle server paid the full window — measured on a synthetic 2 ms model,
**7.74 ms with the default 5 ms window against 2.47 ms after**, nearly four times the
actual work spent waiting for nothing.

If your traffic is sporadic — a query every few minutes — this was the worst case:
batching could never help, and the overhead was paid on every call. It no longer is, at
any window setting.

Subtracting the direct batch-of-one figure from the idle figure gives the serving
path's own cost. On the baseline it is **0.28 ms**: dispatcher, queueing, futures,
metrics and HTTP-free request handling combined. For real models it disappears into
noise.

### Retrieval quality: everything passes, so read the margin

Every real model scores R@1 = 1.00 and MRR = 1.00. The evaluation set — 8 queries over
12 documents, Slovak-dominant with English mixed in — is a **sanity check, not a
discriminator**: it proves a model works for this language pair and that pooling and
prefixes are configured correctly, and nothing more.

The margin column (best relevant score minus best irrelevant score) carries what signal
there is. `jina-v3` separates relevant from irrelevant by +0.578, `qwen3-0.6b` by +0.357,
`e5-base` by only +0.072.

**Do not read that as jina-v3 being eight times better than e5-base.** The E5 family
compresses cosine similarity into a narrow band by design — an unrelated pair still
scores around 0.7 — so margins are only comparable *within* a family. Across families
the number says how spread out a model's scores are, not how good it is. Use SkMTEB for
quality, this table for cost.

An earlier version of this benchmark scored a single "correct" document per query and
every model came out at exactly R@1 = 0.62. They were all failing the same three
questions — the ones answered by both a Slovak and an English document, where returning
the English one was scored as wrong. Relevance is a set here for that reason.

## What to actually run on this machine

- **`e5-small-int8`** — 328 items/s, 602 MB, 3 ms/item. Thirty times the throughput of
  the large models for a quality drop that SkMTEB puts at about 2 points. For a personal
  workload this is the obvious default, and the batch-stability caveat (0.9959) is small
  enough not to matter unless you need reproducible vectors.
- **`e5-base`** if you want more quality and can live with 61 items/s.
- **`e5-sk-large`** if the traffic is overwhelmingly Slovak — 16 items/s is still ample
  for a personal deployment, and it is the strongest Slovak model that fits in 2 GB.
- **Not `jina-v3` or `qwen3-0.6b`** on this CPU. Both are good models being asked to run
  on the wrong hardware: 11 and 8 items/s, with 3–7 GB of RAM. They belong on a GPU.

## Caveats

- Single machine, single run, laptop CPU on `powersave`. Treat the ratios as portable
  and the absolute numbers as not.
- Short inputs (10–20 tokens). Long-context models are penalized least by this and would
  look relatively better on long documents, which is what they are for.
- Quality is measured on 8 queries. That is enough to catch a broken configuration, and
  nowhere near enough to choose between two good models. Use
  `embedforge model compare` on your own corpus for that.
- Latency is best-of-three, so it reflects a warm cache and an otherwise idle machine.
  The `Idle` columns are a p50 of 15 sequential requests and are noisier: at these
  timescales a laptop on `powersave` moves several milliseconds run to run. Read them as
  "the same" or "not the same", not to two decimal places.
- RSS is captured with the model loaded **once**, before the engine measurements, which
  load it again. An earlier version of this table reported the high-water mark across
  all three loads and roughly doubled every figure — enough to have produced badly
  oversized container limits.
