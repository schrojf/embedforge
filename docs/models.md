# Models

A process loads exactly one model, chosen by `EMBEDFORGE_MODEL_ID` at startup. Model
files are downloaded ahead of time with the CLI, never at startup.

```bash
embedforge model list             # the catalog, with download status
embedforge model info e5-base     # one model in full, with its trade-offs
embedforge model download e5-base
embedforge model verify e5-base
```

## The catalog

All of these are multilingual (Slovak and English included), permissively licensed, and
pinned to a specific commit so downloads are reproducible.

| ID | Dim | Context | Symmetric | Download | Slovak (SkMTEB) |
| --- | --- | --- | --- | --- | --- |
| `e5-small-int8` | 384 | 512 | no | 135 MB | 70.32 (before quantization) |
| `e5-small` | 384 | 512 | no | 487 MB | 70.32 |
| `e5-base-int8` | 768 | 512 | no | 296 MB | 72.39 (before quantization) |
| `e5-base` | 768 | 512 | no | 1.1 GB | 72.39 |
| `e5-large-instruct` | 1024 | 512 | no | 2.3 GB | **77.49** |
| `bge-m3` | 1024 | **8192** | yes | 2.3 GB | 74.43 |
| `gte-base` | 768 | **8192** | yes | 1.3 GB | 71.76 |
| `gte-base-int8` | 768 | **8192** | yes | 357 MB | 71.76 (before quantization) |
| `dev-hash` | 384 | — | yes | none | not a real model |

`dev-hash` is the default so the server starts with no downloads. It produces
deterministic but **meaningless** vectors: use it for smoke tests and CI, never for
retrieval.

## Where these numbers come from

Two benchmarks, because they disagree and the disagreement matters.

**MTEB** is the general multilingual leaderboard. **SkMTEB**
([arXiv 2606.13647](https://arxiv.org/html/2606.13647v1)) evaluates 31 models on Slovak
specifically, and its ranking is quite different. Three findings shaped this catalog:

- **The globally-strongest small model is the weakest here.** `embeddinggemma-300m` is
  the top-ranked text-only multilingual model under 500M on MTEB overall, and scores
  69.25 on Slovak — below `multilingual-e5-small`, which is a third of its size. It is
  not in the catalog for that reason.
- **Scale stops paying around 600M.** `jina-embeddings-v4` (3.8B) scores 72.44, *below*
  `snowflake-arctic-embed-l-v2.0` at 568M. There is no reason to serve a multi-billion
  parameter embedding model on a CPU box.
- **Slovak-specific models lose to multilingual ones.** The SlovakBERT-derived embedding
  models underperform plain multilingual alternatives, so there is nothing to gain by
  going that route.

Treat all of this as a starting point. Benchmarks describe someone else's data;
`embedforge model compare` exists so you can settle it on yours.

## Choosing

**Start with `e5-base`.** 768 dimensions, good Slovak and English, comfortably fast on a
CPU. Then move only if something pushes you:

| If | Then |
| --- | --- |
| Latency or memory is tight | `e5-small-int8` — 135 MB, the fastest real model here. |
| Retrieval quality matters most | `e5-large-instruct` — the best open Slovak score, at 2.3 GB and the slowest inference. |
| Documents exceed 512 tokens | `bge-m3` (best quality) or `gte-base-int8` (cheapest long context). |
| You want e5-base but smaller | `e5-base-int8` — compare it first; quantization is not free. |

Dimension is a running cost, not just a download: 1024-dimensional vectors cost roughly
2.7× what 384-dimensional ones cost to store and search. That difference usually
outlives the latency difference.

### Is int8 worth it?

Quantized variants are about a quarter of the size and typically around twice as fast on
CPU, at some accuracy cost that depends on your data. That is exactly what the
comparison tool is for:

```bash
embedforge model download e5-base
embedforge model download e5-base-int8
embedforge model compare e5-base e5-base-int8 -q "your real query" -f your-corpus.txt
```

If the ranking agreement is near 1.0 on your queries, take the quantized one.

## Comparing models before you commit

```bash
embedforge model compare -q "how do I reset my password" -f corpus.txt
```

With no model ids it compares everything currently downloaded. Per model you get ranked
results with cosine scores, load time, and milliseconds per item; across models you get
a rank-agreement matrix (Spearman, 1.0 = identical ordering).

- **High agreement** means the models retrieve the same things for your queries. Take
  the cheaper one; the choice does not matter.
- **Low agreement** means the choice does matter, and no benchmark average will settle
  it. Judge the rankings yourself.

Each model is loaded, used, and released in turn, so peak memory is one model rather
than all of them.

### Why this is a CLI and not a development API

The obvious alternative is an endpoint that loads a model per request. This project
deliberately does not do that:

- Serving several models means either paying the load cost on every request or holding
  every model in memory — the exact costs the one-model-per-process design avoids.
- A "load any model the caller names" code path must never be reachable in production,
  and the safest way to guarantee that is for it not to exist in the server.
- Offline comparison can measure things a request cannot, such as load time, and can
  compare rankings *across* models rather than returning one vector at a time.

If you want comparisons over HTTP later, run one container per model behind the same
API. That keeps every process single-model.

## Symmetric and asymmetric models

An **asymmetric** model encodes a query differently from a stored passage, using a
prefix it was trained with. The E5 models here prepend `query: ` and `passage: `;
`e5-large-instruct` puts a task description on the query side. An **asymmetric** model
therefore returns *different* vectors from `/v1/embed` and `/v1/query`.

A **symmetric** model (`bge-m3`, `gte-base`) uses one representation for everything, so
both endpoints return identical vectors.

Every response reports `"symmetric": true|false`. Use the endpoint that matches your
intent regardless, so switching between the two kinds stays a configuration change.

These prefixes are part of each model's definition rather than something callers pass.
Getting them wrong does not raise an error — it silently produces vectors that look
normal and retrieve badly, which is the worst kind of bug to have in a search system.

**A note on E5 scores.** E5 models compress cosine similarity into a narrow band: an
unrelated pair may still score 0.73 where a good match scores 0.89. The *ordering* is
what carries the signal. Do not port an absolute similarity threshold from another model
family.

## Downloading and verifying

```bash
embedforge model download e5-base          # ~1.1 GB, pinned to a commit sha
embedforge model verify e5-base            # re-hash every file
embedforge model verify e5-base --quick    # presence and size only
embedforge model remove e5-base
```

Downloads record a `manifest.json` of sha256 digests next to the files. `verify` checks
against it, which is what distinguishes a complete model from a truncated download —
otherwise the failure surfaces much later and much less clearly, as a corrupt-graph
error at startup.

Files live under `EMBEDFORGE_MODEL_DIR` (`/var/lib/embedforge/models` in the container),
one directory per model id, mirroring the repository layout. Quantized variants get
their own directory even though they share an upstream repository, so they never
collide.

## Switching models

Changing `EMBEDFORGE_MODEL_ID` requires a restart, and **invalidates every vector you
have stored**: embeddings from different models are not comparable, even at the same
dimension. A model change means re-embedding your corpus.

The `model` field on a request is the guard. Send the model your stored vectors came
from, and the server returns 400 if it has a different one loaded, rather than silently
returning vectors that will not match.

## Adding a model

For another ONNX text model, add an entry to `CATALOG` in
`src/embedforge/engine/catalog.py`:

```python
(
    ModelInfo(id="my-model", name="org/repo", dimension=768, max_input_tokens=512,
              symmetric=False, description=..., pros=(...), cons=(...),
              license="MIT", size_mb=1100),
    OnnxModelConfig(
        repo_id="org/repo",
        revision="<full commit sha>",     # never a branch
        onnx_file="onnx/model.onnx",
        pooling=Pooling.MEAN,             # must match how it was trained
        max_seq_length=512,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
),
```

Check the upstream `1_Pooling/config.json` for the pooling mode and the model card for
the prefixes. Tests enforce that the revision is a full sha, that pros and cons are
filled in, and that prefixes are consistent with the symmetry flag.

For a model that is not ONNX text — a different runtime, or another modality —
implement `EmbeddingBackend` instead; see [architecture.md](architecture.md).
