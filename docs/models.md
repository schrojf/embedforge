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
| `jina-v3` | 1024 | **8192** | no | 2.3 GB | 75.10 |
| `e5-sk-large` | 1024 | 512 | no | 1.5 GB (exported) | 74.70 |
| `bge-m3` | 1024 | **8192** | yes | 2.3 GB | 74.43 |
| `gte-base` | 768 | **8192** | yes | 1.3 GB | 71.76 |
| `gte-base-int8` | 768 | **8192** | yes | 357 MB | 71.76 (before quantization) |
| `dev-hash` | 384 | — | yes | none | not a real model |

Two entries need a note before you pick them:

- **`jina-v3` is CC-BY-NC-4.0** — non-commercial only. Fine for private use; a licence
  trap if this ever becomes a product.
- **`e5-sk-large` publishes no ONNX build**, so the first download exports one locally.
  See [locally exported models](#locally-exported-models).

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

For everything else that was evaluated and left out — Gemini, Qwen3, Jina v4, Nomic —
see [models considered but not shipped](#models-considered-but-not-shipped).

## Choosing

**Start with `e5-base`.** 768 dimensions, good Slovak and English, comfortably fast on a
CPU. Then move only if something pushes you:

| If | Then |
| --- | --- |
| Latency or memory is tight | `e5-small-int8` — 135 MB, the fastest real model here. |
| Retrieval quality matters most | `e5-large-instruct` — the best open Slovak score. |
| Your traffic is almost all Slovak | `e5-sk-large` — nearly the same quality at two-thirds the size. |
| Documents exceed 512 tokens | `bge-m3` (symmetric, simplest) or `gte-base-int8` (cheapest). |
| You want e5-base but smaller | `e5-base-int8` — compare it first; quantization is not free. |

### What this actually costs

Measured on one 8-core CPU, single item per call, so treat it as a ratio rather than a
number for your hardware:

| Model | Load | ms/item | items/s |
| --- | --- | --- | --- |
| `e5-small-int8` | 0.8s | 4.9 | 205 |
| `e5-sk-large` | 3.1s | 44 | 23 |
| `jina-v3` | 1.0s | 331 | 3 |

**`jina-v3` is 68x slower than `e5-small-int8` on CPU.** Its benchmark scores are real,
and so is that number: an 8192-token context model with 572M parameters is not a CPU
model in any practical sense. Batch throughput is better than these per-item figures
suggest (see [performance.md](performance.md)), but the ratio holds. If you want
`jina-v3`, plan for a GPU.

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

## Models considered but not shipped

The catalog is a short list drawn from a much longer one. This section records what else
was evaluated and why it is not here, so the research does not have to be repeated — and
so it is obvious what would change the answer.

Four rules decide inclusion:

1. **Self-hostable.** An API is not a model. This server exists so your corpus stays on
   your machine.
2. **An ONNX export exists**, ideally published by the model's authors.
3. **It runs on a CPU.** In practice that means roughly 600M parameters or fewer.
4. **There is Slovak evidence**, because that is what this deployment needs.

| Model | Size | Slovak (SkMTEB) | Why not |
| --- | --- | --- | --- |
| [gemini-embedding-001](https://ai.google.dev/gemini-api/docs/embeddings) | API | 77.23 | Not self-hostable. |
| text-embedding-3-large | API | 75.07 | Not self-hostable. |
| [snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | 568M | 72.54 | Beaten by e5-large-instruct at the same size. |
| [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 3.8B | 72.44 | Far too large for CPU; restrictive licence. |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 596M | 70.53 | Beaten by e5-base at half the size; no stable ONNX export. |
| [embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) | 308M | 69.25 | Last of the realistic candidates on Slovak. |
| [nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) | 475M (305M active) | not evaluated | No Slovak evidence; MoE routing is awkward in ONNX. |

### gemini-embedding-001 (and text-embedding-3-large)

Google's model is genuinely excellent — 77.23 on Slovak, 3072 dimensions truncatable to
128, top of the MTEB multilingual leaderboard — and it is ruled out on the first rule
rather than on quality. It is an API. Using it means sending every document and every
query to Google, paying $0.15 per million tokens forever, and having no service at all
when you are offline or they deprecate the endpoint.

The comparison worth making: it beats `e5-large-instruct` on Slovak by **0.26 points**.
That is the entire quality argument for giving up self-hosting. OpenAI's
`text-embedding-3-large` (75.07) is behind `e5-large-instruct` outright.

### Qwen3-Embedding

Apache-2.0, 32k context, dimensions selectable from 32 to 1024, instruction-aware, and
the 8B variant was ranked first on the MTEB multilingual leaderboard. It looks like an
obvious pick and is not one:

- **The 0.6B scores 70.53 on Slovak** — below `e5-base`, which is less than half its
  size. The global ranking does not transfer.
- **The 4B and 8B are not CPU models.** They would need a GPU, which is exactly the
  dependency this deployment avoids.
- **No official ONNX export**, and Qwen3-architecture export was not supported in the
  stable `optimum` release at the time of writing. That is an integration risk, not just
  a missing file.

Worth revisiting if you move to GPU and the export path stabilizes.

### jina-embeddings-v4 — the multimodal one

3.8B parameters, and the most architecturally interesting model on this list: one model
for **text and images**, with two output modes — a single 2048-dimensional vector
(truncatable to 128) for ordinary similarity search, and multi-vector output of 128
dimensions per token for late-interaction retrieval, which is worth 7-10% on visual
tasks.

For text on CPU it is a non-starter: 3.8B parameters, and 72.44 on Slovak puts it
*below* several 568M models — a good illustration that scale stops buying quality in
embedding models. Its licence is the Qwen Research License (it derives from
Qwen-2.5-VL-3B), which is fine privately but not an open licence.

It matters here for a different reason. **When image support lands, this is the model to
evaluate first** — a single model embedding text and images into one space is exactly
what that feature needs, and it would run on GPU where its size stops being the problem.
Note that late interaction is not a drop-in: multi-vector retrieval needs a vector store
that supports it, which is a different architecture from the one-vector-per-document
model this API assumes.

### nomic-embed-text-v2-moe

A sparse mixture-of-experts encoder: 475M total parameters but only 305M active per
forward pass, 768 dimensions truncatable to 256, 100+ languages, Apache-2.0, fully
reproducible training. The efficiency idea is sound and the licence is clean.

It is absent for want of evidence rather than on any fault: it was not in the SkMTEB
study, so there is no Slovak number for it, and MoE routing exports to ONNX awkwardly.
Its memory footprint is also the full 475M even though only 305M are active, so on CPU
it costs like a mid-size model while computing like a small one.

### snowflake-arctic-embed-l-v2.0

568M, Apache-2.0, long context, published ONNX, and 72.54 on Slovak. A perfectly
reasonable model that lost a straight comparison: at effectively the same size,
`e5-large-instruct` scores 77.49, and for long context `bge-m3` scores 74.43. It is the
model that would be in the catalog if either of those were unavailable.

### embeddinggemma-300m

Covered above under [where the numbers come from](#where-these-numbers-come-from): the
top-ranked text-only multilingual model under 500M on MTEB overall, and last of the
realistic candidates on Slovak at 69.25. It is also under the Gemma Terms of Use rather
than an OSI-approved licence. It stays out on the numbers, not the licence.

### BGE-M3 is already here

`bge-m3` is in the catalog. Worth knowing what is *not* used: the model natively
produces three retrieval representations — dense, sparse (lexical weights), and
multi-vector ColBERT-style. This server uses the **dense** vector only, because that is
what a one-vector-per-input API can express. Hybrid dense+sparse retrieval is
meaningfully better on some corpora, but it is a property of your retrieval stack rather
than of an embedding endpoint.

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

`jina-v3` does the same thing by a different mechanism: instead of a text prefix it
carries a LoRA adapter per task, selected by a `task_id` input to the graph. The effect
is the same — queries and documents get different vectors — but sending a prefix *as
well* would corrupt the input, so its prefixes are deliberately empty. A test enforces
that a model uses one mechanism or the other, never both.

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

### Locally exported models

`e5-sk-large` publishes weights but no ONNX build, so `model download` produces one:

```bash
embedforge model download e5-sk-large
```

It fetches the source repository at its pinned revision, then runs the exporter through
`uvx` in a throwaway environment — so PyTorch, which is larger than this entire project
and needed exactly once, never becomes a dependency of the server or of CI.

What to expect: a few minutes, `uv` on the machine, and roughly twice the final size in
disk while it runs (the source copy is deleted afterwards). The result is byte-for-byte
reproducible from the pinned revision, and `model verify` records digests for it exactly
as for a downloaded model — an export we produced is an export we own.

Do it once on a machine with the disk and bandwidth, then copy the model directory to
wherever it needs to run; the server itself needs nothing but `onnxruntime`.

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

For a model that selects its task with an adapter rather than a prefix, set
`query_task_id` and `document_task_id` instead of the prefixes. For one that publishes
no ONNX build, set `export_from` to the source repository and give `onnx_file` the flat
name the exporter writes (`model.onnx`).

Check the upstream `1_Pooling/config.json` for the pooling mode and the model card for
the prefixes. Tests enforce that the revision is a full sha, that pros and cons are
filled in, and that prefixes are consistent with the symmetry flag.

For a model that is not ONNX text — a different runtime, or another modality —
implement `EmbeddingBackend` instead; see [architecture.md](architecture.md).

## Sources

- [SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation](https://arxiv.org/html/2606.13647v1)
  — 31 models evaluated on Slovak. Every Slovak score in this document comes from here.
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — the general
  multilingual benchmark, which ranks these models differently.
- [jina-embeddings-v4 paper](https://arxiv.org/abs/2506.18902) — multimodal and
  late-interaction embeddings.
- [Qwen3 Embedding](https://qwenlm.github.io/blog/qwen3-embedding/) — the 0.6B/4B/8B
  family.
- [Nomic Embed v2 preprint](https://static.nomic.ai/nomic_embed_multilingual_preprint.pdf)
  — sparse mixture-of-experts embeddings.
- [Arctic-Embed 2.0](https://arxiv.org/html/2412.04506v2) — multilingual retrieval
  without English regression.
