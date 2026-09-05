# Models

A process loads exactly one model, chosen by `EMBEDFORGE_MODEL_ID` at startup.

```bash
embedforge model list           # the catalog, with the configured model marked
embedforge model info dev-hash  # one model in full, with its trade-offs
```

The same catalog is available over HTTP at `GET /v1/models`.

## Available models

| ID | Dim | Symmetric | Notes |
| --- | --- | --- | --- |
| `dev-hash` | 384 | yes | Deterministic hash, no semantics. Development and tests only. |

`dev-hash` produces a stable pseudo-random unit vector per input. Identical text gives
identical vectors, which makes tests and smoke checks reproducible, but "cat" and
"kitten" are as unrelated as any other pair. It exists so the server runs and deploys
with no model download at all.

**Real ONNX models are the next step.** They will register in the same catalog with
documented pros and cons, and `embedforge model download` / `model verify` will fetch
and check their files.

## Symmetric and asymmetric models

A **symmetric** model uses one representation for everything, so `/v1/embed` and
`/v1/query` return identical vectors. An **asymmetric** model encodes a search query
differently from a stored passage, usually by prepending an instruction, so they differ.

Every response reports `"symmetric": true|false`, and the model catalog reports it per
model. Client code should still use the endpoint that matches its intent, so that
switching the server between the two kinds is a configuration change and not a rewrite.

## Comparing models before you commit

`embedforge model compare` runs the same inputs through several models and shows what
each one does with them:

```bash
embedforge model compare -q "how do I reset my password" -f corpus.txt
```

Per model you get the ranked results with cosine scores, load time, and milliseconds
per item; across models you get a rank-agreement matrix (Spearman, where 1.0 means
identical ordering).

How to read the agreement number:

- **High agreement** means the models would retrieve the same documents for your
  queries. Take the cheaper or smaller one; the choice does not matter.
- **Low agreement** means the choice does matter, and benchmark averages will not settle
  it. Judge the rankings yourself on queries you care about.

Use your own corpus and your own queries. A public benchmark says how a model does on
someone else's data.

### Why this is a CLI and not a development API

The obvious alternative is an endpoint that loads a model per request so models can be
compared over HTTP. This project deliberately does not do that:

- Serving several models means either paying the load cost on every request or holding
  every model in memory — the exact costs the one-model-per-process design avoids.
- A "load any model the caller names" code path must never be reachable in production,
  and the safest way to guarantee that is for it not to exist in the server.
- Comparison offline can measure things a request cannot, such as load time, and can
  compare rankings *across* models rather than returning one vector at a time.

If you do want comparisons over HTTP later, run one container per model behind the same
API. That keeps every process single-model and makes the comparison a client concern.

## Choosing a model

The trade-offs that matter, roughly in order:

- **Retrieval quality on your data.** Benchmark numbers are a starting point, not an
  answer; evaluate on your own queries.
- **Dimension.** Drives storage and search cost in your vector database, not just
  inference. 384 versus 1024 is a large difference at scale.
- **Latency.** Layer count and hidden size dominate. On CPU this is what you feel.
- **Maximum input length.** Longer contexts cost quadratically in attention. If your
  documents are long, chunking may beat a long-context model.
- **Language coverage.** Multilingual models trade some English quality for breadth.

`model compare` gives you the second and third of these directly, and evidence for the
first.

## Switching models

Changing `EMBEDFORGE_MODEL_ID` requires a restart, and **invalidates every vector you
have stored**: embeddings from different models are not comparable, even at the same
dimension. A model change means re-embedding your corpus.

The `model` field on a request is the guard against doing this by accident. Send the
model your stored vectors came from, and the server rejects the request with 400 if it
has a different one loaded, instead of silently returning vectors that will not match.

## Adding a model

Implement `EmbeddingBackend` in `src/embedforge/engine/`:

```python
class MyBackend(EmbeddingBackend):
    def load(self) -> None: ...           # optional, runs once off the event loop
    def close(self) -> None: ...          # optional
    def embed(self, inputs, task) -> np.ndarray: ...   # (len(inputs), dimension) float32
```

`embed` is called from worker threads and must be thread-safe; it should release the GIL
for its real work, as ONNX Runtime and the Hugging Face tokenizers do.

Then register it in `engine/registry.py` with a `ModelInfo` that fills in `pros` and
`cons` honestly — the CLI and the API both surface them, and they are what makes the
catalog useful when choosing.
