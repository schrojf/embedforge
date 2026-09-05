# Command line

One binary, three independent command groups. All of them read the same configuration
as the server ([configuration.md](configuration.md)), so they act on the same token file
and model directory.

```bash
embedforge --help
embedforge --version
```

In a container, the CLI is the entrypoint:
`docker run --rm -v embedforge-data:/var/lib/embedforge embedforge:local token list`.

## `embedforge serve`

Runs the HTTP server.

```bash
embedforge serve
embedforge serve --host 0.0.0.0 --port 8080
embedforge serve --reload            # development only
```

| Option | Effect |
| --- | --- |
| `--host`, `--port`, `--workers` | Override the configured values. |
| `--reload` | Restart on code changes. Forces a single process. |

Overrides are exported into the environment before uvicorn starts, so reloader and
worker subprocesses — which load their own configuration — see the same values.

`serve` also caps the BLAS and OpenMP thread pools before the model runtime is imported;
[performance.md](performance.md) explains why that matters.

## `embedforge token`

Manages API clients. See [authentication.md](authentication.md) for the concepts.

| Command | Purpose |
| --- | --- |
| `token create NAME [--scopes …] [--expires-in …] [--note …]` | Create a token and print it once. |
| `token list` | All tokens with status, scopes, and expiry. Never shows a secret. |
| `token show ID` | One token in detail. |
| `token disable ID` / `token enable ID` | Reversible suspension. |
| `token revoke ID [--yes]` | Permanent deletion. Prompts unless `--yes`. |

```bash
embedforge token create search-frontend --scopes query --expires-in 90d
embedforge token create indexer --scopes embed --note "nightly reindex job"
```

## `embedforge model`

Inspects, downloads, verifies, and compares models. See [models.md](models.md) for the
catalog and how to choose.

| Command | Purpose |
| --- | --- |
| `model list` | Every model this build can serve, with download status. |
| `model info ID` | One model in full, including its pros and cons. |
| `model download ID [--force]` | Fetch a model's files and record their digests. |
| `model verify ID [--quick]` | Check downloaded files against those digests. |
| `model remove ID [--yes]` | Delete a model's files. |
| `model compare [MODELS…]` | Run the same inputs through several models and compare them. |

### Getting a model onto a machine

```bash
embedforge model list                  # what is available, and what is present
embedforge model download e5-base      # ~1.1 GB, pinned to a commit sha
embedforge model verify e5-base        # re-hash every file
EMBEDFORGE_MODEL_ID=e5-base embedforge serve
```

A model that publishes no ONNX build (`e5-sk-large`) is exported locally instead: the
same `download` command fetches the source at its pinned revision and runs the exporter
through `uvx`, which keeps PyTorch out of this project entirely. Expect a few minutes
and about twice the final size in disk while it runs.

Downloads are reproducible: each model is pinned to a commit sha, so the same command
gives the same bytes tomorrow. `download` writes a `manifest.json` of sha256 digests
beside the files, and `verify` checks against it — which is how a truncated download is
caught before it becomes a confusing startup crash. `--quick` skips hashing and checks
presence and size only.

### `model compare`

```bash
# Rank a corpus against real queries, across every registered model.
embedforge model compare -q "how do I reset my password" -f corpus.txt

# Two specific models, three queries, top 3 results each.
embedforge model compare e5-small bge-base \
  -q "reset password" -q "billing question" -q "rate limits" \
  -f corpus.txt --top-k 3

# Machine-readable, for scripting an evaluation.
embedforge model compare --json -q "cats" -f corpus.txt > results.json
```

| Option | Effect |
| --- | --- |
| `MODELS…` | Model ids to compare. Defaults to every model currently downloaded. |
| `--query`, `-q` | A search query. Repeatable. |
| `--text`, `-t` | A document to rank. Repeatable. |
| `--file`, `-f` | A file of documents, one per line. |
| `--top-k`, `-k` | Results shown per query (default 5). |
| `--rounds` | Timed passes to average, for steadier latency numbers (default 1). |
| `--json` | Machine-readable output. |

Each model is loaded, used, and released before the next one starts, so peak memory is
one model rather than all of them. A model that fails to load is reported and skipped
rather than aborting the run.

**With queries** you get, per model, the top-ranked documents with cosine scores, plus a
rank-agreement matrix across models. **Without queries** you get each model's
document-to-document similarity instead, which is useful for checking whether a model
separates things you consider different.

The report also shows load time and milliseconds per item. On a CPU server that is
often the deciding number, so it belongs next to the quality signal rather than in a
separate tool.

## `embedforge config`

Prints the effective configuration — every setting, its resolved value, and what it
does — including the thread count derived from the available cores. Use it to confirm
what a container actually picked up:

```bash
docker compose run --rm embedforge config
```
