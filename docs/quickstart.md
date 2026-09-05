# Quickstart

## With Docker

```bash
docker build -t embedforge:local --build-arg VERSION="$(git describe --tags --always)" .

# Tokens and models live in a volume, so they survive container replacement.
docker run --rm -v embedforge-data:/var/lib/embedforge \
  embedforge:local token create my-client

docker run -d --name embedforge -p 127.0.0.1:8000:8000 \
  -v embedforge-data:/var/lib/embedforge embedforge:local
```

## From a checkout

```bash
make install

embedforge token create my-client   # prints the token once
embedforge serve
```

## Call it

```bash
TOKEN=ef_...   # the token printed above

curl -s http://127.0.0.1:8000/readyz

curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/v1/user

curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"input": ["the first document", "the second document"]}' \
  http://127.0.0.1:8000/v1/embed

curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"input": "what am I looking for?"}' \
  http://127.0.0.1:8000/v1/query
```

Interactive docs are at <http://127.0.0.1:8000/docs>.

## Switch to a real model

The default model is `dev-hash`, which produces **deterministic but meaningless**
vectors. It exists so the server starts with no downloads, which makes deployment easy
to verify — but it cannot do retrieval.

```bash
embedforge model list                  # the catalog and what is downloaded
embedforge model download e5-base      # ~1.1 GB
embedforge model verify e5-base
EMBEDFORGE_MODEL_ID=e5-base embedforge serve
```

In a container, download into the mounted volume first:

```bash
docker compose run --rm embedforge model download e5-base
docker compose up -d   # with EMBEDFORGE_MODEL_ID=e5-base set
```

`e5-base` is the recommended starting point for Slovak and English. See
[models.md](models.md) for the full catalog, the benchmark evidence behind it, and how
to compare models on your own data.
