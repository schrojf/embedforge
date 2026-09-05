# Deployment

The supported deployment is the container image: one process, one model, behind a
reverse proxy that terminates TLS.

## Build

```bash
docker build -t embedforge:0.1.0 --build-arg VERSION="$(git describe --tags --always)" .
```

`VERSION` is worth passing. The version normally comes from git tags, and `.git` is kept
out of the build context, so without it the image reports `0.0.0` — which makes "which
build is this?" unanswerable during an incident.

What the image does:

- Multi-stage build: dependencies install from `uv.lock` in a layer that only changes
  when the lockfile does; the project installs as a real wheel, so the runtime image
  carries only the virtualenv.
- Runs as uid 10001, never root.
- `tini` as PID 1, so SIGTERM reaches the server and shutdown stays graceful.
- `HEALTHCHECK` on `/readyz`, so the container is healthy only once the model is loaded.
- Entrypoint is the CLI: `docker run <image> token list` works as well as serving.

## Run with Compose

`docker-compose.yml` in the repository root is a working starting point:

```bash
export VERSION="$(git describe --tags --always)"
docker compose build
docker compose run --rm embedforge token create my-client   # note the token
docker compose up -d
docker compose logs -f
```

It publishes to `127.0.0.1:8000` only, keeps state in a named volume, drops all Linux
capabilities, runs a read-only root filesystem, and sets `stop_grace_period` above the
server's graceful shutdown timeout so in-flight requests finish instead of being killed
mid-inference.

## Run with plain Docker

```bash
docker volume create embedforge-data
docker run --rm -v embedforge-data:/var/lib/embedforge embedforge:0.1.0 token create my-client

docker run -d --name embedforge \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v embedforge-data:/var/lib/embedforge \
  --cpus 4 --memory 2g \
  --stop-timeout 40 \
  -e EMBEDFORGE_MODEL_ID=dev-hash \
  embedforge:0.1.0
```

`--cpus` is not just a limit: the server reads its CPU affinity at startup to size its
thread pools, so this is also how you tell it how much machine it has.

## Model files

Models are **not baked into the image**. They live in the mounted volume, downloaded
before the server starts:

```bash
docker compose run --rm embedforge model download e5-base
docker compose run --rm embedforge model verify e5-base
# then set EMBEDFORGE_MODEL_ID=e5-base and start
docker compose up -d
```

This keeps the image small (about 500 MB rather than several gigabytes), lets you change
models without rebuilding, and means an image rebuild does not re-download two gigabytes
of weights. The cost is one extra step on a fresh host, and a container that will not
become ready if you forget it — which `/readyz` reports plainly.

If you would rather have a self-contained image (an air-gapped host, or an immutable
deployment with no writable volume), add to the runtime stage:

```dockerfile
ARG MODEL_ID=e5-base
ENV EMBEDFORGE_MODEL_ID=${MODEL_ID}
RUN embedforge model download ${MODEL_ID} && embedforge model verify ${MODEL_ID}
```

Size the container for the model, not the server. The process needs roughly the model's
download size in RAM plus a few hundred megabytes:

| Model | Download | Suggested memory limit |
| --- | --- | --- |
| `e5-small-int8` | 135 MB | 512 MB |
| `e5-base` | 1.1 GB | 2 GB |
| `e5-sk-large` | 1.5 GB | 2 GB |
| `e5-large-instruct`, `bge-m3`, `jina-v3` | 2.3 GB | 4 GB |

Model files are re-downloadable, so they do not need backing up — unlike the token file.

One exception to "download it in the container": `e5-sk-large` is exported rather than
downloaded, which needs `uv` and roughly 3 GB of scratch space. Do that once on a
machine that has both, then copy the model directory into the volume. The server itself
needs only `onnxruntime`.

## GPU

The image installs `onnxruntime` (CPU). For GPU, swap the package and tell the server to
use it:

```dockerfile
RUN uv pip install --python /app/.venv/bin/python onnxruntime-gpu
```

```bash
EMBEDFORGE_DEVICE=cuda   # or `auto` to prefer GPU when present and fall back to CPU
```

`cuda` fails loudly at startup if the CUDA execution provider is unavailable, which is
what you want: silently serving from CPU at a tenth of the speed is worse than not
starting. Run the container with `--gpus all` and a CUDA-capable base image.

For most private deployments this is not worth it. A quantized model on a few CPU cores
serves a personal workload comfortably, and GPU memory is expensive to leave idle.

## Reverse proxy

Terminate TLS in front of the server and do not expose it directly. Caddy, with
automatic certificates:

```caddyfile
embeddings.example.com {
    reverse_proxy 127.0.0.1:8000

    # Embedding requests can be large; the default body limit is often too small.
    request_body {
        max_size 10MB
    }

    # Metrics are for you, not the internet.
    @metrics path /metrics
    respond @metrics 404
}
```

nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name embeddings.example.com;

    ssl_certificate     /etc/letsencrypt/live/embeddings.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/embeddings.example.com/privkey.pem;

    client_max_body_size 10m;

    location /metrics { return 404; }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Longer than EMBEDFORGE_REQUEST_TIMEOUT, so the server's own 504 wins
        # and the client gets a useful error body.
        proxy_read_timeout 60s;
    }
}
```

If the proxy sets `X-Forwarded-*`, tell the server which peer to trust:
`EMBEDFORGE_FORWARDED_ALLOW_IPS=172.18.0.1`. Never set it to `*` on anything reachable
from outside, or clients can forge their apparent address. If you do not need forwarded
headers, set `EMBEDFORGE_PROXY_HEADERS=false`.

## systemd, without Docker

```ini
# /etc/systemd/system/embedforge.service
[Unit]
Description=EmbedForge embedding server
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=embedforge
Group=embedforge
WorkingDirectory=/opt/embedforge
Environment=EMBEDFORGE_HOST=127.0.0.1
Environment=EMBEDFORGE_PORT=8000
Environment=EMBEDFORGE_LOG_FORMAT=json
Environment=EMBEDFORGE_TOKENS_FILE=/var/lib/embedforge/tokens.json
Environment=EMBEDFORGE_MODEL_DIR=/var/lib/embedforge/models
ExecStart=/opt/embedforge/.venv/bin/embedforge serve
Restart=on-failure
RestartSec=5
KillSignal=SIGTERM
# Longer than EMBEDFORGE_TIMEOUT_GRACEFUL_SHUTDOWN.
TimeoutStopSec=40

StateDirectory=embedforge
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/embedforge

[Install]
WantedBy=multi-user.target
```

```bash
sudo -u embedforge /opt/embedforge/.venv/bin/embedforge token create my-client
sudo systemctl enable --now embedforge
```

## Upgrades

The server has no schema and no migrations, so an upgrade is a container replacement:

```bash
docker compose build && docker compose up -d
```

Roll forward with confidence by watching `/readyz`: a replacement container reports
`not_ready` until its model is loaded and warm, so an orchestrator that respects
readiness will not send it traffic early. With a single container there is a gap of a
few seconds; if that matters, run two replicas behind the proxy and replace them one at
a time.

Downgrades work the same way — the token file format is versioned and read by any
version.

## Deployment checklist

- [ ] `VERSION` passed to the build, so `/healthz` reports something meaningful.
- [ ] TLS terminated by a proxy; the server bound to loopback or a private network.
- [ ] `/metrics` blocked at the proxy.
- [ ] `EMBEDFORGE_FORWARDED_ALLOW_IPS` set to the proxy, not `*`.
- [ ] Token file on a volume, and in your backups.
- [ ] Model downloaded into the volume and verified, with `EMBEDFORGE_MODEL_ID` set.
- [ ] Memory limit sized for the model, not the server.
- [ ] CPU and memory limits set; `--cpus` matches the capacity you intend.
- [ ] `stop_grace_period` (or `TimeoutStopSec`) above the graceful shutdown timeout.
- [ ] Log rotation configured, or logs shipped off the box.
- [ ] `EMBEDFORGE_DOCS_ENABLED=false` if you do not want the schema public.
