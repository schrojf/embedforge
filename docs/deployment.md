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
- [ ] CPU and memory limits set; `--cpus` matches the capacity you intend.
- [ ] `stop_grace_period` (or `TimeoutStopSec`) above the graceful shutdown timeout.
- [ ] Log rotation configured, or logs shipped off the box.
- [ ] `EMBEDFORGE_DOCS_ENABLED=false` if you do not want the schema public.
