"""Prometheus metrics.

Metrics are always recorded; `EMBEDFORGE_METRICS_ENABLED` only controls whether the
/metrics endpoint is mounted. Recording is a few nanoseconds and having the numbers
available makes incident debugging possible.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

HTTP_REQUESTS = Counter(
    "embedforge_http_requests_total",
    "HTTP requests handled.",
    ["method", "endpoint", "status"],
)
HTTP_DURATION = Histogram(
    "embedforge_http_request_duration_seconds",
    "End-to-end HTTP request duration.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

INFERENCE_DURATION = Histogram(
    "embedforge_inference_duration_seconds",
    "Time spent inside one model call, excluding queue wait.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
BATCH_SIZE = Histogram(
    "embedforge_inference_batch_size",
    "Items per model call. Low values under load mean batching is not paying off.",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
QUEUE_WAIT = Histogram(
    "embedforge_queue_wait_seconds",
    "Time an item waited between submission and entering a model call.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
ITEMS_TOTAL = Counter(
    "embedforge_items_total",
    "Inputs embedded.",
    ["task"],
)
QUEUE_DEPTH = Gauge(
    "embedforge_queue_items",
    "Items accepted but not yet returned to their caller.",
)
INFLIGHT_BATCHES = Gauge(
    "embedforge_inflight_batches",
    "Model calls currently executing.",
)
REJECTED = Counter(
    "embedforge_rejected_total",
    "Requests shed without a result.",
    ["reason"],
)
INFERENCE_ERRORS = Counter(
    "embedforge_inference_errors_total",
    "Model calls that raised.",
)
MODEL_INFO = Info(
    "embedforge_model",
    "The model currently loaded.",
)
