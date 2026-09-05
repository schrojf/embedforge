# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# onnxruntime ships without type stubs, and this script deals in loosely typed
# measurement dicts throughout.
"""Benchmark every model in the catalog on this machine.

Downloads one model, measures it, deletes it, moves to the next - so peak disk use is
one model rather than the whole catalog (about 15 GB).

Each model runs in a subprocess so its peak memory is measured cleanly and a crash in
one does not lose the rest of the run.

    uv run python devtools/benchmark.py --out results.json
    uv run python devtools/benchmark.py --models e5-base,jina-v3 --keep
    uv run python devtools/benchmark.py --markdown results.json
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---- The evaluation set -------------------------------------------------------------
#
# Small on purpose, and not a substitute for a real benchmark: it is here to catch a
# model that is broken or wrong for this language pair, not to rank models finely.
# Slovak-dominant with English mixed in, because that is this deployment's traffic.

DOCUMENTS: list[str] = [
    "Heslo si zmeníte v nastaveniach účtu po prihlásení.",
    "Bratislava je hlavné a najväčšie mesto Slovenska.",
    "Faktúru nájdete v sekcii Platby, kde si ju môžete stiahnuť ako PDF.",
    "Podpora je dostupná v pracovné dni od 9:00 do 17:00.",
    "Účet zrušíte tlačidlom Zmazať účet v nastaveniach profilu.",
    "Reset your password from the account settings page.",
    "Invoices can be downloaded as PDF from the Payments section.",
    "The capital of Slovakia is Bratislava.",
    "Burza dnes prudko klesla a investori prišli o peniaze.",
    "Mačka spí na gauči celý deň.",
    "Recept na bryndzové halušky vyžaduje zemiaky a bryndzu.",
    "The stock market fell sharply today.",
]

# (query, indices of every document that answers it)
#
# Relevance is a set, not a single document. Three of these questions are answered by
# both a Slovak and an English document, and a multilingual model that returns the
# English one is right, not wrong. Scoring a single "correct" index made every model
# look identical, because they all failed the same three cross-lingual pairs.
QUERIES: list[tuple[str, set[int]]] = [
    ("ako si zmením heslo?", {0, 5}),
    ("kde nájdem faktúru?", {2, 6}),
    ("aké je hlavné mesto Slovenska?", {1, 7}),
    ("kedy je dostupná podpora?", {3}),
    ("chcem zrušiť účet", {4}),
    ("how do I change my password?", {0, 5}),
    ("where can I download my invoice?", {2, 6}),
    ("what is the capital of Slovakia?", {1, 7}),
]

BATCH_SIZES = (1, 8, 32)
TIMED_ROUNDS = 3

IDLE_REQUESTS = 15
"""Sequential single-item requests through the engine, with nothing else in flight."""

CONCURRENT_REQUESTS = 64
"""Single-item requests issued at once, to see batching work end to end."""

LONG_WINDOW_MS = 200.0
"""A deliberately absurd batch window.

The engine only holds a batch open while every worker is busy, so an idle request
should be unaffected by this. Measuring at both 5 ms and 200 ms is what proves it:
before that rule, this column would have read about 200 ms for every model."""

BASELINE_MODEL = "dev-hash"
"""The empty run.

dev-hash does no real inference, so its numbers are the cost of everything that is not
the model: the interpreter, onnxruntime, tokenization, pooling, and this project's own
code. Every other model should be read as its figure minus this one."""


# ---- Machine ------------------------------------------------------------------------


def read_cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:  # pragma: no cover - non-Linux
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def describe_machine() -> dict[str, Any]:
    import platform

    import onnxruntime as ort

    flags = read_cpu_flags()
    model_name = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
                break
    except OSError:  # pragma: no cover
        pass

    total_ram = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                total_ram = int(line.split()[1]) * 1024
                break
    except OSError:  # pragma: no cover
        pass

    return {
        "cpu": model_name or platform.processor(),
        "logical_cpus": os.cpu_count(),
        "usable_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "ram_bytes": total_ram,
        # These decide whether a quantized build is fast or merely small.
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": "avx512_vnni" in flags,
        "avx_vnni": "avx_vnni" in flags,
        "amx_int8": "amx_int8" in flags,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
    }


# ---- Measuring one model ------------------------------------------------------------


def peak_rss_bytes() -> int:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return usage * 1024 if sys.platform.startswith("linux") else usage


async def engine_latency(backend: Any, batch_wait_ms: float) -> dict[str, float]:
    """Measure the serving path, not just the model.

    Everything else here calls the backend directly, which is the model's own cost.
    This goes through the dispatcher a real request goes through, so it includes
    queueing, batching and the wait window - the part a caller actually feels.
    """
    from embedforge.engine.base import TaskType, TextInput
    from embedforge.engine.batching import InferenceEngine

    engine = InferenceEngine(
        backend,
        max_batch_size=32,
        batch_wait_ms=batch_wait_ms,
        queue_max_size=4096,
        workers=1,
    )
    await engine.start()
    try:
        # The same input and task as the direct batch-of-one measurement, so the two
        # are comparable and their difference is the serving path's own cost.
        idle: list[float] = []
        for _ in range(IDLE_REQUESTS):
            started = time.perf_counter()
            await engine.embed([TextInput(DOCUMENTS[0])], TaskType.DOCUMENT)
            idle.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        await asyncio.gather(
            *(
                engine.embed([TextInput(f"document {index}")], TaskType.DOCUMENT)
                for index in range(CONCURRENT_REQUESTS)
            )
        )
        concurrent_seconds = time.perf_counter() - started
    finally:
        await engine.aclose()

    idle.sort()
    return {
        "idle_p50_ms": idle[len(idle) // 2],
        "idle_min_ms": idle[0],
        "concurrent_items_per_second": CONCURRENT_REQUESTS / concurrent_seconds,
    }


def measure(model_id: str, model_dir: Path) -> dict[str, Any]:
    """Run every measurement for one model. Executed in its own process."""
    from embedforge.config import Settings
    from embedforge.engine.base import TaskType, TextInput
    from embedforge.engine.registry import create_backend, get_spec

    settings = Settings(model_dir=model_dir, model_id=model_id)
    info = get_spec(model_id).info
    backend = create_backend(settings)

    # Everything is imported by now, so this is the interpreter plus onnxruntime plus
    # this project - the floor any deployment pays before a model exists.
    rss_before = peak_rss_bytes()
    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started

    documents = [TextInput(text) for text in DOCUMENTS]
    queries = [TextInput(text) for text, _ in QUERIES]

    # Warm up: the first call pays for lazy allocation and would distort everything.
    backend.embed(documents[:1], TaskType.DOCUMENT)

    latency: dict[str, dict[str, float]] = {}
    for size in BATCH_SIZES:
        batch = [documents[index % len(documents)] for index in range(size)]
        timings: list[float] = []
        for _ in range(TIMED_ROUNDS):
            start = time.perf_counter()
            backend.embed(batch, TaskType.DOCUMENT)
            timings.append(time.perf_counter() - start)
        best = min(timings)
        latency[str(size)] = {
            "batch_seconds": best,
            "ms_per_item": best / size * 1000,
            "items_per_second": size / best,
        }

    document_vectors = backend.embed(documents, TaskType.DOCUMENT)
    query_vectors = backend.embed(queries, TaskType.QUERY)

    # Batch stability: this server batches concurrent requests, so a model whose output
    # depends on batch composition cannot embed a document reproducibly.
    alone = backend.embed(documents[:1], TaskType.DOCUMENT)
    stability = float(
        alone[0]
        @ document_vectors[0]
        / (np.linalg.norm(alone[0]) * np.linalg.norm(document_vectors[0]))
    )

    scores = query_vectors @ document_vectors.T
    hits = 0
    reciprocal_ranks: list[float] = []
    margins: list[float] = []
    for position, (_, relevant) in enumerate(QUERIES):
        order = [int(index) for index in np.argsort(-scores[position])]
        rank = next(place for place, index in enumerate(order, start=1) if index in relevant)
        hits += rank == 1
        reciprocal_ranks.append(1.0 / rank)
        best_relevant = max(float(scores[position][index]) for index in relevant)
        best_irrelevant = max(
            float(scores[position][index])
            for index in range(len(DOCUMENTS))
            if index not in relevant
        )
        margins.append(best_relevant - best_irrelevant)

    # Capture memory before the engine section. Peak RSS is a process high-water mark,
    # and the engine measurements below load the model twice more; letting those count
    # would roughly double the figure and make it useless for sizing a container.
    peak_rss = peak_rss_bytes()

    # The serving path, at the default window and at an absurd one. The engine takes
    # ownership of the backend's lifecycle, so this runs last.
    engine_default = asyncio.run(engine_latency(backend, 5.0))
    engine_long_window = asyncio.run(engine_latency(backend, LONG_WINDOW_MS))
    peak_rss_including_engine = peak_rss_bytes()

    backend.close()

    return {
        "engine": {
            "default_window": engine_default,
            "long_window": engine_long_window,
            "long_window_ms": LONG_WINDOW_MS,
        },
        "model": model_id,
        "dimension": info.dimension,
        "max_input_tokens": info.max_input_tokens,
        "symmetric": info.symmetric,
        "declared_size_mb": info.size_mb,
        "disk_bytes": sum(
            path.stat().st_size
            for path in model_dir.joinpath(model_id).rglob("*")
            if path.is_file()
        ),
        "load_seconds": load_seconds,
        "peak_rss_bytes": peak_rss,
        "peak_rss_including_engine_bytes": peak_rss_including_engine,
        "rss_before_load_bytes": rss_before,
        "rss_growth_bytes": peak_rss - rss_before,
        "latency": latency,
        "batch_stability": stability,
        "recall_at_1": hits / len(QUERIES),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_margin": float(np.mean(margins)),
        "min_margin": float(np.min(margins)),
        # Kept so the quality metrics can be recomputed without downloading again.
        "scores": [[round(float(value), 6) for value in row] for row in scores],
    }


# ---- Driver -------------------------------------------------------------------------


def free_disk_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def run_one(model_id: str, model_dir: Path, keep: bool, reserve_bytes: int) -> dict[str, Any]:
    """Download, measure in a subprocess, then delete unless asked to keep."""
    from embedforge.config import Settings
    from embedforge.engine import registry
    from embedforge.engine.download import download_model, is_downloaded, remove_model

    settings = Settings(model_dir=model_dir, model_id=model_id)
    result: dict[str, Any] = {"model": model_id}

    if registry.is_onnx_model(model_id):
        config = registry.onnx_config(model_id)
        already_present = is_downloaded(settings, model_id, config)
        needed = (registry.get_spec(model_id).info.size_mb or 0) * 1_000_000
        if not already_present and free_disk_bytes(model_dir) < needed + reserve_bytes:
            return {**result, "error": "not enough free disk"}
        if not already_present:
            print(f"  downloading ({needed / 1e9:.1f} GB)...", flush=True)
            started = time.perf_counter()
            download_model(settings, model_id, config)
            result["download_seconds"] = time.perf_counter() - started
    else:
        already_present = True

    print("  measuring...", flush=True)
    # The worker writes its result to a file rather than stdout: the server's own
    # structured logging also goes to stdout, and mixing the two makes the output
    # unparsable.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        result_file = Path(handle.name)
    try:
        process = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                model_id,
                "--model-dir",
                str(model_dir),
                "--result-file",
                str(result_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            result["error"] = (process.stderr or "").strip()[-600:]
        else:
            try:
                result.update(json.loads(result_file.read_text()))
            except (OSError, json.JSONDecodeError) as error:
                result["error"] = f"unreadable result: {error}"
    finally:
        result_file.unlink(missing_ok=True)
        # Always reclaim the disk, even when the measurement failed. Not filling the
        # disk is the whole reason this runs one model at a time.
        if not keep and not already_present and registry.is_onnx_model(model_id):
            remove_model(settings, model_id)
            print("  removed", flush=True)
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    machine, results = payload["machine"], payload["results"]
    lines = [
        f"CPU: {machine['cpu']} | {machine['usable_cpus']} usable CPUs | "
        f"AVX2 {'yes' if machine['avx2'] else 'no'}, "
        f"AVX-512 VNNI {'yes' if machine['avx512_vnni'] else 'no'}",
        "",
        "| Model | Disk | RSS | Load | ms/item b=1 | ms/item b=32 | items/s b=32 | Idle p50 | Idle @200ms | Batch-stable | R@1 | Margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        if "error" in row:
            lines.append(f"| `{row['model']}` | — | — | — | — | — | — | — | — | — | — | failed |")
            continue
        one, big = row["latency"]["1"], row["latency"]["32"]
        label = f"`{row['model']}`"
        if row["model"] == BASELINE_MODEL:
            label += " *(baseline)*"
        lines.append(
            f"| {label} | {row['disk_bytes'] / 1e6:,.0f} MB "
            f"| {row['peak_rss_bytes'] / 1e6:,.0f} MB "
            f"| {row['load_seconds']:.1f}s "
            f"| {one['ms_per_item']:.1f} | {big['ms_per_item']:.1f} "
            f"| {big['items_per_second']:,.0f} "
            f"| {row['engine']['default_window']['idle_p50_ms']:.1f} ms "
            f"| {row['engine']['long_window']['idle_p50_ms']:.1f} ms "
            f"| {row['batch_stability']:.4f} "
            f"| {row['recall_at_1']:.2f} | {row['mean_margin']:+.3f} |"
        )
    baseline = next((row for row in results if row["model"] == BASELINE_MODEL), None)
    if baseline and "error" not in baseline:
        lines += [
            "",
            f"Application floor, before any model is loaded: "
            f"{baseline['rss_before_load_bytes'] / 1e6:,.0f} MB resident. "
            f"The baseline row adds {baseline['rss_growth_bytes'] / 1e6:,.0f} MB and "
            f"{baseline['latency']['32']['ms_per_item']:.3f} ms/item of pipeline overhead.",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--models", help="Comma-separated ids. Default: the whole catalog.")
    parser.add_argument("--out", default="benchmark-results.json")
    parser.add_argument("--keep", action="store_true", help="Do not delete models afterwards.")
    parser.add_argument("--reserve-gb", type=float, default=5.0, help="Free disk to leave spare.")
    parser.add_argument("--markdown", help="Render an existing results file and exit.")
    args = parser.parse_args()

    if args.markdown:
        print(render_markdown(json.loads(Path(args.markdown).read_text())))
        return 0

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.worker:
        payload = measure(args.worker, model_dir)
        if args.result_file:
            Path(args.result_file).write_text(json.dumps(payload))
        else:
            json.dump(payload, sys.stdout)
        return 0

    from embedforge.engine import registry

    if args.models:
        model_ids = [name.strip() for name in args.models.split(",") if name.strip()]
    else:
        # Smallest first, so partial results are useful if the run is interrupted.
        model_ids = sorted(
            registry.known_ids(), key=lambda i: registry.get_spec(i).info.size_mb or 0
        )

    machine = describe_machine()
    print(
        f"machine: {machine['cpu']} ({machine['usable_cpus']} CPUs), "
        f"AVX-512 VNNI={machine['avx512_vnni']}",
        flush=True,
    )
    print(f"free disk: {free_disk_bytes(model_dir) / 1e9:.0f} GB\n", flush=True)

    results: list[dict[str, Any]] = []
    output = Path(args.out)
    for position, model_id in enumerate(model_ids, start=1):
        print(f"[{position}/{len(model_ids)}] {model_id}", flush=True)
        row: dict[str, Any]
        try:
            row = run_one(model_id, model_dir, args.keep, int(args.reserve_gb * 1e9))
        except Exception as error:  # keep going; one bad model must not lose the run
            row = {"model": model_id, "error": str(error)}
        if "error" in row:
            print(f"  FAILED: {row['error'][:200]}", flush=True)
        else:
            print(
                f"  {row['latency']['32']['items_per_second']:,.0f} items/s at batch 32, "
                f"R@1 {row['recall_at_1']:.2f}, stability {row['batch_stability']:.4f}",
                flush=True,
            )
        results.append(row)
        # Write after every model, so an interrupted run still leaves usable data.
        output.write_text(json.dumps({"machine": machine, "results": results}, indent=2))

    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
