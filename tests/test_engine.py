"""Batching, backpressure, and failure behavior of the inference engine."""

import asyncio
import time
from collections.abc import Sequence

import numpy as np
import pytest

from embedforge.engine.base import EmbeddingBackend, EmbedInput, ModelInfo, TaskType, TextInput
from embedforge.engine.batching import InferenceEngine
from embedforge.errors import ModelNotReadyError, OverloadedError, RequestTimeoutError

INFO = ModelInfo(id="test", name="test", dimension=4, max_input_tokens=128, symmetric=False)


class RecordingBackend(EmbeddingBackend):
    """Backend that records the batches it is given and can be made slow or fail."""

    def __init__(self, *, delay: float = 0.0, fail: bool = False) -> None:
        super().__init__(INFO, normalize=False)
        self.delay = delay
        self.fail = fail
        self.batches: list[tuple[TaskType, list[str]]] = []

    def reset(self) -> None:
        """Forget the warmup calls, so tests only see the batches they caused."""
        self.batches.clear()

    def embed(self, inputs: Sequence[EmbedInput], task: TaskType) -> np.ndarray:
        self.batches.append((task, [item.text for item in inputs]))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("backend exploded")
        return np.array([[float(len(item.text))] * INFO.dimension for item in inputs], "float32")


async def make_engine(backend: RecordingBackend, **kwargs: object) -> InferenceEngine:
    engine = InferenceEngine(backend, **kwargs)  # pyright: ignore[reportArgumentType]
    await engine.start()
    backend.reset()
    return engine


async def test_concurrent_requests_merge_into_one_batch() -> None:
    """The point of the dispatcher: 16 single-item requests must not be 16 model calls."""
    backend = RecordingBackend(delay=0.02)
    engine = await make_engine(backend, max_batch_size=32, batch_wait_ms=20, workers=1)
    try:
        results = await asyncio.gather(
            *(engine.embed([TextInput(f"text-{index}")], TaskType.DOCUMENT) for index in range(16))
        )
    finally:
        await engine.aclose()

    assert all(result.shape == (1, 4) for result in results)
    assert len(backend.batches) < 16, "requests were not batched"
    assert max(len(texts) for _, texts in backend.batches) > 1


async def test_batches_never_exceed_max_batch_size() -> None:
    backend = RecordingBackend(delay=0.01)
    engine = await make_engine(backend, max_batch_size=4, batch_wait_ms=10, workers=1)
    try:
        await asyncio.gather(
            *(engine.embed([TextInput(f"t{index}")], TaskType.DOCUMENT) for index in range(20))
        )
    finally:
        await engine.aclose()
    assert max(len(texts) for _, texts in backend.batches) <= 4


async def test_a_large_request_is_split_and_reassembled_in_order() -> None:
    backend = RecordingBackend()
    engine = await make_engine(backend, max_batch_size=3, batch_wait_ms=0, workers=1)
    try:
        texts = [f"{'x' * index}" or "y" for index in range(1, 11)]
        matrix = await engine.embed([TextInput(text) for text in texts], TaskType.DOCUMENT)
    finally:
        await engine.aclose()

    assert matrix.shape == (10, 4)
    # The backend encodes text length, so order is verifiable.
    assert [int(row[0]) for row in matrix] == [len(text) for text in texts]
    assert len(backend.batches) >= 4


async def test_document_and_query_work_is_never_mixed_in_one_batch() -> None:
    """Asymmetric models apply different prefixes, so a batch must be single-task."""
    backend = RecordingBackend(delay=0.01)
    engine = await make_engine(backend, max_batch_size=16, batch_wait_ms=15, workers=1)
    try:
        await asyncio.gather(
            *(engine.embed([TextInput(f"d{i}")], TaskType.DOCUMENT) for i in range(8)),
            *(engine.embed([TextInput(f"q{i}")], TaskType.QUERY) for i in range(8)),
        )
    finally:
        await engine.aclose()

    for task, texts in backend.batches:
        prefix = "d" if task is TaskType.DOCUMENT else "q"
        assert all(text.startswith(prefix) for text in texts)


async def test_overload_sheds_load_instead_of_queueing_forever() -> None:
    backend = RecordingBackend(delay=0.05)
    engine = await make_engine(
        backend, max_batch_size=1, batch_wait_ms=0, workers=1, queue_max_size=4
    )
    try:
        tasks = [
            asyncio.create_task(engine.embed([TextInput(f"t{index}")], TaskType.DOCUMENT))
            for index in range(40)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await engine.aclose()

    rejected = [result for result in results if isinstance(result, OverloadedError)]
    assert rejected, "queue_max_size was never enforced"
    assert rejected[0].status_code == 503
    assert rejected[0].headers["Retry-After"] == "1"


async def test_slow_inference_times_out_with_504() -> None:
    backend = RecordingBackend(delay=0.5)
    engine = await make_engine(backend, batch_wait_ms=0, workers=1, request_timeout=0.05)
    try:
        with pytest.raises(RequestTimeoutError) as error:
            await engine.embed([TextInput("slow")], TaskType.DOCUMENT)
    finally:
        await engine.aclose()
    assert error.value.status_code == 504


async def test_abandoned_work_is_not_computed() -> None:
    """A caller that timed out must not keep buying inference time."""
    backend = RecordingBackend(delay=0.05)
    engine = await make_engine(
        backend, max_batch_size=1, batch_wait_ms=0, workers=1, request_timeout=0.01
    )
    try:
        with pytest.raises(RequestTimeoutError):
            await asyncio.gather(
                *(engine.embed([TextInput(f"t{index}")], TaskType.DOCUMENT) for index in range(10))
            )
        await asyncio.sleep(0.2)
    finally:
        await engine.aclose()
    assert len(backend.batches) < 10


async def test_backend_failure_surfaces_to_every_caller_in_the_batch() -> None:
    backend = RecordingBackend()
    engine = await make_engine(backend, batch_wait_ms=5, workers=1)
    backend.fail = True
    try:
        results = await asyncio.gather(
            *(engine.embed([TextInput(f"t{index}")], TaskType.DOCUMENT) for index in range(4)),
            return_exceptions=True,
        )
    finally:
        await engine.aclose()
    assert all(isinstance(result, RuntimeError) for result in results)


async def test_engine_rejects_work_before_start_and_after_close() -> None:
    engine = InferenceEngine(RecordingBackend())
    with pytest.raises(ModelNotReadyError):
        await engine.embed([TextInput("x")], TaskType.DOCUMENT)
    await engine.start()
    await engine.aclose()
    with pytest.raises(ModelNotReadyError):
        await engine.embed([TextInput("x")], TaskType.DOCUMENT)


async def test_empty_input_short_circuits() -> None:
    backend = RecordingBackend()
    engine = await make_engine(backend)
    try:
        matrix = await engine.embed([], TaskType.DOCUMENT)
    finally:
        await engine.aclose()
    assert matrix.shape == (0, 4)
    assert backend.batches == [], "no model call for no work"


async def test_warmup_runs_before_serving() -> None:
    backend = RecordingBackend()
    engine = InferenceEngine(backend)
    await engine.start()
    await engine.aclose()
    assert backend.batches, "the model should be warmed up at startup"


async def test_a_backend_that_cannot_warm_up_fails_startup() -> None:
    """Fail fast at boot rather than serving 500s to real traffic."""
    engine = InferenceEngine(RecordingBackend(fail=True))
    with pytest.raises(RuntimeError):
        await engine.start()
    assert not engine.ready


async def test_closing_a_never_started_engine_is_safe() -> None:
    """A failed startup must still release the thread pool."""
    engine = InferenceEngine(RecordingBackend())
    await engine.aclose()
    assert not engine.ready


async def test_an_idle_request_does_not_wait_for_the_batch_window() -> None:
    """A lone request on an idle server has nothing to batch with, so it must not wait.

    The window exists to let concurrent requests find each other. With every worker
    free there is no company coming that starting now would miss, and waiting is pure
    added latency.
    """
    backend = RecordingBackend()
    # A window long enough that waiting for it would be unmistakable.
    engine = await make_engine(backend, batch_wait_ms=250, workers=1)
    try:
        started = time.perf_counter()
        await engine.embed([TextInput("alone")], TaskType.DOCUMENT)
        elapsed = time.perf_counter() - started
    finally:
        await engine.aclose()
    assert elapsed < 0.05, f"waited {elapsed * 1000:.0f}ms for a batch that could not grow"


async def test_the_window_still_applies_once_every_worker_is_busy() -> None:
    """Under load the window is what makes batches full, so it must still be used."""
    backend = RecordingBackend(delay=0.05)
    engine = await make_engine(backend, max_batch_size=32, batch_wait_ms=40, workers=1)
    try:
        # The first request occupies the only worker; the rest must accumulate.
        first = asyncio.create_task(engine.embed([TextInput("first")], TaskType.DOCUMENT))
        await asyncio.sleep(0.01)
        rest = [
            asyncio.create_task(engine.embed([TextInput(f"t{index}")], TaskType.DOCUMENT))
            for index in range(12)
        ]
        await asyncio.gather(first, *rest)
    finally:
        await engine.aclose()

    assert max(len(texts) for _, texts in backend.batches) > 1, "work was not merged"
    assert len(backend.batches) < 13


async def test_work_arriving_while_a_worker_is_busy_joins_the_next_batch() -> None:
    """Items queued while the dispatcher waits for a slot must not sit out a round."""
    backend = RecordingBackend(delay=0.05)
    engine = await make_engine(backend, max_batch_size=32, batch_wait_ms=0, workers=1)
    try:
        first = asyncio.create_task(engine.embed([TextInput("first")], TaskType.DOCUMENT))
        await asyncio.sleep(0.01)
        later = [
            asyncio.create_task(engine.embed([TextInput(f"t{index}")], TaskType.DOCUMENT))
            for index in range(8)
        ]
        await asyncio.gather(first, *later)
    finally:
        await engine.aclose()
    # Even with no waiting window, the eight that arrived during inference run together.
    assert max(len(texts) for _, texts in backend.batches) >= 8
