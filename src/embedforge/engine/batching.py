"""Dynamic batching engine.

The concurrency model, in one place:

* The HTTP layer is async and never blocks. A request submits its inputs and awaits
  a future.
* A single dispatcher coroutine merges inputs from *different* concurrent requests
  into one batch. Embedding models are massively more efficient per item on a batch
  of 32 than on 32 batches of one, so a wait of a few milliseconds buys a large
  throughput win and disappears next to inference time.
* Batches run in a small `ThreadPoolExecutor`. Inference releases the GIL, so these
  threads do real parallel work. The pool is deliberately small: one ONNX session
  already saturates the cores it is given, and running many sessions at once only
  oversubscribes the CPU and inflates tail latency.
* Admission control is by item count, not request count. When more items are in
  flight than `queue_max_size`, the server sheds load with 503 instead of building
  an unbounded queue that guarantees timeouts for everyone.
"""

import asyncio
import functools
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from embedforge import metrics
from embedforge.engine.base import EmbeddingBackend, EmbedInput, ModelInfo, TaskType
from embedforge.errors import ModelNotReadyError, OverloadedError, RequestTimeoutError
from embedforge.logging import get_logger

log = get_logger(__name__)

# A None on the queue means "stop after draining what is already there".
_SHUTDOWN: None = None


@dataclass(slots=True)
class _Job:
    """One caller's request, possibly spread across several batches."""

    task: TaskType
    inputs: Sequence[EmbedInput]
    future: asyncio.Future[np.ndarray]
    results: list[np.ndarray | None]
    remaining: int
    submitted_at: float

    @property
    def abandoned(self) -> bool:
        """True once the caller gave up (timeout or disconnect)."""
        return self.future.done()


@dataclass(slots=True)
class _Item:
    job: _Job
    index: int


class InferenceEngine:
    """Owns the backend, the batch dispatcher, and the inference threads."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        max_batch_size: int = 32,
        batch_wait_ms: float = 5.0,
        queue_max_size: int = 512,
        workers: int = 1,
        request_timeout: float = 30.0,
    ) -> None:
        self._backend = backend
        self._max_batch_size = max_batch_size
        self._batch_wait = batch_wait_ms / 1000.0
        self._queue_max_size = queue_max_size
        self._workers = workers
        self._request_timeout = request_timeout

        self._pending: dict[TaskType, deque[_Item]] = {task: deque() for task in TaskType}
        self._task_order = list(TaskType)
        self._round_robin = 0
        self._pending_count = 0
        self._inflight_items = 0

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_Job | None] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._slots: asyncio.Semaphore | None = None
        self._ready = False

    # ---- Lifecycle ----

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def info(self) -> ModelInfo:
        return self._backend.info

    @property
    def backend(self) -> EmbeddingBackend:
        return self._backend

    async def start(self) -> None:
        """Load the model, warm it up, and start dispatching. Blocks until ready."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue = asyncio.Queue()
        self._slots = asyncio.Semaphore(self._workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="ef-infer"
        )
        started = time.perf_counter()
        # Loading and warmup are blocking and slow; keep them off the event loop.
        await loop.run_in_executor(self._executor, self._backend.load)
        await loop.run_in_executor(self._executor, self._warmup)
        self._dispatcher = loop.create_task(self._dispatch_loop(), name="ef-dispatcher")
        self._ready = True
        log.info(
            "engine_started",
            model=self._backend.info.id,
            dimension=self._backend.info.dimension,
            workers=self._workers,
            max_batch_size=self._max_batch_size,
            batch_wait_ms=self._batch_wait * 1000,
            load_seconds=round(time.perf_counter() - started, 3),
        )

    def _warmup(self) -> None:
        """Force lazy allocations so the first real request is not the slow one."""
        inputs = self._backend.warmup_inputs()
        if not inputs:
            return
        for task in TaskType:
            self._backend.embed(inputs, task)

    async def aclose(self) -> None:
        """Stop dispatching, let running batches finish, then release the model."""
        self._ready = False
        if self._dispatcher is not None and self._queue is not None:
            self._queue.put_nowait(_SHUTDOWN)
            try:
                await asyncio.wait_for(self._dispatcher, timeout=10)
            except (TimeoutError, asyncio.CancelledError):  # pragma: no cover - slow shutdown
                self._dispatcher.cancel()
        for queue in self._pending.values():
            while queue:
                item = queue.popleft()
                if not item.job.future.done():
                    item.job.future.set_exception(ModelNotReadyError("Server is shutting down."))
        self._pending_count = 0
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._backend.close()
        log.info("engine_stopped", model=self._backend.info.id)

    # ---- Public API ----

    async def embed(self, inputs: Sequence[EmbedInput], task: TaskType) -> np.ndarray:
        """Embed `inputs`, returning a `(len(inputs), dimension)` matrix."""
        if not self._ready or self._queue is None:
            raise ModelNotReadyError("Model is not loaded.")
        count = len(inputs)
        if count == 0:
            return np.zeros((0, self._backend.info.dimension), dtype=np.float32)
        if self._inflight_items + count > self._queue_max_size:
            metrics.REJECTED.labels("overloaded").inc()
            raise OverloadedError(
                f"Too many items in flight ({self._inflight_items}/{self._queue_max_size})."
            )

        loop = asyncio.get_running_loop()
        job = _Job(
            task=task,
            inputs=inputs,
            future=loop.create_future(),
            results=[None] * count,
            remaining=count,
            submitted_at=time.perf_counter(),
        )
        self._inflight_items += count
        metrics.QUEUE_DEPTH.set(self._inflight_items)
        try:
            self._queue.put_nowait(job)
            return await asyncio.wait_for(job.future, self._request_timeout)
        except TimeoutError:
            metrics.REJECTED.labels("timeout").inc()
            raise RequestTimeoutError(
                f"Embedding did not complete within {self._request_timeout:g}s."
            ) from None
        finally:
            self._inflight_items -= count
            metrics.QUEUE_DEPTH.set(self._inflight_items)

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "inflight_items": self._inflight_items,
            "queued_items": self._pending_count,
            "queue_max_size": self._queue_max_size,
            "max_batch_size": self._max_batch_size,
            "batch_wait_ms": self._batch_wait * 1000,
            "inference_workers": self._workers,
        }

    # ---- Dispatcher ----

    async def _dispatch_loop(self) -> None:
        assert self._queue is not None and self._slots is not None
        loop = asyncio.get_running_loop()
        while True:
            job = await self._queue.get()
            stopping = job is None
            if job is not None:
                self._enqueue(job)
            # Whatever else already arrived joins this batch for free.
            stopping = self._drain_nowait() or stopping
            # Then hold briefly for stragglers - but only when it can actually pay off.
            if not stopping and self._should_wait_for_more():
                stopping = await self._collect_until(loop.time() + self._batch_wait) or stopping

            while self._pending_count:
                # Blocking here is intentional: when every worker is busy, new work keeps
                # accumulating and the next batch forms full.
                await self._slots.acquire()
                # Work may have arrived while we waited for a worker. Taking it now costs
                # nothing and makes the batch we are about to run larger.
                stopping = self._drain_nowait() or stopping
                batch = self._take_batch()
                if batch is None:
                    self._slots.release()
                    break
                self._launch(*batch)

            if stopping:
                return

    def _should_wait_for_more(self) -> bool:
        """Whether holding the batch open can still buy anything.

        Waiting is only useful when the alternative is running a small batch that
        delays a larger one. Two cases where it cannot help:

        * A full batch is already waiting - there is nothing to add.
        * A worker is idle. Starting now costs that worker nothing, because it would
          otherwise sit doing nothing for the whole window; and anything that arrives
          during inference forms its own batch anyway.

        The second case is the common one on a lightly loaded server, where the old
        behaviour made a lone request wait the full window for company that never
        came. Under real load the workers are busy, this returns True, and batches
        fill up as before.
        """
        if self._batch_wait <= 0:
            return False
        if self._pending_count >= self._max_batch_size:
            return False
        assert self._slots is not None
        # locked() is True when acquiring would block, i.e. every worker is busy.
        return self._slots.locked()

    def _enqueue(self, job: _Job) -> None:
        queue = self._pending[job.task]
        for index in range(len(job.inputs)):
            queue.append(_Item(job=job, index=index))
        self._pending_count += len(job.inputs)

    def _drain_nowait(self) -> bool:
        """Move every already-queued job into pending. Returns True on shutdown."""
        assert self._queue is not None
        stopping = False
        while True:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return stopping
            if job is None:
                stopping = True
            else:
                self._enqueue(job)

    async def _collect_until(self, deadline: float) -> bool:
        """Wait for more work until the batch window closes. Returns True on shutdown."""
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while self._pending_count < self._max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                job = await asyncio.wait_for(self._queue.get(), remaining)
            except TimeoutError:
                return False
            if job is None:
                return True
            self._enqueue(job)
            if self._drain_nowait():
                return True
        return False

    def _take_batch(self) -> tuple[TaskType, list[_Item]] | None:
        """Fill one batch from a single task queue, rotating tasks for fairness."""
        count = len(self._task_order)
        for offset in range(count):
            task = self._task_order[(self._round_robin + offset) % count]
            queue = self._pending[task]
            batch: list[_Item] = []
            while queue and len(batch) < self._max_batch_size:
                item = queue.popleft()
                self._pending_count -= 1
                if item.job.abandoned:
                    continue  # Caller gave up; do not pay for their inference.
                batch.append(item)
            if batch:
                self._round_robin = (self._round_robin + offset + 1) % count
                return task, batch
        return None

    def _launch(self, task: TaskType, batch: list[_Item]) -> None:
        assert self._loop is not None and self._executor is not None
        inputs = [item.job.inputs[item.index] for item in batch]
        now = time.perf_counter()
        for item in batch:
            metrics.QUEUE_WAIT.observe(now - item.job.submitted_at)
        metrics.BATCH_SIZE.observe(len(batch))
        metrics.INFLIGHT_BATCHES.inc()
        future = self._loop.run_in_executor(self._executor, self._run_batch, inputs, task)
        future.add_done_callback(functools.partial(self._on_batch_done, batch))

    def _run_batch(self, inputs: list[EmbedInput], task: TaskType) -> np.ndarray:
        """Runs on a worker thread."""
        started = time.perf_counter()
        matrix = self._backend.embed(inputs, task)
        metrics.INFERENCE_DURATION.observe(time.perf_counter() - started)
        metrics.ITEMS_TOTAL.labels(task.value).inc(len(inputs))
        if matrix.shape[0] != len(inputs):
            raise RuntimeError(
                f"Backend returned {matrix.shape[0]} vectors for {len(inputs)} inputs."
            )
        return matrix

    def _on_batch_done(self, batch: list[_Item], future: asyncio.Future[np.ndarray]) -> None:
        """Runs on the event loop thread once a batch finishes."""
        metrics.INFLIGHT_BATCHES.dec()
        if self._slots is not None:
            self._slots.release()
        try:
            matrix = future.result()
        except Exception as exc:
            metrics.INFERENCE_ERRORS.inc()
            log.error("inference_failed", error=str(exc), batch_size=len(batch), exc_info=exc)
            for item in batch:
                if not item.job.future.done():
                    item.job.future.set_exception(exc)
            return
        for row, item in enumerate(batch):
            job = item.job
            if job.abandoned:
                continue
            job.results[item.index] = matrix[row]
            job.remaining -= 1
            if job.remaining == 0:
                # Every slot is filled once remaining hits zero.
                rows = cast(list[np.ndarray], job.results)
                job.future.set_result(np.stack(rows))
