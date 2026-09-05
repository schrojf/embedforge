"""Process-level runtime tuning.

These knobs must be applied *before* numpy, ONNX Runtime, or any BLAS library is
imported, because those read the environment once at load time. That is why the
`serve` command touches them before importing the application.
"""

from __future__ import annotations

import os

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def apply_thread_limits(threads: int) -> dict[str, str]:
    """Cap the math libraries' thread pools, unless the operator set them explicitly.

    Without this, every BLAS call spawns one thread per core. With several inference
    workers that means cores^2 threads fighting over the same CPU, which costs far
    more in context switching than it wins in parallelism.
    """
    value = str(max(1, threads))
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, value)
    return {name: os.environ[name] for name in THREAD_ENV_VARS}
