"""Package version, resolved from installed distribution metadata."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("embedforge")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
