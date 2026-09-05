"""EmbedForge: a production-ready embedding server.

The public surface is intentionally small. Import submodules directly:

    from embedforge.config import Settings
    from embedforge.api.app import create_app
"""

from embedforge.version import __version__

__all__ = ["__version__"]
