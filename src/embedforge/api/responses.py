"""Response classes."""

from typing import Any

import orjson
from fastapi.responses import JSONResponse


class ORJSONResponse(JSONResponse):
    """orjson-backed JSON, with native numpy support.

    Embedding responses are large float arrays. Serializing them straight from the
    numpy matrix, instead of converting to Python lists and validating them through
    pydantic, is the difference between microseconds and milliseconds per request.
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)
