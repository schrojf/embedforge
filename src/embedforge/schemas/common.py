"""Shared response shapes."""

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    type: str = Field(description="Stable machine-readable error type.")
    message: str


class ErrorResponse(BaseModel):
    """The envelope every failing request returns."""

    error: ErrorDetail
    request_id: str | None = None
