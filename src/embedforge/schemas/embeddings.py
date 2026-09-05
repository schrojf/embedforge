"""Embedding request and response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from embedforge.engine.base import TaskType


class EmbedRequest(BaseModel):
    """Input for `/v1/embed` and `/v1/query`.

    `input` accepts one string or a list of strings. When other modalities land, they
    arrive as an additional field rather than a change to this one, so existing
    clients keep working.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    input: str | list[str] = Field(
        description="A single text, or a list of texts to embed in one call.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional guard: the model you expect to answer. The request fails if it "
            "does not match the model this server has loaded."
        ),
    )

    def as_texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class Embedding(BaseModel):
    index: int = Field(description="Position of the corresponding input.")
    embedding: list[float]


class Usage(BaseModel):
    items: int
    characters: int
    """Input size in characters. Token counts arrive with the tokenizing backends."""


class EmbedResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    task: TaskType
    dimension: int
    normalized: bool
    symmetric: bool = Field(
        description="True when this model gives queries and documents the same vector, "
        "so /embed and /query return identical results.",
    )
    data: list[Embedding]
    usage: Usage
