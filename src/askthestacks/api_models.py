"""Pydantic models for API request validation and response shaping.

Kept separate from schema.py:
- schema.py models the data domain (DatabaseEntry invariants, embedding text).
- api_models.py models the wire format (what clients send, what they receive).

This separation lets the API evolve independently of the corpus schema.
For example, we could add a 'highlight' field to the response without
touching DatabaseEntry, or version the API without re-versioning data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchHit(BaseModel):
    """One ranked result in a search response."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(..., ge=1,
                      description="1-indexed position in result list.")
    score: float = Field(
        ..., description="Cosine similarity (higher = more relevant)."
    )
    code: str = Field(..., min_length=1, description="WIU redirector code.")
    name: str = Field(..., min_length=1, description="Database display name.")
    url: str = Field(..., description="Redirector URL to the database.")
    subject_hint: str | None = Field(
        default=None, description="Optional subject category."
    )
    coverage: str | None = Field(
        default=None, description="Optional description of contents."
    )
    dates: str | None = Field(
        default=None, description="Optional date coverage."
    )


class SearchResponse(BaseModel):
    """Full search response payload."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Echo of the query that was searched.")
    results: list[SearchHit] = Field(default_factory=list)
    result_count: int = Field(..., ge=0)
    took_ms: float = Field(..., ge=0.0,
                           description="Server-side processing time.")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="'ok' if service is ready.")
    corpus_version: str
    entry_count: int = Field(..., ge=0)
    model: str
    embedding_dim: int = Field(..., ge=1)


class VersionResponse(BaseModel):
    """Version metadata response."""

    model_config = ConfigDict(frozen=True)

    corpus_version: str
    built_at: str
    source_url: str
    entry_count: int = Field(..., ge=0)
    model: str
    embedding_dim: int


class ErrorResponse(BaseModel):
    """Generic error response shape."""

    model_config = ConfigDict(frozen=True)

    error: str
    detail: str | None = None
