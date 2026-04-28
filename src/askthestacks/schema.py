"""Schema for a single library database entry."""
from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator, model_validator

_WIU_URL_RE = re.compile(
    r"^https://www\.wiu\.edu/"
    r"(library/direct/\?[A-Za-z0-9_-]+|libraries/databases/[A-Za-z0-9._/?=&-]+)$"
)


class DatabaseEntry(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=2, max_length=300)
    subject_hint: str | None = None
    subjects: list[str] = Field(default_factory=list)
    dates: str | None = None
    coverage: str | None = None
    full_text: str | None = None
    url: str
    embedding_text: str = ""

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = v.strip()
        if not _WIU_URL_RE.match(v):
            raise ValueError(f"not a valid WIU database URL: {v}")
        return v

    @field_validator("subjects")
    @classmethod
    def _normalize_subjects(cls, v: list[str]) -> list[str]:
        return sorted({s.strip() for s in v if s.strip()})

    @model_validator(mode="after")
    def _build_embedding_text(self) -> DatabaseEntry:
        parts = [self.name]
        if self.subject_hint:
            parts.append(f"Subject: {self.subject_hint}")
        if self.subjects:
            parts.append(f"Subjects: {', '.join(self.subjects)}")
        if self.coverage:
            parts.append(f"Coverage: {self.coverage}")
        if self.dates:
            parts.append(f"Dates: {self.dates}")
        if self.full_text and self.full_text.lower() not in ("no", "n/a"):
            parts.append(f"Full text: {self.full_text}")
        object.__setattr__(self, "embedding_text", " | ".join(parts))
        return self


class Corpus(BaseModel):
    version: str = "1.0"
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_url: str = "https://www.wiu.edu/libraries/databases/"
    entries: list[DatabaseEntry]

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @model_validator(mode="after")
    def _check_unique_codes(self) -> Corpus:
        codes = [e.code for e in self.entries]
        if len(codes) != len(set(codes)):
            from collections import Counter

            dupes = sorted({c for c, n in Counter(codes).items() if n > 1})
            raise ValueError(f"duplicate codes: {dupes}")
        return self
