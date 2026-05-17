"""Application configuration loaded from environment variables.

Uses pydantic-settings so every setting is typed, validated, and documented.
Override any setting by exporting the env var (e.g., RATE_LIMIT_BURST=10/second).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Read once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # CORS
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated origins allowed by CORS, or '*' for any.",
    )

    # Rate limiting
    rate_limit_burst: str = Field(
        default="5/second",
        description="Per-IP burst rate (slowapi syntax).",
    )
    rate_limit_sustained: str = Field(
        default="30/minute",
        description="Per-IP sustained rate (slowapi syntax).",
    )

    # Query validation
    max_query_length: int = Field(default=500, ge=1, le=10000)
    max_top_k: int = Field(default=20, ge=1, le=100)
    default_top_k: int = Field(default=5, ge=1, le=100)

    # Cache
    cache_size: int = Field(default=1024, ge=0, le=100000)

    # Data paths
    corpus_path: Path = Field(default=Path("data/corpus/latest.json"))
    index_dir: Path = Field(default=Path("data/index"))

    # Logging
    log_level: str = Field(default="INFO")

    # Server bind (used by serve.py)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        v_upper = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v_upper

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parsed list of CORS origins. '*' means allow any."""
        if self.allowed_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    """Factory so callers get a fresh Settings instance. Useful for tests."""
    return Settings()
