"""FastAPI application for AskTheStacks semantic database search.

Architecture:
- Lifespan loads the Retriever once at startup (expensive: model + index).
- Middleware: CORS, slowapi rate limit, request-logging.
- Endpoints: /api/search, /api/health, /api/version.
- Errors: ValidationError → 400, RateLimit → 429, anything else → 500
  (sanitized — never leak tracebacks).

Why we don't use FastAPI's Depends() to inject Retriever:
- Retriever holds the model (~80MB) and FAISS index. Re-creating it per request
  would be catastrophic. App state is the right home for singletons.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from askthestacks.api_cache import SearchCache
from askthestacks.api_models import (
    ErrorResponse,
    HealthResponse,
    SearchHit,
    SearchResponse,
    VersionResponse,
)
from askthestacks.config import Settings, get_settings
from askthestacks.embedder import EMBEDDING_DIM, MODEL_NAME
from askthestacks.retrieval import Retriever, load_retriever

log = structlog.get_logger()


def configure_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level)
        ),
    )


# slowapi limiter: per-IP, evaluated at function decoration time.
# We construct it at module load and use its decorators inside endpoints.
limiter = Limiter(key_func=get_remote_address)


def _build_settings_and_state(settings: Settings) -> dict:
    """Construct the app state dict — retriever + cache + metadata."""
    retriever = load_retriever(settings.corpus_path, settings.index_dir)
    cache = SearchCache(max_size=settings.cache_size)

    # Read corpus metadata for /version and /health
    corpus_data = json.loads(settings.corpus_path.read_text(encoding="utf-8"))

    return {
        "retriever": retriever,
        "cache": cache,
        "settings": settings,
        "corpus_version": corpus_data.get("version", "unknown"),
        "corpus_built_at": corpus_data.get("built_at", "unknown"),
        "corpus_source_url": corpus_data.get("source_url", "unknown"),
        "corpus_entry_count": len(corpus_data.get("entries", [])),
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load retriever once at startup. Tear down on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("api_starting", host=settings.host, port=settings.port)

    state = _build_settings_and_state(settings)
    app.state.retriever = state["retriever"]
    app.state.cache = state["cache"]
    app.state.settings = state["settings"]
    app.state.corpus_version = state["corpus_version"]
    app.state.corpus_built_at = state["corpus_built_at"]
    app.state.corpus_source_url = state["corpus_source_url"]
    app.state.corpus_entry_count = state["corpus_entry_count"]

    log.info(
        "api_ready",
        corpus_version=state["corpus_version"],
        entry_count=state["corpus_entry_count"],
    )

    yield

    log.info("api_stopping")


def create_app(state_factory=None) -> FastAPI:
    """Build the FastAPI app. Optional state_factory for tests.

    state_factory: callable returning the app-state dict. Default uses real settings
    and loads the actual retriever. Tests pass a factory returning fake state.
    """

    if state_factory is None:
        app = FastAPI(
            title="AskTheStacks",
            description="Semantic database search for WIU Libraries.",
            version="0.1.0",
            lifespan=lifespan,
        )
    else:
        @asynccontextmanager
        async def test_lifespan(app: FastAPI) -> AsyncIterator[None]:
            state = state_factory()
            for k, v in state.items():
                setattr(app.state, k, v)
            yield

        app = FastAPI(
            title="AskTheStacks",
            description="Semantic database search for WIU Libraries.",
            version="0.1.0",
            lifespan=test_lifespan,
        )

    app.state.limiter = limiter

    # CORS
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Rate limit exception handler
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        log.warning(
            "rate_limit_exceeded",
            client=get_remote_address(request),
            path=request.url.path,
            limit=str(exc.detail),
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=ErrorResponse(
                error="rate_limit_exceeded",
                detail=str(exc.detail),
            ).model_dump(),
            headers={"Retry-After": "60"},
        )

    # Unexpected error handler — never leak tracebacks
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="internal_server_error").model_dump(),
        )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=get_remote_address(request),
        )
        return response

    # ---- Endpoints ----

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            corpus_version=request.app.state.corpus_version,
            entry_count=request.app.state.corpus_entry_count,
            model=MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
        )

    @app.get("/api/version", response_model=VersionResponse)
    async def version(request: Request) -> VersionResponse:
        return VersionResponse(
            corpus_version=request.app.state.corpus_version,
            built_at=request.app.state.corpus_built_at,
            source_url=request.app.state.corpus_source_url,
            entry_count=request.app.state.corpus_entry_count,
            model=MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
        )

    settings_for_decorators = get_settings()

    @app.get("/api/search", response_model=SearchResponse)
    @limiter.limit(settings_for_decorators.rate_limit_sustained)
    @limiter.limit(settings_for_decorators.rate_limit_burst)
    async def search(
        request: Request,
        q: str = Query(
            ...,
            min_length=1,
            description="Natural-language search query.",
        ),
        k: int = Query(
            default=settings_for_decorators.default_top_k,
            ge=1,
            description="Number of results to return.",
        ),
    ) -> SearchResponse:
        s: Settings = request.app.state.settings
        retriever: Retriever = request.app.state.retriever
        cache: SearchCache = request.app.state.cache

        # Validate length explicitly (Query() handles min, we enforce max)
        if len(q) > s.max_query_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query exceeds maximum length of {s.max_query_length}",
            )

        # Clamp k
        k_clamped = min(k, s.max_top_k)
        q_stripped = q.strip()
        if not q_stripped:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty or whitespace.",
            )

        start = time.perf_counter()

        # Cache check
        cached = cache.get(q_stripped, k_clamped)
        if cached is not None:
            results = cached
            cache_hit = True
        else:
            results = retriever.search(q_stripped, top_k=k_clamped)
            cache.set(q_stripped, k_clamped, results)
            cache_hit = False

        took_ms = round((time.perf_counter() - start) * 1000, 2)

        hits = [
            SearchHit(
                rank=r.rank,
                score=r.score,
                code=r.entry.code,
                name=r.entry.name,
                url=r.entry.url,
                subject_hint=r.entry.subject_hint,
                coverage=r.entry.coverage,
                dates=r.entry.dates,
            )
            for r in results
        ]

        log.info(
            "search_request",
            query=q_stripped,
            top_k=k_clamped,
            result_count=len(hits),
            took_ms=took_ms,
            cache_hit=cache_hit,
        )

        return SearchResponse(
            query=q_stripped,
            results=hits,
            result_count=len(hits),
            took_ms=took_ms,
        )

    return app


# Default app instance — uvicorn loads this
app = create_app()
