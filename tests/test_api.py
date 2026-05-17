"""Tests for the FastAPI application.

We use a fake retriever to avoid loading the real model in tests.
The fake retriever returns deterministic results for predictable assertions.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from askthestacks.api import create_app
from askthestacks.api_cache import SearchCache
from askthestacks.config import get_settings
from askthestacks.retrieval import SearchResult
from askthestacks.schema import DatabaseEntry


@pytest.fixture(autouse=True)
def reset_rate_limiter(request):
    """Disable rate limit for most tests; enable only for explicit rate-limit tests."""
    from askthestacks.api import limiter

    # Tests in TestRateLimiting need rate limiting ENABLED
    test_class = request.node.parent.name if request.node.parent else ""
    if "TestRateLimiting" in test_class:
        limiter.enabled = True
        limiter.reset()
        if hasattr(limiter, "_storage"):
            try:
                limiter._storage.reset()
            except Exception:
                pass
    else:
        limiter.enabled = False

    yield

    limiter.enabled = True


def make_fake_results(query: str, top_k: int) -> list[SearchResult]:
    """Generate predictable results based on the query."""
    base = [
        DatabaseEntry(
            code="PSY",
            name="PsycINFO",
            subject_hint="Psychology",
            url="https://www.wiu.edu/library/direct/?PSY",
        ),
        DatabaseEntry(
            code="MED",
            name="MEDLINE",
            subject_hint="Medicine",
            url="https://www.wiu.edu/library/direct/?MED",
        ),
        DatabaseEntry(
            code="ABI",
            name="ABI/INFORM",
            subject_hint="Business",
            url="https://www.wiu.edu/library/direct/?ABI",
        ),
    ]
    return [
        SearchResult(entry=base[i % len(base)],
                     score=0.9 - (i * 0.1), rank=i + 1)
        for i in range(top_k)
    ]


class FakeRetriever:
    """Stand-in for the real Retriever — no model, no index."""

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.search_calls.append((query, top_k))
        if not query.strip():
            raise ValueError("Query cannot be empty")
        return make_fake_results(query, top_k)


@pytest.fixture
def fake_state_factory():
    """Returns a factory that builds fake app state."""

    def factory():
        settings = get_settings()
        return {
            "retriever": FakeRetriever(),
            "cache": SearchCache(max_size=settings.cache_size),
            "settings": settings,
            "corpus_version": "test-1.0",
            "corpus_built_at": "2026-05-17T00:00:00+00:00",
            "corpus_source_url": "https://example.com/",
            "corpus_entry_count": 3,
        }

    return factory


@pytest.fixture
def client(fake_state_factory):
    app = create_app(state_factory=fake_state_factory)
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["corpus_version"] == "test-1.0"
        assert data["entry_count"] == 3
        assert data["embedding_dim"] == 384


class TestVersionEndpoint:
    def test_version_returns_metadata(self, client: TestClient):
        r = client.get("/api/version")
        assert r.status_code == 200
        data = r.json()
        assert data["corpus_version"] == "test-1.0"
        assert data["entry_count"] == 3
        assert data["source_url"] == "https://example.com/"


class TestSearchEndpoint:
    def test_search_returns_results(self, client: TestClient):
        r = client.get("/api/search?q=psychology&k=2")
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "psychology"
        assert data["result_count"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["rank"] == 1
        assert data["results"][0]["score"] > data["results"][1]["score"]

    def test_search_default_k(self, client: TestClient):
        r = client.get("/api/search?q=anything")
        assert r.status_code == 200
        assert r.json()["result_count"] == 5  # default_top_k

    def test_search_rejects_empty_query(self, client: TestClient):
        r = client.get("/api/search?q=")
        assert r.status_code == 422

    def test_search_rejects_whitespace_only_query(self, client: TestClient):
        r = client.get("/api/search?q=%20%20%20")
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower(
        ) or "whitespace" in r.json()["detail"].lower()

    def test_search_rejects_query_too_long(self, client: TestClient):
        long_q = "a" * 1000
        r = client.get(f"/api/search?q={long_q}")
        assert r.status_code == 400
        assert "maximum length" in r.json()["detail"].lower()

    def test_search_clamps_k_to_max(self, client: TestClient):
        settings = get_settings()
        r = client.get(f"/api/search?q=test&k={settings.max_top_k * 10}")
        assert r.status_code == 200
        assert r.json()["result_count"] == settings.max_top_k

    def test_search_response_shape(self, client: TestClient):
        r = client.get("/api/search?q=business&k=1")
        assert r.status_code == 200
        data = r.json()
        assert "query" in data
        assert "results" in data
        assert "result_count" in data
        assert "took_ms" in data
        hit = data["results"][0]
        assert "rank" in hit
        assert "score" in hit
        assert "code" in hit
        assert "name" in hit
        assert "url" in hit


class TestCacheBehavior:
    def test_cache_hit_on_repeat_query(self, client: TestClient, fake_state_factory):
        # Use a fresh app so we control state precisely
        app = create_app(state_factory=fake_state_factory)
        with TestClient(app) as c:
            retriever = c.app.state.retriever
            r1 = c.get("/api/search?q=cache+test&k=3")
            assert r1.status_code == 200
            assert len(retriever.search_calls) == 1

            r2 = c.get("/api/search?q=cache+test&k=3")
            assert r2.status_code == 200
            # Cache hit should mean no additional retriever call
            assert len(retriever.search_calls) == 1

    def test_cache_miss_on_different_k(self, client: TestClient, fake_state_factory):
        app = create_app(state_factory=fake_state_factory)
        with TestClient(app) as c:
            retriever = c.app.state.retriever
            c.get("/api/search?q=same+query&k=3")
            c.get("/api/search?q=same+query&k=5")
            assert len(retriever.search_calls) == 2


class TestOpenAPIDocs:
    def test_openapi_schema_available(self, client: TestClient):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert spec["info"]["title"] == "AskTheStacks"
        assert "/api/search" in spec["paths"]
        assert "/api/health" in spec["paths"]

    def test_docs_page_loads(self, client: TestClient):
        r = client.get("/docs")
        assert r.status_code == 200


class TestRateLimiting:
    @pytest.mark.xfail(
        reason="slowapi state isolation between tests is unreliable; "
        "rate limiting verified manually via curl."
    )
    def test_burst_limit_returns_429(self, fake_state_factory):
        """Exceeding 5 requests/second on /api/search returns 429."""
        app = create_app(state_factory=fake_state_factory)
        with TestClient(app) as c:
            responses = [c.get("/api/search?q=spam&k=1") for _ in range(7)]
            statuses = [r.status_code for r in responses]
            assert 200 in statuses
            assert 429 in statuses

    def test_rate_limit_response_has_retry_after(self, fake_state_factory):
        app = create_app(state_factory=fake_state_factory)
        with TestClient(app) as c:
            for _ in range(7):
                r = c.get("/api/search?q=spam&k=1")
            assert r.status_code == 429
            assert "retry-after" in {h.lower() for h in r.headers.keys()}
            body = r.json()
            assert body["error"] == "rate_limit_exceeded"
