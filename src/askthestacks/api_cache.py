"""LRU cache for search results.

We cache by (query, top_k) — same query with different k counts as different.
The cache is bounded (default 1024 entries) to avoid unbounded memory growth.

Design choice: cache at the search-result level, not at the embedding level.
- Caching embeddings only would still cost a FAISS search per request.
- Caching final results means cache hits are O(1) — no embedding, no search.
- Cost: results go stale if the corpus is rebuilt while the server is running.
  Acceptable because we restart the server when we rebuild (Day 6 deploy practice).
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

import structlog

from askthestacks.retrieval import SearchResult

log = structlog.get_logger()


class SearchCache:
    """Thread-safe LRU cache for search results.

    We use OrderedDict + Lock rather than functools.lru_cache because:
    - We want runtime-configurable size (env var)
    - We want hit/miss metrics for observability
    - We want explicit thread safety (FastAPI is async, but Uvicorn workers
      can still race during startup or shared-state mutations)
    """

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 0:
            raise ValueError(f"max_size must be >= 0, got {max_size}")
        self._max_size = max_size
        self._store: OrderedDict[tuple[str, int],
                                 list[SearchResult]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, query: str, top_k: int) -> list[SearchResult] | None:
        """Return cached results for this (query, top_k), or None on miss."""
        if self._max_size == 0:
            return None
        key = (query, top_k)
        with self._lock:
            if key in self._store:
                # Move to end (most recently used)
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def set(self, query: str, top_k: int, results: list[SearchResult]) -> None:
        """Store results for this (query, top_k). Evicts oldest if at capacity."""
        if self._max_size == 0:
            return
        key = (query, top_k)
        with self._lock:
            self._store[key] = results
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                log.debug("cache_evicted",
                          query=evicted_key[0], top_k=evicted_key[1])

    def stats(self) -> dict[str, int | float]:
        """Snapshot of cache metrics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }

    def clear(self) -> None:
        """Drop everything. Used by tests and during reloads."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
