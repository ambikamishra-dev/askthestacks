"""Retrieval: end-to-end query → ranked DatabaseEntry results.

Wires together: query → Embedder.embed_query → FAISS.search → corpus lookup → results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from askthestacks.embedder import Embedder
from askthestacks.index import load_index, search_index
from askthestacks.schema import Corpus, DatabaseEntry

log = structlog.get_logger()


@dataclass(frozen=True)
class SearchResult:
    """One ranked result from a search."""

    entry: DatabaseEntry
    score: float
    rank: int


class Retriever:
    """Holds the loaded corpus, index, and embedder. Stateful but reusable."""

    def __init__(self, corpus: Corpus, index_dir: Path, embedder: Embedder) -> None:
        self._corpus = corpus
        self._embedder = embedder
        self._index, self._id_map = load_index(index_dir)

        if self._index.ntotal != len(corpus.entries):
            raise ValueError(
                f"Index size {self._index.ntotal} doesn't match corpus size "
                f"{len(corpus.entries)} — corpus and index are out of sync"
            )

        self._entries_by_position = {
            i: entry for i, entry in enumerate(corpus.entries)
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return top-K ranked results for the query."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_vec = self._embedder.embed_query(query)
        scores, positions = search_index(self._index, query_vec, top_k=top_k)

        results: list[SearchResult] = []
        for rank, (score, position) in enumerate(zip(scores, positions, strict=True), start=1):
            if position < 0:
                continue
            entry = self._entries_by_position[int(position)]
            results.append(SearchResult(
                entry=entry, score=float(score), rank=rank))

        log.info(
            "search_complete",
            query=query,
            results_count=len(results),
            top_score=results[0].score if results else None,
        )
        return results


def load_retriever(corpus_path: Path, index_dir: Path) -> Retriever:
    """Convenience: load a Retriever from disk paths."""
    corpus_data = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus = Corpus(**corpus_data)
    embedder = Embedder()
    return Retriever(corpus, index_dir, embedder)
