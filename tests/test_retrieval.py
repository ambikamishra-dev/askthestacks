"""Tests for the Retriever class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from askthestacks.embedder import EMBEDDING_DIM, Embedder, embed_corpus
from askthestacks.index import build_index, save_index
from askthestacks.retrieval import Retriever, SearchResult, load_retriever
from askthestacks.schema import Corpus, DatabaseEntry


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    return Embedder()


@pytest.fixture
def sample_corpus() -> Corpus:
    return Corpus(
        entries=[
            DatabaseEntry(
                code="ABI",
                name="ABI/INFORM",
                subject_hint="Business",
                coverage="indexes 3,700+ business and trade periodicals",
                url="https://www.wiu.edu/library/direct/?ABI",
            ),
            DatabaseEntry(
                code="PSY",
                name="PsycINFO",
                subject_hint="Psychology",
                coverage="psychology and behavioral sciences journals",
                url="https://www.wiu.edu/library/direct/?PSY",
            ),
            DatabaseEntry(
                code="MED",
                name="MEDLINE",
                subject_hint="Medicine",
                coverage="medical research and clinical literature",
                url="https://www.wiu.edu/library/direct/?MED",
            ),
        ]
    )


@pytest.fixture
def built_index_dir(
    sample_corpus: Corpus, embedder: Embedder, tmp_path: Path
) -> Path:
    """Build a real FAISS index for the sample corpus, save it, return the directory."""
    embeddings = embed_corpus(sample_corpus, embedder)
    index = build_index(embeddings)
    save_index(index, sample_corpus, tmp_path)
    return tmp_path


class TestRetriever:
    def test_search_returns_results(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        retriever = Retriever(sample_corpus, built_index_dir, embedder)
        results = retriever.search("psychology research", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_top_result_is_relevant(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        retriever = Retriever(sample_corpus, built_index_dir, embedder)
        results = retriever.search("psychology and mental health", top_k=1)
        assert results[0].entry.code == "PSY"

    def test_search_business_query(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        retriever = Retriever(sample_corpus, built_index_dir, embedder)
        results = retriever.search("corporate business strategy", top_k=1)
        assert results[0].entry.code == "ABI"

    def test_search_medical_query(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        retriever = Retriever(sample_corpus, built_index_dir, embedder)
        results = retriever.search("clinical medicine research", top_k=1)
        assert results[0].entry.code == "MED"

    def test_search_ranks_are_consecutive(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        retriever = Retriever(sample_corpus, built_index_dir, embedder)
        results = retriever.search("any query", top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_search_scores_descending(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        retriever = Retriever(sample_corpus, built_index_dir, embedder)
        results = retriever.search("any query", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_rejects_empty_query(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        retriever = Retriever(sample_corpus, built_index_dir, embedder)
        with pytest.raises(ValueError):
            retriever.search("")
        with pytest.raises(ValueError):
            retriever.search("   ")

    def test_index_corpus_mismatch_raises(
        self, sample_corpus: Corpus, built_index_dir: Path, embedder: Embedder
    ):
        # Build a corpus with extra entries that doesn't match the index size
        bigger = Corpus(
            entries=[
                *sample_corpus.entries,
                DatabaseEntry(
                    code="EXTRA",
                    name="Extra DB",
                    url="https://www.wiu.edu/library/direct/?EXTRA",
                ),
            ]
        )
        with pytest.raises(ValueError, match="out of sync"):
            Retriever(bigger, built_index_dir, embedder)


class TestLoadRetriever:
    def test_load_retriever_from_disk(
        self, sample_corpus: Corpus, built_index_dir: Path, tmp_path: Path
    ):
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text(
            sample_corpus.model_dump_json(), encoding="utf-8")

        retriever = load_retriever(corpus_path, built_index_dir)
        results = retriever.search("business", top_k=1)
        assert results[0].entry.code == "ABI"
