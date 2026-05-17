"""Tests for the FAISS index module."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest

from askthestacks.embedder import EMBEDDING_DIM
from askthestacks.index import build_index, load_index, save_index, search_index
from askthestacks.schema import Corpus, DatabaseEntry


@pytest.fixture
def sample_corpus() -> Corpus:
    return Corpus(
        entries=[
            DatabaseEntry(
                code="ABI",
                name="ABI/INFORM",
                subject_hint="Business",
                url="https://www.wiu.edu/library/direct/?ABI",
            ),
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
        ]
    )


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Deterministic dummy embeddings — orthogonal unit vectors for predictable behavior."""
    rng = np.random.default_rng(seed=42)
    raw = rng.standard_normal((3, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


class TestBuildIndex:
    def test_build_index_correct_size(self, sample_embeddings: np.ndarray):
        index = build_index(sample_embeddings)
        assert index.ntotal == 3
        assert index.d == EMBEDDING_DIM

    def test_build_index_rejects_1d_array(self):
        with pytest.raises(ValueError):
            build_index(np.zeros(EMBEDDING_DIM, dtype=np.float32))

    def test_build_index_handles_empty(self):
        empty = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        index = build_index(empty)
        assert index.ntotal == 0


class TestSaveLoadIndex:
    def test_round_trip(
        self,
        sample_corpus: Corpus,
        sample_embeddings: np.ndarray,
        tmp_path: Path,
    ):
        index = build_index(sample_embeddings)
        index_path, id_map_path = save_index(index, sample_corpus, tmp_path)

        assert index_path.exists()
        assert id_map_path.exists()

        loaded_index, id_map = load_index(tmp_path)
        assert loaded_index.ntotal == 3
        assert loaded_index.d == EMBEDDING_DIM

        assert id_map["corpus_version"] == sample_corpus.version
        assert len(id_map["entries"]) == 3
        assert id_map["entries"][0]["code"] == "ABI"
        assert id_map["entries"][1]["code"] == "PSY"
        assert id_map["entries"][2]["code"] == "MED"

    def test_load_missing_index_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_index(tmp_path)

    def test_load_missing_id_map_raises(
        self,
        sample_corpus: Corpus,
        sample_embeddings: np.ndarray,
        tmp_path: Path,
    ):
        index = build_index(sample_embeddings)
        faiss.write_index(index, str(tmp_path / "index.faiss"))
        # id_map.json deliberately not written
        with pytest.raises(FileNotFoundError):
            load_index(tmp_path)


class TestSearchIndex:
    def test_search_returns_top_k(self, sample_embeddings: np.ndarray):
        index = build_index(sample_embeddings)
        query = sample_embeddings[0]
        scores, positions = search_index(index, query, top_k=2)
        assert scores.shape == (2,)
        assert positions.shape == (2,)

    def test_search_top_result_is_exact_match(self, sample_embeddings: np.ndarray):
        index = build_index(sample_embeddings)
        # Searching with an exact vector from the index should return that position
        for i in range(3):
            scores, positions = search_index(
                index, sample_embeddings[i], top_k=1)
            assert positions[0] == i
            assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_search_handles_2d_input(self, sample_embeddings: np.ndarray):
        index = build_index(sample_embeddings)
        query = sample_embeddings[0:1]  # (1, 384)
        scores, positions = search_index(index, query, top_k=1)
        assert positions[0] == 0
