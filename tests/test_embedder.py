"""Tests for the Embedder class."""

from __future__ import annotations

import numpy as np
import pytest

from askthestacks.embedder import EMBEDDING_DIM, Embedder, embed_corpus
from askthestacks.schema import Corpus, DatabaseEntry


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Load the model once for the whole test session — it's expensive."""
    return Embedder()


@pytest.fixture
def sample_corpus() -> Corpus:
    return Corpus(
        entries=[
            DatabaseEntry(
                code="ABI",
                name="ABI/INFORM",
                subject_hint="Business",
                coverage="indexes 3,700+ periodicals",
                url="https://www.wiu.edu/library/direct/?ABI",
            ),
            DatabaseEntry(
                code="PSY",
                name="PsycINFO",
                subject_hint="Psychology",
                coverage="psychology and behavioral sciences",
                url="https://www.wiu.edu/library/direct/?PSY",
            ),
        ]
    )


class TestEmbedder:
    def test_embed_query_returns_384_dim_vector(self, embedder: Embedder):
        vec = embedder.embed_query("test query")
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32

    def test_embed_query_normalized(self, embedder: Embedder):
        vec = embedder.embed_query("anything")
        norm = float((vec ** 2).sum() ** 0.5)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_query_rejects_empty(self, embedder: Embedder):
        with pytest.raises(ValueError):
            embedder.embed_query("")
        with pytest.raises(ValueError):
            embedder.embed_query("   ")

    def test_embed_documents_returns_batch(self, embedder: Embedder):
        texts = ["first doc", "second doc", "third doc"]
        out = embedder.embed_documents(texts)
        assert out.shape == (3, EMBEDDING_DIM)
        assert out.dtype == np.float32

    def test_embed_documents_normalized(self, embedder: Embedder):
        out = embedder.embed_documents(["any text"])
        norm = float((out[0] ** 2).sum() ** 0.5)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_documents_empty_list(self, embedder: Embedder):
        out = embedder.embed_documents([])
        assert out.shape == (0, EMBEDDING_DIM)

    def test_embed_documents_deterministic(self, embedder: Embedder):
        a = embedder.embed_documents(["the same text"])
        b = embedder.embed_documents(["the same text"])
        assert np.allclose(a, b)

    def test_query_and_document_differ(self, embedder: Embedder):
        """Query has instruction prefix; document doesn't. Vectors should differ."""
        q = embedder.embed_query("PTSD in veterans")
        d = embedder.embed_documents(["PTSD in veterans"])[0]
        # They embed the same text differently due to the query prefix
        assert not np.allclose(q, d)


class TestEmbedCorpus:
    def test_embed_corpus_shape(self, embedder: Embedder, sample_corpus: Corpus):
        out = embed_corpus(sample_corpus, embedder)
        assert out.shape == (2, EMBEDDING_DIM)

    def test_embed_corpus_order_matches_entries(
        self, embedder: Embedder, sample_corpus: Corpus
    ):
        out = embed_corpus(sample_corpus, embedder)
        # Embed individually and confirm row 0 == ABI text, row 1 == PSY text
        manual = embedder.embed_documents(
            [e.embedding_text for e in sample_corpus.entries]
        )
        assert np.allclose(out, manual)
