"""Embedder: turns DatabaseEntry text into 384-dim vectors using bge-small-en-v1.5.

Design notes:
- Model: BAAI/bge-small-en-v1.5 (384-dim, ~80MB, optimized for English semantic search).
- We use sentence-transformers' SentenceTransformer wrapper for convenience.
- bge-small recommends a query instruction prefix for queries (not for documents).
- Vectors are L2-normalized so we can use IndexFlatIP (inner product) which equals
  cosine similarity for normalized vectors. This is faster and standard practice.
- Single source of truth for embedding_text comes from DatabaseEntry.embedding_text.
- The model loads once and is reused (expensive: ~1-2s load time).
"""

from __future__ import annotations

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from askthestacks.schema import Corpus, DatabaseEntry

log = structlog.get_logger()

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384


class Embedder:
    """Wraps the sentence-transformers model. Stateful: holds the loaded model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        log.info("embedder_loading_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        log.info("embedder_model_loaded", model=model_name, dim=EMBEDDING_DIM)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed document texts (no query prefix). Returns (N, 384) L2-normalized array."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        log.info("embedder_embedding_documents", count=len(texts))
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (with instruction prefix). Returns (384,) L2-normalized array."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        prefixed = QUERY_INSTRUCTION + query
        embedding = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)


def embed_corpus(corpus: Corpus, embedder: Embedder) -> np.ndarray:
    """Embed every entry in a corpus. Returns (N, 384) array in corpus.entries order."""
    texts = [entry.embedding_text for entry in corpus.entries]
    return embedder.embed_documents(texts)
