"""FAISS index: builds searchable index from corpus embeddings.

Design notes:
- IndexFlatIP (inner product) — for L2-normalized vectors this equals cosine similarity.
- Flat (exhaustive) search — at 186 entries it's instant and gives exact results.
  Approximate indexes (IVF, HNSW) only pay off at 10k+ vectors.
- Persistence: index.faiss (binary) + id_map.json (code-to-position mapping).
  The id_map is the only way to translate FAISS's internal int positions back to db codes.
- Reproducibility: we save corpus_version into id_map so loaders can verify the index
  matches the corpus they have.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import structlog

from askthestacks.schema import Corpus

log = structlog.get_logger()


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from (N, dim) embeddings."""
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("faiss_index_built", entries=n, dim=dim)
    return index


def save_index(
    index: faiss.IndexFlatIP,
    corpus: Corpus,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist the index and an id_map alongside it. Returns (index_path, id_map_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    id_map_path = output_dir / "id_map.json"
    id_map = {
        "corpus_version": corpus.version,
        "built_at": corpus.built_at.isoformat(),
        "source_url": corpus.source_url,
        "entries": [
            {"position": i, "code": entry.code, "name": entry.name}
            for i, entry in enumerate(corpus.entries)
        ],
    }
    id_map_path.write_text(json.dumps(id_map, indent=2), encoding="utf-8")

    log.info(
        "faiss_index_saved",
        index_path=str(index_path),
        id_map_path=str(id_map_path),
        entries=len(corpus.entries),
    )
    return index_path, id_map_path


def load_index(index_dir: Path) -> tuple[faiss.IndexFlatIP, dict]:
    """Load a persisted index and its id_map. Returns (index, id_map_dict)."""
    index_path = index_dir / "index.faiss"
    id_map_path = index_dir / "id_map.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not id_map_path.exists():
        raise FileNotFoundError(f"id_map not found at {id_map_path}")

    index = faiss.read_index(str(index_path))
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

    log.info(
        "faiss_index_loaded",
        index_path=str(index_path),
        entries=index.ntotal,
    )
    return index, id_map


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Search index with a (384,) or (1, 384) query. Returns (scores, positions)."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    scores, positions = index.search(query_embedding, top_k)
    return scores[0], positions[0]
