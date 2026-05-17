"""Tests for the build_corpus script."""

import asyncio
import json
from pathlib import Path

import pytest

from askthestacks.scraper import parse_databases_html
from scripts.build_corpus import build_corpus


@pytest.fixture
def tmp_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output = tmp_path / "corpus"
    output.mkdir()
    monkeypatch.setattr("scripts.build_corpus.OUTPUT_DIR", output)
    return output


@pytest.fixture
def stub_scrape_all(monkeypatch: pytest.MonkeyPatch, fixture_html: str):
    from askthestacks.scraper import parse_databases_html

    async def fake_scrape_all(_url: str):
        return parse_databases_html(fixture_html)

    monkeypatch.setattr("scripts.build_corpus.scrape_all", fake_scrape_all)

    # Also stub embedder + index so build_corpus tests don't actually run the model
    class _FakeEmbedder:
        def __init__(self): pass

    def _fake_embed_corpus(_corpus, _embedder):
        import numpy as np
        return np.zeros((len(_corpus.entries), 384), dtype=np.float32)

    def _fake_build_index(_embeddings):
        class _FakeIndex:
            ntotal = len(_embeddings)
        return _FakeIndex()

    def _fake_save_index(_index, _corpus, _dir):
        from pathlib import Path
        Path(_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("scripts.build_corpus.Embedder", _FakeEmbedder)
    monkeypatch.setattr(
        "scripts.build_corpus.embed_corpus", _fake_embed_corpus)
    monkeypatch.setattr("scripts.build_corpus.build_index", _fake_build_index)
    monkeypatch.setattr("scripts.build_corpus.save_index", _fake_save_index)
