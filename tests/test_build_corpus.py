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
    async def fake_scrape_all(_url: str):
        return parse_databases_html(fixture_html)

    monkeypatch.setattr("scripts.build_corpus.scrape_all", fake_scrape_all)

    class _FakeEmbedder:
        def __init__(self):
            pass

    def _fake_embed_corpus(_corpus, _embedder):
        import numpy as np
        return np.zeros((len(_corpus.entries), 384), dtype=np.float32)

    def _fake_build_index(_embeddings):
        class _FakeIndex:
            ntotal = len(_embeddings)
        return _FakeIndex()

    def _fake_save_index(_index, _corpus, _dir):
        Path(_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("scripts.build_corpus.Embedder", _FakeEmbedder)
    monkeypatch.setattr(
        "scripts.build_corpus.embed_corpus", _fake_embed_corpus)
    monkeypatch.setattr("scripts.build_corpus.build_index", _fake_build_index)
    monkeypatch.setattr("scripts.build_corpus.save_index", _fake_save_index)


def test_build_corpus_writes_three_files(tmp_output_dir: Path, stub_scrape_all):
    exit_code = asyncio.run(build_corpus())
    assert exit_code == 0

    versioned = list(tmp_output_dir.glob("corpus_v1_*.json"))
    reports = list(tmp_output_dir.glob("build_report_*.json"))
    latest = tmp_output_dir / "latest.json"

    assert len(versioned) == 1
    assert len(reports) == 1
    assert latest.exists()


def test_build_corpus_latest_matches_versioned(tmp_output_dir: Path, stub_scrape_all):
    asyncio.run(build_corpus())

    versioned = list(tmp_output_dir.glob("corpus_v1_*.json"))[0]
    latest = tmp_output_dir / "latest.json"

    assert versioned.read_text() == latest.read_text()


def test_build_corpus_report_has_entry_count(tmp_output_dir: Path, stub_scrape_all):
    asyncio.run(build_corpus())

    report_path = list(tmp_output_dir.glob("build_report_*.json"))[0]
    report = json.loads(report_path.read_text())

    assert report["entry_count"] > 100
    assert "built_at" in report
    assert report["source_url"]


def test_build_corpus_sanity_check_fails_on_too_few_entries(
    tmp_output_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    async def fake_scrape_tiny(_url: str):
        return []

    monkeypatch.setattr("scripts.build_corpus.scrape_all", fake_scrape_tiny)

    exit_code = asyncio.run(build_corpus())
    assert exit_code == 2
