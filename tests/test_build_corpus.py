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
def stub_fetch(monkeypatch: pytest.MonkeyPatch, fixture_html: str):
    async def fake_fetch(_url: str) -> str:
        return fixture_html

    monkeypatch.setattr("scripts.build_corpus.fetch_page", fake_fetch)


def test_build_corpus_writes_three_files(tmp_output_dir: Path, stub_fetch):
    exit_code = asyncio.run(build_corpus())
    assert exit_code == 0

    versioned = list(tmp_output_dir.glob("corpus_v1_*.json"))
    reports = list(tmp_output_dir.glob("build_report_*.json"))
    latest = tmp_output_dir / "latest.json"

    assert len(versioned) == 1
    assert len(reports) == 1
    assert latest.exists()


def test_build_corpus_latest_matches_versioned(tmp_output_dir: Path, stub_fetch):
    asyncio.run(build_corpus())

    versioned = list(tmp_output_dir.glob("corpus_v1_*.json"))[0]
    latest = tmp_output_dir / "latest.json"

    assert versioned.read_text() == latest.read_text()


def test_build_corpus_report_has_entry_count(tmp_output_dir: Path, stub_fetch):
    asyncio.run(build_corpus())

    report_path = list(tmp_output_dir.glob("build_report_*.json"))[0]
    report = json.loads(report_path.read_text())

    assert report["entry_count"] > 100
    assert "built_at" in report
    assert report["source_url"]


def test_build_corpus_sanity_check_fails_on_too_few_entries(
    tmp_output_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    async def fake_fetch_tiny(_url: str) -> str:
        return "<html><body>nothing here</body></html>"

    monkeypatch.setattr("scripts.build_corpus.fetch_page", fake_fetch_tiny)

    exit_code = asyncio.run(build_corpus())
    assert exit_code == 2
