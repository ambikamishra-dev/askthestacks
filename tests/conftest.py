"""Shared pytest fixtures for the test suite."""

from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "databases_page.html"


@pytest.fixture(scope="session")
def fixture_html() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")
