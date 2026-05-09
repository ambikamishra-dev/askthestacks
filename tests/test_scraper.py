import httpx
import respx

import pytest
from pathlib import Path

from askthestacks.scraper import parse_databases_html

from askthestacks.scraper import fetch_page


@pytest.mark.asyncio
@respx.mock
async def test_fetch_page_returns_html_on_success():
    url = "https://www.wiu.edu/libraries/databases/"
    respx.get(url).mock(
        return_value=httpx.Response(200, text="<html>fake page</html>")
    )
    result = await fetch_page(url)
    assert result == "<html>fake page</html>"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_page_does_not_retry_on_404():
    url = "https://www.wiu.edu/libraries/this-page-does-not-exist"
    route = respx.get(url).mock(return_value=httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_page(url)

    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_fetch_page_retries_on_503_then_succeeds():
    url = "https://www.wiu.edu/libraries/databases/"
    route = respx.get(url).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, text="<html>recovered</html>"),
        ]
    )

    result = await fetch_page(url)

    assert "recovered" in result
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_fetch_page_retries_on_network_error_then_raises():
    url = "https://www.wiu.edu/libraries/databases/"
    route = respx.get(url).mock(
        side_effect=httpx.ConnectError("simulated network failure")
    )

    with pytest.raises(httpx.NetworkError):
        await fetch_page(url)

    assert route.call_count == 2


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "databases_page.html"


@pytest.fixture(scope="session")
def fixture_html() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def parsed_entries(fixture_html: str):
    return parse_databases_html(fixture_html)


class TestParseDatabasesHtml:
    def test_parses_a_meaningful_number_of_entries(self, parsed_entries):
        assert len(parsed_entries) > 100, (
            f"Expected 100+ databases, got {len(parsed_entries)}"
        )

    def test_extracts_known_database_abi_inform(self, parsed_entries):
        entries_by_code = {e.code: e for e in parsed_entries}
        assert "ABI" in entries_by_code
        abi = entries_by_code["ABI"]
        assert abi.name == "ABI/INFORM"
        assert abi.subject_hint == "Business"
        assert abi.url == "https://www.wiu.edu/library/direct/?ABI"

    def test_extracts_known_database_academic_search_complete(self, parsed_entries):
        entries_by_code = {e.code: e for e in parsed_entries}
        assert "ASC" in entries_by_code
        asc = entries_by_code["ASC"]
        assert asc.name == "Academic Search Complete"
        assert asc.subject_hint == "Multi-Disciplinary"

    def test_skips_sub_collection_rows(self, parsed_entries):
        names = {e.name for e in parsed_entries}
        assert "Alexander Street Press Videos" not in names
        assert "E-Book Collections" not in names
        assert "E-Journal Collections" not in names

    def test_no_duplicate_codes(self, parsed_entries):
        codes = [e.code for e in parsed_entries]
        assert len(codes) == len(set(codes)), "Duplicate codes leaked through"

    def test_all_entries_have_valid_wiu_urls(self, parsed_entries):
        for e in parsed_entries:
            assert "wiu.edu" in e.url, f"Non-WIU URL leaked: {e.url}"

    def test_handles_empty_html(self):
        result = parse_databases_html("")
        assert result == []

    def test_handles_garbage_html(self):
        result = parse_databases_html(
            "<html><body>not a database page</body></html>")
        assert result == []
