import httpx
import respx

import pytest

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
