import asyncio

import httpx
import structlog

log = structlog.get_logger()


async def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": "AskTheStacks/0.1 (WIU Library Database Navigator)"
    }
    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        for attempt in (1, 2):
            try:
                log.info("http_get", url=url, attempt=attempt)
                response = await client.get(url)
                response.raise_for_status()
                log.info(
                    "http_get_success",
                    url=url,
                    bytes=len(response.text),
                )
                return response.text
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt == 2:
                    log.error("http_get_failed", url=url, error=str(e))
                    raise
                log.warning(
                    "http_get_retry",
                    url=url,
                    error=str(e),
                    backoff_seconds=2,
                )
                await asyncio.sleep(2)
            except httpx.HTTPStatusError as e:
                if 500 <= e.response.status_code < 600 and attempt == 1:
                    log.warning(
                        "http_get_retry_5xx",
                        url=url,
                        status=e.response.status_code,
                        backoff_seconds=2,
                    )
                    await asyncio.sleep(2)
                    continue
                log.error(
                    "http_get_failed",
                    url=url,
                    status=e.response.status_code,
                )
                raise

    raise RuntimeError("unreachable")
