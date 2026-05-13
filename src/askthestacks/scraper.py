import asyncio

import httpx
import structlog
import re
from askthestacks.schema import DatabaseEntry
from selectolax.parser import HTMLParser, Node

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

_REDIRECT_CODE_RE = re.compile(r"/library/direct/\?([A-Za-z0-9_-]+)$")


def _parse_row(row: Node) -> DatabaseEntry | None:
    cells = row.css("td")
    if len(cells) != 5:
        return None

    name_cell = cells[1]

    link = name_cell.css_first("big a[href]")
    if link is None:
        return None

    href = link.attributes.get("href", "")
    if not href:
        return None

    match = _REDIRECT_CODE_RE.search(href)
    if match is None:
        return None
    code = match.group(1)

    name = link.text(strip=True)
    if not name:
        return None

    subject_hint_node = name_cell.css_first("small")
    subject_hint = None
    if subject_hint_node is not None:
        raw = subject_hint_node.text(strip=True)
        if raw.startswith("(") and raw.endswith(")"):
            subject_hint = raw[1:-1].strip() or None

    dates = cells[2].text(strip=True) or None
    coverage = cells[3].text(strip=True) or None
    full_text = cells[4].text(strip=True) or None

    try:
        return DatabaseEntry(
            code=code,
            name=name,
            subject_hint=subject_hint,
            dates=dates,
            coverage=coverage,
            full_text=full_text,
            url=href,
        )
    except Exception as e:
        log.warning("row_validation_failed", name=name, url=href, error=str(e))
        return None


def parse_databases_html(html: str) -> list[DatabaseEntry]:
    parser = HTMLParser(html)
    rows = parser.css("tr")
    log.info("parse_start", row_candidates=len(rows))

    entries: list[DatabaseEntry] = []
    seen_codes: set[str] = set()

    for row in rows:
        entry = _parse_row(row)
        if entry is None:
            continue
        if entry.code in seen_codes:
            log.debug("duplicate_row_skipped",
                      code=entry.code, name=entry.name)
            continue
        seen_codes.add(entry.code)
        entries.append(entry)

    log.info("parse_complete", entries_extracted=len(entries))
    return entries


def _parse_ul_subcollection(html: str) -> list[DatabaseEntry]:
    parser = HTMLParser(html)
    main_content = parser.css_first("#mainContentFull")
    if main_content is None:
        log.warning("ul_subcollection_no_main_content")
        return []

    entries: list[DatabaseEntry] = []
    seen_codes: set[str] = set()

    for link in main_content.css("ul li a[href]"):
        href = link.attributes.get("href", "")
        match = _REDIRECT_CODE_RE.search(href)
        if match is None:
            continue
        code = match.group(1)
        name = link.text(strip=True)
        if not name or code in seen_codes:
            continue
        try:
            entries.append(DatabaseEntry(code=code, name=name, url=href))
            seen_codes.add(code)
        except Exception as e:
            log.warning("ul_row_validation_failed",
                        name=name, url=href, error=str(e))

    log.info("ul_subcollection_parsed", entries_extracted=len(entries))
    return entries


def _parse_table_subcollection(html: str) -> list[DatabaseEntry]:
    parser = HTMLParser(html)
    main_content = parser.css_first("#mainContentFull")
    if main_content is None:
        log.warning("table_subcollection_no_main_content")
        return []

    entries: list[DatabaseEntry] = []
    seen_codes: set[str] = set()

    for row in main_content.css("table tbody tr"):
        link = row.css_first("a[href].norm2")
        if link is None:
            continue

        href = link.attributes.get("href", "")
        match = _REDIRECT_CODE_RE.search(href)
        if match is None:
            continue
        code = match.group(1)

        name = link.text(strip=True)
        if not name or code in seen_codes:
            continue

        hint_node = link.parent.css_first("small") if link.parent else None
        subject_hint = None
        if hint_node is not None:
            raw = hint_node.text(strip=True)
            if raw.startswith("(") and raw.endswith(")"):
                subject_hint = raw[1:-1].strip() or None

        try:
            entries.append(
                DatabaseEntry(
                    code=code,
                    name=name,
                    subject_hint=subject_hint,
                    url=href,
                )
            )
            seen_codes.add(code)
        except Exception as e:
            log.warning("table_row_validation_failed",
                        name=name, url=href, error=str(e))

    log.info("table_subcollection_parsed", entries_extracted=len(entries))
    return entries


SUBCOLLECTION_SOURCES: tuple[tuple[str, str], ...] = (
    (
        "https://www.wiu.edu/libraries/databases/AlexanderStPressVideos.php",
        "ul",
    ),
    (
        "https://www.wiu.edu/libraries/databases/?ebkCollections=1&showebooks=1",
        "table",
    ),
    (
        "https://www.wiu.edu/libraries/databases/?ejCollections=1&showej=1",
        "table",
    ),
)


async def scrape_all(main_url: str) -> list[DatabaseEntry]:
    fetch_tasks = [fetch_page(main_url)]
    fetch_tasks.extend(fetch_page(url) for url, _ in SUBCOLLECTION_SOURCES)

    htmls = await asyncio.gather(*fetch_tasks)

    main_html = htmls[0]
    entries = parse_databases_html(main_html)

    seen_codes = {e.code for e in entries}

    for html, (_, parser_type) in zip(htmls[1:], SUBCOLLECTION_SOURCES, strict=True):
        if parser_type == "ul":
            sub_entries = _parse_ul_subcollection(html)
        else:
            sub_entries = _parse_table_subcollection(html)

        for entry in sub_entries:
            if entry.code in seen_codes:
                log.debug("subcollection_duplicate_skipped", code=entry.code)
                continue
            seen_codes.add(entry.code)
            entries.append(entry)

    log.info("scrape_all_complete", total_entries=len(entries))
    return entries
