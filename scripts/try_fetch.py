"""Manual smoke test: fetch the live WIU databases page.

Optional: pass --save to also write the HTML to tests/fixtures/databases_page.html
for use as a parser test fixture.
"""

import asyncio
import sys
from pathlib import Path

from askthestacks.scraper import fetch_page

URL = "https://www.wiu.edu/libraries/databases/"


async def main() -> None:
    html = await fetch_page(URL)
    print(f"Got {len(html)} characters")
    print(html[:200])

    if "--save" in sys.argv:
        fixture_path = Path("tests/fixtures/databases_page.html")
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(html, encoding="utf-8")
        print(f"\nSaved fixture to {fixture_path}")


if __name__ == "__main__":
    asyncio.run(main())
