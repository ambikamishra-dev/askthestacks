"""Launch the AskTheStacks API server with uvicorn.

Usage: python scripts/serve.py

Configuration is read from environment variables via the Settings class.
For development, you can put values in a .env file at the repo root.
"""

import uvicorn

from askthestacks.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "askthestacks.api:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
