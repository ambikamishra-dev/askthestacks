"""Build a versioned corpus + FAISS index of WIU library databases."""

import asyncio
import json
import logging
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog

from askthestacks.embedder import Embedder, embed_corpus
from askthestacks.index import build_index, save_index
from askthestacks.schema import Corpus
from askthestacks.scraper import scrape_all


WIU_DATABASES_URL = "https://www.wiu.edu/libraries/databases/"
OUTPUT_DIR = Path("data/corpus")
INDEX_DIR = Path("data/index")
MIN_EXPECTED_ENTRIES = 100


log = structlog.get_logger()


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


async def build_corpus() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    try:
        entries = await scrape_all(WIU_DATABASES_URL)
    except Exception as e:
        log.error("scrape_failed", error=str(e), error_type=type(e).__name__)
        return 1

    if len(entries) < MIN_EXPECTED_ENTRIES:
        log.error(
            "sanity_check_failed",
            entries_extracted=len(entries),
            minimum_expected=MIN_EXPECTED_ENTRIES,
            message="Fewer entries than expected — possible parsing failure",
        )
        return 2

    try:
        corpus = Corpus(entries=entries)
    except Exception as e:
        log.error("corpus_validation_failed", error=str(e))
        return 2

    # Persist corpus JSON
    corpus_path = OUTPUT_DIR / f"corpus_v1_{today}.json"
    corpus_path.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")

    latest_path = OUTPUT_DIR / "latest.json"
    shutil.copy(corpus_path, latest_path)

    # Build and persist FAISS index
    try:
        embedder = Embedder()
        embeddings = embed_corpus(corpus, embedder)
        index = build_index(embeddings)
        save_index(index, corpus, INDEX_DIR)
    except Exception as e:
        log.error("index_build_failed", error=str(
            e), error_type=type(e).__name__)
        return 3

    # Build report
    report = {
        "built_at": corpus.built_at.isoformat(),
        "source_url": corpus.source_url,
        "entry_count": corpus.entry_count,
        "entries_with_subject_hint": sum(1 for e in corpus.entries if e.subject_hint),
        "entries_with_dates": sum(1 for e in corpus.entries if e.dates),
        "entries_with_coverage": sum(1 for e in corpus.entries if e.coverage),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "index_type": "IndexFlatIP",
    }
    report_path = OUTPUT_DIR / f"build_report_{today}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info(
        "build_complete",
        corpus_path=str(corpus_path),
        latest_path=str(latest_path),
        index_dir=str(INDEX_DIR),
        entry_count=corpus.entry_count,
    )
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(asyncio.run(build_corpus()))
