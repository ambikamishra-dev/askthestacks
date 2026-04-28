# AskTheStacks

A retrieval-augmented search tool that helps WIU Library patrons find the right database for their research topic in seconds, instead of manually browsing 190+ subscriptions.

## The Problem

When a patron asks "where do I find articles on PTSD in veterans?", a librarian currently spends 30–60 minutes manually scanning the database directory to recommend the right starting points. AskTheStacks turns that into a 2-second answer.

## How It Works (high level)

```
Patron query
    ↓
Embedding model (bge-small-en-v1.5)
    ↓
FAISS vector search over WIU's database catalog
    ↓
Top 5–7 most relevant databases with names, descriptions, and direct links
```

## Status

🚧 **In active development — Day 1 of 7.** Targeting first demo May 3, 2026.

| Component               | Status      |
| ----------------------- | ----------- |
| Corpus scraper & schema | In progress |
| Subject enrichment      | Planned     |
| Embedding pipeline      | Planned     |
| FAISS retrieval         | Planned     |
| FastAPI backend         | Planned     |
| Embeddable frontend     | Planned     |
| Load testing (k6)       | Planned     |
| Docker + CI             | Planned     |

## Tech Stack

- **Python 3.12** with `src/` layout
- **Pydantic v2** for schema validation
- **FAISS** for vector similarity search
- **bge-small-en-v1.5** for embeddings
- **FastAPI** with async endpoints
- **structlog** for production-grade structured logging
- **Docker** for reproducible deployment

## Local Development

Setup instructions will be added once the package is installable. See `LEARNING.md` for an ongoing log of design decisions and tradeoffs.

## License

MIT
