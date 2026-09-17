# AskTheStacks

Natural-language semantic search over an academic library's database directory.

Type "PTSD in veterans" and get the 5 most relevant research databases, ranked
by semantic similarity — not keyword match. Built as a graduate assistantship
project for a university library that subscribes to 186 research databases
(PubMed, JSTOR, PsycINFO, and 183 others). Formally accepted by the library
with public release approval; awaiting institutional deployment schedule.

## The problem

Patrons — students, professors, thesis researchers — had to manually scan an
alphabetical list of 186 databases to figure out which one fit their topic.
A student writing about PTSD in veterans had to know that PTSDpubs existed,
or scroll through the whole directory hoping to spot it.

This was a slow research step everyone had normalized. The system replaces it
with a single natural-language query that returns ranked, direct-link results
in under 200ms.

## What it does

- **Semantic search** over the full database directory using sentence-transformer
  embeddings. Understands that "heart disease" and "cardiovascular research"
  mean similar things, even when the words don't overlap.
- **Sub-200ms responses** end-to-end from a warm cache; ~100ms cold.
- **Ranked results** with database name, subject area, coverage description,
  and a direct link to the resource.
- **Embedded widget** dropped into a single HTML page — no framework build step,
  no bundler. Ships as one file.

## Architecture

Three layers, each with a single responsibility:
                ┌──────────────────────────────┐
                │  Vanilla JS Widget           │
                │  (single HTML file, no deps) │
                └──────────────┬───────────────┘
                               │  HTTP GET /api/search
                               ▼
                ┌──────────────────────────────┐
                │  FastAPI Service             │
                │  - Two-tier rate limiting    │
                │  - LRU query cache           │
                │  - Structured logging        │
                │  - Pydantic validation       │
                │  - CORS via env var          │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Retriever                   │
                │  bge-small-en-v1.5 embedder  │
                │  FAISS IndexFlatIP (cosine)  │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Corpus (186 entries)        │
                │  Built by async scraper      │
                │  Versioned JSON + FAISS bin  │
                └──────────────────────────────┘

### Data pipeline

An async Python scraper (`httpx` + `selectolax`) pulls the current database
list from the library's four directory pages (main directory + 3 sub-collection
pages) concurrently. Categorized retry policy: retry on 5xx and network errors,
never on 4xx. HTML parsed with `selectolax` (10-100x faster than BeautifulSoup —
matters when re-running the scraper on a schedule).

Each database becomes a `DatabaseEntry` — a Pydantic model that validates
schema at construction. The `embedding_text` field is computed in the model
itself, so downstream stages (embedder, index) can't accidentally embed
different text than what the schema promises. Single source of truth.

The build script produces a versioned JSON corpus (`corpus_v1_<date>.json`)
plus a `latest.json` symlink, alongside a build report tracking entry counts,
fetch duration, and dropped rows. Sanity check fails loudly if the entry count
drops below a threshold — catches "page structure changed and we silently
scraped nothing" before it ships.

### Retrieval

Embeddings via `BAAI/bge-small-en-v1.5` — 384 dimensions, ~80MB, CPU-only,
highly rated on MTEB for English semantic search. L2-normalized at embedding
time so FAISS `IndexFlatIP` (inner product) gives cosine similarity for free.

FAISS `IndexFlatIP` (exhaustive search) rather than approximate (HNSW, IVF).
At 186 vectors, exhaustive is instant and exact — approximate indexes only
pay off at 10k+ vectors and trade accuracy for speed.

bge-small's recommended query-instruction prefix ("Represent this sentence
for searching relevant passages: ") is applied to queries but not documents.

### API layer

FastAPI service with three endpoints:

| Endpoint         | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| `GET /api/search`| Ranked results for a query                     |
| `GET /api/health`| Corpus version + entry count, for monitoring   |
| `GET /api/version`| Build metadata: model, corpus version, index  |

Production concerns applied by layer, not universally:

- **Lifespan management**: Retriever loads once at startup, held in app state.
  Every request reuses the loaded model and index. Without this pattern, every
  user would eat a 1-2 second cold start.
- **Two-tier rate limiting** via `slowapi`: 5 req/sec burst + 30 req/min sustained
  per IP. Burst alone doesn't stop sustained pressure; sustained alone doesn't
  stop burst attacks. Both together match real usage patterns.
- **LRU query cache**: repeated queries during exam weeks skip the embedding +
  search entirely. Measured 4400x speedup on cache hits.
- **CORS** configurable via `ALLOWED_ORIGINS` env var (default `*` for dev).
- **Structured request logging** via `structlog`: method, path, status, duration,
  client IP, query — every request is a structured JSON event.
- **Sanitized error handling**: internal exceptions never leak stack traces to
  clients. Client sees `{"error": "internal server error"}`; full traceback
  logs server-side.
- **Pydantic input validation**: query length capped at 500 chars, `k` clamped
  to `max_top_k`. Bad input rejected with clean 400 before hitting business logic.

### Frontend

Single `index.html` file with embedded CSS and JS. Debounced search (300ms after
typing stops), loading and empty states, error states for network failures and
429s, keyboard support (Enter triggers immediate search). Mobile-responsive.

No React, no build step, no bundler. Deliberate: the widget will eventually
embed into the library's CMS, which doesn't play well with framework tooling.
Vanilla JS drops into any HTML page cleanly.

## Quality — how I know it works

Automated retrieval quality doesn't have a ground truth for this domain, so I
built a hand-rated evaluation harness. Six representative queries spanning
health, history, education, and interdisciplinary topics. For each query, I
personally graded the top-5 results as `relevant`, `partial`, or `not_relevant`,
mapped to numeric scores (1.0 / 0.5 / 0.0).

Three IR metrics:

| Metric        | Score | Interpretation                                       |
| ------------- | ----- | ---------------------------------------------------- |
| Hit@1         | 100%  | Every query surfaced a useful database at rank 1     |
| NDCG@5        | 97%   | Ranking is near-optimal — best matches at the top    |
| Precision@5   | 65%   | Average relevance across top-5 results               |

Precision@5 reflects collection size, not retrieval quality. WIU has 186
databases; for niche queries, five perfect matches simply don't exist in the
collection. The system correctly returns partial matches instead of padding
with irrelevant noise.

The eval script (`scripts/eval.py`) also detects **drift**: if the live retriever
returns different top-5 codes than what was hand-rated (because the corpus was
rebuilt or the model changed), the query is flagged and excluded from metrics
with a clear warning. Keeps the eval set honest over time.

## Test suite

79 passing tests + 1 documented `xfail`. Coverage across all layers:

- Schema validation (13 tests)
- HTTP fetch behavior — mocked with `respx`, no real network (4 tests)
- HTML parsing — mocked with fixture files, deterministic (13 tests)
- Build orchestration (4 tests)
- Embedder shape and normalization (7 tests)
- FAISS index build, save, load, search (8 tests)
- Retrieval end-to-end (12 tests)
- API layer — happy path, validation, cache, rate limit, error handling (17 tests)
- Static file serving (3 tests)

The `xfail` is on a burst-rate-limit test where `slowapi`'s module-level state
doesn't isolate cleanly between tests. Rate limiting itself is verified manually
via curl and by a companion test checking response shape. Documenting the
tooling limitation with `xfail` was cheaper and more honest than building
elaborate infrastructure to work around it.

Suite runs in under 20 seconds. Deterministic — no real network calls in any test.

## Design decisions worth calling out

**Semantic > keyword search**: user language rarely matches curator language
word-for-word. "Heart disease" needs to find "cardiovascular research."

**bge-small over a bigger model**: 186 documents didn't need a large model.
Bigger would add latency and cost with no measurable retrieval gain.

**FAISS `IndexFlatIP` over approximate indexes**: exhaustive is exact and
instant at this scale. Approximate would trade accuracy for speed we don't need.

**FastAPI over Flask/Django**: async support, Pydantic-based validation,
auto-generated OpenAPI docs. Django is overkill (no admin, no ORM); Flask
requires more manual work for the same result.

**Vanilla JS over React**: the widget will embed in the library's CMS.
Framework tooling would fight the CMS; a single HTML file drops in cleanly.

**Hand-rated eval over automated metrics**: no ground truth exists for this
domain. A librarian's judgment is the ground truth. Six diverse queries with
per-result grading > 1000 auto-scored queries against a synthetic label set.

**Rate limiting via `slowapi`, not custom**: leverages proven middleware
integration with FastAPI. Documented the one test-isolation quirk with
`xfail` rather than reimplementing.

**No LLM in the retrieval path**: deliberate. In an institutional context,
generated answers about library resources could mislead patrons in ways that
matter. Retrieval half is complete; generation was out of scope for the
institutional need.

## What I'd add next

- **Docker image + GitHub Actions CI** — scoped out for the acceptance timeline
- **Load testing with k6** — measure sustained throughput and P95 latency
- **Query analytics** — log unanswered or low-confidence queries for the
  librarians to review, expand the collection or refine descriptions
- **Optional generation layer** — for contexts where grounded LLM answers are
  acceptable, add generation over retrieved databases

## Stack

- **Language**: Python 3.12
- **API**: FastAPI, Uvicorn
- **Embeddings**: sentence-transformers (`BAAI/bge-small-en-v1.5`)
- **Vector search**: FAISS (`faiss-cpu`)
- **Scraping**: `httpx` (async), `selectolax`
- **Validation**: Pydantic v2
- **Rate limiting**: `slowapi`
- **Logging**: `structlog`
- **Testing**: `pytest`, `respx`
- **Config**: `pydantic-settings` (12-factor env vars)
- **Frontend**: Vanilla JS + HTML + CSS (single file)

## Running locally

```bash
# Setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Build the corpus + index (one command, ~5 seconds)
python scripts/build_corpus.py

# Run the eval harness
python scripts/eval.py

# Run the test suite
pytest -v

# Start the API + widget
python scripts/serve.py
# Open http://127.0.0.1:8000/
```

## Status

Formally accepted by the library with signed acceptance and public release
approval. Deployment awaits the institution's own schedule — they are
rebuilding their site and it will be integrated then.

## Repository structure

askthestacks/
├── src/askthestacks/
│ ├── schema.py # Pydantic models — corpus contract
│ ├── scraper.py # Async httpx + selectolax
│ ├── embedder.py # bge-small wrapper
│ ├── index.py # FAISS build/save/load/search
│ ├── retrieval.py # Retriever — glues embedder + index + corpus
│ ├── api.py # FastAPI app, lifespan, middleware, endpoints
│ ├── api_models.py # Request/response Pydantic models
│ ├── api_cache.py # LRU query cache
│ ├── config.py # pydantic-settings — env-driven config
│ └── static/
│ └── index.html # Widget (HTML + CSS + JS in one file)
├── scripts/
│ ├── build_corpus.py # Orchestrator: scrape → embed → index → save
│ ├── eval.py # Hand-rated eval harness with drift detection
│ └── serve.py # Uvicorn launcher
├── eval/
│ ├── queries.json # Hand-rated golden set
│ └── last_run.json # Latest eval report
├── tests/
│ ├── fixtures/ # Frozen HTML for deterministic parser tests
│ ├── test_schema.py
│ ├── test_scraper.py
│ ├── test_embedder.py
│ ├── test_index.py
│ ├── test_retrieval.py
│ ├── test_build_corpus.py
│ └── test_api.py
├── data/
│ ├── corpus/ # Versioned JSON output
│ └── index/ # FAISS binary + id_map.json
└── pyproject.toml
