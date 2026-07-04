# Local Setup and Test Guide

This guide covers local development setup, behavior-first test workflow, and API smoke testing for the Alpha Financial Analyst system.

This is a support guide. The source of truth remains `docs/SPEC.md`, `docs/sprint-plan.md`, `docs/test-plan.md`, `docs/retrieval-benchmark.md`, and the active task in `tasks/todo.md`.

## Scope

- Use this guide for local development, verification, and smoke testing.
- Do not use this guide as production deployment guidance.
- Do not commit `.env`, local runtime files, downloaded filings, vector data, or secrets.
- Do not read or print an existing `.env` during agent work unless the user explicitly approves it.

## Prerequisites

Install these before running the project:

- Git.
- Python 3.12, or another interpreter supported by `pyproject.toml`.
- `uv` for Python dependency management.
- Docker, optional but recommended for Qdrant.
- Ollama, optional for live LLM and embedding flows.
- A SEC EDGAR user-agent contact string.
- A Tavily API key only if running news search.

The static test suite does not require live Ollama, Tavily, or SEC access. Ingestion and analysis smoke tests do.

## Step 1 - Open the repository

```bash
git clone <repo-url>
cd Financial-Analyst-system
```

If the repository already exists, start from the project root:

```bash
pwd
git status --short
```

Review existing local changes before editing. Do not revert unrelated changes.

## Step 2 - Install dependencies

Install the Python version and synchronize the default dependency groups:

```bash
uv python install 3.12
uv sync --python 3.12
```

If the local environment has a restricted home directory or cache path, use a writable cache:

```bash
uv --cache-dir /tmp/uv-cache sync --python 3.12
```

## Step 3 - Configure local environment

Create `.env` from this template and fill in only the values needed for the workflow you are running:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3.5:9b
OLLAMA_EMBED_MODEL=qwen3-embedding:4b
OLLAMA_TIMEOUT=600
OLLAMA_TEMPERATURE=0.7

VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=sec_filings

SEC_USER_AGENT="Your Name your-email@example.com"
TAVILY_API_KEY=

LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=financial-analyst-system
```

Use `VECTOR_BACKEND=chroma` for local fallback when Qdrant is not running.

## Step 4 - Start Qdrant for retrieval tests

Start the local Qdrant service:

```bash
docker compose -f docker-compose.qdrant.yml up -d
docker ps --filter name=qdrant-financial-analyst-agent-system
```

Follow logs if Qdrant does not appear healthy:

```bash
docker logs qdrant-financial-analyst-agent-system
```

## Step 5 - Optional Ollama setup

Install Ollama and pull the configured models if you want to run live ingestion, embedding, or analysis flows:

```bash
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:4b
ollama list
```

If model names change, update `.env` and keep the test evidence in `tasks/todo.md`.

## Step 6 - Follow the behavior-first test protocol

Before changing behavior:

1. Trace the change to `docs/test-plan.md`, `docs/sprint-plan.md`, `docs/retrieval-benchmark.md`, SPEC, or an ADR.
2. Add or update the task entry in `tasks/todo.md`.
3. Write the failing or protective behavior test before implementation.
4. Assert through public interfaces: API routes, service functions, retrieval contracts, or documented evaluators.
5. Avoid assertions against private attributes, implementation-only call order, or mocked internals.
6. Use faithful fakes that compute from input rather than echoing expected answers.
7. Run `python scripts/ci/check_test_hygiene.py` before marking the task done.

## Step 7 - Run verification

Run the focused test first, then the broader suite for the changed area.

Core checks:

```bash
uv run ruff check .
uv run mypy app evaluation
uv run pytest -q
```

Document governance checks:

```bash
python scripts/ci/check_no_scope_residue.py
python scripts/ci/check_sprint_map.py
python scripts/ci/check_doc_sync.py
python scripts/ci/check_test_hygiene.py
```

Ingestion and retrieval checks:

```bash
uv run pytest tests/ingestion -q
uv run pytest tests/unit/test_vector_store_factory.py -q
VECTOR_BACKEND=qdrant QDRANT_URL=http://localhost:6333 uv run pytest tests/unit/test_qdrant_store.py -q
uv run python evaluation/validate_retrieval_fixture.py evaluation/fixtures/retrieval_shared_benchmark_v1.json
```

Run all relevant commands before updating a task result to PASS. If a command fails, record the failure honestly in `tasks/todo.md`.

## Step 8 - Run the API locally

Start the FastAPI app:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, run smoke checks:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
curl http://127.0.0.1:8000/metrics
```

`/health` may report `degraded` when Ollama or LangSmith is unavailable. That is acceptable for static development checks, but not for live analysis validation.

## Step 9 - Optional live ingestion smoke test

This requires network access to SEC EDGAR, a valid `SEC_USER_AGENT`, a running vector backend, and an embedding model.

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL"}'

curl http://127.0.0.1:8000/ingest/AAPL
```

Check `/stats` after ingestion:

```bash
curl http://127.0.0.1:8000/stats
```

## Step 10 - Optional live analysis smoke test

This requires Ollama, market data access, and retrieval data if filing analysis is enabled. Tavily is required only when `include_news_sentiment` is true.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","include_filing_analysis":true,"include_news_sentiment":false,"max_news_articles":1}'
```

For long-running checks, use the async API:

```bash
curl -X POST http://127.0.0.1:8000/analyze/async \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","include_filing_analysis":true,"include_news_sentiment":false,"max_news_articles":1}'
```

Poll the returned job:

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
```

## Step 11 - Cleanup

Stop Qdrant when done:

```bash
docker compose -f docker-compose.qdrant.yml down
```

Local generated data usually lives under `data/` and `.runtime/`. Remove it only when you intentionally want a clean local state and after checking that no needed fixture or task artifact is stored there.

## Troubleshooting

- `uv` cannot write to its cache: rerun with `uv --cache-dir /tmp/uv-cache ...`.
- `/health` is degraded: check Ollama, vector store, and optional LangSmith configuration.
- Qdrant tests fail to connect: confirm `docker ps`, `QDRANT_URL`, and port `6333`.
- Ingestion fails: confirm network access, SEC user agent, and embedding model availability.
- News search fails: set `TAVILY_API_KEY` or run analysis with `include_news_sentiment=false`.
- Behavior tests are blocked by hygiene checks: rewrite tests to assert public behavior instead of private implementation shape.
