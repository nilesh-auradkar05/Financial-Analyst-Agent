# Financial Analyst Agent System

AI-powered financial analyst agent that researches companies, analyzes SEC filings, evaluates market sentiment, retrieves filing evidence, and generates investment memos with citations.

Built with **LangGraph**, **FastAPI**, **RAG**, **Pydantic**, local-first LLM tooling, and a retrieval evaluation workflow designed for measurable grounding improvements.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Project Status

**Stage: instrumented prototype → production-readiness sprint.** A production-readiness review (2026-08-24) is now canonical in `docs/SPEC.md` v1.3 (application record `docs/adr/ADR-0008-production-readiness-order.md`) and `docs/sprint-plan.md`. The eval/grounding harness is the strongest asset and the service shell the weakest. The current work is the ordered eight-step plan below; nothing later in the list opens before the previous step has commit evidence.

What exists and is measured:

- LangGraph single-agent workflow with citation registry built before the LLM call
- SEC 10-K ingestion (edgartools), section-aware chunk metadata, `RetrievalStore` abstraction (Chroma default, Qdrant behind the interface)
- FastAPI sync/async endpoints, file-backed run store, Prometheus metrics, LangSmith tracing
- Heuristic memo verifier (claim extraction, tolerance number match, cosine ≥ 0.45) — latest live-evidence baseline `grounded_claim_rate 0.904 ± 0.062`, `citation_coverage 0.925 ± 0.049`, classification **candidate** (live evidence cannot reach `approved`)
- Retrieval benchmark fixtures, paired case-level comparator, quality/latency baseline runners

Known limitations at HEAD (verified against code, not aspiration):

- A ticker that was never ingested produces a memo with no SEC evidence and still reports `completed`
- Verification failure is logged, not enforced; there is no repair loop
- Evidence nodes run serially; FinBERT inference blocks the event loop
- No guardrails, no caching, no auth, no rate limiting, in-process background jobs
- Eval results carry no lineage (commit, model, snapshot) and no eval runs in CI

### Implementation order (authoritative copy: `docs/sprint-plan.md`, S2 preamble)

| Step | Task | Exit evidence |
| --- | --- | --- |
| 1 | S2-T00a — governance docs into `docs/`, CI governance job, hooks committed | `git ls-files docs/` non-empty; governance job green; deliberate break fails CI |
| 2 | S2-T00c — fan-out of independent nodes, `to_thread` for FinBERT/store, graph singleton, `errors` reducer | latency baseline before/after (same model/temp); 4 concurrent requests < 1.5× single |
| 3 | S2-T00b/T00d — `EvidenceSnapshot` freeze/replay with zero-network test; `degraded` / `evidence_missing` statuses | replay green under `unshare -n`; `grep "from evaluation" app/` empty |
| 4 | S6 — eval registry with lineage; `eval-replay` CI regression gate; verifier↔judge κ | a PR that regresses grounding fails CI |
| 5 | S6 — bounded draft→verify→revise loop (max 2) | paired comparison vs no-loop on the frozen snapshot |
| 6 | S7 — guardrails (input, untrusted content, output policy), API auth + rate limit | adversarial fixture in CI |
| 7 | S7 — evidence / query-embedding / memo caching keyed on `snapshot_hash` | cache hit-rate in `/metrics` |
| 8 | S7 → S10 — queue + worker, Postgres job store, FinBERT out-of-process, circuit breaker; cloud gated on ADR-0006 | `JobQueue`/`RunStore` protocols swapped without app changes |

## What This System Does

Given a ticker, the system can:

1. fetch market context,
2. collect recent company news,
3. ingest and retrieve SEC filing sections,
4. build structured evidence packets,
5. run sentiment and structured analysis,
6. generate an investment memo with citations,
7. verify memo grounding and citation coverage,
8. expose the workflow through a FastAPI service.

## Features

### Multi-Source Research

- **News search** through Tavily
- **Stock data** through YFinance
- **SEC filings** through EDGAR APIs
- **Filing-section retrieval** for business, risk factors, MD&A, and market-risk sections

### Evidence-Centered RAG

- backend-agnostic retrieval contract
- metadata-rich filing chunks
- section-aware retrieval filters
- normalized `EvidencePacket` objects
- citation-friendly memo generation
- grounding and citation verification

### Evaluation and Observability

- retrieval fixtures and benchmark result files
- paired retrieval comparison tooling
- precision, recall, MRR, NDCG, section-recall, first-rank, and latency-oriented metrics
- health, stats, and metrics endpoints
- LangSmith / RAGAS / DeepEval-oriented evaluation direction

### Service Layer

- FastAPI sync and async analysis endpoints
- file-backed async run state
- graceful handling of partial tool failures
- environment-driven configuration
- Docker/local-stack support

## Architecture

![System Architecture](assets/images/architecture.jpg)

### Agent Workflow

![Agent Workflow](assets/images/agent-workflow.jpg)

### Architecture at a Glance

```text
Ticker Request
  -> Validate request and runtime configuration
  -> Fetch market data and recent news
  -> Retrieve SEC filing evidence
  -> Build structured evidence packets
  -> Run sentiment and structured analysis
  -> Draft memo with citations
  -> Verify grounding and citation coverage
  -> Return memo, evidence, citations, and verification payload
```

## Repository Layout

```text
Financial-Analyst-Agent/
├── agents/
├── api/
│   ├── main.py
│   ├── run_store.py
│   └── schemas.py
├── configs/
├── evaluation/
│   ├── fixtures/
│   ├── results/
│   └── compare_retrieval_results.py
├── models/
├── monitoring/
├── observability/
├── rag/
│   ├── embeddings.py
│   ├── evidence.py
│   ├── ingestion.py
│   └── vector_store.py
├── scripts/
├── tests/
├── tools/
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── docker-compose.qdrant.yml
└── pyproject.toml
```

## Requirements

- Python 3.12+
- Ollama running locally
- Tavily API key for web/news search
- SEC-compliant user agent string
- local write access for vector-store persistence and file-backed run storage
- Qdrant local stack when running Qdrant-backed retrieval experiments

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/nilesh-auradkar05/Financial-Analyst-Agent.git
cd Financial-Analyst-Agent
```

### 2. Install dependencies

```bash
uv install
```

Optional full local setup:

```bash
make install
```

### 3. Pull local models

```bash
ollama pull qwen3-vl:8b
ollama pull qwen3-embedding:4b
```

### 4. Configure environment

Create a `.env` file with at least:

```bash
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3-vl:8b
OLLAMA_EMBED_MODEL=qwen3-embedding:4b
SEC_USER_AGENT="your-name your-email@example.com"
CHROMA_PERSIST_DIR=./data/chroma
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
```

Use `VECTOR_BACKEND=chroma` when comparing against the Chroma baseline.

## Running the API

### Development server

```bash
make serve
```

Equivalent direct command:

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production-style local run

```bash
make serve-prod
```

## Docker and Local Stack

Bring up the default stack:

```bash
make docker-up
```

Bring up Qdrant locally when running Qdrant experiments:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

Stop services:

```bash
make docker-down
```

View logs:

```bash
make docker-logs
```

## API Reference

### Base URL

```text
http://localhost:8000
```

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Basic API info |
| `GET` | `/health` | Component health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/stats` | Vector store and run-store stats |
| `POST` | `/analyze` | Run synchronous analysis |
| `POST` | `/analyze/async` | Start async analysis job |
| `GET` | `/jobs/{job_id}` | Fetch async job status/result |
| `POST` | `/ingest` | Ingest SEC filing data |
| `GET` | `/ingest/{ticker}` | Check whether a ticker is indexed |
| `GET` | `/docs` | Swagger UI |

## Example Usage

### Ingest a ticker

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

### Run synchronous analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

### Run async analysis

```bash
curl -X POST http://localhost:8000/analyze/async \
  -H "Content-Type: application/json" \
  -d '{"ticker": "MSFT"}'
```

Then poll:

```bash
curl http://localhost:8000/jobs/<job_id>
```

### Use optional analysis controls

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "include_filing_analysis": true,
    "include_news_sentiment": false,
    "max_news_articles": 5
  }'
```

## Retrieval and Ingestion Notes

The retrieval layer is designed around a backend-facing abstraction in `rag/vector_store.py`, not direct ad hoc calls into one database.

Supported retrieval concepts include:

- metadata-rich `IndexDocument` objects
- `SearchFilters` for ticker, filing type, section key/name, and filing date
- section-focused retrieval helpers
- vector-store stats and document counting
- `EvidencePacket` as the atomic retrieval unit for downstream grounding

The ingestion path tracks:

- filing date
- total chunks
- requested sections
- sections found
- sections skipped
- documents written

## Testing

Run focused API and run-store tests:

```bash
uv run pytest tests/test_run_store.py tests/test_api_integration.py
```

Run the full test suite:

```bash
make test
```

Equivalent direct command:

```bash
uv run pytest tests/ -v
```

Run unit tests only:

```bash
uv run pytest tests/unit -v
```

Run integration tests that need external services/API keys:

```bash
uv run pytest tests/integration -v --run-integration
```

## Retrieval Evaluation

The next sprint task is to run every retrieval method against the same shared benchmark fixture:

```bash
uv run python evaluation/retrieval_main.py \
  --fixture evaluation/fixtures/retrieval_shared_benchmark_v1.json \
  --mode section_aware \
  --output evaluation/results/qdrant_section_aware_shared_v1.json
```

Repeat the run for each retrieval mode, then compare paired results:

```bash
uv run python evaluation/compare_retrieval_results.py \
  evaluation/results/qdrant_section_aware_shared_v1.json \
  evaluation/results/qdrant_reranked_hybrid_shared_v1.json \
  --baseline-mode section_aware \
  --candidate-mode reranked_hybrid \
  --candidate-method dense_bm25_cross_encoder_rerank \
  --strict-case-ids
```

Result files should identify their source fixture and retrieval method. Anything less is how fake benchmarks are born, and they grow up to become slide-deck lies.

Expected result metadata:

```json
{
  "fixture_file": "evaluation/fixtures/retrieval_shared_benchmark_v1.json",
  "mode": "section_aware",
  "retrieval_method": "qdrant_section_aware"
}
```

### Retrieval Decision Rules

- Keep the section-aware baseline if hybrid/reranked methods reduce recall or section coverage.
- Do not adopt a method just because precision@5 improves while recall@5 collapses.
- Treat latency, first relevant rank, section recall, and pass@k as first-class metrics.
- Compare only shared `case_id` values with paired deltas.
- Separate retrieval-method improvements from answer-generation prompt improvements.

## Current Sprint Checklist

- [x] Shared retrieval fixture, paired comparator, measured baseline selected
- [x] Grounding instrument fixed at root cause (tolerance number match, heading-aware claim extraction) and unit-tested
- [x] Production-readiness review applied: SPEC v1.3 canonical; ADR-0008 records the application
- [ ] S2-T00a — docs tracked, CI governance job, hooks committed
- [ ] S2-T00c — fan-out and event-loop hygiene
- [ ] S2-T00b — evidence snapshot freeze/replay, zero-network assertion
- [ ] S2-T00d — verification/evidence-completeness status semantics
- [ ] S2 execution — anchored fixture v3 → Gate A

## Roadmap Direction

Steps 1–3 above, then S2 (Gate A), then S6 eval hardening + repair loop, then S7 service readiness. Multi-agent decomposition, frontend, and cloud remain deferred; each is gated on a documented exit criterion, not on enthusiasm.

Explicitly **not** planned: semantic caching of memo outputs (unsafe for time-sensitive financial content — caching is keyed on the evidence snapshot hash instead) and Kafka (no second consumer type exists; Redis Streams behind a `JobQueue` protocol until one does).

## Agent-Tooling Hooks

Governance rules that can be checked mechanically are enforced at the coding-agent boundary by `.claude/settings.json` and the scripts in `.claude/hooks/` (Claude Code; the same scripts register for Codex CLI's six-event subset). Blocking hooks exist only on `PreToolUse`, `UserPromptSubmit`, and `Stop`.

| Id | Event | Rule |
| --- | --- | --- |
| H1 | SessionStart | inject `tasks/lessons.md`, recent commits, active sprint task, tree status |
| H2 | UserPromptSubmit | implementation prompts require an unchecked plan item in `tasks/todo.md` |
| H3 | PreToolUse Edit/Write | governance docs read-only unless `ALLOW_SPEC_EDIT=1` |
| H4 | PreToolUse Edit/Write | frozen fixtures and datasets immutable |
| H5 | PreToolUse Edit/Write | eval result files must carry lineage keys |
| H6 | PreToolUse Bash | benchmark runs refused when uncommitted changes span more than one axis |
| H7 | PreToolUse Bash | replay test commands rewritten to run without network |
| H8 | PreToolUse Bash/Read | secrets files and credential patterns blocked |
| H9 | PreToolUse Bash | force-push, hard reset, destructive `rm`, collection deletion blocked |
| H10 | PostToolUse Edit/Write | `ruff` + `mypy` on the written file; failures fed back |
| H11 | PostToolUse Bash | after a user correction followed by a failure, stub appended to `tasks/lessons.md` |
| H12 | Stop | turn cannot end with a dirty tree, failing unit tests, or failing doc-sync |

Smoke-test any hook with `echo '<event json>' | .claude/hooks/<script>`; see `docs/test-plan.md §15` for the cases.

## Recommended Repo Status Statement

> Financial Analyst Agent is a single-agent, evidence-grounded financial analysis system (LangGraph, FastAPI, SEC/news/market tools, citation registry, heuristic verifier, backend-abstracted retrieval) with a measured grounding baseline of 0.90 ± 0.06 on live evidence. It is a prototype, not a production service: a 2026-08 review recorded the gaps (silent evidence omission, unenforced verification, serial nodes, no guardrails/caching/queue) and an eight-step, evidence-gated order for closing them, starting with governance-in-CI and evidence-snapshot replay.

## License

MIT License.
