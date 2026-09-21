# Task 1 validation report

Implemented focused REST contracts and access controls; preserved the interrupted implementer's changes. Trace: task-1-brief, Global Constraints, SPEC §§3/5/8/11/12; test-plan §§1/8/11/12; authorized sprint-plan exception.

## Changes
- API-key auth (Bearer or X-API-Key), fail closed when unconfigured; public root/health, protected remaining data routes and metrics. OpenAPI security schemes and typed safe error schemas.
- 202 async acceptance and Location; safe common 401/404/405/409/422/429/500/502/503 envelopes with appropriate challenge/Allow/retry headers; no-store financial/job/error responses.
- Validated normalized tickers, UUIDs, names, extra fields and 10-K support; force_refresh passes through to replacement ingestion.
- Terminal status precedence: fatal/no memo, missing required stock or enabled filing/news/sentiment evidence, absent/failed verification, verified completion. Poll and result agree; graph error messages sanitized.
- Per-key 10/60s sliding-window limiter, stale-key cleanup, and atomic file-store async idempotency using hashed principal/key plus normalized-body fingerprint. Replay retains the original result; changed requests conflict.
- Health separately checks Ollama embeddings and local chat availability; Bedrock model/region checks explicitly identify configuration-only, never inference success. Retrieval outage returns 503.
- Removed obsolete MEMO_TEMPLATE import and call-order spy tests; current public model construction and HTTP model availability tests now run.
- README documents auth setup, metrics scrape credentials_file, retry/CORS semantics and single-process limits; SPEC/test-plan/sprint-plan updated. Original CRLF conventions preserved.

## Verification
Commands run from /tmp/alpha-rest-tracing, prefixed `rtk proxy env PYTHONPATH=/tmp/alpha-rest-tracing LANGCHAIN_TRACING_V2=false /media/cosmic-muffin/wd-black/git/Gen-AI_Prj/Financial-Analyst-system/.venv/bin/python`:

- Baseline inherited focused suite: 51 passed (controller's pre-task baseline: 43).
- New behavior tests before fixes: 4 failed, 23 passed, reproducing absent-verification completion, missing stock evidence, router envelope mismatch, and misleading health metadata.
- `-m pytest tests/test_api_integration.py tests/unit/test_api_cors.py tests/unit/test_config.py tests/unit/test_run_store.py tests/unit/test_api_schema_defaults.py tests/unit/test_llm.py -q`: 74 passed.
- `-m ruff check app/main.py app/models.py app/config.py app/services/run_store.py app/services/llm.py tests/test_api_integration.py tests/unit/test_api_cors.py tests/unit/test_config.py tests/unit/test_run_store.py tests/unit/test_llm.py`: PASS.
- `-m mypy app/main.py app/models.py app/config.py app/services/run_store.py app/services/llm.py`: PASS (5 source files).
- Four `python3 scripts/ci/check_{no_scope_residue,sprint_map,doc_sync,test_hygiene}.py`: PASS.
- `git -c core.whitespace=cr-at-eol diff --check`: PASS. Plain diff --check flags retained original CRLF bytes as whitespace; no source newline migration performed.

## Limits and handoff
No .env read/load and no live provider request. Task2 owns LangSmith/request correlation; controller owns final independent audit, broad suite and approved live request. Unrelated embedding model override defect intentionally unchanged. Bedrock health does not probe credentials, connectivity or inference permission. BackgroundTasks, file run store and limiter remain one-process only; no distributed guarantee or retention policy added. Controller owns tasks/todo.md and tasks/rest-tracing-plan.md; these are excluded from the task commit.
