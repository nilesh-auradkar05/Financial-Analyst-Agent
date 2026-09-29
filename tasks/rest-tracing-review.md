# REST and request tracing verification

Branch: `codex/rest-tracing-resume`, based on `287e092`. Implementation commits: `7199ff1`, `a3c487c`, `a7e558a`, `7ba7216`, `969667c`. This report covers the explicitly authorized REST/tracing slice; it does not close broader S7 infrastructure work.

## Route inventory

| Method | Path | Success contract | Authorization |
| --- | --- | --- | --- |
| GET | `/` | 200, typed JSON information | Public |
| GET | `/health` | 200 healthy / 503 unavailable, typed JSON | Public |
| GET | `/metrics` | 200, Prometheus text | API key |
| GET | `/stats` | 200, typed JSON statistics | API key |
| POST | `/analyze` | 200, typed JSON analysis with terminal outcome | API key |
| POST | `/analyze/async` | 202, typed accepted job and Location | API key |
| GET | `/jobs/{job_id}` | 200, typed stored job and original correlation | API key |
| POST | `/ingest` | 200, typed ingestion result | API key |
| GET | `/ingest/{ticker}` | 200, typed ingestion status | API key |

OpenAPI includes success schemas/media types and typed error contracts. Validation, authorization, method errors and submission limits use safe envelopes and relevant HTTP headers. Submission idempotency applies to the async route. Current URLs are retained for compatibility; several remain action-oriented rather than resource-noun URLs. FastAPI documentation/OpenAPI endpoints are separate from these nine business routes.

## Trace contract

The expected native hierarchy is HTTP → background job (async only) → financial analyst → compiled graph → executed nodes/tools/retrieval/model/verification. Stored job correlation survives polling and idempotency replay; each polling HTTP request has its own request ID. The tool catalog is labeled `available_tools`; executed tools are child spans. Native model spans carry prompts, generations, token metadata and provider-visible reasoning when supplied. Hidden reasoning is not claimed.

Application exceptions have safe failure markers. Verification execution failure yields explicit failed verification and a degraded outcome. Default-off tracing, client reuse, exporter fault isolation and bounded shutdown are covered by offline tests.

## Evidence and limits

Independent task review found three blocking issues; its fixes received final local review under the user's two-attempt cap. No fresh subagent was spawned because the permitted gpt-6-sol model is unavailable. A real exported request has **not** been executed or verified. Runtime loading of the existing `.env` requires explicit approval under AGENTS §5.

The service remains one process with BackgroundTasks, a file job store and an in-process limiter; it is not a distributed microservice deployment. Retrieval evaluation and memo-grounding evaluation are separate instruments. The sequential latency runner exists at `evaluation/latency_baseline.py`; no `latency_baseline_*.json` was found in the original checkout's `evaluation/results` during this verification. Saved memo-quality artifacts are not proof of a latency baseline.

## Decisions retained from recovery

- Restore saved commits into a durable isolated worktree; preserve original main and its user edits. Cost if wrong: branch reconciliation before merge.
- Requested `gpt-6-sol` was unavailable; use the available `gpt-5.6-sol` at high effort, disclosed to the user. Cost: the requested model itself was not used.
- Disable automatic dotenv loading for offline tests: an imported evaluation module calls `load_dotenv()` during collection, and a nested worktree can discover the parent checkout's file. Earlier runs cannot support a claim that no implicit file loading occurred; no credential values were displayed. Future offline commands use `PYTHON_DOTENV_DISABLED=1` and explicit disabled tracing switches.

## Fresh verification at 969667c

- Focused REST/tracing/workflow suite: **85 passed**; nested failure/timeout regressions: **3 passed**.
- Full pytest: **278 passed, 1 failed, 2 skipped**. Remaining failure is the existing `TestGetEmbeddings.test_custom_model` override defect.
- Focused Ruff and four governance checks: **PASS**.
- Full mypy: **5 existing errors**, in `web_search_tool.py`, `quality_baseline.py` (three), and `latency_baseline.py`. Smoke-script module collision resolved.
- Whole-repository Ruff previously had four unrelated findings; focused success does not mean all repository lint passes.

Final full command used the original `.venv/bin/python -m pytest -q` from this worktree, `PYTHONPATH` pointing here, `PYTHON_DOTENV_DISABLED=1`, and `LANGCHAIN_HANDLER`, `LANGCHAIN_TRACING`, `LANGSMITH_TRACING`, `LANGCHAIN_TRACING_V2` all set to `false`. Exact command/evidence is committed in `.superpowers/sdd/rest-tracing-plan/resume-implementation-report.md` (`ade4924`).

## Final review — 2026-09-23

The controller inspected the pending second correction, all route handlers and schemas, auth/errors/cache middleware, limiter, async idempotency/store behavior, and tracing boundaries. The requested model-only constraint supersedes the skill's separate final reviewer dispatch; this final review was local, not a new independent-agent review.

- C1 addressed for reviewed production paths: SEC and ingestion exception results now contain stable safe messages; collector tests reject private exception markers in all captured spans.
- I1 addressed: caught provider exceptions mark actual stock/news/SEC/ingestion spans failed with safe codes. An actual empty-success news response remains distinguishable from failure.
- I2 addressed: each executed filing query has its own child retriever span with query/ticker/count/evidence IDs; store objects and chunk text are excluded from these child spans.
- M1 deferred: HTTP duration excludes the final body-send delay because span finalization precedes `await send(message)`.
- Per-query backend metadata is not separately added. Existing query spans establish invocation lineage; richer retrieval diagnostics remain outside this bounded correction.

Second-correction evidence:2newcollectorprobespassed;69focusedregressionspassed;focusedRuffpassed;focusedmypyonlytheexistingweb-searchreturn-anyfinding;fourgovernancecheckspassed. Final controller default full suite: **279 passed, 1 failed, 2 skipped**, in27.66s. Only failure: `TestGetEmbeddings.test_custom_model`, expected nomic-embed-text but factory selects qwen3-embedding:4b. No additional runtime changes were made after this final run.

The final offline OpenAPI check imported the application with dotenv and all tracing switches disabled, asserted the exact nine-route inventory, verified every success schema/media type, public-vs-protected routes, both alternative authorization schemes, 405 documentation, and async202/Location. It passed. This establishes reviewed REST contracts; action-oriented URLs and single-process deployment limits remain explicit qualifications.

No live request was executed. The ready live checker will submit one async AAPL analysis and record IDs, actual span hierarchy and LangSmith URL after explicit approval to load existing credentials. Do not treat offline collectors as proof of remote delivery.

Final explicit offline integration command: `python -m pytest tests/integration --run-integration -q` with the same dotenv-disabled environment -> **1 passed, 1 failed** in13.29s. Remaining failure is `test_runtime_hardened_pipeline_end_to_end`: its `fake_check_ollama_health` rejects the new `model` keyword at startup. This stale integration fixture is unresolved; no third correction attempt was made. The other real-graph verification integration passed. The branch is not fully green or claimed production-ready.

Final Doc Sync Check2026-09-23: scope-residue, sprint-map, doc-sync and test-hygiene all **PASS**; CRLF-aware git diff whitespace check **PASS**. No scope, benchmark methodology, architecture or sprint-map change in the bounded second correction.
