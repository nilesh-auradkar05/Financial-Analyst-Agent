# REST/tracing resume implementation report

Implementation commit: `969667c`
Branch: `codex/rest-tracing-resume`

## Changes

- Added a type-preserving `app_traceable` wrapper using LangSmith's native `exceptions_to_handle` so application exceptions retain their original type while exported spans contain only `app_stage_failed`.
- Applied that wrapper to application-managed agent, sentiment, tool, and ingestion spans. Native provider model spans remain unchanged.
- Converted verifier execution exceptions into the documented degraded workflow result with `passed=false` and the safe `verification_failed` marker before native LangGraph callbacks can serialize raw exception text.
- Renamed static root metadata from `tools` to `available_tools`; invoked tools remain represented by child spans.
- Added behavior-first offline collector tests for nested application failures and verifier failures, including full-run scans for secret exception text.
- Added `scripts/__init__.py` so mypy assigns the smoke client one module identity.

## Verification

- Focused REST/tracing/workflow suite with `--run-integration`: **85 passed**.
- Leak and timeout regression selection: **3 passed**.
- Focused Ruff on changed Python files: **PASS**.
- Governance checks (`check_no_scope_residue`, `check_sprint_map`, `check_doc_sync`, `check_test_hygiene`): **PASS**.
- Full mypy: **5 existing errors in 3 files** (`app/services/tools/web_search_tool.py`, `evaluation/quality_baseline.py` ×3, `evaluation/latency_baseline.py`). The smoke-client duplicate-module error is gone and changed tracing modules add no errors.
- Full pytest command (all values shown; no secret values were inspected):

```bash
env -u LANGCHAIN_HANDLER -u LANGCHAIN_TRACING -u LANGSMITH_TRACING \
  PYTHON_DOTENV_DISABLED=1 \
  PYTHONPATH=/media/cosmic-muffin/wd-black/git/Gen-AI_Prj/Financial-Analyst-system/.worktrees/rest-tracing \
  LANGCHAIN_HANDLER=false LANGCHAIN_TRACING=false LANGSMITH_TRACING=false \
  LANGCHAIN_TRACING_V2=false \
  /media/cosmic-muffin/wd-black/git/Gen-AI_Prj/Financial-Analyst-system/.venv/bin/python -m pytest -q
```

Result: **278 passed, 1 failed, 2 skipped**.
- The remaining known existing failure is `tests/unit/test_embeddings_client.py::TestGetEmbeddings::test_custom_model`, which ignores the requested model override.
- Earlier broad runs had seven additional legacy tracing failures. They are invalid as final evidence because `evaluation/judge_models_interface.py` calls `load_dotenv()` at import and the nested worktree can auto-discover the parent repository environment file. With `PYTHON_DOTENV_DISABLED=1`, all seven pass in the full suite.
- No environment-file values were inspected or printed. Because the earlier commands did not disable dotenv, this report does not claim that those processes avoided automatic loading.

## Limits

- No intentional live validation or external-service execution was performed in this resume; earlier dotenv-affected runs were not globally network-denied.
- No environment file was directly read or printed; final verification disabled python-dotenv before collection.
- Live trace export remains controller-owned because required credentials were absent.

## Reviewer follow-up fixes

- Added one native child retriever span per filing-store query. Each span exports only the query, ticker, requested result count, returned count, and evidence identifiers; store/client objects and chunk text remain excluded.
- Marked caught stock, news, SEC CIK/list/download, and ingestion failures with stable safe failure codes while preserving their public fallback values. Legitimate empty news evidence remains a successful span.
- Replaced raw SEC download and ingestion exception text in returned error fields with stable public messages.
- Updated T2-02 and T2-07 to state the query-level lineage and caught-failure behavior.
- Deferred the reviewer’s minor HTTP-duration finalization observation to a separate change because it is outside the critical/important failure and lineage fixes.

### Follow-up verification

All Python commands set `PYTHON_DOTENV_DISABLED=1`, `LANGCHAIN_HANDLER=false`, `LANGCHAIN_TRACING=false`, `LANGSMITH_TRACING=false`, `LANGCHAIN_TRACING_V2=false`, and `PYTHONPATH` to this durable worktree.

- New compiled-graph retrieval and caught-failure collector probes: **2 passed**.
- Focused tracing, news/stock evidence, graph-news, and API integration regression suite: **69 passed**.
- Focused Ruff across all changed Python files: **PASS**.
- Focused mypy across all changed Python files: the only error is the pre-existing `app/services/tools/web_search_tool.py:159 [no-any-return]`; no new type errors were reported.
- Governance checks (`check_no_scope_residue`, `check_sprint_map`, `check_doc_sync`, `check_test_hygiene`): **PASS**.
- No full-suite rerun was performed for this bounded follow-up; the preceding clean-environment full-suite result remains recorded above.

## Controller final verification — 2026-09-23

User capped corrections at two and restricted new agents to unavailable gpt-6-sol(high); no new agents were spawned, and no additional runtime fixes were made. Controller reviewed the pending second correction locally.

- Final default full suite: **279 passed, 1 failed, 2 skipped** in27.66s; remaining embedding override failure.
- Explicit offline integration suite: **1 passed, 1 failed** in13.29s; stale fake_check_ollama_health rejects model keyword. This supersedes historical broad-pass claims for that integration and remains unresolved at the correction cap.
- All nine OpenAPI route contracts checked: **PASS** (inventory, success schemas/media, authorization schemes,405,async202+Location).
- Live execution/export still pending explicit runtime credential-loading approval.
