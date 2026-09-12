# tasks/TASKS.md — Project Task Register (status snapshot)

> **Snapshot date:** 2026-09-12 · **Commit reviewed:** `ebfd13b` (clean tree) · **Reviewer:** Claude Code (current-state review)
> **Authority:** Derived, non-canonical view. It ranks *below* `tasks/todo.md` (SPEC §0 #7). Task definitions stay in `docs/sprint-plan.md`, behaviour oracle in `docs/test-plan.md`, and scope in SPEC §3. If this file disagrees with those docs, those docs win, and this file needs refreshing.
> **Sources read:** `docs/SPEC.md` v1.3, `docs/sprint-plan.md`, `docs/test-plan.md`, `docs/retrieval-benchmark.md`, `tasks/todo.md`, `tasks/sprint-review.md`, `tasks/test-suite-audit.md`, `.github/workflows/ci.yml`, `.claude/hooks/*`, and the code under `app/`, `dataops/`, `evaluation/`, `tests/`.

## Status legend

| Status | Meaning |
| --- | --- |
| `DONE` | Implemented and backed by recorded verification, still true at `ebfd13b`. |
| `DONE-DOC` | Done at documentation level only. No code is expected yet. |
| `DONE*` | Was done, but something has drifted since and needs an update (see §4). |
| `PARTIAL` | Some of the task's scope exists. The remaining scope is listed. |
| `AHEAD` | Code exists but its sprint/gate is not open, so it is unverified against the plan's criteria. |
| `TODO` | Not started. |
| `GATED` | Horizon work that must not start until its predecessor gate clears. |

---

## 1. Verification snapshot (run in this review, 2026-09-12)

| Check | Command | Result |
| --- | --- | --- |
| Scope residue | `uv run python scripts/ci/check_no_scope_residue.py` | PASS |
| Sprint map | `uv run python scripts/ci/check_sprint_map.py` | PASS |
| Doc sync | `uv run python scripts/ci/check_doc_sync.py` | PASS |
| Test hygiene | `uv run python scripts/ci/check_test_hygiene.py` | **FAIL**: `tests/unit/test_llm.py:25,32` call-order spies (pre-existing since ≤2026-06-27) |
| Lint | `uv run ruff check .` | **FAIL, 4 errors**: `app/llm/deepseek_thinking_check.py:12` W292, `app/llm/provider_check.py:11` W292, `.claude/hooks/stop_gate.py:23` E741, `.claude/skills/excalidraw-diagram/references/render_excalidraw.py:97` F541 |
| Types | `uv run python -m mypy .` (CI runs `mypy .`) | **FAIL, 6 errors in 4 files** (127 checked): `app/services/tools/web_search_tool.py:159`, `evaluation/quality_baseline.py:354,359,363`, `evaluation/latency_baseline.py:211`, `tests/unit/test_llm.py:11` |
| Tests | `uv run python -m pytest -q --continue-on-collection-errors` | **218 passed, 1 failed, 2 skipped, 1 collection error**. Failed: `test_embeddings_client.py::TestGetEmbeddings::test_custom_model` (explicit `model=` argument ignored; see UPD-02). Error: `tests/unit/test_llm.py` (`ImportError: MEMO_TEMPLATE`) |
| Precursor fixture | `uv run python evaluation/validate_retrieval_fixture.py …_v1.json` | PASS (72 cases, 18 per section) |
| v2 candidate fixture | same validator on `…_v2_candidate.json` | PASS |
| CI on `main` | `gh run list --branch main` | **Not verifiable** (`gh` not authenticated). CI runs `mypy .` and `pytest` without `--continue-on-collection-errors`, so it is **expected red** based on the local results above |
| Environment | `uv run pytest` / `uv run mypy` | **Broken**: the repo moved from `git/prj/` to `git/Gen-AI_Prj/`, and the `.venv/bin/*` shebangs still point to the old path. `python -m …` works. This also breaks hook H12 (`stop_gate.py` calls `uv run pytest`). See ENV-01 |

**Bottom line:** documentation governance is green. The code quality bar that S0-T04 closed green on (full pytest, ruff, and mypy all green) has **regressed**. Phase 0 and Phase 1 are done. S2 pre-tasks T00c and T00d have not started, and T00a and T00b are partial. The retrieval/Qdrant/hybrid code is well ahead of the sprint gates.

---

## 2. Sprint / phase summary

| Sprint / Phase | Purpose (SPEC §12) | Status | Notes |
| --- | --- | --- | --- |
| S0 | Spec coherence + baseline verification | `DONE*` | Closed green 2026-06-04. The quality bar has since regressed (UPD-01..03) |
| S1 | SEC ingestion identity + section coverage | `DONE` | `tests/ingestion` green in this run |
| Phase 0 (v2 plan) | Governance docs into repo + CI | `DONE` | Maps to S2-T00a, which is only partial (see below) |
| Phase 1 (v2 plan) | Evidence snapshot + replay + approved baseline | `DONE` | Maps to S2-T00b, which is only partial (see below) |
| S2 pre-tasks T00a–T00d | Production-readiness steps 1–3 | `PARTIAL` | T00a partial, T00b partial, T00c TODO, T00d TODO |
| S2 (Gate A) | Anchored retrieval benchmark oracle | `PARTIAL` | Methodology doc done. Precursor tooling only; no anchored fixture |
| S3 (Gate B prep) | Qdrant behind the interface | `AHEAD` | Store, indexes, and compose exist. Parity suite, `healthcheck()`, and `tests/rag` do not |
| S4 (Gate B) | Chroma vs Qdrant decision | `TODO` | Needs Gate A |
| S5 (Gate C) | Hybrid / rerank | `GATED` + `AHEAD` | Hybrid, reranked, and section-aware code already exists |
| S6 (Gate D) | Eval registry, regression gate, repair loop | `GATED` | Memo-grounding instrument (BENCH-FIX) and RAGAS/DeepEval harness exist |
| S7 | Guardrails, caching, queue, auth | `GATED` | None implemented |
| S8 / S9 / S10 | Frontend / polish / cloud | `GATED` | ADR-0006 is still Proposed |

---

## 3. Task register

Columns: **ID** · **Task / context** · **Trace** · **Status** · **Evidence** · **Remaining / next action** · **Depends on**

### S0 — Spec coherence & baseline verification

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| S0-A00 | Read-only current-state audit; opened gaps G1–G8 | sprint-plan ledger | `DONE` | `tasks/sprint-review.md §1–5`; todo audit log 2026-06-04 | — | — |
| S0-T01 | Doc topology, dedup, residue removal, doc-check scripts | SPEC §0/§3/§13 | `DONE` | 3 doc scripts PASS (this run) | — | — |
| S0-T02 | `retrieval-benchmark.md` comparison policy | SPEC §10; test-plan §6 | `DONE-DOC` | Doc §8 present; SPEC §10 points to it | Implementation is S2 (G6) | — |
| S0-T03 | Sprint map S0–S10 consistent | SPEC §12 | `DONE` | `check_sprint_map.py` PASS (this run) | — | — |
| S0-T04 | Baseline verification; C0–C3 classified | SPEC §1.1 | `DONE*` | 2026-06-04: pytest 178 pass, ruff clean, mypy success. **This run:** 1 failed + 1 collection error, ruff 4, mypy 6 | Restore the green bar: UPD-01, UPD-02, UPD-03 | — |
| GOV-TH | `check_test_hygiene.py` call-order guard | test-plan Principles; CLAUDE §5 | `PARTIAL` | Script exists, FAILS on `test_llm.py:25,32` | Fix via UPD-01; then wire into CI (S2-T00a) | UPD-01 |

### S1 — SEC ingestion & section-coverage lock

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| S1-T01 | ADR-0004 edgartools parser decision | SPEC §9/§16 | `DONE` | `docs/adr/ADR-0004-edgartools-parser.md` Accepted | — | — |
| S1-T02 | Canonical metadata + deterministic `chunk_id = f(accession, section_key, chunk_index)` | SPEC §7/§9; test-plan §3 | `DONE` | `tests/ingestion/test_chunk_metadata.py`, `test_ingestion_idempotency.py` pass (this run) | Embedding identity is not yet recorded in ingest metadata (test-plan §8), which is S3-T05 | — |
| S1-T03 | AAPL/MSFT/NVDA × 4 critical sections, coverage through the store interface | SPEC §9; test-plan §3 | `DONE` | `tests/ingestion/test_section_coverage.py` pass (this run); C3/G5 closed | Qdrant run is S3-T05 | — |

### Phase 0 / Phase 1 (INTEGRATION_PLAN_v2), feeding S2-T00a / S2-T00b

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| PF1–PF3 | Pre-flight: docs tracked? clean baseline? fixes committed? | SPEC §0; v2 §0 | `DONE` | todo.md PF1–PF3 results | — | — |
| P0-T01 | Commit `docs/` + `tasks/` | SPEC §0 | `DONE` | `git ls-files docs/` non-empty at `ebfd13b` | — | — |
| P0-T02 | Wire doc checks into CI | SPEC §13/§14.4 | `DONE*` | `ci.yml` step "Run document governance checks" (3 scripts) | `check_test_hygiene.py` not wired (UPD-04) | — |
| P0-T03 | Prove CI goes red on doc drift | lessons 2026-06 | `DONE` | GH run `28713339887` red; revert `275db60` | — | — |
| ADR-0005 | Unified DataOps release registry | SPEC §16 | `DONE` | ADR Accepted | — | — |
| ADR-0006 | Cloud deployment | SPEC §3.y | `DONE-DOC` (Proposed) | ADR Proposed only | Owner decision on provider/budget before S10 | — |
| P1-T01 | `EvidenceSnapshot` / `DatasetReleaseManifest` contracts | SPEC §7 | `DONE` | `dataops/contracts.py`; `test_dataops_contracts.py` pass | Aggregate `snapshot_hash` (SPEC §7.x) not yet modelled (S2-T00b) | — |
| P1-T02 | Append-only release registry + active pointers | ADR-0005 | `DONE` | `dataops/registry.py`; `artifacts/dataops/releases.jsonl` | — | — |
| P1-T03 | `evidence_snapshot` quality gate | SPEC §7 | `DONE` | `dataops/gates.py`; `test_dataops_gates.py` pass | — | — |
| P1-T04 | `record_evidence` on the 4 tools, default path unchanged | SPEC §7 | `DONE` | `test_tool_evidence_recording.py` pass | — | — |
| P1-T05 | Register `alpha-evidence:0.1.0` (66 snapshots, AAPL/MSFT/NVDA) | SPEC §7 | `DONE` | `artifacts/dataops/active/evidence.yaml`; quality report | — | — |
| P1-T06 | `--evidence-release` replay in `quality_baseline.py` | SPEC §10 | `DONE` | `dataops/replay.py`; `test_evidence_replay.py`, `test_quality_baseline_evidence_replay.py` pass | Replay is offline-harness only; runtime `run_agent` has no replay path (S2-T00b) | — |
| P1-T07 | Frozen re-baseline; register approved `alpha-quality-baseline:0.1.0` | SPEC §11.1 | `DONE` | grounded 0.9346 ± 0.0447, coverage 0.9234 ± 0.0473; `artifacts/dataops/quality_baselines/…0.1.0.json` | Artifacts named `retrieval_baseline_*` are actually memo-grounding (UPD-09) | — |

### Tooling / bug-fix tasks already closed

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| BENCH-FIX | Sound memo-grounding instrument (number matcher, claim filter, eval version, filename helper) | test-plan §7/§8 | `DONE*` | Commit `6826034`; PF3 PASS; `test_grounding_eval.py` pass | todo.md still shows it **unchecked** (UPD-06). 4 latent edges tracked (banker's rounding, borderline threshold, unterminated bullets, artifact prefix) | — |
| GROUND-FIX | Header lines not scored as claims | test-plan §7 | `DONE` | Superseded by BENCH-FIX implementation | — | — |
| NEWS-FIX | Tavily `topic=news`, snippet cleaning, dedup | SPEC §3; test-plan §2 | `DONE*` | `test_web_search_tool.py` pass | mypy error `web_search_tool.py:159` (UPD-03); leading site-chrome residual | — |
| HOTFIX-LLM-TIMEOUT | Bounded `draft_memo` LLM wait | test-plan §2 | `DONE` | `test_graph_verification_integration.py` pass | — | — |
| HOTFIX-IMPORT-CYCLE | `inspect_grounding` import cycle | test-plan §7 | `DONE` | `test_inspect_grounding.py` pass | — | — |
| DOC-SETUP | Local + AWS test-env guides | SPEC §3 | `DONE` | `docs/setup-and-test.md`, `docs/aws-test-environment.md` | — | — |
| PROD-DESIGN | Production diagrams, interview doc, review, ADR-0007 (Proposed) | SPEC §3.y; ADR-0006/0007 | `DONE-DOC` | Commits `73c056d`, `ebfd13b` | todo.md has 2 unchecked items waiting on a governance-green run (blocked by GOV-TH) | UPD-01 |
| SPEC-v1.3 | Apply amendment, bump SPEC, ADR-0008 | SPEC §0 | `DONE` | Commit `6bfc9ce`; doc checks PASS | Stale "active sprint" markers remain in other docs (UPD-05) | — |

### S2 — pre-tasks (production-readiness steps 1–3)

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| S2-T00a | Governance docs in repo + CI governance gate (step 1) | SPEC §0/§13/§14; G11 | `PARTIAL` | Done: docs tracked, `.claude/settings.json` + hooks committed, 3 doc checks in CI, red-path proven | (1) Add `check_test_hygiene.py` to CI, as its own `governance` job. (2) Untrack `.runtime/run_store.json` and `test_image/`; add `test_image/` to `.gitignore` (`.runtime` is already ignored but still tracked). (3) `tests/unit/test_hooks.py` for H2–H9, H12 (test-plan §15) is missing. (4) Make CI green (UPD-01..03). (5) Record `Result:` in sprint-plan | UPD-01..03 |
| S2-T00c | Fan-out + event-loop hygiene (step 2) | SPEC §8.1–8.4; G12; test-plan §10 | `TODO` | Code at `ebfd13b`: graph is serial (`graph.py:689–712`); `add_error` mutates `state["errors"]` (`state.py:125`); `create_agent()` is called per request (`graph.py:759`); `analyze_sentiment_batch` runs synchronously inside an async node; retrieval pulls 4×3 chunks but the registry keeps 5 | Implement per sprint-plan; latency before/after on frozen release; test-plan §10 cases | S2-T00a |
| S2-T00b | EvidenceSnapshot + zero-network runtime replay (step 3a) | SPEC §7.x/§7.y/§11.1; test-plan §13 | `PARTIAL` | Phase 1 delivered snapshot contracts, registry, gate, and offline replay in `quality_baseline.py` | (1) Aggregate `snapshot_hash` + `evidence_as_of` in `AgentState` and `AnalysisResponse`. (2) `run_agent(…, evidence_release=…)`. (3) `replay` pytest marker plus a zero-network run (`unshare -n`, hook H7). (4) Location decision: SPEC §5 says `app/dataops/`, code is `dataops/`; sprint-plan says `evaluation/evidence_releases/`, code uses `artifacts/dataops/` (UPD-07) | S2-T00c (sequencing rule) |
| S2-T00d | `degraded` / `evidence_missing` statuses; move verifier to `app/verification/` (step 3b) | SPEC §8.5; G10; test-plan §11 | `TODO` | `app/agents/graph.py:49` imports `evaluation.grounding` (G10 open). `JobStatus` = pending/running/completed/failed only. `app/main.py:369–373` returns `completed` even with recoverable errors or failed verification | Implement per sprint-plan; test-plan §11 cases; `grep -rn "from evaluation" app/` must be empty | S2-T00b |

### S2 — Shared retrieval benchmark oracle → Gate A

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| S2-T01 | Label schema, matcher, metric formulas (K∈{5,10}, graded NDCG) | retrieval-benchmark §2–4 | `DONE-DOC` (unrecorded) | `docs/retrieval-benchmark.md` §2–4 fully specify schema, matcher, and graded NDCG | Record `Result:` in sprint-plan after a review pass | — |
| S2-T02 | Anchored fixture validator ("lie detector") | retrieval-benchmark §5–6; test-plan §6 | `PARTIAL` (precursor) | `evaluation/validate_retrieval_fixture.py` checks schema, sections, `MIN_CASES=50`, coverage. No `gold_evidence`, `anchor_text`, `relevance`, anchor-in-source check, or `--min-cases` flag. `evaluation/build_source_section_cache.py` and `evaluation/fixtures/source_sections/` are missing | Extend the validator; build the source-section cache from the frozen `alpha-evidence:0.1.0` SEC snapshots; add the 10 test-plan §6 validator cases | S2-T00d |
| S2-T03 | 50–100 anchored, graded cases | test-plan §6 | `TODO` | v1 (72) and v2_candidate are keyword/section precursors (G6) | Author on frozen release; new `fixture_version`; coverage report | S2-T02 |
| S2-T04 | Benchmark runner with provenance | retrieval-benchmark §7 | `PARTIAL` (precursor) | `evaluation/run_shared_retrieval_benchmark.py` + `retrieval_eval.py` emit latency, null `first_relevant_rank`, MRR/NDCG@5 with **binary** relevance | Graded gains; K=10; `fixture_version` + embedding + git provenance; cold/warm latency; determinism test | S2-T03 |
| S2-T05 | Hardened paired comparator | retrieval-benchmark §8; test-plan §6 | `PARTIAL` | `compare_retrieval_results.py`: `--strict-case-ids`, paired delta, W/T/L, bootstrap CI, null=miss, latency separate, mode/method checks; `test_compare_retrieval_results_paired.py` pass | Reject mismatched `fixture_version`; reject "both backend and method differ" (single-axis) | S2-T04 |
| S2-T06 | Gate A baseline (Chroma, section_aware) | SPEC §11 Gate A | `TODO` | — | Run + self-compare sanity; record in sprint-review | S2-T05 |

### S3 — Qdrant behind the interface → Gate B prep

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| S3-T01 | ADR-0003, Qdrant compose, `VECTOR_BACKEND` selection | SPEC §6/§16; test-plan §5 | `AHEAD` | ADR-0003 Accepted; `docker-compose.qdrant.yml` healthcheck; `_get_vector_backend()` raises on invalid value (default `qdrant`, ratified G4); `test_vector_store_factory.py` pass | `tests/rag/test_vector_backend_selection.py`; "Qdrant unavailable → clear degraded failure" test | Gate A |
| S3-T02 | `QdrantVectorStore` protocol parity | SPEC §6 | `AHEAD` | `app/components/retrieval/qdrant_store.py`; `test_qdrant_store.py` pass | `healthcheck()` absent on both stores (G3 remnant) | Gate A |
| S3-T03 | Payload indexes (5 fields) | test-plan §5 | `AHEAD` | `PAYLOAD_INDEX_FIELDS` includes `accession_number` (G7 fixed); idempotent creation | `tests/rag/test_qdrant_payload_indexes.py` against a live Qdrant | Gate A |
| S3-T04 | Backend contract parity suite (both backends) | test-plan §4 | `TODO` | `tests/rag/` does not exist | Create the parity oracle; run under both `VECTOR_BACKEND` values | S3-T02 |
| S3-T05 | Coverage + idempotency under Qdrant; embedding identity | test-plan §3/§5/§8 | `TODO` | No `embedding_model` in ingest metadata | Run S1 tests with `VECTOR_BACKEND=qdrant`; record embedding identity | S3-T04 |

### S4 — Chroma vs Qdrant → Gate B

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| S4-T01 | Qdrant candidate result | SPEC §11 Gate B | `TODO` | — | Same fixture/method/top_k/embedding | S3 exit |
| S4-T02 | Paired quality comparison | retrieval-benchmark §8–9 | `TODO` | — | Strict comparator; expect a tie within CI | S4-T01 |
| S4-T03 | Filtering + operability comparison | test-plan §5 | `TODO` | — | — | S4-T01 |
| S4-T04 | Gate B decision + ADR-0003 update | SPEC §11 Gate B | `TODO` | — | Honest framing: quality tie plus operability | S4-T02/T03 |

### Horizon (gated; do not start)

| ID | Task / context | Trace | Status | Evidence | Remaining / next action | Depends on |
| --- | --- | --- | --- | --- | --- | --- |
| S5 | Diagnostics → hybrid/rerank/section prior (Gate C) | SPEC §3 deferred | `GATED` + `AHEAD` | `hybrid_retrieve.py`, `reranked_hybrid_retrieve.py`, `section_aware_search.py`, `run_retrieval_method_matrix.py` + unit tests exist | No method claims until paired on the Gate A fixture | Gate B |
| S6-a | Eval registry with lineage (`evaluation/registry/runs/`) | SPEC §11.2; test-plan §14 | `GATED` | Hook H5 enforces lineage keys on writes; no registry writer; result JSONs lack `git_sha`/`prompt_version`/`snapshot_hash`/`fixture_version` | — | Gate B |
| S6-b | `eval-replay` CI regression gate (2σ) | SPEC §11.2 | `GATED` | — | — | S6-a |
| S6-c | Verifier↔judge agreement, n≥50, κ | SPEC §11.2 | `GATED` | `evaluation/judge_models_interface.py`, `run_rag_quality_eval.py` exist | — | S6-b |
| S6-d | Bounded draft→verify→revise loop (max 2) | SPEC §8.1 | `GATED` | `verify_memo → END` | — | S6-c |
| S6-e | LLM-as-judge/RAGAS, then GEPA on holdout | SPEC §12 | `GATED` | RAGAS/DeepEval harness exists (`eval` group) | — | S6-d |
| S7-a | Guardrails (input regex, untrusted-content framing, output policy, disclaimer) | SPEC §11.4; test-plan §12 | `GATED` | Ticker only `min_length=1,max_length=10` (no regex); no `app/guardrails/` | — | S6 |
| S7-b | API auth + rate limit + idempotency key | test-plan §12 | `GATED` | None | — | S7-a |
| S7-c | Caching (evidence TTL, query-embedding, memo) + metrics | SPEC §11.3; test-plan §13 | `GATED` | No `app/cache/` | — | S7-a |
| S7-d | JobQueue→Redis Streams, RunStore→Postgres, checkpointer | SPEC §12 | `GATED` | `BackgroundTasks` + `.runtime/run_store.json` | — | S7-c |
| S7-e | FinBERT out-of-process; provider circuit breaker | SPEC §8.2/§12 | `GATED` | — | — | S7-d |
| S7-f | G13: `/health` degraded on Ollama outage regardless of provider | SPEC §12 G13 | `GATED` | `app/main.py:193–198` checks only Ollama | — | — |
| S8 | Frontend MVP | SPEC §1.2 | `GATED` | — | — | S7 |
| S9 | Portfolio polish | SPEC §12 | `GATED` | — | — | S8 |
| S10 | Cloud (v2 Phases 4–5) | SPEC §3.y; ADR-0006 | `GATED` | ADR-0006 Proposed | — | ADR-0006 Accepted |

---

## 4. Updates needed to already-completed components

These close drift in work that is marked done. They are small and should come **before** any new S2 step. The sprint-plan sequencing rule and G11 both depend on a green bar.

| ID | Component | Problem (evidence) | Needed update | Trace | Priority |
| --- | --- | --- | --- | --- | --- |
| ENV-01 | Local `.venv` | Shebangs point to `/media/…/git/prj/…` after the repo move, so `uv run pytest`/`mypy` fail to spawn and hook H12 cannot run tests | Recreate the venv (`uv sync --reinstall`, or remove `.venv` then `uv sync`). Local-only; nothing to commit | CLAUDE §4, §8 | P0 |
| UPD-01 | `tests/unit/test_llm.py` | Collection `ImportError: MEMO_TEMPLATE`; call-order spies at L25/L32 fail `check_test_hygiene.py`; mypy L11 | Rewrite against the current `app/services/llm.py` public surface with behaviour assertions (test-plan §7), or delete if it has no oracle case. Do not weaken the hygiene check | test-plan Principles; S0-T04 | P0 |
| UPD-02 | `app/components/retrieval/embeddings.py:91` (`get_embeddings`) | **Code bug; the test is correct.** `model = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")` overwrites the explicit `model` argument and bypasses `settings.ollama.embed_model`, so `get_embeddings(model="nomic-embed-text")` returns `qwen3-embedding:4b`. This undermines embedding identity for the S3-T05/S4 backend comparison | Honour the argument first, then settings (`model or settings.ollama.embed_model`); keep `test_custom_model` unchanged | test-plan §8 (embedding identity) | P0 |
| UPD-03 | Lint + types | ruff 4 (2 in `app/llm/*_check.py`, 1 hook, 1 vendored skill); mypy 6 (`web_search_tool.py:159`, `quality_baseline.py:354/359/363`, `latency_baseline.py:211`, `test_llm.py:11`) | Fix the `app/` and `evaluation/` errors. Exclude `.claude/skills/` from ruff or fix it. Rename `l` in `stop_gate.py` | CLAUDE §5/§8 | P0 |
| UPD-04 | CI workflow (P0-T02) | `check_test_hygiene.py` not run; no separate `governance` job; `.runtime/run_store.json` and `test_image/` still tracked | Finish S2-T00a scope items (1)–(2) | S2-T00a | P1 |
| UPD-05 | Active-sprint markers | `CLAUDE.md §1` says "S1 now active". The sprint-plan S0 header still says "active". The `tasks/todo.md` header says Phase 0→1 active, but both phases exited. S2-T00a/T00b `Result: Pending` hides the Phase 0/1 delivery | Name one active step: **S2-T00a (finish)**. Record partial Results for T00a/T00b and a `Result:` for S2-T01. Mirror in AGENTS.md; run Doc Sync. CLAUDE/AGENTS/sprint-plan edits need `ALLOW_SPEC_EDIT=1` where protected | lessons "confirm active plan doc"; CLAUDE §2 | P1 |
| UPD-06 | `tasks/todo.md` checkboxes | BENCH-FIX is still `[ ]` though its Result is PASS; two 2026-09-06 items are waiting on governance-green | Tick BENCH-FIX; close the two items after UPD-01 turns hygiene green | CLAUDE §3.10 | P2 |
| UPD-07 | Path drift, DataOps + retrieval layout | SPEC §5 `app/dataops/` vs `dataops/`; sprint-plan `evaluation/evidence_releases/` vs `artifacts/dataops/`; sprint-plan S3 and CLAUDE §5 say `rag/` and `tests/rag/`, but code is `app/components/retrieval/` (G8 remnant) | Decide the canonical paths in S2-T00b (move code or amend docs via ADR/SPEC edit); update the CLAUDE/AGENTS "never escape `rag/`" rule to the real package | SPEC §5; G8 | P1 |
| UPD-08 | SPEC §2.1 #7 | Says CI wiring "is G11 / S2-T00a", but 3 of 4 checks are already wired | Update after S2-T00a closes (SPEC edit under H3) | SPEC §2.1 | P2 |
| UPD-09 | Memo-grounding artifact naming | `evaluation/results/retrieval_baseline_00N_*.json` are memo-grounding runs, easy to confuse with the S2 retrieval oracle; they also lack the §11.2 lineage keys | Rename the prefix (e.g. `memo_grounding_baseline_`) when S6-a lands; don't rename the approved artifacts in place | SPEC §10/§11.2 | P3 |
| UPD-10 | Hook tests | test-plan §15 names `tests/unit/test_hooks.py`; it does not exist | Part of S2-T00a scope (3) | test-plan §15 | P1 |

---

## 5. Open gap register

| Gap | Description | Status | Resolves in |
| --- | --- | --- | --- |
| G1 | Stale API test stub | Closed 2026-06-04 | — |
| G2 | `EvidencePacket` naming vs SPEC §7 | Closed (re-baselined) | — |
| G3 | `RetrievalStore` shape; `healthcheck()` missing | **Open** (no `healthcheck` on either store) | S3-T02 |
| G4 | Default backend = qdrant | Closed (ratified, ADR-0003) | — |
| G5 | Section-coverage regression test | Closed (S1-T03) | — |
| G6 | Benchmark not content-anchored | **Open** | S2-T02/T03 |
| G7 | Qdrant `accession_number` index | Closed | — |
| G8 | Doc/path name drift | **Partially open** (`rag/`, DataOps paths; UPD-07) | S2-T00b / doc pass |
| G10 | `app/` imports `evaluation/` | **Open** (`graph.py:49`) | S2-T00d |
| G11 | CI does not run all governance checks | **Partially open** (hygiene missing) | S2-T00a |
| G12 | `add_error` mutates in place | **Open** | S2-T00c |
| G13 | `/health` Ollama-only | **Open** | S7 |

---

## 6. What's next (ordered; respects the sprint-plan rule that no step opens before the previous one records a `Result:` with commit evidence)

1. **ENV-01**: recreate `.venv` so `uv run` and hook H12 work again (local, minutes).
2. **UPD-01 → UPD-03**: restore the green bar: fix `test_llm.py` behaviour-first, fix `get_embeddings` ignoring its `model` argument, clear ruff 4 and mypy 6. Verify with the full CLAUDE §8 core and governance commands.
3. **S2-T00a (finish, step 1)**: add hygiene to CI (governance job), untrack `.runtime/run_store.json` + `test_image/`, add `tests/unit/test_hooks.py`, confirm CI green on GitHub, record `Result:`. Fold in UPD-05/UPD-06 doc markers.
4. **S2-T00c (step 2)**: parallel fan-out, `errors` reducer, graph/LLM singletons, `to_thread` for FinBERT/store calls; latency before/after on `alpha-evidence:0.1.0`.
5. **S2-T00b (step 3a)**: aggregate `snapshot_hash` + `evidence_as_of`, runtime `evidence_release` replay, zero-network `replay` suite; settle UPD-07 paths.
6. **S2-T00d (step 3b)**: move the verifier to `app/verification/` (closes G10); `degraded`/`evidence_missing` statuses.
7. **S2-T01 → S2-T06 (Gate A)**: record T01, then source-section cache from frozen SEC snapshots → anchored validator → 50–100 graded cases → runner provenance → comparator `fixture_version` + single-axis → Chroma baseline.
8. Then **S3 → S4 (Gate B)**. Most S3 code exists; the work is the `tests/rag` parity suite, `healthcheck()`, index test, and embedding identity.

Hard stops to watch: no retrieval/Qdrant/method comparison on live evidence or on the v1/v2 precursor fixtures; no gold labels outside the frozen release; do not open S5–S7 work even though S5 code already exists.
