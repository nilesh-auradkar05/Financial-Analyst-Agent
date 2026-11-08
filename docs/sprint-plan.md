# sprint-plan.md — Sequencing & Task Ledger

> **Authority:** Source-of-truth rank 3 (SPEC §0). Sprint IDs match SPEC §12 exactly. Committed sprints (S0–S4) are detailed; horizon sprints (S5–S10, plus S11 added 2026-11-01 by ADR-0018) are gated one-liners, detailed only when the predecessor gate clears.
> **Task format:** Trace · Goal · Scope · Expected behavior · Tests · Verification · Done · Non-goals · Result.
> **Discipline:** No task is marked done without recorded verification output. Run the Doc Sync Check before closing any task.

---

## Completed ledger: prior work, unverified until S0-T04

These exist from prior work but are **not yet proven by a recorded test run in this repo**. S0-T04 reclassifies each as `verified` with a proving test or `gap` with a ticket. No committed task may treat these as load-bearing until then.

> **Audit classification (2026-06-04, S0 current-state audit; evidence in `tasks/sprint-review.md`).** None reaches full `verified`: the complete `pytest` run is not green (gap G1) and C2/C3 carry substantive gaps. The audit also found the prototype is **several gated sprints ahead of the docs** — Qdrant, hybrid/reranked/section-aware retrieval, a 72-case fixture, paired comparator, and a RAGAS/DeepEval harness already exist — and that two gated decisions were already taken in code (G4, G6). Classifications below.

- **C0** — LangGraph workflow baseline; persistent run state; request controls (`include_filing_analysis`, `include_news_sentiment`, `max_news_articles`). → **Partially verified.** Workflow + all three controls implemented (`app/agents/graph.py`); `tests/unit` green for state/routing/run-store. Public-API proof blocked by gap **G1** (stale `fake_check_ollama_health` stub errors all of `tests/test_api_integration.py`).
- **C1** — Structured `EvidencePacket` contract; verification wired into workflow and API. Prior lesson: invalid retrieval conclusions were once drawn from mismatched fixtures, so the benchmark discipline in `retrieval-benchmark.md §8` exists to prevent recurrence. → **Partially verified.** Packet + verifier present and unit-tested, but field names diverge from SPEC §7 (`chunk_id`/`section_name` vs `evidence_id`/`section_key`) → gap **G2**.
- **C2** — Retrieval abstraction: the `RetrievalStore` contract with Chroma implementation. → **Gap.** Protocol surface diverges from SPEC §6 (no `upsert_documents`/`delete_by_filter`/`healthcheck`; different return types) → **G3**; and `get_vector_store()` **defaults to `qdrant`**, contradicting SPEC §1 and ADR-0003 → **G4** (hard stop).
- **C3** — Section-aware ingestion baseline; `edgartools` reportedly fixed prior MSFT/NVDA zero-section issue. Asserted by manual observation; regression-locked only at S1-T03. → **Unverified (as designed).** No AAPL/MSFT/NVDA section-coverage test through the store interface yet → gap **G5**; regression lock remains S1-T03.

### Discovered gaps (audit 2026-06-04)

Full detail in `tasks/sprint-review.md §4`. Hard-stop surfaces are **G4** and **G6**.

- **G10** `app/agents/graph.py` imports `evaluation.grounding` — production depends on the eval package (layering inversion). Resolve S2-T00d · **G11** `ci.yml` never invokes `scripts/ci/check_*.py`; governance docs not tracked at HEAD, so no doc rule is enforced. Resolve S2-T00a · **G12** `add_error` mutates `state["errors"]` in place — unsafe under fan-out. Resolve S2-T00c · **G13** `/health` reports `degraded` on Ollama outage regardless of configured provider. Resolve CM-6a · **G14** (2026-11-01) no checkpointer; jobs run in-process, so a crash restarts the run. Resolve S7.3, S7.4 · **G15** (2026-11-01) stock `DynamoDBSaver` writes are not fenced. Resolve S7.8.

- **G1** stale API test stub (11 errors) · **G2** `EvidencePacket` naming vs SPEC §7 · **G3** `RetrievalStore` shape vs SPEC §6 · **G4** default backend = qdrant vs SPEC §1/ADR-0003 (hard stop) · **G5** no section-coverage regression test (S1-T03) · **G6** benchmark is keyword/section-based, not the content-anchored graded methodology in `retrieval-benchmark.md`; no `source_sections/` cache or `build_source_section_cache.py` (hard stop) · **G7** Qdrant payload indexes miss `accession_number` (4 of 5) · **G8** doc/path name drift (eval scripts, test dirs, API routes) makes several documented verification commands non-runnable as written.

---

# COMMITTED

## Sprint 0 — Spec coherence & baseline verification, active

Goal: make the governance layer internally consistent and convert the C-ledger from assertion to evidence before any implementation sprint trusts it.

```text
S0-T01 — Apply doc topology + dedup edits
Trace: SPEC §0/§3/§13; README document model
Goal: One fact, one home; remove residue; install scripted doc checks.
Scope:
- SPEC §0 is the sole home for source-of-truth order; CLAUDE/AGENTS point to it.
- SPEC §3 is the sole home for scope; CLAUDE/AGENTS point to it.
- Remove PulsePress/Terraform/AWS deployment residue from Alpha agent-control docs.
- Add/verify these scripts: scripts/ci/check_no_scope_residue.py, check_sprint_map.py, check_doc_sync.py.
- Keep task loop, hard stops, coding conventions, verification discipline, and correction handling in both CLAUDE.md and AGENTS.md as semantically equivalent mirrors. Claude-specific mechanics may live in Claude-only sections, but no operating instruction may be omitted from AGENTS.md.
Expected: no rule is defined twice except intentional mirrors; no stale deployment residue remains in core Alpha docs.
Tests:
- python scripts/ci/check_no_scope_residue.py
- python scripts/ci/check_doc_sync.py
- python scripts/ci/check_sprint_map.py
Verification:
- grep -Rni "terraform" CLAUDE.md AGENTS.md docs/SPEC.md docs/test-plan.md docs/retrieval-benchmark.md docs/adr || true
- The grep above must return no Alpha-control residue; sprint-plan.md is intentionally excluded so the command does not match itself. Humanity survives one more shell command.
Done: contradictions resolved; mirror sections semantically equivalent; scripts exist and pass.
Non-goals: no app code; no new methodology.
Result: PASS (audit 2026-06-04). check_no_scope_residue / check_doc_sync / check_sprint_map all pass; residue grep clean; CLAUDE↔AGENTS semantically equivalent.
```

```text
S0-T02 — Establish retrieval-benchmark.md comparison policy
Trace: SPEC §10; test-plan §6; retrieval-benchmark.md
Goal: single home for benchmark methodology exists with comparison policy.
Scope: doc exists; SPEC §10 and test-plan §6 point to it; paired-comparison policy and single-axis rule present.
Expected: any benchmark-policy question has one authoritative answer.
Tests: grep — no metric formulas outside retrieval-benchmark.md except brief pointers.
Done: retrieval-benchmark.md exists; pointers in place. Schema/metrics implementation happens in S2.
Non-goals: no fixture authoring.
Result: PASS at doc level (audit 2026-06-04) — methodology home + single-axis/paired policy present; SPEC §10 and test-plan §6 point to it. CAVEAT: the *implemented* fixture/validator do not follow this methodology (keyword/section-based, no anchors, no source cache) — gap G6 (hard stop).
```

```text
S0-T03 — Finalize sprint map + committed/horizon split
Trace: SPEC §12; this file
Goal: one numbering scheme; gate mapping; committed vs horizon.
Scope: SPEC §12 table and this file use S0–S10 identically; gates A→S2, B→S4, C→S5, D→S6, E→S10.
Expected: "Sx" resolves to the same work in every doc.
Tests: python scripts/ci/check_sprint_map.py
Done: numbering consistent; split formalized.
Non-goals: no task detail for horizon beyond one-liners.
Result: PASS (audit 2026-06-04). check_sprint_map confirms S0–S10 present in SPEC §12 and this file; horizon described as S5–S10.
```

```text
S0-T04 — Baseline verification pass
Trace: SPEC §1.1; completed ledger C0–C3; test-plan
Goal: turn "complete (asserted)" into "verified" or "gap".
Scope:
- Run document-governance checks first:
  - python scripts/ci/check_no_scope_residue.py
  - python scripts/ci/check_doc_sync.py
  - python scripts/ci/check_sprint_map.py
- Run repository checks:
  - uv run ruff check .
  - uv run pyright . || uv run mypy .
  - uv run pytest
- Record actual pass counts/failures.
- Mark each C-ledger item verified with the proving test or gap with a follow-up ticket.
- Capture all evidence in tasks/sprint-review.md as the baseline-status report.
Expected: no downstream sprint trusts an unverified foundation.
Verification: command outputs pasted or summarized honestly in tasks/sprint-review.md.
Done: every C-ledger line verified or ticketed.
Non-goals: do not fix non-trivial gaps here; log them.
Result: DONE — CLOSED GREEN (2026-06-04). Hard stops resolved by user decision (G4 ratify Qdrant; G6 anchored labels at S2; G2/G3/G7/G8 pragmatic re-baseline). G1 + G7 fixed in code; docs reconciled. Verification all green: doc scripts PASS · `ruff` clean · `mypy app evaluation` Success (44 files) · `pytest` 178 passed, 2 skipped · fixture validator PASS (72). Reclassified: C0 verified, C1 verified, C2 verified (re-baselined; healthcheck + parity suite → S3), C3 gap → S1-T03 (G5). Full record in tasks/sprint-review.md §6.
```

Exit: doc scripts pass, sprint IDs match, baseline-status report complete.

---

## Sprint 1 — SEC ingestion & section-coverage lock

Goal: lock section extraction and stable identity so the S2 benchmark can label evidence that will not move under it.

```text
S1-T01 — ADR-0004: edgartools parser decision
Trace: SPEC §9/§16; test-plan §3
Goal: stop re-litigating parser strategy.
Scope: docs/adr/ADR-0004-edgartools-parser.md records prior zero-section failure, why edgartools, fallback trigger, and coverage gate.
Tests: ADR exists and references the coverage gate.
Done: ADR committed.
Non-goals: no paid API.
Result: DONE (2026-06-04) — ADR-0004 exists, is Accepted, and ties edgartools to the S1 section-coverage gate.
```

```text
S1-T02 — Normalize section metadata + deterministic chunk IDs
Trace: SPEC §7/§9; test-plan §3
Goal: every chunk carries canonical, stable identity across backends.
Scope: canonical section keys; ticker uppercase; filing_type, filing_date, accession_number, source_url present; chunk_id = f(accession_number, section_key, chunk_index), stable across re-ingest.
Expected: both backends get identical payload metadata; re-ingest produces same IDs.
Tests: chunk-metadata schema test; re-ingest twice gives identical chunk_ids.
Verification: uv run pytest tests/ingestion/test_chunk_metadata.py tests/ingestion/test_ingestion_idempotency.py
Done: metadata and idempotency tests pass.
Non-goals: no scoring changes; no new sections.
Result: DONE (2026-06-04) — `build_index_documents()` now requires accession/source metadata, derives `source_url` from `filing_url` when needed, and mints `chunk_id = f(accession_number, section_key, chunk_index)`. Verification included `tests/ingestion` 5 passed and full `pytest` 181 passed, 2 skipped.
```

```text
S1-T03 — Section coverage tests, regression lock for C3
Trace: SPEC §9; test-plan §3
Goal: a critical section with zero chunks fails CI permanently.
Scope: AAPL/MSFT/NVDA × {business, risk_factors, md&a, market_risk}; assert counts through the store interface. Missing optional section gets explicit xfail/skip with reason.
Expected: the MSFT/NVDA zero-section failure cannot recur undetected.
Verification: VECTOR_BACKEND=chroma uv run pytest tests/ingestion/test_section_coverage.py
Done: coverage tests pass; C3 reclassified verified.
Non-goals: Qdrant coverage is S3.
Result: DONE (2026-06-04) — representative AAPL/MSFT/NVDA fixtures assert every critical section has >0 chunks via `count_documents(SearchFilters(...))` through the store interface. Verification: `VECTOR_BACKEND=chroma uv --cache-dir /tmp/uv-cache run pytest tests/ingestion -q` -> 5 passed.
```

Exit: DONE (2026-06-04) — critical sections non-zero through interface; chunk IDs stable across re-ingest; C3/G5 closed. Full evidence in `tasks/sprint-review.md §7`.

---

## Sprint 2 — Shared retrieval benchmark oracle → Gate A

Goal: a trusted, self-validating oracle before any backend or method is judged.

> **Phase overlay (INTEGRATION_PLAN_v2, 2026-07).** Two local-first phases precede S2 *execution* and are prerequisites for it, not replacements:
> - **Phase 0** — governance `docs/` committed + doc checks wired into CI. Verify actual git state first (`git ls-files docs/`); v2's "no docs/" finding may be stale given SPEC §0 already specifies the `docs/` tree.
> - **Phase 1** — evidence-snapshot freeze/replay (`--evidence-release`). This is the fixture-freeze rule below, generalized to *all* evidence (SEC + news + market + FinBERT), not just retrieval fixtures. **S2's anchored fixture and Gate A baseline are built on Phase 1's frozen release**, so the source-section cache is derived from a committed snapshot rather than re-fetched. S2 does not start until Phase 1 exit criteria are green.

> **Production-readiness overlay (review 2026-08-24, SPEC amendment v1.3 §12).** The review ordered eight steps. Steps 1–3 are S2 pre-tasks below (T00a–T00d) and are prerequisites for S2 execution. Steps 4–8 are absorbed into S6/S7/S10 (see HORIZON). **Sequencing rule:** no step opens before the previous step has a recorded `Result:` with commit evidence. Opening a later step early is the scope-expansion pattern and is a hard stop.
>
> | Step | Home | What |
> | --- | --- | --- |
> | 1 | S2-T00a | Governance docs into `docs/`, `scripts/ci/check_*.py` wired into CI (Phase 0) |
> | 2 | S2-T00c | Fan-out + event-loop hygiene; graph singleton; `errors` reducer |
> | 3 | S2-T00b, S2-T00d | EvidenceSnapshot + zero-network replay (Phase 1); verification-fail and evidence-missing status semantics |
> | 4 | S6 | Eval registry with lineage, CI regression gate, verifier↔judge agreement |
> | 5 | S6 | Bounded draft→verify→revise loop (max 2), paired against no-loop on frozen snapshot |
> | 6 | S7 | Guardrails: input schema, untrusted-content framing, output policy, API auth + rate limit |
> | 7 | S7 | Caching: evidence (per-source TTL), query-embedding, memo (snapshot_hash) |
> | 8 | S7 → S10 | Queue/worker, durable job store (DynamoDB + SQS since ADR-0015; was Postgres), Qdrant, FinBERT out-of-process, circuit breaker; cloud gated on the ADR-0006 entry criteria (ADR accepted 2026-10-08) |

```text
S2-T00a — Governance docs into repo + CI governance gate  (Phase 0; step 1)
Trace: SPEC §0/§13/§14; INTEGRATION_PLAN_v2 Phase 0; gap G11
Goal: every governance rule the repo claims is enforced by a script that CI actually runs.
Scope:
- git-track docs/SPEC.md, docs/sprint-plan.md, docs/test-plan.md, docs/retrieval-benchmark.md, old_artifacts/docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md, docs/adr/*.
- Add a `governance` job to .github/workflows/ci.yml running check_no_scope_residue, check_sprint_map, check_doc_sync, check_test_hygiene.
- Remove .runtime/run_store.json and test_image/ from tracking; add to .gitignore.
- Commit .claude/settings.json + .claude/hooks/* (SPEC §14).
Expected: `git ls-files docs/` non-empty; governance job green on main; a PR that edits SPEC §12 without sprint-plan fails CI.
Tests: test-plan §9, §15.
Verification:
- git ls-files docs/ | wc -l  (≥ 5)
- gh run view --job governance  (pass)
- Deliberately break a sprint ID in a throwaway branch → governance job must fail.
Done: all three verification outputs recorded below; hooks H3/H4/H8/H9/H12 pass manual stdin smoke tests.
Non-goals: no app code; no doc content changes beyond path fixes.
Result: DONE locally (2026-09-29) — governance job runs all four checks; 35 hook subprocess tests cover H2–H9/H12; ruff clean; mypy 128 files clean; pytest 262 passed, 2 skipped; four governance checks pass; 36 docs tracked; `.runtime/` and `test_image/` ignored/untracked. Local red paths pass. Remote GitHub `governance` job confirmation remains pending until push.
```

```text
S2-T00b — Evidence snapshot freeze/replay with zero-network assertion  (Phase 1; step 3a)
Trace: SPEC §7.x/§11.1; INTEGRATION_PLAN_v2 Phase 1; test-plan §13
Goal: any memo can be re-produced from a committed evidence release with the network disabled.
Scope:
- app/dataops/: build_snapshot(ticker, request) → EvidenceSnapshot{snapshot_hash, evidence_as_of, packets, manifest}; freeze to evaluation/evidence_releases/<release>/; replay loader.
- run_agent(..., evidence_release=...) bypasses live tools when set.
- snapshot_hash + evidence_as_of surfaced in AgentState and AnalysisResponse.
Expected: two replays of the same release produce byte-identical citation registries.
Tests: test-plan §13 (replay determinism, zero-network).
Verification:
- EVIDENCE_MODE=replay pytest tests/unit -m replay  under `unshare -n` (or HTTP_PROXY dead-proxy fallback on macOS) → green.
- Hook H7 rewrites the command; record the rewritten command line.
Done: zero-network run green; hash stable across two runs; release manifest committed.
Non-goals: no cloud storage; no scheduler.
Result: DONE (2026-10-11), branch `feat/s2-t00b-runtime-replay`. Phase 1 (2026-07) delivered contracts, registry, gate, writer, release `alpha-evidence:0.1.0` and eval-side replay. This task added runtime replay: `run_agent(..., evidence_release=...)` runs only draft and verify from a registered release (`app/dataops/replay.py`), and every run carries `snapshot_hash` and `evidence_as_of` in `AgentState` and `AnalysisResponse` (`app/dataops/snapshot.py`; SPEC §7.y, §7.z). Replay suite `tests/replay/` 71 passed with an in-test socket guard; full suite 452 passed, 2 skipped (five consecutive runs); ruff, mypy and the four governance checks pass; two loads of the committed release give the same hash. Deviations: code is in `app/dataops/` and `dataops/`, artifacts in `artifacts/dataops/`. `unshare -n` is not permitted on this machine, so isolation rests on the socket guard plus dead-proxy variables, not a network namespace. Follow-up 2026-10-11 (user correction): `evidence_as_of` is the oldest fetch time, never the run start; live nodes record per-source fetch times, ingestion stores `fetched_at` on every chunk, and chunks indexed earlier fall back to their filing date (SPEC §7.y). Not in this task: `snapshot_hash` in eval run records (S6a) and cache keys (S7).
```

```text
S2-T00c — Fan-out and event-loop hygiene  (step 2)
Trace: SPEC §8.1–8.4; gap G12; latency-baseline discipline (single axis)
Goal: independent evidence nodes run concurrently; no node blocks the loop; graph and clients are singletons.
Scope:
- research_news ‖ fetch_stock ‖ retrieve_filings via LangGraph branch/Send; join before analyze_sentiment.
- analyze_sentiment_batch and Chroma/Qdrant calls under asyncio.to_thread.
- create_agent() at module import; get_llm cached per process.
- AgentState.errors declared with an append reducer; add_error returns a fresh list.
- Retrieval returns exactly the number of chunks the registry uses (no dead top-10 → top-5 waste).
Expected: p50 wall-time of the evidence phase ≈ max(node latencies) instead of their sum; concurrent /analyze requests do not serialize on FinBERT.
Tests: test-plan §10.
Verification:
- evaluation/latency_baseline.py on the frozen release before and after; SAME model/temperature; record both JSON filenames.
- Concurrency probe: 4 simultaneous /analyze on replay → total wall-time < 1.5× single.
Done: both artifacts recorded; unit tests for reducer + parallel routing green.
Non-goals: no multi-agent; no caching; no repair loop.
Result: DONE (2026-10-22). Latency evidence recorded in `evaluation/results/s2_t00c_live_probe_2026-10-22.json` per D1 and the 2026-10-22 owner decision D-A3: the probe ran live, not on replay, because replay skips the evidence nodes and so cannot show FinBERT blocking; total time is not compared because the draft model differs from the BEFORE trace. Seven live `/analyze/async` runs for APP (D1 inputs) on the running api container, whose evidence-path code matches main. Fan-out (warm run 7): evidence phase 4.12 s = slowest branch 1.80 s + sentiment 2.25 s, against 5.90 s if serial; BEFORE trace 01a0f4dc was serial (8.65 s, different network timings). Concurrency probe: 4 simultaneous runs finish the evidence phase 6.15 s after the first submission, 1.44× the single run's 4.27 s (threshold 1.5×, PASS by a thin margin, n=1 baseline over the live network); sentiment per request rises from 2.25 s to about 3.5 s under load (CPU contention, not serialization). Run 1 was cold (model loads) and is excluded; run 2 had an 11 s yfinance outlier. All seven runs ended `degraded` (verification failed), which is outside this task. Earlier: Partial as of 2026-10-06. Implemented in 946f1de (fan-out, errors reducer, to_thread, AGENT singleton, retrieval sizing) and b407868 (provider client built once per process). Unit tests for reducer + parallel routing green (test-plan §10): three 1 s branches complete in 1.02 s (2.83 s before); /health answers in under 200 ms while sentiment scores 50 snippets (blocked 0.60 s before). Full suite 356 passed, 2 skipped; ruff, mypy and governance checks pass. NOT recorded: the latency artifacts and the 4-concurrent probe. Frozen-release latency cannot be measured until S2-T00b (replay skips the evidence nodes); the agreed substitute, one live run compared with LangSmith trace 01a0f4dc-8b2b-7021-b7de-5958220c0ad1 (66.73 s total, 8.65 s evidence phase), has not been run.
```

```text
S2-T00d — Verification and evidence-completeness status semantics  (step 3b)
Trace: SPEC §8.5; gap G10; test-plan §11
Goal: the API never reports `completed` for a memo that failed verification or lacks a required evidence class.
Scope:
- Move evaluation/grounding.py runtime path to app/verification/; evaluation/ imports it, not the reverse.
- New terminal statuses: degraded, evidence_missing. _format_response derives status from verification.passed and get_data_availability.
- Ticker with zero indexed filings and include_filing_analysis=True → evidence_missing with an actionable `missing=[...]` list.
Expected: JobStatus enum extended; existing tests updated to new semantics; no silent completion path remains.
Tests: test-plan §11.
Verification:
- grep -rn "from evaluation" app/  → empty.
- pytest tests/unit/test_graph_verification_integration.py tests/unit/test_api_schema_defaults.py → green with new statuses.
Done: grep empty; tests green; SPEC §8.5 statuses match models.JobStatus exactly.
Non-goals: no repair loop (S6); no guardrails (S7).
Result: DONE (2026-10-22). G10 closed: the runtime verifier moved from `evaluation/grounding.py` to `app/verification/grounding.py` (behaviour-preserving, no shim); `app/agents/graph.py`, `evaluation/` and the tests import it from there; `grep -rn "from evaluation" app/` is empty; new `tests/unit/test_layering.py` (test-plan §11 Layering) fails on the old import and passes now. Suite 493 passed before and 494 passed after (the new test), 2 skipped; ruff and `mypy .` clean (apart from the user's gitignored `probe_bedrock_access.py`); four governance checks pass. Earlier: Partial (reconciled 2026-10-10). Done: `JobStatus.DEGRADED` and `EVIDENCE_MISSING` in `app/models.py`; status and `missing=[...]` derived in `app/main.py`; cases in `tests/test_api_integration.py` (verification failed → degraded; no filings or no stock data → evidence_missing); both statuses shown in the UI (user-confirmed 2026-10-10). NOT done: gap G10 layering — `app/agents/graph.py` still imports `evaluation.grounding` and `app/verification/` does not exist, so the `grep "from evaluation" app/` check is not empty.
```

> **Implementation note (2026-06-04, audit reconciliation).** Decision: *upgrade the existing keyword/section benchmark to the content-anchored graded methodology* in `retrieval-benchmark.md` (raise code to spec). Bind the doc's planned filenames to the actual files to **extend, not recreate** (G8 path drift):
> - `evaluation/validate_benchmark_fixture.py` → **extend** existing `evaluation/validate_retrieval_fixture.py` (add `gold_evidence` with `accession_number`/`section_key`/verbatim `anchor_text`≥8 words/`relevance∈{1,2}`, `answer_intent`, and the anchor-in-source check). Note: the current validator's `--min-cases` is hardcoded (`MIN_CASES=50`), not a flag.
> - `evaluation/run_retrieval_benchmark.py` → existing `evaluation/run_shared_retrieval_benchmark.py`.
> - `tests/evaluation/test_*` → create under `tests/unit/` (the repo's actual test home) unless a `tests/evaluation/` dir is added.
> - The current `retrieval_shared_benchmark_v1.json` (72 keyword/section cases) is the **precursor**; the anchored fixture gets a new `fixture_version` and a source-section cache + builder (`evaluation/build_source_section_cache.py`, reusing `app/services/tools/edgartools_sec_extractor.py`). Gate A is **not** satisfied by the v1 precursor.
>
> **Owner decisions 2026-10-22 (GATE-A).** (D-A1) The Gate A baseline runs on **Qdrant** section_aware, not Chroma: Chroma and S4 were retired 2026-10-10. (D-A2) The benchmark corpus is a separate Qdrant collection built from the frozen release `alpha-evidence:0.1.0` (AAPL, MSFT, NVDA) with the ingestion chunker; the source-section cache holds the same cleaned section text; the live `sec_filings` collection is not used or modified (its market_risk sections run 66–73 chunks per ticker, a boundary defect logged for a later ingestion fix). (D-A4) The owner reviews the fixture's coverage report and a case sample after S2-T03, before S2-T04…T06. (D-A5) The anchored fixture is `evaluation/fixtures/retrieval_anchored_benchmark_v1.json` with `fixture_version` `retrieval_anchored_benchmark_v1`; the S2-T03 verification command applies to that file.

```text
S2-T01 — Specify labeling schema, matcher, metric definitions
Trace: SPEC §10/§11; test-plan §6; retrieval-benchmark.md
Scope: fill/verify retrieval-benchmark.md §§2–4: case schema, content-anchored graded labels, matcher, metric formulas, K∈{5,10}.
Expected: every metric has one unambiguous formula tied to graded labels.
Done: schema + matcher + metrics fully specified; NDCG graded.
Non-goals: no fixture authoring; no LLM-judged relevance.
Result: DONE (2026-10-22). §§2–4 checked against the Done line: schema, matcher and metric formulas complete, NDCG graded, K ∈ {5, 10}. Three gaps closed by an owner-approved amendment to §4: 1-based ranks; `keyword_hit_rate@K` matching, K and empty-keyword rule; `evidence_id` = `chunk_id` with its tie order. No metric changes meaning.
```

```text
S2-T02 — Fixture validator: lie detector
Trace: SPEC §11 Gate A; test-plan §6; retrieval-benchmark.md §5–§6
Scope: evaluation/validate_benchmark_fixture.py validates schema, relevance∈{1,2}, anchor≥8 words, expected_sections match, min-cases, per-ticker/section coverage, and anchor-in-source against committed source-section cache.
Tests: valid fixture passes; duplicate id, missing gold, unknown section, invalid relevance, short anchor, absent anchor, expected_sections mismatch, and below-min-cases all fail.
Verification: uv run pytest tests/evaluation/test_benchmark_fixture_schema.py
Done: validator and behavior tests pass.
Non-goals: no scoring logic.
Result: DONE (2026-10-22). Red c309b13: `tests/unit/test_anchored_fixture_validator.py`, 46 cases over every test-plan §6 validator row plus the §5 answer_intent and coverage checks and §3 norm() matching. `evaluation/validate_retrieval_fixture.py` gains the anchored API (`validate_anchored_fixture`, `load_anchored_fixture`, `normalize_text`, `FixtureValidationError` listing every failure by case_id and rule; CLI `--source-dir`, `--min-cases`); the precursor path is unchanged. `evaluation/build_source_section_cache.py` writes `evaluation/fixtures/source_sections/` (12 files) from `alpha-evidence:0.1.0` with the ingestion cleaner (`clean_section_text`, made public): all 532 chunks the ingestion chunker builds from the release appear verbatim in the cache, and a rebuild is byte-identical. pytest 540 passed, 2 skipped; ruff and mypy clean (apart from the user's gitignored local script); four governance checks pass. Test path bound to tests/unit per the implementation note.
```

```text
S2-T03 — Author/expand fixture to 50–100 anchored, graded cases
Trace: SPEC §3/§10/§11; test-plan §6
Scope: AAPL/MSFT/NVDA plus more if available × all 4 sections; mixed answer_intent; 1–3 graded anchors per case; validate continuously; emit fixture coverage report.
Verification: uv run python evaluation/validate_benchmark_fixture.py evaluation/fixtures/retrieval_shared_benchmark_v1.json --min-cases 50
Done: 50–100 validated cases; coverage report committed; thin cells noted.
Non-goals: no tuning toward any backend or method; no answer-quality labels.
Result: DONE (2026-10-22). `evaluation/fixtures/retrieval_anchored_benchmark_v1.json` (`fixture_version` retrieval_anchored_benchmark_v1), frozen after owner review: 75 cases (AAPL 24, MSFT 24, NVDA 27) over all four sections of `alpha-evidence:0.1.0`, 138 graded anchors (93 relevance 2, 45 relevance 1), five answer intents, 6 multi-section cases; anchors authored from the source-section cache only. Validator passes with `--min-cases 50`. Coverage report `evaluation/fixtures/retrieval_anchored_benchmark_v1_coverage.md`: thin cells are the three market_risk cells (4 cases each); 0 anchors straddle a chunk. Risk Factors text is capped at 50,000 characters by the extractor, so later risk factors are outside the corpus.
```

```text
S2-T04 — Benchmark runner
Trace: SPEC §10; test-plan §6; retrieval-benchmark.md §7
Scope: evaluation/run_retrieval_benchmark.py emits result file per backend/method with provenance, labels, metrics, cold/warm latency, null first_relevant_rank, deterministic tie-breaks.
Tests: all case_ids present; labels present; null rank emitted; provenance and latency fields present.
Verification: VECTOR_BACKEND=chroma uv run python evaluation/run_retrieval_benchmark.py --fixture evaluation/fixtures/retrieval_shared_benchmark_v1.json --mode section_aware --output evaluation/results/chroma_section_aware_shared.json
Done: schema-valid, provenance-bearing result.
Non-goals: no comparison; no hybrid/rerank.
Result: DONE (2026-10-22). Red e025bb5: `tests/unit/test_anchored_benchmark_runner.py` (34 tests) covers every test-plan §6 runner row, including the owner-approved metrics row (hand-computed graded NDCG, straddle miss, section-only hit, precision over K, null rank and keyword rate). `evaluation/run_shared_retrieval_benchmark.py` gains `score_retrieval`, `run_anchored_benchmark` (methods `dense`, `section_aware`; K ∈ {5, 10}; cold/warm latency; per-chunk provenance) and an anchored CLI path; the precursor path is unchanged. `evaluation/build_benchmark_index.py` indexes the release's SEC snapshots with the ingestion chunker into `sec_filings_bench_alpha_evidence_0_1_0` (532 chunks; refuses `sec_filings`; live collection unchanged at 1,104); every fixture anchor lies wholly in one indexed chunk. Ingestion exposes `CHUNK_SIZE`/`CHUNK_OVERLAP` for provenance. End-to-end smoke runs of both methods produce all 75 cases with provenance (not a baseline; no comparison drawn). pytest 574 passed, 2 skipped; ruff, mypy (apart from the user's gitignored local script) and the four governance checks pass. Binding: the doc's `run_retrieval_benchmark.py` is this file, per the implementation note.
```

```text
S2-T05 — Harden paired comparator
Trace: SPEC §10; test-plan §6; retrieval-benchmark.md §8
Scope: evaluation/compare_retrieval_results.py enforces fixture_version match, single-axis rule, --strict-case-ids, paired deltas, relative deltas, win/tie/loss, bootstrap CI, null=miss, latency separated.
Tests: mismatched fixture/mode/method fail; both-differ fails; missing case fails; null=miss; CI present.
Verification: uv run pytest tests/evaluation/test_compare_retrieval_results.py
Done: comparator passes behavior tests.
Non-goals: no default switch.
Result: DONE (2026-10-22). Red 4762699: `tests/unit/test_compare_anchored_results.py` (30 tests) covers every test-plan §6 comparator row. `evaluation/compare_retrieval_results.py` gains `compare_anchored_results` for anchored result files: rejects a fixture_version mismatch, a mode that disagrees with retrieval_method, differing K sets, duplicate case ids, and more than one differing axis among backend, retrieval_method and embedding_model (§8, §9a, §10); `--strict-case-ids` names baseline-only and candidate-only ids; reports per-K paired means, paired and relative deltas, win/tie/loss, a seeded 95% bootstrap CI (10,000 resamples by default), miss_rate with null rank as a miss, and cold/warm latency separately. Precursor path unchanged (its 4 tests pass). pytest 604 passed, 2 skipped; ruff, mypy (apart from the user's gitignored local script) and the four governance checks pass; CLI exercised on two S2-T04 smoke outputs (axis = retrieval_method), no quality claim drawn. Noted, not changed: the precursor path ranks `latency_ms_warm` as higher-is-better (pre-existing).
```

```text
S2-T06 — Generate Gate A baseline
Trace: SPEC §11 Gate A
Scope: run runner on Chroma section_aware over validated fixture; self-compare sanity; record baseline + coverage/CI-width report in tasks/sprint-review.md.
Tests: result validates; comparator self-compare produces zero deltas and full shared count.
Done: Gate A satisfied.
Non-goals: no Qdrant run; no method adoption.
Result: DONE (2026-10-22) — Gate A satisfied. Baseline `evaluation/results/gate_a_baseline_qdrant_section_aware_retrieval_anchored_benchmark_v1.json` on Qdrant section_aware (owner decision D-A1 replaces the Chroma wording; the "no Qdrant run" non-goal predates the 2026-10-10 Chroma retirement), collection built from `alpha-evidence:0.1.0`, code 0e7ef14, fixture retrieval_anchored_benchmark_v1. recall@10 0.602 [0.498, 0.702], ndcg@10 0.421, pass@10 0.667 (25 of 75 misses). Self-compare: 75 shared, every quality delta 0, CI [0, 0]; an independent rerun retrieved identical chunk ids for every case. Coverage, CI widths and the power caveat are in tasks/sprint-review.md §8. The fixture freeze rule now applies.
```

**Fixture freeze rule:** once Gate A baseline is generated, fixture edits require a `fixture_version` bump, re-validation, and regenerated baseline before any candidate comparison. No quiet label cleanup after results. That is not cleanup; that is benchmark contamination wearing a fake mustache.

Exit = Gate A.

---

## Sprint 3 — Qdrant backend behind the interface → Gate B prep

Goal: a second `RetrievalStore` that passes the exact tests Chroma passes, with zero change above `rag/`.

```text
S3-T01 — ADR-0003 + Qdrant local infra & backend selection
Trace: SPEC §6/§11/§15; test-plan §5
Scope: ADR-0003; docker-compose qdrant service + healthcheck wait; VECTOR_BACKEND=qdrant via DI; invalid backend causes config error; Qdrant unavailable causes degraded/clear failure.
Tests: tests/rag/test_vector_backend_selection.py
Verification: docker compose up -d qdrant ; uv run pytest tests/rag/test_vector_backend_selection.py
Done: ADR + compose + selection + degraded test.
Non-goals: no comparison or switch.
Result: DONE (reconciled 2026-10-10). ADR-0003 accepted; Qdrant service with healthcheck in `docker-compose.yml`; factory defaults to Qdrant, selects by `VECTOR_BACKEND`, rejects unknown values (`tests/unit/test_vector_store_factory.py`, 3 tests); `/health` reports `vector_store.ok=false` when the store is unavailable (`tests/test_api_integration.py`). Tests live under `tests/unit/`, not `tests/rag/`.
```

```text
S3-T02 — Implement QdrantVectorStore, Protocol parity
Trace: SPEC §6; test-plan §4/§5
Scope: implement all five contract methods; idempotent collection; SearchFilters→Qdrant filter; results→EvidencePacket; no qdrant_client escapes rag/.
Tests: covered by S3-T04 parity suite.
Done: Protocol implemented; no public signature above rag/ changed.
Non-goals: no hybrid; no indexes yet.
Result: DONE (reconciled 2026-10-10). `QdrantVectorStore` implements every `RetrievalStore` method; `qdrant_client` is imported only under `app/components/retrieval/`; `tests/unit/test_qdrant_store.py` (6 tests, fake client). Running default backend since S0-T04 G4.
```

```text
S3-T03 — Qdrant payload indexes
Trace: SPEC §6; test-plan §5
Scope: indexes for ticker, filing_type, section_key, filing_date, accession_number; idempotent creation; documented.
Tests: tests/rag/test_qdrant_payload_indexes.py
Verification: VECTOR_BACKEND=qdrant uv run pytest tests/rag/test_qdrant_payload_indexes.py
Done: indexes verified.
Non-goals: no scoring changes.
Result: DONE at unit level (reconciled 2026-10-10). `create_payload_indexes()` runs on collection setup; the fake-client tests assert the indexed fields. Not asserted against a live Qdrant server.
```

```text
S3-T04 — Backend contract parity tests
Trace: SPEC §6; test-plan §4
Scope: tests/rag/test_vector_store_contract.py runs under both VECTOR_BACKEND values; parity cases per test-plan §4. If S0-T04 found no contract test, this creates it as the parity oracle; Chroma must pass it too.
Verification:
- VECTOR_BACKEND=chroma uv run pytest tests/rag/test_vector_store_contract.py
- VECTOR_BACKEND=qdrant uv run pytest tests/rag/test_vector_store_contract.py
Done: identical pass under both; differences fixed in adapter, never by forking test.
Non-goals: no method-quality comparison.
Result: Green 2026-11-06 in CM-2 (65f3e27): `tests/rag/test_vector_store_contract.py` runs test-plan §4 against the local OpenSearch container (two indexes, one corpus). Retargeted 2026-10-23 to OpenSearch (ADR-0016); Chroma and Qdrant legs dropped. Earlier: Pending. No contract suite runs against a real backend: `tests/rag/` does not exist; current store tests use a fake Qdrant client and Chroma monkeypatches. With S4 retired the suite is the Qdrant contract oracle; the Chroma leg is required only while Chroma stays a supported backend (SPEC §6).
```

```text
S3-T05 — Section coverage + idempotent ingest under Qdrant
Trace: SPEC §9/§6; test-plan §3/§5
Scope: run S1 coverage + idempotency under VECTOR_BACKEND=qdrant; guard that the same embedding model/version is used on both backends; record embedding identifier in ingest metadata.
Verification: VECTOR_BACKEND=qdrant uv run pytest tests/ingestion/test_section_coverage.py tests/ingestion/test_ingestion_idempotency.py
Done: coverage + idempotency pass under Qdrant; embedding identity recorded.
Non-goals: no benchmark run; no switch.
Result: Green 2026-11-06 in CM-2 (65f3e27): `tests/rag/test_opensearch_coverage.py` runs section coverage, idempotent re-ingest and replace_existing against the container; every chunk now carries `embedding_id` (`ollama:<model>:<dimension>`) in its stored metadata. Retargeted 2026-10-23 to OpenSearch (ADR-0016). Earlier: Pending. `tests/ingestion/test_section_coverage.py` and `test_ingestion_idempotency.py` pass, but against an in-memory store, not Qdrant. The embedding identifier is recorded in trace metadata only, not in ingest metadata.
```

Exit: Qdrant passes the same contract/coverage/idempotency tests Chroma passes; indexes verified; embeddings matched; nothing above `rag/` changed.

---

## Sprint 4 — Chroma vs Qdrant measured comparison → Gate B (RETIRED 2026-10-10)

> **Retired by owner decision, 2026-10-10.** No Chroma-vs-Qdrant paired comparison will be run. Qdrant stays the default on the operational grounds recorded in ADR-0003; no retrieval-quality claim is made for the backend choice. Gate B closes at S3 exit (SPEC §11.1). The task definitions below are kept as the historical record.

Goal (historical): decide the default backend on evidence and record the decision honestly.

```text
S4-T01 — Generate Qdrant candidate result
Trace: SPEC §10/§11 Gate B; test-plan §6
Scope: run S2-T04 runner under VECTOR_BACKEND=qdrant on the same fixture/method/top_k/embedding; labeled backend=qdrant, same fixture_version.
Tests: result validates; all case_ids; fixture_version == baseline.
Done: comparable Qdrant candidate exists.
Non-goals: no tuning to win.
Result: Retired (owner decision 2026-10-10). Not run.
```

```text
S4-T02 — Paired quality comparison
Trace: SPEC §10/§11 Gate B; retrieval-benchmark.md §8–§9
Scope: strict comparator on Chroma baseline vs Qdrant candidate; report deltas/CI/win-tie-loss/effective-N; state honest expectation: identical embeddings imply quality tie within CI.
Verification: uv run python evaluation/compare_retrieval_results.py evaluation/results/chroma_section_aware_shared.json evaluation/results/qdrant_section_aware_shared.json --strict-case-ids
Done: paired report committed.
Non-goals: no decision yet.
Result: Retired (owner decision 2026-10-10). Not run.
```

```text
S4-T03 — Filtering & operability comparison
Trace: SPEC §6/§11 Gate B; test-plan §5
Scope: filtering correctness/expressiveness; cold/warm + filtered-query latency; operability notes; S5 hybrid/sparse readiness.
Done: operability comparison recorded.
Non-goals: no hybrid implementation.
Result: Retired (owner decision 2026-10-10). Not run.
```

```text
S4-T04 — Gate B decision + ADR-0003 update
Trace: SPEC §11 Gate B/§15
Scope: decide against explicit criteria: no quality regression, filtering ≥ Chroma, latency acceptable, S5-readiness. Record decision and numbers in sprint-review.md and ADR-0003. Honest framing is mandatory: quality tie + better operability/S5-readiness is a valid migration reason, not a fabricated quality win.
Done: Gate B report + operability comparison + recorded decision.
Non-goals: no S5 work; no retro-editing fixture.
Result: Retired (owner decision 2026-10-10). Not run.
```

Exit = Gate B. (Superseded 2026-10-10: Gate B closes at S3 exit.)

---

# HORIZON (S5–S10): restructured 2026-11-01 into one-feature sprints

- **S5 — Retrieval quality diagnostics → hybrid/rerank (Gate C).** Per-query diagnostics and failure taxonomy first; optional section prior, hybrid dense+sparse, and rerank candidates evaluated on same fixture. Unchanged.
- **S6 — Answer quality & eval hardening (Gate D). Steps 4–5.** (a) Eval registry `evaluation/registry/runs/*.json` with lineage keys (SPEC §11.2); hook H5 + CI reject unlineaged results. (b) `eval-replay` CI job on the frozen release, fails on >2σ regression of grounded_claim_rate / citation_coverage. (c) Verifier↔judge agreement on ≥50 claims; κ recorded. (d) Bounded draft→verify→revise loop (max 2), `attempts` in response, paired vs no-loop on the same snapshot/model/temperature. (e) Then LLM-as-judge/RAGAS, then GEPA on holdout. Order inside S6 is fixed: (a)→(b)→(c)→(d)→(e).
- **S7 — Service readiness. Steps 6–8 (local).** (a) Guardrails per SPEC §11.4 with adversarial fixture in CI. (b) API auth + rate limit + idempotency key; JWT issuer allowlist, customer-scoped authorization, `guest` (Turnstile-gated, BFF-minted token, quotas) and `admin` roles, `analyst` defined for the scale tier, per ADR-0010 and its 2026-10-11 amendment (local development issuer). (c) Caching per SPEC §11.3 (v1.4 keys, ADR-0013) with hit/miss metrics and single-flight memo generation. (d) `RunStore` protocol → DynamoDB adapter (conditional-write claim with `attempt` fencing, lazy deadline expiry); `JobQueue` protocol → SQS adapter fed by a DynamoDB Streams dispatcher; LangGraph `DynamoDBSaver` checkpointer; local stand-ins DynamoDB Local and ElasticMQ (ADR-0013 as amended by ADR-0015 on 2026-10-11; was a Postgres `SKIP LOCKED` queue, before that Redis Streams). No Postgres adapter is built; it belongs to the production scale tier. (e) FinBERT off the API event loop: ONNX inside the worker for the demo tier (ADR-0015), with recorded label parity; out-of-process service only in the scale tier; provider circuit breaker (separate from ADR-0009 fallback; Ollama development-only). (f) Fix G13. CI matrix, correlation IDs, structured logs retained from prior S7 scope.
- **S8 — Frontend MVP.** Thin API-driven SPA against the hardened API. Unchanged.
- **S9 — Portfolio polish.** README, measured results, honest limitations, benchmark report, demo, fresh-clone smoke. Unchanged.
- **S10 — Cloud (INTEGRATION_PLAN_v2 Phases 4–5). Step 8 (remote).** ADR-0006 accepted 2026-10-08 with ADR-0010 to ADR-0013; ADR-0015 (2026-10-11) makes the demo tier scale-to-zero; SPEC §3 amended in v1.4. Starts only when the ADR-0006 entry criteria have recorded evidence. Task outline (detailed when S10 opens): S10-T01 Terraform demo tier (Lambda, DynamoDB, SQS, Cloudflare Worker), apply → smoke → destroy twice; S10-T02 deploy pipeline with the eval gate and alias shift; S10-T03 Foundry fallback and model parity (S7-AZURE-FALLBACK); S10-T04 embedding parity gate and Gate C outcome; S10-T05 MWAA Serverless ingestion DAG (monthly + admin trigger); S10-T06 guest mode, Cognito admin and RBAC; S10-T07 observability and budget kill switch; S10-T08 region-loss runbook (documentation only); S10-T09 OpenSearch Serverless store, paired retrieval comparison against the Qdrant baseline and the daily awake cap (ADR-0014, added 2026-10-11); S10-T10 edge signing and cold start: unsigned function-URL requests refused, Worker CPU < 10 ms, first-query latency with and without the warm-up recorded (ADR-0015). See `docs/cloud-deployment.md` §9. Moves to the production scale tier follow the ADR-0015 triggers (`docs/production-scale-architecture.md`); Kafka only for a stated retention or replay requirement. This replaces the earlier second-consumer trigger (review 2026-08-24), which the 2026-09-06 review found insufficient.
- **S11 — Agent architecture revision (ADR-0018, proposed 2026-11-01, renumbered from "ADR-0016" on 2026-11-02; SPEC v1.5 §3.z; test-plan §23; `docs/agent-architecture.md`).** Opens only when the owner accepts ADR-0018, after S7.12 and Gate D; independent of S10 (it changes only the inside of the worker). It refines ADR-0017: each task is behavior-preserving until Gate F, and ADR-0017's three-agent graph stays the default behind a flag until then. Task outline: S11-T01 `SpecialistAgent` protocol and ADR-0017's Researcher split into `market`, `news` and `filings`, with the Analyst and Reviewer becoming `analyst` and `verifier` (sentiment inside `news`, concurrent filing queries; replay registries byte-identical, no new model call); S11-T02 tool gateway with in-process tools, per-specialist allowlists, budgets, C3 cache and the `tool_calls` log in the release manifest (MCP-0); S11-T03 cache stages C7–C9 with metrics; S11-T04 issuer-profile memory (M2) in T3, MEMORY block, verifier rule, replay pins `memory_ref`; episodic memory (M1), the code-computed delta and `DELETE /memory` only if the owner reverses ADR-0017 §1; S11-T05 MCP consumption behind flags (`news-search`, then the `sec-edgar` wrapper) with paired evidence-equality checks and breaker and schema-drift drills; S11-T06 `/mcp` exposure; S11-T07 Gate F paired comparison (single-axis rule) and the game days. New dependency (MCP SDK) enters at S11-T02 via `uv add` with an advisory check.

| Earlier item | Feature sprint(s) |
| --- | --- |
| S5 diagnostics, hybrid dense+sparse, rerank, section prior | S5.1 diagnostics · S5.2 hybrid · S5.3 rerank (section prior folded into S5.1 as a diagnosed candidate) |
| S6 (a) eval registry · (b) eval-replay CI · (c) verifier↔judge κ · (d) bounded repair loop · (e) judge/RAGAS, then GEPA | S6.1 · S6.2 · S6.3 · S6.4 (adds the ADR-0017 early stop) · S6.5, S6.6 |
| S7 (a) guardrails · (b) auth, rate limit, guest and admin roles | S7.1 · S7.2 |
| S7 (c) caching | S7.9 (adds the ADR-0017 T3 write policy) |
| S7 (d) `RunStore` / `JobQueue` / `DynamoDBSaver` | S7.7 (job store and queue) · S7.8 (fenced checkpointer and journal on DynamoDB) |
| S7 (e) FinBERT off the loop · provider circuit breaker | S7.11 · S7.10 (adds per-run model pinning) |
| S7 (f) G13 | CM-6a (already scheduled with the model-plane close-out) |
| — (new, ADR-0017, 2026-10-31) | S7.3 checkpoints and resume · S7.4 effect journal and pinning · S7.5 error taxonomy · S7.6 three-agent split · S7.12 execution observability · S10-T11 deployed resume drill |
| S8 frontend MVP | S8.1 live run stream · S8.2 guest mode |
| S9 portfolio polish | S9.1 measured results · S9.2 fresh-clone smoke and demo |
| S10-T01 … S10-T10 | Unchanged ids, one feature each (cloud-deployment.md §9) |

### Authorized REST/tracing sequencing exception — 2026-09-21

The user authorized the focused S7 API authorization, rate limits, async idempotency,
REST/error contracts and provider-aware health slice, plus S2-T00d response status
corrections and request correlation. See `tasks/rest-tracing-plan.md`. This does
not open caching, queue replacement, repair loops, or unrelated sprint work.
The service remains a single process with its existing file-backed run store.


### Authorized provider and missing-filing slice (2026-09-29)

User authorizes PROVIDER-FALLBACK / FILING-DEGRADATION before remaining S7 gates, traced to SPEC §4 runtime provider policy, §8.5, test-plan §16 and ADR-0009. Implement Bedrock → configured personal Anthropic → configured personal OpenAI, bounded attempts and native traces; explicit Ollama development mode. Missing filings preserve usable evidence and memo with evidence_missing status. Azure invocation remains deferred as S7-AZURE-FALLBACK until its deployment/model configuration is specified. Other sequencing and benchmark gates stay unchanged.

Result as of 2026-09-30: implementation in frontend-impl, focused provider checks 26 passed, filing checks 14 passed, frontend 10 tests plus type/lint/build passed, Compose 4 cases passed. Final backend suite: 344 passed, 1 failed, 2 skipped; residual tracing assertion expects the old draft-span model label. Full lint, mypy app/evaluation and four Doc Sync checks pass. Overall verification remains open at the user's two-correction limit. Changes are uncommitted; no deployment/live-inference success is claimed. Azure remains deferred. Detailed commands and findings belong to tasks/todo.md.

### Authorized ingest-on-miss slice (2026-10-10)

User authorizes INGEST-ON-MISS after the Qdrant volume was found recreated empty (2026-10-08) and every analysis reported filings unavailable. Traced to SPEC §4 ingest on miss and unavailable filings, §8.2, §8.5 and test-plan §16. Analysis ingests a ticker's latest 10-K once when zero chunks are indexed for it, off the event loop and under a time bound; an ingestion failure is reported as filing_ingestion_failed, distinct from an absent 10-K.

Result as of 2026-10-10: implementation on branch auto-ingest-on-miss. Full suite 371 passed, 2 skipped; ruff, mypy and the four governance checks pass. Live check (first analysis of an unindexed ticker ingests and returns filing chunks; a second run does not ingest; a ticker with no 10-K still reports filings unavailable) not yet run.

### Ledger reconciliation and S4 retirement (2026-10-10)

User decision, 2026-10-10: the Chroma-vs-Qdrant paired comparison (S4) is retired; Gate B closes at S3 exit. Recorded in SPEC §11.1 and §12, ADR-0003 (addendum) and `docs/retrieval-benchmark.md` §9. In the same pass the S2-T00b, S2-T00d and S3 results above were reconciled against the code and `tasks/todo.md`; each result names its evidence and what is still open.

Remaining order after reconciliation: S2-T00c (latency artifacts, concurrency probe) → S2-T00b remainder → S2-T00d remainder (G10) → S2-T01…T06 (Gate A) → S3-T04, S3-T05 (Gate B) → S5 including the ADR-0011 embedding bake-off (Gate C) → S6 (a)…(e) (Gate D) → S7 (a)…(f) → S9 → S10 entry evidence → S10-T01…T08.

Update 2026-10-22: Gate A closed (S2-T00c, S2-T00d, S2-T01…T06 recorded above; baseline commit c1fcd21). Remaining order: S3-T04, S3-T05 (Gate B) → S5 including the ADR-0011 embedding bake-off (Gate C) → S6 (a)…(e) (Gate D) → S7 (a)…(f) → S9 → S10 entry evidence → S10-T01…T08. The authorized EMBED-ARMS plumbing slice (2026-10-11) is still pending; with the fixture frozen, arms may now be measured in S5. Open for EMBED-ARMS: the baseline records the embedding as `qwen3-embedding:4b` while the fixture records `ollama:qwen3-embedding:4b:2560`; align the identifier there and regenerate the baseline (a baseline regeneration, not a fixture change).


### Authorized embedding-provider plumbing slice (2026-10-11)

User decision, 2026-10-11: the Gate C embedding bake-off has four arms (`qwen3-embedding:4b`, Cohere Embed v4, Titan Text Embeddings V2, `text-embedding-3-small`; ADR-0011 amendment, `docs/retrieval-benchmark.md` §9a). The plumbing is built now, ahead of S5: an embedding provider switch behind the existing retrieval boundary, one index per arm built from the same chunks, and the embedding identifier written to ingest metadata (also the open part of S3-T04). Measurement stays in S5, after Gate A; no arm is measured or cited before the fixture is frozen. Tests: test-plan §21. The remaining order above is otherwise unchanged.

Result: Pending.

### Authorized Azure provider slice (2026-10-17)

User decision, 2026-10-17 (ADR-0009 amendment 2026-10-17; SPEC §4 runtime provider policy; test-plan §16 rows dated 2026-10-17): the Azure tier, previously S10-T03 / S7-AZURE-FALLBACK, is pulled forward and is the next task. Scope: Foundry DeepSeek-V4-Flash as the first draft attempt when configured, Bedrock `deepseek.v3.2` next, personal fallbacks unchanged; the evaluation judge ladder GPT-5.6 Luna on Bedrock (`bedrock-mantle`) → Foundry; Compose pass-through for the new variable. Out of scope: the STS-to-Entra keyless sign-in (stays in S10), circuit breakers (S7 (e)), parity and fallback-quality measurement (S10 / S6). The in-flight EMBED-ARMS plumbing slice pauses and resumes after this slice. Ordering of the embedding bake-off and the DeepSeek-V4-Flash memo baseline after this slice: pending the owner's answer (Gate A is still open, and the bake-off is gated on it above).

### Owner-requested architecture revision (2026-11-01)

The owner asked for a system architecture revision adding a memory architecture, a five-specialist split of the analysis task, an MCP integration plan, asynchronous agents and tools for concurrent requests at scale, caching at several stages to reduce model and tool calls, and hardening of agent-specific non-functional requirements. Recorded as ADR-0018 (proposed), SPEC v1.5 (§§0, 2, 3.z, 5, 8.1a, 11.1 Gate F, 11.3, 12, 16), test-plan §23, the S11 horizon entry above and the design companion `docs/agent-architecture.md`. Documentation only; no runtime code changed.

Result as of 2026-11-01: documents written; four governance checks pass. Diagrams added as Archify sources with HTML, SVG and PNG renders under `docs/System-design/agents/`, `docs/svg/agents/` and `docs/png/agents/`.

Reconciliation 2026-11-02 (doc review after merging main): the record was first numbered "ADR-0016", which the accepted cloud data and model plane record already held; it is now ADR-0018. It was drafted without ADR-0017's text, which landed on main the same day; it is now framed as a refinement of ADR-0017 (five specialists split the Researcher; resume uses ADR-0017's checkpoints and journal; Gate F compares against ADR-0017's three-agent graph). The merge into main had replaced ADR-0017's test-plan §22 with this record's cases; §22 is restored verbatim from commit `3b7b379` and this record's cases are §23. SPEC §§8.1b, 8.6–8.8 and gaps G14–G15, which the S7.x cards cite, are recorded from ADR-0017. Diagrams re-rendered with the ADR-0018 title; they draw the proposal, not the ADR-0017 baseline.

Pending owner decisions: (1) episodic memory — ADR-0017 §1 excludes it, ADR-0018 D2 proposes it; (2) acceptance of ADR-0018 as a refinement of ADR-0017.

Result as of 2026-10-18: implemented on branch feat/embedding-provider-switch. Offline: 492 passed, 2 skipped; ruff, mypy and the four governance checks pass; test-plan §16 rows dated 2026-10-17 (Judge ladder row revised 2026-10-18) covered offline by `tests/unit/test_provider_azure.py` and `tests/unit/test_judge_ladder.py`, including a structured verdict through the DeepEval adapter that falls back to Foundry. Live Azure draft run not yet done; Bedrock Luna is blocked by a zero account quota on `bedrock-runtime`, so only the Foundry judge is live-usable.

### Authorized cloud data and model plane migration (2026-10-23)

Owner decisions, 2026-10-23 (ADR-0016): migrate the data and model plane to managed services first, then benchmark there; Qdrant and Chroma are retired and OpenSearch is the only store (local container for development and tests, Serverless in the cloud); all three cloud embedding arms are measured on OpenSearch Serverless; the backend comparison copies the Gate A vectors instead of re-embedding; sentiment moves to a hosted model. Scope is the data and model plane only; the S10 demo-tier runtime and the S7 seams keep the ADR-0006 entry criteria. Every apply that creates billable resources needs the owner's approval first.

Order (one `task/<id>` branch each; test-author → implementer loop for code):

1. CM-0 — ADR-0016 and doc amendments (SPEC §§1, 3, 5, 6, 11.1, 12, 16; ADR-0003/0006/0007/0011/0014; test-plan §§4, 5, 20; retrieval-benchmark §§8, 9.1, 9a; this note).
2. VERIFY-SCALE — bug fix, in parallel with CM-1…CM-3: the runtime verifier misses 10-K figures stated in million/billion against tables in thousands (156 of 243 unsupported claims across the nine `degraded` runs; diagnosis in `tasks/todo.md` DEGRADED-DIAG). Owner rule: scale-aware matching for SEC filing evidence only, rounded at the claim's stated precision; a wrong figure must still fail. Trace: SPEC §8.5, test-plan §7 (row added first). Must land before any eval baseline is recorded.
3. CM-1 — cost controls: AWS Budgets ($10, $25, $100 action), tags, teardown command; official OpenSearch Serverless prices and idle timeout recorded in ADR-0014.
4. CM-2 — `OpenSearchVectorStore`; test-plan §§4, 5 against a local container (S3-T04, S3-T05 retargeted); embedding identifier in ingest metadata.
5. CM-3 — Terraform for the Serverless collection group; apply → smoke → destroy once.
6. CM-4 — copy the Gate A vectors; backend-only comparison (retrieval-benchmark §9.1) → Gate B.
7. CM-5 — cloud embedding provider switch (EMBED-ARMS, test-plan §21) and the three-arm bake-off (retrieval-benchmark §9a) → Gate C embedding outcome.
8. CM-6 — model-plane close-out (Azure draft live check, Bedrock Luna quota, G13). CM-6b — hosted sentiment with its own ADR.
9. CM-7 — remove Qdrant and Chroma; CLAUDE.md/AGENTS.md §8 and `check_doc_sync.py` updated together.

Reordered 2026-11-07 (owner: the project is cloud-only; no local CUDA, torch or Ollama): after CM-2, run CM-6b (sentiment on Foundry Claude Sonnet 5.5) → CM-6d (rerank on Foundry Cohere Rerank v5 fast; torch and every CUDA package removed) → CM-5a (cloud embedding provider switch, build only) → CM-3 → CM-4 → CM-5b → CM-6e (Ollama removed). Ollama stays on the host until CM-4 and CM-5b have run, because both search the copied Qwen3 index and embed each benchmark query with `qwen3-embedding:4b`. CM-6b keeps its one FinBERT label-parity run before FinBERT is deleted.

After CM-7: S5 (hybrid/rerank on OpenSearch), S6 (a)…(e), S7 (a)…(f), S9, then the S10 runtime tasks. Superseded 2026-11-01: CM-5 and CM-6 are split into one-feature sprints and the order after CM-7 is the FORWARD PLAN below.

Result: Pending.

---

# FORWARD PLAN: one-feature sprints (from 2026-11-01)

Owner instruction, 2026-11-01: each sprint ships one feature, implemented and tested (unit, integration, regression and the other layers it touches). This replaces the one-line HORIZON for S5–S10 and the post-CM-7 order of the 2026-10-23 note. Architecture inputs: SPEC v1.5, ADR-0017 (Researcher / Analyst / Reviewer agents, three-tier memory, crash resume), ADR-0013 to ADR-0016.

## Sprint contract

**One feature.** A feature is one capability a user, operator or evaluator can observe, with one acceptance oracle (its test-plan rows). If a card needs two oracles, it is two sprints. Refactors ride inside the feature that needs them; they are not sprints.

**Loop per sprint** (unchanged from CLAUDE.md / AGENTS.md): plan in `tasks/todo.md` → test-plan rows exist first (add them in the same branch if missing) → `test(<id>): red` commit with failing tests → implementation → green → regression suites → `Result:` recorded here with commit SHA → `tasks/todo.md` review. One `task/<id>` branch per sprint.

**Test layers** are defined in test-plan "Test layers": U unit · C contract · I integration (local stand-ins) · F fault injection · E2E · Q paired quality · P performance · L live · R regression. A card lists only the layers it touches; every listed layer must be green.

**Standing regression suites.** A sprint closes only when every suite that applies to it is green, run on the sprint's final commit:

| Suite | What | Command (repo root) | Applies |
| --- | --- | --- | --- |
| R1 Offline | Lint, types and the full offline test suite (U, C, I with stand-ins) | `uv run ruff check . && uv run mypy . && uv run pytest` | Every sprint |
| R2 Governance | The four doc checks | `for c in no_scope_residue sprint_map doc_sync test_hygiene; do python scripts/ci/check_$c.py; done` | Every sprint |
| R3 Replay | Zero-network replay; two replays of `alpha-evidence:0.1.0` give the same `snapshot_hash` and citation registry | `uv run pytest -m replay` (socket guard) | Every sprint |
| R4 Retrieval oracle | Gate A fixture against the current baseline; zero deltas unless retrieval is the sprint's axis | `uv run python evaluation/run_shared_retrieval_benchmark.py …` then `compare_retrieval_results.py --strict-case-ids` | Sprints that change retrieval, ingestion, embeddings or the store, and every gate exit |
| R5 Answer quality | `eval-replay` gate: `grounded_claim_rate` and `citation_coverage` within 2σ of the approved baseline | CI job `eval-replay` | Every sprint from S6.2 |
| R6 Resume matrix | `FAIL_AT` at every agent and node, `pre` and `post` effect: same memo, zero duplicate provider calls after `post` | `uv run pytest -m fault` | Every sprint from S7.3 (from S7.4 for the zero-duplicate assertion) |

**Done, for every card:** the card's tests by layer green; applicable R-suites green; SPEC, test-plan and ADR text match the code; `Result:` names the commit, test counts by layer and suite outputs; nothing claimed that was not run. Billable applies need the owner's approval first; live (L) results record cost and are never approved baselines.

**Order.** Linear: VERIFY-SCALE → CM-1 … CM-7 → S5.1 … S5.3 → S6.1 … S6.6 → S7.1 … S7.12 → S8.1, S8.2 → S9.1, S9.2 → S10 entry evidence → S10-T01 … S10-T11. A sprint opens only when its predecessor has a `Result:`; reordering needs an owner decision recorded here. S11 (ADR-0018, proposed) sits outside this order until the owner accepts ADR-0018; it cannot open before S7.12 and Gate D.

## Carried over from the 2026-10-23 cloud data and model plane slice

Same scope as the numbered list in that note; reformatted as feature cards, with CM-5 and CM-6 split so each sprint is one feature.

```text
VERIFY-SCALE — Scale-aware verification of filing figures (bug fix)
Feature: the runtime verifier supports "$356.7 million" against a 10-K table in thousands, and still fails a wrong figure.
Trace: SPEC §8.5; test-plan §7 VERIFY-SCALE and VERIFY-UNITS rows; tasks/todo.md DEGRADED-DIAG
Depends on: —
Scope: scale-aware matching for sec_filing body text only, at the claim's stated precision; scale words part of the figure for every source.
Non-goals: any other verifier change; prompt changes.
Tests:  U every §7 VERIFY-SCALE / VERIFY-UNITS row · E2E the nine degraded runs replayed: unsupported-claim count recorded before/after
Regression: R1 R2 R3
Done: all §7 rows green; existing supported claims unchanged; lands before any eval baseline is recorded.
Result: Pending.
```

```text
CM-1 — Cost controls
Feature: no cloud spend can exceed the owner's ceiling unnoticed.
Trace: ADR-0016; cloud-deployment §7 budget controls
Scope: AWS Budgets ($10 and $25 alerts, $100 action), resource tags, teardown command; official OpenSearch Serverless prices and idle timeout recorded in ADR-0014.
Tests:  L a test budget threshold fires its alert; teardown leaves only the evidence bucket
Regression: R1 R2
Done: alerts received; prices cited with date.
Result: DONE (2026-11-06), commit 8d6ed8e. `terraform apply` added 5 resources as the deployer role; re-plan exit 0 (no drift). Budget `alpha-analyst-monthly`: $100 limit, alerts at $10/$25/$100 actual and $100 forecast; owner confirmed the alert email arrived. Kill switch STANDBY, automatic at $100, on the runtime role and both model-calling users. Prices cited in ADR-0014 (2026-10-23). Teardown command in infra/terraform/README.md; "leaves only the evidence bucket" is exercised at CM-3, when a second stack exists. Open: the five `Resource: "*"` statements in the manually created IAM policies stay for now by owner decision; constrain them and add the ADR note after full deployment.
```

```text
CM-2 — OpenSearch store behind the retrieval contract
Feature: VECTOR_BACKEND=opensearch serves retrieval through the unchanged RetrievalStore contract.
Trace: SPEC §6, §11.1 Gate B; test-plan §4, §5, §21 "Embedding identifier recorded"; S3-T04, S3-T05 retargeted
Scope: OpenSearchVectorStore; idempotent mapping; chunk_id as document id; embedding identifier in ingest metadata.
Tests:  U config selection, invalid backend · C test-plan §4 suite against a local container · I §5 rows incl. coverage and idempotent re-ingest
Regression: R1 R2 R3 R4
Done: §4 and §5 green against the container. Result: Done criterion met 2026-11-06, commit 65f3e27 (tests 72d1db5): `tests/rag` + factory 45 passed against OpenSearch 2.19.2. R1 (710 passed, 2 skipped; ruff; mypy) R2 R3 (87 passed) green. R4 not run — pending owner: the Gate A Qdrant volume `qdrant-financial-analyst-agent-system` is not on the Docker Desktop engine and the native engine is not running. Default backend still qdrant. Lucene HNSW cosinesimil, filters inside k-NN; Serverless-compatible APIs only. CM-3 checks: refresh=wait_for and delete_by_ticker loop under the 5 s refresh; NEXTGEN 32x compression.
```

```text
CM-3 — OpenSearch Serverless collection as code
Feature: the Serverless collection group is created and destroyed from code.
Trace: ADR-0014, ADR-0016; cloud-deployment §11
Scope: Terraform for the collection group, access and network policies; apply → smoke → destroy once. Amended 2026-11-07 (owner): the existing NEXTGEN collection `zcech-banking-emb` is used; the stack owns only the data access policy, so create/destroy applies to that policy, not the collection.
Tests:  L apply, CM-2 smoke queries against the collection, destroy; idle timeout and cold start recorded
Regression: R1 R2
Done: one clean cycle recorded with cost. Result: DONE 2026-11-08. Cycle run 2026-11-07/08 (commits 789bb27, cf412c8, d11d1c1, 4766792, cf95663, cbad608, e6919e0, 8076fdd); destroy→apply rehearsal 2026-11-08 09:01–09:15Z as the deployer role, owner-approved. Destroy removed 1 resource; with no policy naming the user, an index create returned 403. Apply imported, added and changed 1 resource each. Afterwards a `sec_` index could be created and deleted, a non-`sec_` create returned 403, and the re-plan was clean. Both data policies are narrowed to `index/zcech-banking-emb/sec_*`. The runtime role's `opensearch-access-policy` is fixed (it named `zech-banking-emb`) and is now in the stack. The runtime role's own access was not exercised, because the role cannot be assumed from here. Scope change (owner 2026-11-07): the project uses the owner's existing NEXTGEN collection `zcech-banking-emb` (group max lowered to 2 OCU) instead of creating one, so the stack reads the collection and group as data sources with a cost-guard postcondition and owns only the data access policy for IAM user `financial_analyst_agent`; `destroy` removes that policy, never the collection. Apply 1 added, re-plan clean. Store fixed for Serverless (SigV4; no `engine`/`mode`; `compression_level` 32x default, 1x for tests and CM-4; `rescore`; writes without refresh; one-pass delete; index health probe; 30 s timeout and one retry under SigV4). Smoke through the store alone: writes, filtered k-NN, count, scan, delete, compression-mismatch refusal all correct. Idle → zero about 7–10 min; cold start 13–20 s (write and search pools separately); write visibility 0.2–5.6 s. Cost measured from CloudWatch: 1.20 search + 0.50 indexing OCU-h = 1.70 OCU-h × $0.24 ≈ $0.41 (billed figure from Cost Explorer to follow). R1 888 passed, 2 skipped; ruff; mypy (2 known probe errors); R2 4/4. Details: ADR-0014 "Verified 2026-11-08 (CM-3)".
How CM-4 will differ from Gate A, and why: (1) the index is faiss HNSW on Serverless instead of Qdrant HNSW — the intended single axis; (2) precision is equal: the CM-4 index is built at 1x, like Qdrant, because 32x would add a second axis (retrieval-benchmark §9.1); production 32x is compared to 1x separately; (3) Serverless splits an index into 2 shards and merges each shard's top hits, where Qdrant searched one collection, so chunks with exactly equal scores can come back in a different order (seen in the smoke) — a rank metric can move by tie order alone, rarely with real embeddings; (4) distance means the same on both (1 − cosine, scale verified); (5) write lag and cold start change timing, not results, provided the run starts only after `_count` equals the source count (the copy step must poll; `evaluation/build_embedding_arm.py` counts immediately and would need that poll on Serverless); (6) filters apply inside k-NN on both.
```

```text
CM-4 — Backend-only comparison → Gate B
Feature: OpenSearch Serverless is adopted only if it matches the Gate A baseline with the vectors held constant.
Trace: SPEC §11.1 Gate B; retrieval-benchmark §9.1; test-plan §20 "Retrieval adoption gate"
Scope: copy the Gate A vectors unchanged; run the anchored benchmark; paired comparison, backend the only axis.
Tests:  Q paired result within §9 tolerance · L collection cost recorded
Regression: R1 R2 R4
Done: Gate B recorded; on failure, stop for an owner decision. Result: DONE 2026-11-08, Gate B PASS. Rows e64a215; red d56fe32 (15 failed); feat 6caafb7 (`evaluation/copy_gate_vectors.py`, runner `--backend opensearch`; tests unchanged since red); result 4d6ca3b. Owner approvals 2026-11-08: tolerance −0.03 confirmed, one copy + one run, keep the index. `copy_gate_vectors` wrote the 532 Gate A points (chunk id, text, metadata, vector; no embedding call) into Serverless index `sec_filings_bench_alpha_evidence_0_1_0` at 1x in 46 s; `_count` 532 = source and every gold anchor matched. Paired comparison (`--strict-case-ids`, shared 75): every metric ties on every case at K=5 and K=10; CI [+0, +0] for recall@10 0.6022 and section_recall@10 0.7733. Item level: 750/750 ranked chunk IDs identical in order, scores within 3.6e-7, no tie reorders. Warm p50 133 → 321 ms (cloud round trip; not gated). Cost (CloudWatch, financial-agent): search 2 OCU 18:29–18:39Z = 0.37 OCU-h, indexing 1 OCU 18:28–18:37Z = 0.17 OCU-h; 0.53 OCU-h × $0.24 ≈ $0.13; both pools 0 by 18:40Z. Index kept as the CM-5b source. Details: retrieval-benchmark §9.1 Result. Agents: `.claude/agents/` is missing, so general-purpose Sonnet (high effort) stood in for test-author and implementer.
```

```text
CM-5a — Cloud embedding provider switch (EMBED-ARMS plumbing)
Feature: any of the four embedding arms is selected by settings alone, one index per arm.
Trace: ADR-0011 amendment; test-plan §21
Scope: provider switch behind the retrieval boundary; per-arm index naming; dimension check; arm built from the same chunks.
Tests:  U every §21 row with faithful fake transports · C §4 suite on one cloud-arm index (container) · I arm build copies chunk ids unchanged
Regression: R1 R2 R3 R4
Done: §21 green; identifier mismatch noted on 2026-10-22 aligned and the baseline regenerated. Result: DONE (2026-11-07). Red 101e8dd and 53706fe; feature 47e9f63, d5213a7 and 6951362; baseline acce359. EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBEDDING_DIMENSIONS select the arm (default ollama qwen3-embedding:4b at 2560). Each arm gets its own OpenSearch index named from `embedding_identifier()`. An index with another vector size is refused before any write or search. `evaluation/build_embedding_arm.py` copies the baseline chunks into an arm's index. The Bedrock arm authenticates with the settings bearer token. Benchmark provenance now records `ollama:qwen3-embedding:4b:2560`. The Gate A collection had been lost, so it was rebuilt from alpha-evidence:0.1.0 (532 points) and the baseline regenerated: all 75 cases return the same ranked chunk ids and the metrics are exactly equal. pytest 834 passed, 2 skipped; ruff; mypy apart from the git-ignored probe file; governance 4/4. Replay under `unshare -n` is blocked on this host (AppArmor user-namespace restriction); the replay tests pass in the full suite. The default stays ollama; arms are measured in CM-5b.
```

```text
CM-5b — Embedding bake-off → Gate C embedding outcome
Feature: the embedding model is chosen on the frozen fixture with the embedding as the only axis.
Trace: retrieval-benchmark §9a; ADR-0011; SPEC §11.1 Gate C
Scope: three cloud arms on OpenSearch Serverless against the Gate A arm.
Tests:  Q one paired comparison per arm, single axis · L cost per arm recorded
Regression: R1 R2 R4
Done: outcome recorded in ADR-0011. Result: DONE 2026-11-08 — no arm qualified under §9a; owner override adopts Cohere Embed v4 (1024 d). Test-plan §21 row amended 08104eb (arm build waits for Serverless write visibility); red c8e593f; feat 971961a (tests unchanged since red); prices and tie-break declared before the run 1d15fba; results 8dd1d4a. Three arms built from the CM-4 index (532 chunks each, per-ticker counts equal, `embedding_id` verified), Serverless 1x, run 20:33–20:42Z. Paired vs Qwen3 at K=10: Cohere recall +0.000 [−0.060, +0.060], section_recall +0.000 [−0.040, +0.040]; Titan recall −0.053 [−0.127, +0.020]; Azure recall −0.033 [−0.107, +0.033]. Cost 0.92 OCU-h ≈ $0.22 + ≈ $0.02 tokens. Arm indexes kept. Details: ADR-0011 "Outcome 2026-11-08". Amended 2026-11-08 (owner request): a fourth arm, Cohere Embed v5 fast on Foundry (Cohere-native route, one-off harness adapter), qualifies under §9a — vs Qwen3 recall@10 +0.067 [+0.007, +0.133], section_recall +0.000 [+0, +0]; vs Cohere v4 recall +0.067 [+0.007, +0.133]; latency not measured (HTTP 429 throttling). The owner adopts it as the default, superseding the v4 override; results 19c634a; details ADR-0011 "Outcome 2026-11-08 (amended)". CM-6e unblocked: default = Cohere Embed v5 fast.
```

```text
CM-6a — Provider-aware health (G13)
Feature: /health reports the configured model plane, not Ollama, as the chat dependency.
Trace: SPEC §12 G13; test-plan §1 REST hardening (health rows)
Tests:  U health per provider configuration · I health with Ollama down and Bedrock configured → not degraded
Regression: R1 R2 R3
Done: G13 closed. Result: Pending.
```

```text
CM-6b — Hosted sentiment
Feature: sentiment scores come from a hosted model with recorded label parity, off the event loop. Owner choice 2026-11-07: Claude Sonnet 5.5 on Microsoft Foundry, deployment `sonnet-5-5-sentiment` (FinBERT is not offered on AWS or Azure); FinBERT, transformers and FINBERT_* settings are removed after the parity run.
Trace: ADR-0016 (own ADR required first); test-plan §10 sentiment rows
Tests:  U adapter with a faithful fake · Q label parity against FinBERT on the frozen news set · P event loop not blocked
Regression: R1 R2 R3
Done: ADR accepted; parity recorded. Result: Implemented 2026-11-07 (red 8d2a235; tests unchanged since). Sentiment runs on Foundry `sonnet-5-5-sentiment`; FinBERT code, FINBERT_* settings and the startup preload removed (torch/transformers leave in CM-6d). R1: 731 passed, 2 skipped; ruff; mypy (only the git-ignored probe file). R2 4/4. R3 replay 87 passed. Parity recorded in ADR-0019: 16/30 = 53.3% agreement with the frozen FinBERT labels; owner accepted the rate 2026-11-07 and ADR-0019 is Accepted. DONE.
```

```text
CM-6d — Hosted rerank; torch and CUDA removed
Feature: reranked hybrid retrieval scores pairs with Cohere Rerank v5 fast on Microsoft Foundry (deployment `cohere-rerank-v5-fast`) instead of a local CrossEncoder; no torch or CUDA package remains.
Trace: ADR-0011 D12 (amended 2026-11-07: was Bedrock Cohere Rerank 3.5); retrieval-benchmark rerank axis (amended); test-plan §25 (new; no rerank rows existed before); owner decision 2026-11-07
Scope: rerank adapter behind the retrieval boundary with timeout and bounded retries; rerank stays off by default until Gate C shows lift; remove sentence-transformers, torch, transformers, docker-compose.gpu.yml, the model-cache volume and CUDA notes in Dockerfile/Compose.
Tests:  U adapter with a faithful fake transport · I reranked-hybrid eval mode end to end on the fake · R1 dependency check (no torch/nvidia in uv.lock)
Regression: R1 R2 R3
Done: uv.lock has no torch or nvidia-*/triton package; image builds without them. Result: Implemented 2026-11-07 (red bd824c5; tests unchanged since). Rerank calls Foundry `cohere-rerank-v5-fast` through `FoundryCohereReranker` (httpx, 30 s timeout, 3 attempts); sentence-transformers, torch and transformers removed: 27 locked packages gone (torch, transformers, sentence-transformers, triton, 15 nvidia-*, 3 cuda-*, plus safetensors, scikit-learn, joblib, threadpoolctl, cloudpickle), none added; pip-audit shows no new advisory. docker-compose.gpu.yml deleted; model-cache mount and HF/TORCH cache env removed (the Docker volume itself is kept). R1: 756 passed, 2 skipped; ruff; mypy (only the git-ignored probe file). R2 4/4. R3 replay 87 passed. Image build not run in this task. Open: the RAGAS `huggingface` embeddings option in evaluation/ now needs a package the project no longer has (owner decision).
```

```text
CM-6e — Ollama removed
Feature: no code, config, Compose entry or health check depends on Ollama.
Trace: ADR-0016 decision 7; SPEC §5 provider policy; owner decision 2026-11-07; ADR-0011 "Outcome 2026-11-08 (amended)"
Blocked until: CM-4 and CM-5b have run (both embed benchmark queries with qwen3 on the copied index).
Scope: remove langchain-ollama, OllamaSettings, the Ollama startup and `/health` checks (with CM-6a), OLLAMA_* Compose variables; default embedding = the Gate C winner, Cohere Embed v5 fast on Foundry. It is served only on Foundry's Cohere-native `/providers/cohere/v2/embed` route, so CM-6e adds that embedding arm and re-runs the CM-5b anchored benchmark through it to confirm the amended Gate C result.
Tests:  U health and config without Ollama · I ingestion and search on the winning cloud arm (fake transport)
Regression: R1 R2 R3 R4
Done: no `ollama` reference in app/, dataops/, evaluation/, Compose or pyproject. Result: Pending.
```

```text
CM-6c — Model ladder live qualification
Feature: every configured draft and judge tier is proven invocable from this account, or removed from the ladder.
Trace: SPEC §4; ADR-0009 amendment 2026-10-17; test-plan §16 Azure and judge rows
Blocked: the configured first draft tier (Foundry DeepSeek-V4-Flash) has no Global Standard quota (increase rejected 2026-10-17) and GPT-5.6 Luna is not available to this AWS account on Bedrock (2026-10-18). Needs an owner decision on the ladder before it opens.
Tests:  L one bounded call per tier, cost recorded · U ladder skips an unavailable tier
Regression: R1 R2 R3
Done: ladder in SPEC §4 equals the tiers proven live. Result: Blocked (owner decision).
```

```text
CM-7 — Remove Qdrant and Chroma
Feature: OpenSearch is the only store in code, compose and docs.
Trace: ADR-0016; CLAUDE.md/AGENTS.md §8; scripts/ci/check_doc_sync.py
Tests:  U factory rejects qdrant/chroma · C §4 suite unchanged · R2 doc sync updated together
Regression: R1 R2 R3 R4
Done: no import of qdrant_client or chromadb remains. Result: Pending.
```

## S5 — Retrieval quality → Gate C

```text
S5.1 — Retrieval diagnostics and failure taxonomy
Feature: every Gate A miss is explained by a named failure class, so S5.2 and S5.3 test a diagnosed cause.
Trace: SPEC §12 S5; retrieval-benchmark §§7–8; test-plan §6
Scope: per-query report (rank of each gold anchor, section hit, lexical overlap, chunk boundary); taxonomy (vocabulary gap, section miss, boundary split, ranking); section prior evaluated here as a candidate.
Tests:  U classifier on hand-built cases · I report over the 25 baseline misses
Regression: R1 R2 R3 R4
Done: every miss classified; candidate list ranked by misses addressed. Result: Pending.
```

```text
S5.2 — Hybrid dense + BM25 retrieval
Feature: retrieval combines dense and keyword scores on OpenSearch.
Trace: SPEC §11.1 Gate C; retrieval-benchmark §8; test-plan §6 comparator
Scope: OpenSearch hybrid search pipeline; method label `hybrid`; same fixture, K and embedding.
Tests:  U fusion normalisation · C §4 suite with method=hybrid · Q paired vs Gate A baseline, method the only axis · P cold/warm latency
Regression: R1 R2 R3 R4
Done: adopt only if recall@10 does not drop beyond tolerance; decision recorded. Result: Pending.
```

```text
S5.3 — Rerank
Feature: an optional reranker reorders the top candidates.
Trace: ADR-0011 (Bedrock Cohere Rerank 3.5); Gate C
Tests:  U rerank adapter with a faithful fake · Q paired vs the S5.2 winner · P added latency and cost per query
Regression: R1 R2 R3 R4
Done: Gate C recorded; rerank off unless it shows lift within the cost budget. Result: Pending.
```

## S6 — Answer quality → Gate D

```text
S6.1 — Eval registry with lineage
Feature: every eval run is an append-only record with full lineage, and unlineaged results are refused.
Trace: SPEC §11.2; test-plan §14; hook H5
Tests:  U registry writer rejects each missing key; live runs cannot be approved · C hook H5 and the CI check agree
Regression: R1 R2 R3
Result: Pending.
```

```text
S6.2 — eval-replay CI regression gate
Feature: a PR that lowers grounding on frozen evidence beyond 2σ fails CI.
Trace: SPEC §11.2; test-plan §14 "Regression gate"
Tests:  I synthetic run 2.5σ below baseline fails; at baseline passes · E2E gate runs with network disabled
Regression: R1 R2 R3 (R5 starts here)
Done: approved baseline recorded after VERIFY-SCALE. Result: Pending.
```

```text
S6.3 — Verifier ↔ judge agreement
Feature: the heuristic grounding rate is quoted only with its measured agreement against a judge.
Trace: SPEC §11.2; test-plan §14, §20 "Judge adoption"
Tests:  Q κ on ≥ 50 owner-reviewed claims · U agreement report schema
Regression: R1 R2 R3 R5
Result: Pending (judge tier depends on CM-6c).
```

```text
S6.4 — Bounded revise loop with early stop
Feature: the Reviewer returns failed claims for at most two revisions, stopping early when a revision does not improve grounding.
Trace: SPEC §8.1, §8.6; ADR-0017 §2; test-plan §11 repair-loop rows
State: implemented and tested locally, not pushed (owner, 2026-10-31). This sprint lands it and adds the early stop if missing.
Tests:  U bound, early stop, not-converged best draft, draft count · E2E degraded on non-convergence · Q paired loop vs no-loop on the frozen release, loop the only axis
Regression: R1 R2 R3 R5
Done: §11 rows green; paired result recorded with per-case regressions listed. Result: Pending.
```

```text
S6.5 — Judge-based answer-quality suite (LLM-as-judge / RAGAS)
Feature: answer quality is measured beyond citation grounding on the frozen eval set.
Trace: SPEC §12 S6; Gate D
Tests:  U metric adapters with faithful fakes · Q baseline recorded with lineage
Regression: R1 R2 R3 R5
Done: Gate D recorded. Result: Pending.
```

```text
S6.6 — Prompt optimization on holdout (GEPA)
Feature: the draft prompt is optimized on a train split and accepted only on a held-out split.
Trace: SPEC §12 S6; Gate D cleared
Tests:  Q paired holdout comparison, prompt the only axis · U split is disjoint and frozen
Regression: R1 R2 R3 R5
Result: Pending.
```

## S7 — Service readiness, durable execution and the three agents

```text
S7.1 — Guardrails
Feature: untrusted content cannot steer the memo, and output policy holds on every returned memo.
Trace: SPEC §11.4; test-plan §12
Tests:  U input schema, framing, output screen · E2E adversarial fixture in CI (injection, price target, guaranteed return)
Regression: R1 R2 R3 R5
Result: Pending.
```

```text
S7.2 — Identity, roles and guest quotas
Feature: guests and the admin are authenticated and isolated; quotas hold at admission.
Trace: ADR-0010 (+ 2026-10-11 amendment); test-plan §20 Issuer allowlist, Customer isolation, Guest access, Guest quotas, Warm-up
Tests:  U token validation · I development issuer + BFF session; every listed §20 row (local)
Regression: R1 R2 R3 R5
Result: Pending.
```

```text
S7.3 — Durable checkpoints and crash resume
Feature: a crashed run resumes at the agent that failed, from its last checkpoint, instead of starting over.
Trace: SPEC §8.8, §12 G14; ADR-0017 §§3–5; test-plan §22.2, §22.4 "Stale checkpoint refused" (SQLite), §22.6 graph_version rows
Scope: FencedCheckpointer over the SQLite saver; thread_id = job_id; durability="sync"; graph_version on every checkpoint; local job record with attempt; FAIL_AT harness and the `fault` pytest marker.
Non-goals: effect journal (S7.4); DynamoDB (S7.8); agent split (S7.6).
Tests:  U thread identity, compatible resume · C stale-write refusal on the SQLite adapter · F resume at every node (pre-effect), finished siblings kept, durable-before-continuing, incompatible checkpoint · E2E uninterrupted run unchanged (memo hash, < 0.5 s added)
Regression: R1 R2 R3 R5 (R6 starts here)
Done: §22.2 green. Result: Pending.
```

```text
S7.4 — Effect journal and evidence pinning
Feature: a resumed agent never repeats a completed model or tool call, and never re-fetches evidence.
Trace: SPEC §8.7, §8.8; ADR-0017 §§3–4, scenarios 3, 4, 6; test-plan §22.3, §22.7 rows 2–3
Scope: EffectJournal (SQLite) with the ADR-0017 key; every external call routed through memoize; partial streams discarded; snapshot read by hash on resume.
Tests:  U key coverage, per-run isolation · F zero duplicate calls after post-effect crash at every node; partial stream; resume after the quote TTL · E2E memo unchanged with the journal on
Regression: R1 R2 R3 R5 R6
Done: §22.3 green; R6 asserts zero duplicates from here on. Result: Pending.
```

```text
S7.5 — Error taxonomy and node retry policies
Feature: each failure is retried, turned into a typed outcome or ends the attempt, by class.
Trace: SPEC §8.8; ADR-0017 §5; test-plan §22.5
Tests:  U classification table, deterministic not retried, retry bounded by deadline · E2E non-critical source exhausted → degraded · F checkpoint store down → abort and redeliver
Regression: R1 R2 R3 R5 R6
Result: Pending.
```

```text
S7.6 — Researcher / Analyst / Reviewer agents with handoff envelopes
Feature: the run is three agents with their own failure domains, handing off through durable typed envelopes.
Trace: SPEC §8.1, §8.6; ADR-0017 §2, §4; test-plan §22.1, §11 "Revision carries only failed claims"
Scope: three subgraphs; HandoffEnvelope committed before each receiving agent; revise edge carries failed_claim_ids; progress events name the agent.
Non-goals: any prompt, model or retrieval change (single axis: topology).
Tests:  U envelope schema, references only · I envelope before next agent · F R6 matrix re-run over the new node names · Q topology quality-neutral (paired, frozen release)
Regression: R1 R2 R3 R5 R6
Done: §22.1 green; paired result within tolerance. Result: Pending.
```

```text
S7.7 — Durable job execution on DynamoDB and SQS
Feature: an accepted job always reaches exactly one terminal result, even when the API or a worker dies.
Trace: ADR-0013 (+ ADR-0015 amendment); test-plan §20 Acceptance transaction, Fenced commit, Lease expiry, Late worker, Enqueue outbox, Lazy expiry, Concurrency ceiling
Scope: RunStore protocol + DynamoDB adapter (conditional claim, attempt fence, lazy expiry); JobQueue SQS adapter fed by a Streams dispatcher; DLQ handler; worker entrypoint replacing BackgroundTasks; DynamoDB Local and ElasticMQ in compose.
Tests:  C RunStore suite on file and DynamoDB adapters · I every listed §20 row · F kill API after acceptance; kill worker after claim and after result write
Regression: R1 R2 R3 R5 R6
Result: Pending.
```

```text
S7.8 — Execution memory on DynamoDB (fenced checkpointer, journal, envelopes)
Feature: T2 runs on DynamoDB with the same guarantees as locally, including zombie safety.
Trace: SPEC §8.8, §12 G15; ADR-0017 §4; test-plan §22.4
Scope: FencedCheckpointer over DynamoDBSaver (TransactWriteItems + ConditionCheck); journal and envelope items with 14-day TTL; S3 offload stays off the hot path.
Tests:  C §22.4 stale-write suite on SQLite and DynamoDB adapters · I atomic fence, DLQ fences out · F no forked history (paused zombie) · P checkpoint write p99 recorded
Regression: R1 R2 R3 R5 R6 (R6 now also on DynamoDB Local)
Done: G15 closed. Result: Pending.
```

```text
S7.9 — Caching with the T3 write policy
Feature: repeated work is served from versioned caches, and only verified artifacts enter long-term memory.
Trace: SPEC §11.3; ADR-0013 D08; ADR-0017 §1; test-plan §13 cache rows, §20 Cache versioning, Single-flight, Cache bypass, Awake cap, §22.7
Tests:  U key composition per cache · I hit/miss counters, single-flight, verified-only writes, cache failure runs cold · F crash between generation and cache write leaves no partial entry
Regression: R1 R2 R3 R5 R6
Result: Pending.
```

```text
S7.10 — Provider circuit breaker and per-run model pinning
Feature: a failing provider is isolated without mixing models inside one run.
Trace: ADR-0009; ADR-0017 §5; test-plan §20 LLM fallback, §22.6 pinning rows
Tests:  U breaker state machine · I breaker open → new attempt on the fallback host, lineage records both models, run not approvable · F resume uses the pinned model
Regression: R1 R2 R3 R5 R6
Result: Pending.
```

```text
S7.11 — Sentiment off the API event loop in the worker
Feature: sentiment never blocks request handling, in the form CM-6b decided (hosted model, or ONNX FinBERT in the worker per D13).
Trace: SPEC §8.2; cloud-deployment D13; test-plan §10
Tests:  P event loop not blocked · Q label parity recorded · U adapter
Regression: R1 R2 R3 R5 R6
Result: Pending.
```

```text
S7.12 — Execution observability
Feature: an operator can see a resumed run end to end and what each resume cost.
Trace: ADR-0017 §7; test-plan §22.8
Tests:  I every §22.8 row · U metric names and labels
Regression: R1 R2 R3 R5 R6
Done: a dashboard shows one injected crash and its resume. Result: Pending.
```

## S8 — Frontend

```text
S8.1 — Live run stream
Feature: the workspace follows a run over server-sent events, per agent, including a resumed attempt, without ever showing unverified text.
Trace: test-plan §20 Event replay, Stream segment; §17; §18
Scope: GET /jobs/{id}/events (Last-Event-ID, 15 s heartbeat, 5-minute segments); UI replaces polling; agent lanes; "resumed" marker from attempt.
Tests:  I event replay and segment rows · U UI reducers · E2E reconnect mid-run loses and duplicates nothing
Regression: R1 R2 R3 R5 R6 + frontend test, type, lint, build
Result: Pending.
```

```text
S8.2 — Guest mode
Feature: a recruiter can try one live memo without an account, within quotas.
Trace: ADR-0010 amendment; test-plan §20 Guest access, Guest quotas, Warm-up
Tests:  I Turnstile-gated session via a development verifier · E2E quota exhaustion shows example memos
Regression: R1 R2 R3 R5 R6 + frontend suites
Result: Pending.
```

## S9 — Portfolio polish

```text
S9.1 — Measured-results report
Feature: every number in the README traces to a registry record or result file.
Trace: SPEC §2 goals; §11.2
Tests:  U a script resolves every README metric to its record; unresolved fails CI
Regression: R1 R2 R3 R5 R6
Result: Pending.
```

```text
S9.2 — Fresh-clone smoke and demo
Feature: a fresh clone reaches a verified memo on replay with one documented command.
Tests:  E2E fresh-clone script in CI on replay · L recorded demo
Regression: R1 R2 R3 R5 R6
Result: Pending.
```

## S10 — Cloud (Gate E)

Entry criteria and the task outline live in `docs/cloud-deployment.md` §9. Each `S10-Txx` is one feature sprint under this contract; the ids are kept because cloud-deployment.md and test-plan §20 cite them. Added 2026-11-01:

```text
S10-T11 — Deployed resume drill
Feature: the deployed demo tier resumes a crashed run at the failed agent and stays zombie-safe.
Trace: ADR-0017; test-plan §22.9
Depends on: S10-T01, S7.8
Tests:  L Lambda timeout mid-agent resumes at the Analyst; forced duplicate delivery commits once with one checkpoint chain; cost of both attempts recorded
Regression: R1 R2 R3 R5 R6
Result: Pending.
```
