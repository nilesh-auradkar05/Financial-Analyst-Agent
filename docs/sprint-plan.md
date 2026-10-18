# sprint-plan.md — Sequencing & Task Ledger

> **Authority:** Source-of-truth rank 3 (SPEC §0). Sprint IDs match SPEC §12 exactly. Committed sprints (S0–S4) are detailed; horizon sprints (S5–S10) are gated one-liners, detailed only when the predecessor gate clears.
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

- **G10** `app/agents/graph.py` imports `evaluation.grounding` — production depends on the eval package (layering inversion). Resolve S2-T00d · **G11** `ci.yml` never invokes `scripts/ci/check_*.py`; governance docs not tracked at HEAD, so no doc rule is enforced. Resolve S2-T00a · **G12** `add_error` mutates `state["errors"]` in place — unsafe under fan-out. Resolve S2-T00c · **G13** `/health` reports `degraded` on Ollama outage regardless of configured provider. Resolve S7.

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
Result: Partial as of 2026-10-06. Implemented in 946f1de (fan-out, errors reducer, to_thread, AGENT singleton, retrieval sizing) and b407868 (provider client built once per process). Unit tests for reducer + parallel routing green (test-plan §10): three 1 s branches complete in 1.02 s (2.83 s before); /health answers in under 200 ms while sentiment scores 50 snippets (blocked 0.60 s before). Full suite 356 passed, 2 skipped; ruff, mypy and governance checks pass. NOT recorded: the latency artifacts and the 4-concurrent probe. Frozen-release latency cannot be measured until S2-T00b (replay skips the evidence nodes); the agreed substitute, one live run compared with LangSmith trace 01a0f4dc-8b2b-7021-b7de-5958220c0ad1 (66.73 s total, 8.65 s evidence phase), has not been run.
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
Result: Partial (reconciled 2026-10-10). Done: `JobStatus.DEGRADED` and `EVIDENCE_MISSING` in `app/models.py`; status and `missing=[...]` derived in `app/main.py`; cases in `tests/test_api_integration.py` (verification failed → degraded; no filings or no stock data → evidence_missing); both statuses shown in the UI (user-confirmed 2026-10-10). NOT done: gap G10 layering — `app/agents/graph.py` still imports `evaluation.grounding` and `app/verification/` does not exist, so the `grep "from evaluation" app/` check is not empty.
```

> **Implementation note (2026-06-04, audit reconciliation).** Decision: *upgrade the existing keyword/section benchmark to the content-anchored graded methodology* in `retrieval-benchmark.md` (raise code to spec). Bind the doc's planned filenames to the actual files to **extend, not recreate** (G8 path drift):
> - `evaluation/validate_benchmark_fixture.py` → **extend** existing `evaluation/validate_retrieval_fixture.py` (add `gold_evidence` with `accession_number`/`section_key`/verbatim `anchor_text`≥8 words/`relevance∈{1,2}`, `answer_intent`, and the anchor-in-source check). Note: the current validator's `--min-cases` is hardcoded (`MIN_CASES=50`), not a flag.
> - `evaluation/run_retrieval_benchmark.py` → existing `evaluation/run_shared_retrieval_benchmark.py`.
> - `tests/evaluation/test_*` → create under `tests/unit/` (the repo's actual test home) unless a `tests/evaluation/` dir is added.
> - The current `retrieval_shared_benchmark_v1.json` (72 keyword/section cases) is the **precursor**; the anchored fixture gets a new `fixture_version` and a source-section cache + builder (`evaluation/build_source_section_cache.py`, reusing `app/services/tools/edgartools_sec_extractor.py`). Gate A is **not** satisfied by the v1 precursor.

```text
S2-T01 — Specify labeling schema, matcher, metric definitions
Trace: SPEC §10/§11; test-plan §6; retrieval-benchmark.md
Scope: fill/verify retrieval-benchmark.md §§2–4: case schema, content-anchored graded labels, matcher, metric formulas, K∈{5,10}.
Expected: every metric has one unambiguous formula tied to graded labels.
Done: schema + matcher + metrics fully specified; NDCG graded.
Non-goals: no fixture authoring; no LLM-judged relevance.
Result: Pending
```

```text
S2-T02 — Fixture validator: lie detector
Trace: SPEC §11 Gate A; test-plan §6; retrieval-benchmark.md §5–§6
Scope: evaluation/validate_benchmark_fixture.py validates schema, relevance∈{1,2}, anchor≥8 words, expected_sections match, min-cases, per-ticker/section coverage, and anchor-in-source against committed source-section cache.
Tests: valid fixture passes; duplicate id, missing gold, unknown section, invalid relevance, short anchor, absent anchor, expected_sections mismatch, and below-min-cases all fail.
Verification: uv run pytest tests/evaluation/test_benchmark_fixture_schema.py
Done: validator and behavior tests pass.
Non-goals: no scoring logic.
Result: Pending
```

```text
S2-T03 — Author/expand fixture to 50–100 anchored, graded cases
Trace: SPEC §3/§10/§11; test-plan §6
Scope: AAPL/MSFT/NVDA plus more if available × all 4 sections; mixed answer_intent; 1–3 graded anchors per case; validate continuously; emit fixture coverage report.
Verification: uv run python evaluation/validate_benchmark_fixture.py evaluation/fixtures/retrieval_shared_benchmark_v1.json --min-cases 50
Done: 50–100 validated cases; coverage report committed; thin cells noted.
Non-goals: no tuning toward any backend or method; no answer-quality labels.
Result: Pending
```

```text
S2-T04 — Benchmark runner
Trace: SPEC §10; test-plan §6; retrieval-benchmark.md §7
Scope: evaluation/run_retrieval_benchmark.py emits result file per backend/method with provenance, labels, metrics, cold/warm latency, null first_relevant_rank, deterministic tie-breaks.
Tests: all case_ids present; labels present; null rank emitted; provenance and latency fields present.
Verification: VECTOR_BACKEND=chroma uv run python evaluation/run_retrieval_benchmark.py --fixture evaluation/fixtures/retrieval_shared_benchmark_v1.json --mode section_aware --output evaluation/results/chroma_section_aware_shared.json
Done: schema-valid, provenance-bearing result.
Non-goals: no comparison; no hybrid/rerank.
Result: Pending
```

```text
S2-T05 — Harden paired comparator
Trace: SPEC §10; test-plan §6; retrieval-benchmark.md §8
Scope: evaluation/compare_retrieval_results.py enforces fixture_version match, single-axis rule, --strict-case-ids, paired deltas, relative deltas, win/tie/loss, bootstrap CI, null=miss, latency separated.
Tests: mismatched fixture/mode/method fail; both-differ fails; missing case fails; null=miss; CI present.
Verification: uv run pytest tests/evaluation/test_compare_retrieval_results.py
Done: comparator passes behavior tests.
Non-goals: no default switch.
Result: Pending
```

```text
S2-T06 — Generate Gate A baseline
Trace: SPEC §11 Gate A
Scope: run runner on Chroma section_aware over validated fixture; self-compare sanity; record baseline + coverage/CI-width report in tasks/sprint-review.md.
Tests: result validates; comparator self-compare produces zero deltas and full shared count.
Done: Gate A satisfied.
Non-goals: no Qdrant run; no method adoption.
Result: Pending
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
Result: Pending. No contract suite runs against a real backend: `tests/rag/` does not exist; current store tests use a fake Qdrant client and Chroma monkeypatches. With S4 retired the suite is the Qdrant contract oracle; the Chroma leg is required only while Chroma stays a supported backend (SPEC §6).
```

```text
S3-T05 — Section coverage + idempotent ingest under Qdrant
Trace: SPEC §9/§6; test-plan §3/§5
Scope: run S1 coverage + idempotency under VECTOR_BACKEND=qdrant; guard that the same embedding model/version is used on both backends; record embedding identifier in ingest metadata.
Verification: VECTOR_BACKEND=qdrant uv run pytest tests/ingestion/test_section_coverage.py tests/ingestion/test_ingestion_idempotency.py
Done: coverage + idempotency pass under Qdrant; embedding identity recorded.
Non-goals: no benchmark run; no switch.
Result: Pending. `tests/ingestion/test_section_coverage.py` and `test_ingestion_idempotency.py` pass, but against an in-memory store, not Qdrant. The embedding identifier is recorded in trace metadata only, not in ingest metadata.
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

# HORIZON: gated, detailed only when predecessor gate clears

- **S5 — Retrieval quality diagnostics → hybrid/rerank (Gate C).** Per-query diagnostics and failure taxonomy first; optional section prior, hybrid dense+sparse, and rerank candidates evaluated on same fixture. Unchanged.
- **S6 — Answer quality & eval hardening (Gate D). Steps 4–5.** (a) Eval registry `evaluation/registry/runs/*.json` with lineage keys (SPEC §11.2); hook H5 + CI reject unlineaged results. (b) `eval-replay` CI job on the frozen release, fails on >2σ regression of grounded_claim_rate / citation_coverage. (c) Verifier↔judge agreement on ≥50 claims; κ recorded. (d) Bounded draft→verify→revise loop (max 2), `attempts` in response, paired vs no-loop on the same snapshot/model/temperature. (e) Then LLM-as-judge/RAGAS, then GEPA on holdout. Order inside S6 is fixed: (a)→(b)→(c)→(d)→(e).
- **S7 — Service readiness. Steps 6–8 (local).** (a) Guardrails per SPEC §11.4 with adversarial fixture in CI. (b) API auth + rate limit + idempotency key; JWT issuer allowlist, customer-scoped authorization, `guest` (Turnstile-gated, BFF-minted token, quotas) and `admin` roles, `analyst` defined for the scale tier, per ADR-0010 and its 2026-10-11 amendment (local development issuer). (c) Caching per SPEC §11.3 (v1.4 keys, ADR-0013) with hit/miss metrics and single-flight memo generation. (d) `RunStore` protocol → DynamoDB adapter (conditional-write claim with `attempt` fencing, lazy deadline expiry); `JobQueue` protocol → SQS adapter fed by a DynamoDB Streams dispatcher; LangGraph `DynamoDBSaver` checkpointer; local stand-ins DynamoDB Local and ElasticMQ (ADR-0013 as amended by ADR-0015 on 2026-10-11; was a Postgres `SKIP LOCKED` queue, before that Redis Streams). No Postgres adapter is built; it belongs to the production scale tier. (e) FinBERT off the API event loop: ONNX inside the worker for the demo tier (ADR-0015), with recorded label parity; out-of-process service only in the scale tier; provider circuit breaker (separate from ADR-0009 fallback; Ollama development-only). (f) Fix G13. CI matrix, correlation IDs, structured logs retained from prior S7 scope.
- **S8 — Frontend MVP.** Thin API-driven SPA against the hardened API. Unchanged.
- **S9 — Portfolio polish.** README, measured results, honest limitations, benchmark report, demo, fresh-clone smoke. Unchanged.
- **S10 — Cloud (INTEGRATION_PLAN_v2 Phases 4–5). Step 8 (remote).** ADR-0006 accepted 2026-10-08 with ADR-0010 to ADR-0013; ADR-0015 (2026-10-11) makes the demo tier scale-to-zero; SPEC §3 amended in v1.4. Starts only when the ADR-0006 entry criteria have recorded evidence. Task outline (detailed when S10 opens): S10-T01 Terraform demo tier (Lambda, DynamoDB, SQS, Cloudflare Worker), apply → smoke → destroy twice; S10-T02 deploy pipeline with the eval gate and alias shift; S10-T03 Foundry fallback and model parity (S7-AZURE-FALLBACK); S10-T04 embedding parity gate and Gate C outcome; S10-T05 MWAA Serverless ingestion DAG (monthly + admin trigger); S10-T06 guest mode, Cognito admin and RBAC; S10-T07 observability and budget kill switch; S10-T08 region-loss runbook (documentation only); S10-T09 OpenSearch Serverless store, paired retrieval comparison against the Qdrant baseline and the daily awake cap (ADR-0014, added 2026-10-11); S10-T10 edge signing and cold start: unsigned function-URL requests refused, Worker CPU < 10 ms, first-query latency with and without the warm-up recorded (ADR-0015). See `docs/cloud-deployment.md` §9. Moves to the production scale tier follow the ADR-0015 triggers (`docs/production-scale-architecture.md`); Kafka only for a stated retention or replay requirement. This replaces the earlier second-consumer trigger (review 2026-08-24), which the 2026-09-06 review found insufficient.


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


### Authorized embedding-provider plumbing slice (2026-10-11)

User decision, 2026-10-11: the Gate C embedding bake-off has four arms (`qwen3-embedding:4b`, Cohere Embed v4, Titan Text Embeddings V2, `text-embedding-3-small`; ADR-0011 amendment, `docs/retrieval-benchmark.md` §9a). The plumbing is built now, ahead of S5: an embedding provider switch behind the existing retrieval boundary, one index per arm built from the same chunks, and the embedding identifier written to ingest metadata (also the open part of S3-T04). Measurement stays in S5, after Gate A; no arm is measured or cited before the fixture is frozen. Tests: test-plan §21. The remaining order above is otherwise unchanged.

Result: Pending.

### Authorized Azure provider slice (2026-10-17)

User decision, 2026-10-17 (ADR-0009 amendment 2026-10-17; SPEC §4 runtime provider policy; test-plan §16 rows dated 2026-10-17): the Azure tier, previously S10-T03 / S7-AZURE-FALLBACK, is pulled forward and is the next task. Scope: Foundry DeepSeek-V4-Flash as the first draft attempt when configured, Bedrock `deepseek.v3.2` next, personal fallbacks unchanged; the evaluation judge ladder GPT-5.6 Luna on Bedrock (`bedrock-mantle`) → Foundry; Compose pass-through for the new variable. Out of scope: the STS-to-Entra keyless sign-in (stays in S10), circuit breakers (S7 (e)), parity and fallback-quality measurement (S10 / S6). The in-flight EMBED-ARMS plumbing slice pauses and resumes after this slice. Ordering of the embedding bake-off and the DeepSeek-V4-Flash memo baseline after this slice: pending the owner's answer (Gate A is still open, and the bake-off is gated on it above).

Result as of 2026-10-18: implemented on branch feat/embedding-provider-switch. Offline: 492 passed, 2 skipped; ruff, mypy and the four governance checks pass; test-plan §16 rows dated 2026-10-17 (Judge ladder row revised 2026-10-18) covered offline by `tests/unit/test_provider_azure.py` and `tests/unit/test_judge_ladder.py`, including a structured verdict through the DeepEval adapter that falls back to Foundry. Live Azure draft run not yet done; Bedrock Luna is blocked by a zero account quota on `bedrock-runtime`, so only the Foundry judge is live-usable.
