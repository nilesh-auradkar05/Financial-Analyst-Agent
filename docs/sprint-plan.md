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
> | 8 | S7 → S10 | Queue/worker, Postgres job store, Qdrant, FinBERT out-of-process, circuit breaker; cloud gated on ADR-0006 |

```text
S2-T00a — Governance docs into repo + CI governance gate  (Phase 0; step 1)
Trace: SPEC §0/§13/§14; INTEGRATION_PLAN_v2 Phase 0; gap G11
Goal: every governance rule the repo claims is enforced by a script that CI actually runs.
Scope:
- git-track docs/SPEC.md, docs/sprint-plan.md, docs/test-plan.md, docs/retrieval-benchmark.md, docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md, docs/adr/*.
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
Result: Pending
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
Result: Pending
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
Result: Pending
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
Result: Pending
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
Result: Pending
```

```text
S3-T02 — Implement QdrantVectorStore, Protocol parity
Trace: SPEC §6; test-plan §4/§5
Scope: implement all five contract methods; idempotent collection; SearchFilters→Qdrant filter; results→EvidencePacket; no qdrant_client escapes rag/.
Tests: covered by S3-T04 parity suite.
Done: Protocol implemented; no public signature above rag/ changed.
Non-goals: no hybrid; no indexes yet.
Result: Pending
```

```text
S3-T03 — Qdrant payload indexes
Trace: SPEC §6; test-plan §5
Scope: indexes for ticker, filing_type, section_key, filing_date, accession_number; idempotent creation; documented.
Tests: tests/rag/test_qdrant_payload_indexes.py
Verification: VECTOR_BACKEND=qdrant uv run pytest tests/rag/test_qdrant_payload_indexes.py
Done: indexes verified.
Non-goals: no scoring changes.
Result: Pending
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
Result: Pending
```

```text
S3-T05 — Section coverage + idempotent ingest under Qdrant
Trace: SPEC §9/§6; test-plan §3/§5
Scope: run S1 coverage + idempotency under VECTOR_BACKEND=qdrant; guard that the same embedding model/version is used on both backends; record embedding identifier in ingest metadata.
Verification: VECTOR_BACKEND=qdrant uv run pytest tests/ingestion/test_section_coverage.py tests/ingestion/test_ingestion_idempotency.py
Done: coverage + idempotency pass under Qdrant; embedding identity recorded.
Non-goals: no benchmark run; no switch.
Result: Pending
```

Exit: Qdrant passes the same contract/coverage/idempotency tests Chroma passes; indexes verified; embeddings matched; nothing above `rag/` changed.

---

## Sprint 4 — Chroma vs Qdrant measured comparison → Gate B

Goal: decide the default backend on evidence and record the decision honestly.

```text
S4-T01 — Generate Qdrant candidate result
Trace: SPEC §10/§11 Gate B; test-plan §6
Scope: run S2-T04 runner under VECTOR_BACKEND=qdrant on the same fixture/method/top_k/embedding; labeled backend=qdrant, same fixture_version.
Tests: result validates; all case_ids; fixture_version == baseline.
Done: comparable Qdrant candidate exists.
Non-goals: no tuning to win.
Result: Pending
```

```text
S4-T02 — Paired quality comparison
Trace: SPEC §10/§11 Gate B; retrieval-benchmark.md §8–§9
Scope: strict comparator on Chroma baseline vs Qdrant candidate; report deltas/CI/win-tie-loss/effective-N; state honest expectation: identical embeddings imply quality tie within CI.
Verification: uv run python evaluation/compare_retrieval_results.py evaluation/results/chroma_section_aware_shared.json evaluation/results/qdrant_section_aware_shared.json --strict-case-ids
Done: paired report committed.
Non-goals: no decision yet.
Result: Pending
```

```text
S4-T03 — Filtering & operability comparison
Trace: SPEC §6/§11 Gate B; test-plan §5
Scope: filtering correctness/expressiveness; cold/warm + filtered-query latency; operability notes; S5 hybrid/sparse readiness.
Done: operability comparison recorded.
Non-goals: no hybrid implementation.
Result: Pending
```

```text
S4-T04 — Gate B decision + ADR-0003 update
Trace: SPEC §11 Gate B/§15
Scope: decide against explicit criteria: no quality regression, filtering ≥ Chroma, latency acceptable, S5-readiness. Record decision and numbers in sprint-review.md and ADR-0003. Honest framing is mandatory: quality tie + better operability/S5-readiness is a valid migration reason, not a fabricated quality win.
Done: Gate B report + operability comparison + recorded decision.
Non-goals: no S5 work; no retro-editing fixture.
Result: Pending
```

Exit = Gate B.

---

# HORIZON: gated, detailed only when predecessor gate clears

- **S5 — Retrieval quality diagnostics → hybrid/rerank (Gate C).** Per-query diagnostics and failure taxonomy first; optional section prior, hybrid dense+sparse, and rerank candidates evaluated on same fixture. Unchanged.
- **S6 — Answer quality & eval hardening (Gate D). Steps 4–5.** (a) Eval registry `evaluation/registry/runs/*.json` with lineage keys (SPEC §11.2); hook H5 + CI reject unlineaged results. (b) `eval-replay` CI job on the frozen release, fails on >2σ regression of grounded_claim_rate / citation_coverage. (c) Verifier↔judge agreement on ≥50 claims; κ recorded. (d) Bounded draft→verify→revise loop (max 2), `attempts` in response, paired vs no-loop on the same snapshot/model/temperature. (e) Then LLM-as-judge/RAGAS, then GEPA on holdout. Order inside S6 is fixed: (a)→(b)→(c)→(d)→(e).
- **S7 — Service readiness. Steps 6–8 (local).** (a) Guardrails per SPEC §11.4 with adversarial fixture in CI. (b) API auth + rate limit + idempotency key. (c) Caching per SPEC §11.3 with hit/miss metrics. (d) `JobQueue` protocol → Redis Streams; `RunStore` protocol → Postgres; LangGraph checkpointer. (e) FinBERT out-of-process; provider circuit breaker (Bedrock→Ollama at runtime, not config). (f) Fix G13. CI matrix, correlation IDs, structured logs retained from prior S7 scope.
- **S8 — Frontend MVP.** Thin API-driven SPA against the hardened API. Unchanged.
- **S9 — Portfolio polish.** README, measured results, honest limitations, benchmark report, demo, fresh-clone smoke. Unchanged.
- **S10 — Cloud (INTEGRATION_PLAN_v2 Phases 4–5). Step 8 (remote).** Gated on ADR-0006 + SPEC §3 amendment. Kafka is explicitly *not* adopted until a second consumer type exists for the same event stream (review 2026-08-24).
