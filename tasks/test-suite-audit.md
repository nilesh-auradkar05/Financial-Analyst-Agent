# tasks/test-suite-audit.md — Test-suite oracle-alignment audit

> Created 2026-06-04. Trace: `docs/test-plan.md` (rank-2 behavioral oracle), `CLAUDE.md §5`.
> Question answered: "is the suite testing intended behavior, or just satisfying the generated code's shape?"

## Verdict

The suite (40 files, ~180 tests, green) is **mixed**. The credibility-critical paths are genuinely behavior-first; a concentrated cluster of retrieval/eval/config tests was implementation-shaped or tautological — exactly what `test-plan.md` forbids. Those have been reworked; an enforcement guard now blocks regression. Remaining work is a set of **coverage gaps** (behaviors in the oracle with no genuine test), ticketed below for a follow-up pass.

## Anti-patterns found → disposition

| Pattern | Evidence (pre-fix) | Disposition |
| --- | --- | --- |
| **Call-order spying** (`assert store.search_called is True`) | `tests/test_retrieval_eval.py`, `test_section_aware_search.py`, `test_retrieval_eval_section_aware_mode.py` | **FIXED** — spy flags + asserts removed; assertions now on observable output. |
| **Mock call-order** (`assert_awaited_once`, `assert_called_once`) | `test_llm.py:59`, `test_embeddings_client.py:42` | **FIXED** — removed; replaced with output assertions (e.g. the actual vectors / health result). |
| **Baked-answer stubs** (stub returns curated chunks → `mrr==1.0`/`passed is True`) | eval-mode family: `test_retrieval_eval_{section_aware,hybrid}_mode.py` | **FIXED** — re-scoped to observable *dispatch* checks; metric *math* lives in `tests/test_retrieval_eval.py` (faithful stub, non-perfect cases: rank-2 hit → MRR 0.5, precision 0.2); retrieval *quality* → S2 anchored benchmark. |
| **Re-implemented logic in test** (`orphans = used - valid` then assert) | `test_citations.py::TestOrphanDetection` | **FIXED** — removed; orphan detection is covered through the real `verify_memo_node` path in `test_graph_verification_integration.py`. |
| **Vacuous / change-detector** | `test_config.py::test_missing_api_keys_produce_warnings` (asserted only `isinstance(list)`) | **FIXED** — now forces the absent-key condition and asserts the real warning text; added `test_invalid_config_value_fails_validation` (test-plan §8 negative case). |
| **White-box wire dict** (`{"$and":[...]}`) | `test_retrieval_contract.py`, `tests/test_retrieval_eval.py` | **KEPT (reconsidered)** — `to_backend_filter()` is a pure transformation whose return value *is* its public contract; asserting it is legitimate, not implementation-shape. |

## Per-file classification

Deep-reviewed in full (high-risk cluster): classified precisely. The remainder were screened by the anti-pattern grep (no call-order spies, no baked-pass) and are **grep-cleared**, with a review ticket for completeness.

| Status | Files |
| --- | --- |
| **REWORKED** (this session) | `tests/test_retrieval_eval.py`, `test_retrieval_eval_section_aware_mode.py`, `test_retrieval_eval_hybrid_mode.py`, `test_section_aware_search.py`, `test_citations.py`, `test_config.py`, `test_llm.py`, `test_embeddings_client.py` |
| **KEEP — behavior-first (model tests)** | `test_graph_verification_integration.py`, `test_compare_retrieval_results_paired.py`, `test_ingestion.py`, `test_api_integration.py`, `test_api_schema_defaults.py`, `test_graph_routing.py`, `test_retrieval_contract.py`, `test_retrieval_eval_reranked_hybrid_mode.py`, `test_retrieve_eval_lc_hybrid_mode.py` |
| **GREP-CLEARED — review-recommended** (no spies/baked-pass found; not yet deep-read) | `test_run_store.py`, `test_state.py`, `test_state_context.py`, `test_sections.py`, `test_section_intent.py`, `test_evidence.py`, `test_vector_store.py`, `test_qdrant_store.py`, `test_vector_store_factory.py`, `test_grounding_eval.py`, `test_judge_model_config.py`, `test_reranked_hybrid_retrieve.py`, `test_lc_hybrid_search.py`, `test_rag_quality_eval_retries.py`, `test_rag_quality_fixture_contract.py`, `test_retrieval_shared_fixture_contract.py`, `test_research_news_node.py`, `test_api_cors.py`, `tests/test_eval_suite.py`, `tests/integration/*` |

## test-plan.md → coverage map (gaps = follow-up tickets)

| Oracle case | Covered by | Status |
| --- | --- | --- |
| §1 API endpoints | `test_api_integration.py` | ✅ |
| §2 controls (filing/news/max) | `test_api_integration.py` | ✅ |
| §2 graceful-degradation routing | `test_graph_routing.py` | ✅ |
| §2 **empty retrieval → limitation + low support** | — | ❌ **T-G1** |
| §2 **tool/LLM failure → degraded run persisted** | partial (`test_graph_routing` routes only) | ❌ **T-G2** |
| §3 ingestion identity/metadata/skip | `test_ingestion.py` | ✅ |
| §3 section-coverage through store | `tests/ingestion/test_section_coverage.py` | ✅ **Closed by S1-T03** |
| §6 paired comparator | `test_compare_retrieval_results_paired.py` | ✅ |
| §6 **runner determinism / null-rank / provenance** | partial | ⚠️ **T-G3** |
| §7 verification / orphan / coverage | `test_graph_verification_integration.py` | ✅ |
| §7 **advice refusal / cautious language** | — | ❌ **T-G4** (needs a deterministic seam) |
| §8 config load + invalid fails | `test_config.py` (after rework) | ✅ |

## Done this session (Phases 1 + 3)

- Reworked 8 files to remove call-order spies, baked-answer tautologies, re-implemented logic, and vacuous asserts; suite stays green (**176 passed, 2 skipped** — net −3 tautological orphan tests, +1 real config negative test).
- Added `scripts/ci/check_test_hygiene.py` (blocks `*_called is True/False` and `assert_called`/`call_count` spying); wired into the Doc Sync Check + §8 governance in `CLAUDE.md`/`AGENTS.md`.
- Added the convention to `CLAUDE.md §5` / `AGENTS.md §5` and strengthened `test-plan.md` Principles (faithful-fakes rule + per-test traceability).

## Remaining (Phase 2 — gap tests, follow-up)

- **T-G1** §2 empty retrieval: real draft→verify with empty `filing_chunks`/news → assert verifier reports low/no support (deterministic; LLM stubbed).
- **T-G2** §2 failure modes: tool/LLM failure → terminal run state is failed/degraded **and persisted** via `run_store` (integration-marked if the compiled graph is needed).
- **T-G3** §6 runner: extend `evaluation` tests for determinism (same fixture+backend+method → identical metrics), null first-rank emitted, provenance preserved.
- **T-G4** §7 advice guard: needs a deterministic seam (assert the memo prompt carries the no-advice/caution guard and/or a post-gen guard flags buy/sell directives) — do not assert live-LLM prose.
- **Review pass** the GREP-CLEARED files individually to confirm behavior-first (low risk; none tripped the spy/baked-pass screens).
