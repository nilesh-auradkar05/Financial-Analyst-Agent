# tasks/sprint-review.md — Baseline-status report

> Created by the S0 current-state audit on 2026-06-04. This is the evidence record S0-T04 mandates.
> It records what was actually run, what passed, and the honest classification of the C0–C3 completed ledger.
> Nothing here marks a C-ledger item `verified` without a recorded proving command. A staff engineer should be able to re-run every command below.

---

## 1. Context

The SPEC/sprint/test/benchmark governance pack (`docs/`, `tasks/`, `CLAUDE.md`, `AGENTS.md`, `scripts/ci/`) is **new and untracked** in git. It was layered on top of a pre-existing, far-more-advanced prototype under `app/` and `evaluation/` (both tracked). The docs describe an S0 baseline; the code already contains Qdrant, hybrid/reranked/section-aware retrieval, a 72-case benchmark fixture, paired comparator, and a RAGAS/DeepEval answer-quality harness. The audit's central job is to reconcile that gap honestly without lowering the SPEC or trusting unverified prior work.

---

## 2. Commands run and results (2026-06-04)

### Document governance (S0-T01/T02/T03 scope)

```
python scripts/ci/check_no_scope_residue.py   -> PASS  ("Scope-residue check passed.")
python scripts/ci/check_sprint_map.py         -> PASS  ("Sprint-map check passed.")
python scripts/ci/check_doc_sync.py           -> PASS  ("Doc-sync check passed.")
grep -Rni "terraform|pulsepress" CLAUDE.md AGENTS.md docs/SPEC.md docs/test-plan.md \
     docs/retrieval-benchmark.md docs/adr   -> no residue
```

### Repository checks

```
uv run pytest --collect-only          -> 180 tests collected
uv run pytest tests/unit              -> 162 passed, 0 failed
uv run pytest tests/integration tests/test_api_integration.py \
              tests/test_eval_suite.py tests/test_retrieval_eval.py
                                      -> 5 passed, 2 skipped (need --run-integration),
                                         11 ERRORS in tests/test_api_integration.py
uv run python evaluation/validate_retrieval_fixture.py \
     evaluation/fixtures/retrieval_shared_benchmark_v1.json
                                      -> PASS: 72 cases (AAPL/MSFT/NVDA 24 each;
                                         business/risk_factors/md&a/market_risk 18 each)
```

`uv run ruff check .` and `uv run pyright .`/`mypy .` were **not** run in this pass (audit was read-only of source; lint/type sweep deferred to the S0-T04 close-out so failures are recorded against a clean baseline, not mixed with audit edits).

### Why the 11 API-integration ERRORs

`app/main.py:134` calls `await check_ollama_health(log_failure=True)`. The test double `fake_check_ollama_health()` in `tests/test_api_integration.py` does not accept `log_failure`, so every test in that module errors at lifespan setup with `TypeError: fake_check_ollama_health() got an unexpected keyword argument 'log_failure'`. The production code is fine; the **test stub is stale**. This blocks all public-API-surface verification (test-plan §1) until fixed → gap **G1**.

---

## 3. C-ledger classification (honest)

| Item | Claim | Classification | Evidence / gap |
| --- | --- | --- | --- |
| **C0** | LangGraph workflow; persistent run state; request controls | **Partially verified** | `app/agents/graph.py` implements all six nodes; controls honored at `graph.py:71/173/246`; `tests/unit/test_run_store.py`, `test_graph_routing.py`, `test_state*.py` pass. Public-API proof of the controls is **blocked** by G1. |
| **C1** | `EvidencePacket` contract; verification wired into workflow+API | **Partially verified** | `EvidencePacket` in `app/components/retrieval/evidence.py`; `verify_memo_node` wired `draft_memo→verify_memo→END`; `test_evidence.py`, `test_citations.py`, `test_graph_verification_integration.py` pass. **Field names diverge from SPEC §7** (`chunk_id` not `evidence_id`; `section_name` not `section_key`) → gap **G2**. |
| **C2** | `RetrievalStore` contract + Chroma implementation | **Gap** | Protocol + `ChromaDBVectorStore` exist and unit-test green, but: (a) Protocol surface diverges from SPEC §6 — no `upsert_documents`/`delete_by_filter`/`healthcheck`, returns `SearchResult`/`int` not `RetrievalResult`/`UpsertSummary`/`DeleteSummary`/`BackendHealth` → gap **G3**; (b) `get_vector_store()` **defaults to `qdrant`**, contradicting SPEC §1 and ADR-0003 "Chroma as default until Gate B" → gap **G4**; (c) no backend-parity suite at the spec'd path. |
| **C3** | Section-aware ingestion; edgartools fixed MSFT/NVDA zero-section | **Unverified (as designed)** | `ingestion.py`, `sections.py`, `edgartools_sec_extractor.py` exist; `test_ingestion.py`/`test_sections.py` pass at unit level. There is **no section-coverage test through the store interface** for AAPL/MSFT/NVDA (that is S1-T03, still Pending). The "fixed" claim remains asserted, not regression-locked → gap **G5**. |

No C-ledger item is `verified` in the full sense the SPEC demands: the full `pytest` run is not green (G1) and C2/C3 carry substantive gaps.

---

## 4. Gap tickets opened by this audit

- **G1** — Stale test double breaks all of `tests/test_api_integration.py` (11 errors). Fix `fake_check_ollama_health` to accept `log_failure`. Blocks test-plan §1 verification. Owner: S0-T04 close-out.
- **G2** — `EvidencePacket` uses `chunk_id`/`section_name`; SPEC §7 specifies `evidence_id`/`section_key` as the canonical join key. Reconcile naming (code→spec or documented alias). Touches benchmark join semantics.
- **G3** — `RetrievalStore` Protocol shape diverges from SPEC §6 (method names, return types, no `healthcheck`). Decide whether SPEC §6 is the target contract (then S3/S4 must close it) or whether SPEC §6 should be annotated as aspirational.
- **G4** — Default backend is `qdrant`, not `chroma`. This is a Gate B decision made in code with ADR-0003 still "Proposed" and no recorded Chroma↔Qdrant comparison. **Hard-stop surface** (undocumented architecture decision / SPEC↔code contradiction). Needs user direction.
- **G5** — No AAPL/MSFT/NVDA section-coverage regression test through the store interface; C3 cannot be regression-locked until S1-T03.
- **G6** — Benchmark methodology mismatch: the implemented fixture/validator are keyword+section based; `retrieval-benchmark.md` mandates content-anchored graded labels with an anchor-in-source cache. No `source_sections/` cache and no `build_source_section_cache.py` exist. The Gate A "lie detector" is not implemented. Needs an explicit decision (adopt anchored methodology, or amend the benchmark doc via the documented bump process). **Hard-stop surface** (benchmark methodology).
- **G7** — Qdrant payload indexes cover 4 fields (`ticker, filing_type, section_key, filing_date`); SPEC/test-plan §5/S3-T03 require 5 (missing `accession_number`).
- **G8** — Path/name drift: docs reference `evaluation/validate_benchmark_fixture.py`, `evaluation/run_retrieval_benchmark.py`, `tests/ingestion/`, `tests/rag/`, `tests/evaluation/`, `/runs/{id}`, `/healthz`; actual repo uses `validate_retrieval_fixture.py`, `run_shared_retrieval_benchmark.py`, `tests/unit|integration/`, `/jobs/{job_id}`, `/health`. Documented verification commands will not run as written.

---

## 5. Bottom line (initial audit)

The governance layer (S0-T01/02/03 doc topology, residue, sprint map, doc-sync) is internally consistent and its scripts pass. The unit layer is healthy (162/162). But the **code is several gated sprints ahead of where the docs place it**, and in two places the code has already made gated decisions (Qdrant default; non-anchored benchmark methodology) that the docs reserve for Gate B/Gate A. Those are surfaced as hard stops G4 and G6 and must be resolved by explicit decision before any S1+ work treats the baseline as trustworthy. C0–C3 remain non-load-bearing per SPEC §1.1.

---

## 6. Reconciliation round (2026-06-04, post-audit) — S0-T04 closed green

User decisions (this session): **G4 → ratify Qdrant in docs**; **G6 → upgrade to content-anchored graded labels (raise code to spec, scheduled S2)**; **G2/G3/G7/G8 → pragmatic re-baseline**.

### Changes applied
- **Code (G1):** `tests/test_api_integration.py::fake_check_ollama_health` now accepts the `log_failure` kwarg → the 11 errors became **11 passed**.
- **Code (G7):** added `accession_number` to `qdrant_store.py::PAYLOAD_INDEX_FIELDS` (4→5) and tightened `tests/unit/test_qdrant_store.py` to assert it.
- **Lint:** auto-fixed 3 pre-existing `I001` import-sort errors in `scripts/ci/check_*.py` (part of the S0 governance layer); scripts still pass.
- **Docs (G4):** ADR-0003 Proposed→**Accepted** (Qdrant default on operational grounds, no quality win, Gate-B obligation retained); SPEC §1/§1.1 corrected (Qdrant default, Chroma fallback).
- **Docs (G2/G3):** SPEC §6 split into target vs implemented surface, `healthcheck()` kept as the one scheduled add (S3); SPEC §7 documents `chunk_id`=canonical `evidence_id` alias and makes `section_key`+`accession_number` on chunk metadata non-negotiable for the matcher.
- **Docs (G6):** retrieval-benchmark.md flags v1 keyword/section as a non-Gate-A precursor; sprint-plan S2 note binds the anchored upgrade + real filenames to extend.
- **Docs (G8):** test-plan §1 routes → `/jobs/{job_id}`,`/health`,`/metrics`; CLAUDE.md/AGENTS.md §8 benchmark command → `validate_retrieval_fixture.py` (no `--min-cases`), forward-looking test paths annotated.

### Verification (green)
```
python scripts/ci/check_no_scope_residue.py / check_sprint_map.py / check_doc_sync.py  -> all PASS
uv run ruff check .                                                                     -> All checks passed!
uv run mypy app evaluation                                                              -> Success: no issues in 44 files
uv run pytest                                                                           -> 178 passed, 2 skipped (integration, need --run-integration)
uv run pytest tests/test_api_integration.py                                             -> 11 passed
uv run python evaluation/validate_retrieval_fixture.py .../retrieval_shared_benchmark_v1.json -> 72 cases PASS (precursor)
```

### C-ledger — final classification
| Item | Classification | Basis |
| --- | --- | --- |
| **C0** workflow/run-state/controls | **Verified** | `tests/test_api_integration.py` (controls disable filing/news, limit articles) + `test_run_store.py` + `test_graph_routing.py` + `test_state*.py` all pass in a green suite. |
| **C1** EvidencePacket + verification | **Verified** | `test_evidence.py`, `test_citations.py`, `test_graph_verification_integration.py`, API verification payload pass; SPEC §7 naming re-baselined (chunk_id=evidence_id). |
| **C2** RetrievalStore + backends | **Verified (re-baselined)** | Contract surface recorded in SPEC §6; Qdrant default ratified (ADR-0003); `test_vector_store_factory.py`/`test_retrieval_contract.py` pass. Open follow-ups: backend-parity suite + `healthcheck()` → **S3** (not baseline). |
| **C3** section-aware ingestion | **Gap → S1-T03** | Ingestion/sections unit tests pass, but non-zero section coverage for AAPL/MSFT/NVDA is **not** asserted through the store interface. Stays gap **G5**; regression-locked at S1-T03. |

### Gap status after this round
Resolved/dispositioned: **G1** fixed · **G7** fixed · **G4** ratified (docs) · **G2** re-baselined · **G3** documented + `healthcheck` scheduled S3 · **G6** committed as S2 target (precursor flagged) · **G8** doc paths reconciled. Still open: **G5** (S1-T03 coverage lock) · **G3-code** (`healthcheck()` implementation, S3).

### S0 exit
S0-T04 is **closed green**. S0 exit criteria met (doc scripts pass, sprint IDs match, baseline-status report complete, baseline verified or ticketed). The project may proceed to **S1 planning** (ingestion & section-coverage lock), with the anchored-benchmark upgrade pre-committed into S2 scope.

---

## 7. S1 execution (2026-06-04) — closed green

### Scope

Trace: SPEC §7/§9; test-plan §3; sprint-plan S1-T01/S1-T02/S1-T03; ADR-0004.

S1 locked stable filing identity and critical-section coverage so S2 can label evidence without chunk IDs moving underneath the benchmark.

### Changes applied

- Added `tests/ingestion/test_chunk_metadata.py` for canonical metadata and accession-based chunk IDs.
- Added `tests/ingestion/test_ingestion_idempotency.py` for same-filing re-ingest stability and duplicate prevention through the store interface.
- Added `tests/ingestion/test_section_coverage.py` for representative AAPL/MSFT/NVDA critical-section coverage through `count_documents(SearchFilters(...))`.
- Updated `app/components/retrieval/ingestion.py` to require `ticker`, `filing_type`, `filing_date`, `accession_number`, and `source_url`/`filing_url`, and to mint `chunk_id = f(accession_number, section_key, chunk_index)`.
- Updated the existing ingestion unit test to the new accession-based identity contract.

### Verification

```text
uv --cache-dir /tmp/uv-cache run pytest tests/ingestion -q
  -> 5 passed

uv --cache-dir /tmp/uv-cache run pytest \
  tests/ingestion tests/unit/test_ingestion.py tests/unit/test_vector_store.py \
  tests/unit/test_qdrant_store.py tests/unit/test_retrieval_contract.py -q
  -> 23 passed

uv --cache-dir /tmp/uv-cache run ruff check .
  -> All checks passed

uv --cache-dir /tmp/uv-cache run mypy app evaluation
  -> Success: no issues found in 44 source files

python scripts/ci/check_no_scope_residue.py
python scripts/ci/check_sprint_map.py
python scripts/ci/check_doc_sync.py
python scripts/ci/check_test_hygiene.py
  -> all PASS

uv --cache-dir /tmp/uv-cache run pytest -q
  -> 181 passed, 2 skipped

VECTOR_BACKEND=chroma uv --cache-dir /tmp/uv-cache run pytest tests/ingestion -q
  -> 5 passed

uv --cache-dir /tmp/uv-cache run python evaluation/validate_retrieval_fixture.py \
  evaluation/fixtures/retrieval_shared_benchmark_v1.json
  -> 72 precursor retrieval cases validated
```

Pytest required unsandboxed execution because `pytest-rerunfailures` opens a local socket; this is an execution-environment permission issue, not a product failure.

### C-ledger update

| Item | Classification | Basis |
| --- | --- | --- |
| **C3** section-aware ingestion | **Verified** | S1 metadata, idempotency, and section-coverage tests pass through the retrieval-store interface. The prior AAPL/MSFT/NVDA zero-section failure class is now regression-locked at the ingestion/store boundary. |

### S1 exit

S1 is **closed green**. G5/C3 is closed. The next active sprint is **S2 — Shared retrieval benchmark oracle**, which upgrades the current keyword/section precursor fixture into the content-anchored graded Gate-A oracle.
