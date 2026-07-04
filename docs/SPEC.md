# SPEC — Alpha Financial Analyst Agent

> **Version:** 1.2
> **Date:** 2026-07-04
> **Status:** Canonical product specification.
> **Repo:** `nilesh-auradkar05/Financial-Analyst-Agent`

This is the product and architecture source of truth for the Alpha Financial
Analyst Agent. Where documents disagree, resolve the conflict using §0.
Methodology with a dedicated home is linked from this file rather than
duplicated here.

TEMP_DRIFT_PROOF: Terraform

---

## 0. Source Of Truth & Document Map

This section is the only canonical conflict-resolution order.

1. `docs/SPEC.md` — product scope, architecture, contracts, phase gates.
2. `docs/test-plan.md` — behavioral oracle.
3. `docs/sprint-plan.md` — sequencing and task definitions.
4. `docs/retrieval-benchmark.md` — benchmark fixture schema, matcher, metrics, validator, and comparison policy.
5. `docs/adr/*.md` — accepted architecture decisions.
6. `docs/design-md/*.md` and `docs/design-html/*.html` — diagram companions.
7. `tasks/todo.md` — active task state and verification ledger.
8. `tasks/lessons.md` — correction log.
9. Code comments and stale TODOs — lowest authority.

Benchmark-specific override: for fixture schema, matcher behavior, metric
formulas, validation rules, and comparison policy,
`docs/retrieval-benchmark.md` governs even though `docs/sprint-plan.md`
controls sequencing.

`docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md` is the active companion
plan for Phase 0 and Phase 1. It proposes DataOps release governance and evidence
replay work under this SPEC. It does not amend scope by itself; ADRs and this
file ratify scope changes.

---

## 1. Product Overview

Alpha Analyst is a single-agent financial-analysis RAG system. Given a ticker,
it gathers market data, recent news, optional sentiment, and SEC filing evidence;
builds a structured evidence context; drafts a citation-bearing investment memo;
and verifies the memo against the returned evidence.

The system is built around LangGraph orchestration, typed retrieval contracts,
`edgartools` SEC ingestion, local or hosted LLM providers selected by config, and
Chroma/Qdrant vector stores behind one retrieval boundary.

### 1.1 Current State

The implementation already contains:

- LangGraph workflow with persistent run state.
- Request controls for filing analysis, news sentiment, and news count.
- Structured evidence packets and memo verification.
- Section-aware SEC ingestion with deterministic chunk identity.
- Chroma and Qdrant implementations behind a retrieval contract.
- Retrieval benchmark precursor fixtures and comparison tooling.
- Memo-grounding evaluation tooling.

The July 2026 DataOps arc exists because memo-quality baselines were still run
against live-changing evidence. Any live-evidence result is at most a candidate
until replayed from an approved frozen evidence release.

### 1.2 Frontend

Frontend work is deferred to S8. Nothing in S0-S4 or Phase 0-Phase 3 depends on
frontend work.

---

## 2. Goals & Non-Goals

Goals:

- Build a reliable, testable, spec-driven financial RAG system.
- Keep evidence grounded, replayable, and verifiable.
- Make retrieval quality measurable with a trusted benchmark oracle.
- Keep vector backends behind stable interfaces.
- Publish only metrics that trace to code, config, model, and evidence versions.

Non-goals for the committed local-first arc:

- Broad multi-agent or swarm orchestration.
- Long-term memory.
- Trading, brokerage execution, or personalized investment advice.
- Frontend before S8.
- Prompt optimization before a stable evaluation oracle exists.
- Production cloud deployment before ADR-0006 is accepted and §3 is amended.

---

## 3. Scope

This section is the only canonical home for scope.

In scope:

- `edgartools` SEC parser stabilization.
- Section-coverage regression tests.
- Canonical section metadata and deterministic chunk IDs.
- Typed retrieval contracts and retrieval services.
- Chroma and Qdrant implementations behind those contracts.
- Qdrant local setup and payload indexes.
- Shared retrieval benchmark: fixture, validator, runner, paired comparator.
- Evidence-grounded memo generation and verification.
- Evidence snapshots, release registry, quality gates, and replay mode for local-first evaluation governance.
- CI and service-readiness checks that support the above.

Deferred, gated work:

- S5: hybrid dense+sparse retrieval, reranking, section-prior search.
- S6: answer-quality evaluation and prompt optimization.
- S7: service-readiness hardening.
- S8: frontend MVP.
- S9: portfolio polish.
- S10: optional multi-agent work after Gate E.
- Phase 4 cloud deployment only after ADR-0006 is accepted with provider, budget, teardown, and deploy-gate details.

Out of scope unless this section is explicitly changed and backed by an ADR:

- Broad multi-agent or swarm architectures.
- Long-term memory.
- Production cloud deployment.
- Trading or brokerage execution.
- Personalized investment advice.

---

## 4. Architecture Overview

Two paths share one evidence and retrieval spine:

1. Runtime path: request -> workflow -> tools/retrieval -> evidence context -> memo -> verification -> response.
2. Offline path: evidence capture and ingestion -> release artifacts -> evaluation -> approved baselines.

Runtime application code remains boring: API and agent layers consume configured
services and approved active pointers; release creation and validation live in
offline DataOps/evaluation tooling.

---

## 5. Module Design

Dependency rule: backend client objects never escape the retrieval boundary.

- `app/` — runtime API, workflow, services, tools, config, observability.
- `app/components/retrieval/` — retrieval contracts, evidence packets, Chroma/Qdrant stores, retrieval methods.
- `evaluation/` — offline benchmark, memo-grounding, and quality evaluation tooling.
- `dataops/` — local-first release contracts, registry, and quality gates.
- `tests/` — behavior-first tests.
- `scripts/ci/` — executable governance checks.

`dataops/` may define artifacts and validate manifests, but it must not import
Qdrant, Chroma, provider SDKs, or cloud SDKs.

---

## 6. Retrieval Contract

Retrieval consumers use typed packets and filters, not backend clients. The
contract must support document upsert/count/search, filtering by canonical
metadata, idempotent re-ingest, and backend selection by config.

Chroma and Qdrant may differ operationally, but they must remain substitutable
through public retrieval interfaces.

---

## 7. Evidence & Release Contracts

Every evidence-bearing artifact must preserve stable identity and provenance.

- SEC chunks use deterministic identity:
  `chunk_id = f(accession_number, section_key, chunk_index)`.
- Evidence snapshots use deterministic identity derived from source type,
  natural key, and payload hash.
- Snapshot payload changes create new snapshot IDs; no silent overwrite.
- Downstream release manifests reference the snapshot IDs they derive from.
- Quality baselines can be approved only when they record code version, model,
  config, evidence release, and a passing quality report.

---

## 8. Runtime Workflow

The workflow must:

- Respect request controls.
- Build evidence before drafting a memo.
- Return safe errors on tool or LLM failure.
- Map every memo citation to returned evidence.
- Avoid buy/sell directives and guaranteed-return language.

---

## 9. SEC Ingestion

SEC ingestion uses `edgartools` and canonical section keys. The S1 coverage lock
requires representative AAPL/MSFT/NVDA critical sections to produce non-zero
chunks through the store interface.

---

## 10. Evaluation & Benchmarking

Retrieval benchmark methodology lives in `docs/retrieval-benchmark.md`.

Memo-grounding evaluation is separate from the retrieval benchmark. It may assess
citation coverage and claim support, but it must not be used to claim retrieval
quality improvements.

Live-feed memo baselines are probes. Approved baselines require frozen evidence
replay.

---

## 11. Phase Gates

Gate A: shared retrieval benchmark oracle is validated and has a baseline.

Gate B: Qdrant backend passes parity and local operational checks behind the
retrieval contract.

Gate C: retrieval-quality method changes are proven by paired benchmark
comparisons.

Gate D: answer-quality methodology is stable enough for prompt/model changes.

Gate E: service readiness and optional later expansions have met their own
documented gates.

Phase 1 evidence freeze is a prerequisite for S2/Gate A work in the v2 DataOps
plan. No Qdrant or retrieval-method comparison may run on live evidence.

---

## 12. Sprint Map

Committed and gated sequencing:

| Sprint | Purpose |
| --- | --- |
| S0 | Spec coherence and baseline verification |
| S1 | SEC ingestion identity and section coverage |
| S2 | Shared retrieval benchmark oracle / Gate A |
| S3 | Qdrant backend parity / Gate B prep |
| S4 | Backend comparison and default decision / Gate B |
| S5 | Retrieval quality improvements / Gate C |
| S6 | Answer quality evaluation / Gate D |
| S7 | Service readiness |
| S8 | Frontend MVP |
| S9 | Portfolio polish |
| S10 | Optional multi-agent work / Gate E |

`docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md` adds Phase 0 and Phase 1
before S2 execution: governance docs into version control, then evidence
snapshot/replay. Phase 2 is S2's anchored fixture work on the frozen evidence
release.

---

## 13. Documentation Governance

Governance checks are executable:

- `scripts/ci/check_no_scope_residue.py`
- `scripts/ci/check_sprint_map.py`
- `scripts/ci/check_doc_sync.py`
- `scripts/ci/check_test_hygiene.py`

Before marking a task done, run the relevant verification subset and record the
result in `tasks/todo.md`.

---

## 14. Correction Handling

Corrections are project data. When a user corrects the agent, record correction,
root cause, prevention rule, whether it was applied, and verification result in
`tasks/lessons.md`, then apply the correction.

---

## 15. Architecture Decisions

Accepted ADRs:

- `docs/adr/ADR-0003-qdrant-behind-interface.md`
- `docs/adr/ADR-0004-edgartools-parser.md`
- `docs/adr/ADR-0005-unified-dataops-release-registry.md`

Proposed or planned ADRs:

- ADR-0006: production cloud deployment. Proposed only until accepted and this
  SPEC's scope is amended.
