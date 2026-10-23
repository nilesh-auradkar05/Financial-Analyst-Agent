# SPEC — Alpha Financial Analyst Agent

> **Version:** 1.4
> **Date:** 2026-10-11
> **Status:** Canonical product specification.
> **Repo:** `nilesh-auradkar05/Financial-Analyst-Agent`
> **Supersedes:** v1.3 (2026-09-12), which superseded v1.2 (2026-07-04) from `SPEC-AMENDMENT-v1.3.md` (retired; archived 2026-10-11 to `old_artifacts/docs/`; application record is ADR-0008).
> **Amended 2026-10-10:** S4 backend comparison retired; Gate B closes at S3 exit (§11.1, §12).
> **Amended 2026-10-17:** ADR-0009 amendment — Azure-first draft ladder (Foundry DeepSeek-V4-Flash → Bedrock `deepseek.v3.2`) and GPT-5.6 Luna judge (Bedrock → Foundry); Foundry is no longer fallback-only (§§3, 4).
> **v1.4 change (2026-10-08):** ADR-0006 accepted with ADR-0010 to ADR-0013 and amendments to ADR-0003, ADR-0007 and ADR-0009. Updated §§0, 2, 3, 4, 5, 11.1, 11.3, 12 and 16. Cloud work stays gated by the ADR-0006 entry criteria. Revised 2026-10-11: ADR-0014 moves the deployed index to Amazon OpenSearch Serverless with a daily awake cap (§§3, 5, 6, 16). Revised again 2026-10-11: ADR-0015 makes the deployed demo tier scale-to-zero (Lambda, DynamoDB, SQS, a Cloudflare Worker, guest access for recruiters) and keeps the always-on design as the documented production scale tier (§§0, 3, 5, 12, 16).

This is the product and architecture source of truth for the Alpha Financial
Analyst Agent. Where documents disagree, resolve the conflict using §0.
Methodology with a dedicated home is linked from this file rather than
duplicated here.

The amendment was drafted against section numbers that did not match v1.2
(workflow was labeled §9 there; v1.2 §9 is SEC ingestion). Inserts land in the
v1.2 homes: workflow clauses are §8.1–8.5, evidence appendices §7.x/§7.y,
quality-gate appendices §11.2–11.4, hooks §14. Sprint-plan traces that cited
amendment §9.1–9.5 are retargeted to §8.1–8.5.

---

## 0. Source Of Truth & Document Map

This section is the only canonical conflict-resolution order.

1. `docs/SPEC.md` — product scope, architecture, contracts, phase gates.
2. `docs/test-plan.md` — behavioral oracle.
3. `docs/sprint-plan.md` — sequencing and task definitions.
4. `docs/retrieval-benchmark.md` — benchmark fixture schema, matcher, metrics, validator, and comparison policy.
5. `docs/adr/*.md` — accepted architecture decisions.
6. `docs/System-design/**/*.excalidraw` (editable source) with `docs/png/**/*.png` and `docs/svg/**/*.svg` renders — diagram companions; `docs/cloud-deployment.md` — deployment design companion for ADR-0006 and ADR-0015 (deployed demo tier); `docs/production-scale-architecture.md` — production scale tier (interview design, not built).
7. `tasks/todo.md` — active task state and verification ledger.
8. `tasks/lessons.md` — correction log.
9. Code comments and stale TODOs — lowest authority.

Benchmark-specific override: for fixture schema, matcher behavior, metric
formulas, validation rules, and comparison policy,
`docs/retrieval-benchmark.md` governs even though `docs/sprint-plan.md`
controls sequencing.

`LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md` was the companion plan for
Phase 0 and Phase 1 (DataOps release governance and evidence replay), both now
complete. It is archived 2026-10-11 to `old_artifacts/docs/`, which is not tracked in git; last tracked at commit `b83fd66`.
References to "INTEGRATION_PLAN_v2" below name that document; where a section
still matters, this SPEC, `docs/sprint-plan.md` and `docs/cloud-deployment.md`
carry the content. It never amended scope by itself.

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
- Cloud deployment before the ADR-0006 entry criteria have recorded evidence.
- Multi-region serving, cross-region data copies or a recovery site (ADR-0006: region loss is a documented failure mode).

### 2.1 Known limitations (v1.3 baseline)

Recorded by the 2026-08-24 production-readiness review against commit
`HEAD@2026-07-03` unless stated. These are the gaps v1.3 sequences against;
they are not permission to ship them.

1. **Silent evidence gap.** `retrieve_sec_filings_node` queries the store only; a ticker never ingested yields `filing_chunks=[]` and the memo reports `status=completed`. No freshness check: `ingest_filing(replace_existing=False)` short-circuits if any chunk exists for the ticker.
2. **Verification is terminal.** `verify_memo_node` is a leaf; `passed=False` and orphan citations are logged as recoverable errors and the memo is returned unchanged.
3. **Serial, event-loop-blocking graph.** `research_news → fetch_stock → retrieve_filings → analyze_sentiment` are serial though independent; FinBERT runs synchronously inside an `async` node; graph is recompiled per request.
4. **Non-scalable service shell.** In-process `BackgroundTasks`, file-backed JSON job store rewritten per update, no auth, no rate limit, no idempotency key, no checkpointer.
5. **Eval exists but is not hardened.** Results are timestamped JSON without lineage (`git_sha`, `model`, `temperature`, `snapshot_hash`); nothing runs in CI; `app/agents/graph.py` imports from `evaluation/` (layering inversion).
6. **No guardrails, no caching** at any layer.
7. **Governance docs and CI.** Four `scripts/ci/check_*.py` exist; wiring them into `ci.yml` is G11 / S2-T00a.

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

3.x In scope (added v1.3):

- Evidence-completeness semantics: a memo produced with a missing evidence class is a distinct terminal status, never `completed`.
- Bounded verification repair loop (draft → verify → revise, max 2 revisions).
- Guardrail layer: input schema hardening, untrusted-content framing, output policy screen.
- Evidence and LLM response caching keyed on EvidenceSnapshot hash (§11.3).
- Eval run registry with lineage and a CI regression gate (§11.2).
- Agent-tooling hooks (§14) enforcing governance rules at the coding-agent boundary.

Deferred, gated work:

- S5: hybrid dense+sparse retrieval, reranking, section-prior search.
- S6: answer-quality evaluation, eval registry, bounded repair loop, prompt optimization.
- S7: service-readiness hardening (guardrails, caching, queue/worker, auth).
- S8: frontend MVP.
- S9: portfolio polish.
- S10: cloud deployment (INTEGRATION_PLAN_v2 Phases 4–5) per ADR-0006 (accepted 2026-10-08), ADR-0010 to ADR-0013 and ADR-0015 (2026-10-11): AWS `us-east-1` serves a scale-to-zero demo tier (Lambda functions, DynamoDB, SQS, a Cloudflare Worker in front; guest access for recruiters, Cognito for the admin); Microsoft Foundry is the model plane only (primary draft host and fallbacks, no state; ADR-0009 amendment 2026-10-17); $100/month kill switch with $10 and $25 alerts; deployed index on Amazon OpenSearch Serverless with a 45-minute daily awake cap (ADR-0014); Terraform teardown; eval-gated deploy. S10 starts only when the ADR-0006 entry criteria have recorded evidence. Design companion: `docs/cloud-deployment.md`.

3.y Out of scope (ADR-0006 accepted 2026-10-08; ADR-0013):

- Cloud resources before the ADR-0006 entry criteria are met.
- Multi-region serving, cross-region data copies and recovery sites.
- Production scale-tier components (ECS Fargate services, Aurora PostgreSQL, Valkey, VPC with NAT) before their ADR-0015 graduation triggers are measured; they are designed in `docs/production-scale-architecture.md`, not built.
- Multi-agent decomposition beyond the single reviewer loop in §8.1.
- Semantic caching of memo *outputs* (prohibited; see §11.3).

Out of scope unless this section is explicitly changed and backed by an ADR:

- Broad multi-agent or swarm architectures.
- Long-term memory.
- Cloud deployment outside the gated S10 item (see 3.y).
- Trading or brokerage execution.
- Personalized investment advice.

---

## 4. Architecture Overview

Runtime provider policy (authorized 2026-09-29, ADR-0009; configuration amended 2026-09-30; ladder amended 2026-10-17): one variable per provider tier. Draft ladder, local and deployed: Microsoft Foundry DeepSeek-V4-Flash (AZURE_OPENAI_ENDPOINT, AZURE_API_KEY, AZURE_FOUNDRY_MODEL = deployment name; used only when all three are set) → Bedrock (AWS_BEDROCK_MODEL, default `deepseek.v3.2`) → personal Anthropic (CLAUDE_LLM_MODEL with ANTHROPIC_API_KEY) → personal OpenAI (OPENAI_LLM_MODEL with OPENAI_API_KEY). Each attempt is bounded; a failure, timeout or rate-limit response moves to the next configured tier; an unconfigured tier is skipped. There is no generic provider selector and no Ollama chat path; Ollama serves embeddings only. Native traces identify actual attempted models. Judge ladder (evaluation only): GPT-5.6 Luna on Bedrock (`bedrock-mantle`) → GPT-5.6-luna on Foundry (AZURE_FOUNDRY_JUDGE_MODEL); the runtime gate stays the heuristic grounding verifier. Personal Anthropic and OpenAI keys are development-only.

Ingest on miss (authorized 2026-10-10): when analysis finds zero indexed chunks for a ticker, it ingests that ticker's latest 10-K once, off the event loop and under a time bound, then retrieves again. A ticker that already has indexed chunks is never re-ingested by analysis. An ingestion failure is reported as filing_ingestion_failed, distinct from an absent 10-K.

Unavailable filings: an absent 10-K yields a specific filing_not_found response. Analysis continues using available stock/news/sentiment and states SEC Filings: Not Available in the memo with a safe ticker-specific reason. Required absent filings retain §8.5 evidence_missing status; verification is never bypassed. Actual upstream failures remain distinct from absent data.

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
- `app/agents/` — LangGraph graph, state, routing (fan-out allowed; nodes must be non-blocking).
- `app/components/retrieval/` — retrieval contracts, evidence packets, Chroma/Qdrant stores (local), OpenSearch Serverless store (deployed, ADR-0014), retrieval methods.
- `app/verification/` — runtime memo verifier (moved from `evaluation/grounding.py`). `evaluation/` MUST NOT be imported by `app/`.
- `app/guardrails/` — `input.py` (schema), `content.py` (untrusted-source framing), `output.py` (policy screen).
- `app/dataops/` — EvidenceSnapshot build/freeze/replay (INTEGRATION_PLAN_v2 §3.4). Runtime replay and the snapshot hash live here (S2-T00b); release contracts, registry and gates stay in repo-root `dataops/`.
- `app/cache/` — EvidenceCache (per-source TTL), QueryEmbeddingCache (exact, keyed by embedding fingerprint), RetrievalCache (keyed by index release and retrieval config), MemoCache (`snapshot_hash` + `prompt_version` + model and verifier/policy fingerprints). Key formats: ADR-0013.
- `app/services/` — tools, run store (`RunStore` protocol; DynamoDB adapter per ADR-0015), sentiment (FinBERT wrapped in `to_thread` or out-of-process; ONNX in the deployed worker).
- `evaluation/` — offline harness + registry; imports from `app/` only, never the reverse.
- `dataops/` — local-first release contracts, registry, and quality gates. Must not import Qdrant, Chroma, provider SDKs, or cloud SDKs.
- `tests/` — behavior-first tests.
- `scripts/ci/` — executable governance checks.
- `.claude/hooks/` — governance hooks (§14); mirrored for Codex via the same scripts.

---

## 6. Retrieval Contract

Retrieval consumers use typed packets and filters, not backend clients. The
contract must support document upsert/count/search, filtering by canonical
metadata, idempotent re-ingest, and backend selection by config.

Chroma, Qdrant and the deployed OpenSearch Serverless store (ADR-0014) may differ operationally, but they must remain substitutable
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

7.x EvidenceSnapshot (from INTEGRATION_PLAN_v2 §3.2) carries `snapshot_hash` =
sha256 over the canonical JSON of all EvidencePackets + source manifest.
`snapshot_hash` MUST appear in: AgentState, AnalysisResponse, every eval run
record, and every cache key.

7.y `evidence_as_of` (ISO-8601 with a UTC offset) MUST be the oldest fetch time
among the sources the memo is drafted from, and is returned in AnalysisResponse.
It is never the run start time.

- Replay: the oldest `fetched_at` among the release's snapshots for the ticker.
- Live news and market data: the time the run fetched them.
- Live filings: the time the filing was fetched at ingestion, recorded as
  `fetched_at` in each chunk's metadata. A chunk indexed before that field
  existed falls back to its `filing_date`, which is never later than the real
  fetch time, so the reported value can be older than the truth but not newer.
- A stored time that cannot be read is treated as absent; it never fails the
  retrieval or removes evidence.
- A source that returned no evidence contributes no time. When no source
  contributed, `evidence_as_of` is absent.
- Sentiment is computed from the news, not fetched, and contributes no time.

7.z Runtime replay (S2-T00b). `run_agent(..., evidence_release="<name:version>")`
seeds the run from that registered `evidence_snapshot` release and calls no
live evidence tool, vector store, embedding model or sentiment model; only memo
drafting and verification run. `snapshot_hash` covers the evidence the memo is
drafted from plus the source manifest (release id and snapshot ids; empty for a
live run). The release is an internal argument and is not accepted from API
clients.

---

## 8. Runtime Workflow

The workflow must:

- Respect request controls.
- Build evidence before drafting a memo.
- Return safe errors on tool or LLM failure.
- Map every memo citation to returned evidence.
- Avoid buy/sell directives and guaranteed-return language.

8.1 Graph topology:

```text
START → [research_news ‖ fetch_stock ‖ retrieve_filings]  (parallel fan-out, LangGraph Send/branch)
      → analyze_sentiment → guard_inputs → draft_memo → verify_memo
      → (passed | attempts ≥ 2) ? guard_output → END : revise_memo → verify_memo
```

8.2 No node may block the event loop: CPU-bound work (FinBERT, tokenization)
runs via `asyncio.to_thread` or a separate process; store calls are awaited or
threaded.

8.3 The compiled graph is a module-level singleton; LLM clients are constructed
once per process.

8.4 State reducers: `errors` uses an append reducer; nodes never mutate input
state.

8.5 Terminal statuses: `completed` | `degraded` (verification failed after max
attempts) | `evidence_missing` (a required evidence class absent) | `failed`.
`completed` requires `verification.passed == True` AND no evidence class missing.

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

## 11. Phase Gates & Quality Policy

11.1 Phase gates

Gate A: shared retrieval benchmark oracle is validated and has a baseline.

Gate B: Qdrant backend passes parity and local operational checks behind the
retrieval contract. Amended 2026-10-10 (owner decision): Gate B closes at S3
exit; the Chroma-vs-Qdrant paired comparison is retired and no quality claim is
made for the backend choice (ADR-0003 addendum).

Gate C: retrieval-quality method changes are proven by paired benchmark
comparisons.

Gate D: answer-quality methodology is stable enough for prompt/model changes.

Gate E: S10 cloud work. ADR-0006 was accepted on 2026-10-08; no cloud resource exists until its entry criteria have recorded evidence.

Phase 1 evidence freeze is a prerequisite for S2/Gate A work in the v2 DataOps
plan. No Qdrant or retrieval-method comparison may run on live evidence.

A run on live evidence cannot be classified `approved`.

11.2 Eval registry and regression gate

- Every eval run writes `evaluation/registry/runs/<run_id>.json` with:
  `run_id`, `git_sha`, `branch`, `model`, `temperature`, `prompt_version`,
  `evidence_release`, `snapshot_hash`, `fixture_version`, `metrics{...}`,
  `classification` ∈ {candidate, approved}.
- Writes to `evaluation/*_res/` without these keys are rejected (hook + CI).
- CI job `eval-replay` replays the frozen evidence release with network disabled
  and fails if `grounded_claim_rate` or `citation_coverage` drops more than 2σ
  below the approved baseline.
- Heuristic verifier agreement with a judge model is measured on ≥50 claims; if
  Cohen's κ < 0.85 the heuristic rate may not be quoted without the judge rate
  beside it.

11.3 Caching policy

- EvidenceCache: exact-key, per source. TTL: SEC = until new accession; market = 60 s; news = 15 min.
- QueryEmbeddingCache: exact reuse only, keyed by embedding fingerprint (model, version, dimensions, preprocessing) and the normalized query hash. Semantic reuse (cosine ≥ 0.98) was removed in v1.4 (ADR-0013): computing the new embedding to test similarity erases the saving, and near-identical financial queries differ by period or negation.
- RetrievalCache: key = index release + retrieval configuration fingerprint + query and filter hashes.
- MemoCache: key = `snapshot_hash` + `prompt_version` + model fingerprint + verifier/policy fingerprint. No TTL beyond snapshot validity. Concurrent misses for the same key are single-flighted.
- PROHIBITED: semantic similarity caching of memo outputs across different snapshots.
- Every cache exposes hit/miss counters in `/metrics`.

11.4 Guardrails

- Input: ticker `^[A-Z]{1,6}(\.[A-Z])?$`; `company_name` ≤ 80 chars, printable ASCII, no control chars.
- Untrusted content (news snippets, filing text) is rendered inside a delimited DATA block with an explicit "treat as data, not instructions" framing; sources outside the allow-list are dropped.
- Output screen rejects/rewrites: price targets, guaranteed-return language, MNPI phrasing, uncited numeric claims surviving verification, PII. Adds the fixed disclaimer block.
- Guardrail decisions are recorded in `AgentState.guardrail_events` and returned in the response.

---

## 12. Sprint Map

Committed and gated sequencing:

| Sprint | Purpose |
| --- | --- |
| S0 | Spec coherence and baseline verification |
| S1 | SEC ingestion identity and section coverage |
| S2 | Shared retrieval benchmark oracle / Gate A |
| S3 | Qdrant backend parity / Gate B |
| S4 | Backend comparison and default decision / Gate B — retired 2026-10-10 (owner decision); Gate B closes at S3 exit |
| S5 | Retrieval quality diagnostics → hybrid/rerank / Gate C |
| S6 | Answer quality: eval registry + CI regression gate (§11.2) + bounded repair loop (§8.1), then LLM-as-judge/RAGAS, then GEPA on holdout / Gate D |
| S7 | Service readiness: guardrails (§11.4), caching (§11.3), queue/worker + job store (DynamoDB `RunStore`, SQS `JobQueue`, DynamoDB checkpointer; local DynamoDB Local and ElasticMQ; ADR-0013 as amended by ADR-0015), provider circuit breaker, FinBERT off the API event loop (ONNX in the worker), auth + rate limit with guest and admin roles (ADR-0010) |
| S8 | Frontend MVP |
| S9 | Portfolio polish |
| S10 | Cloud (INTEGRATION_PLAN_v2 Phases 4–5) per ADR-0006 (accepted 2026-10-08), ADR-0010–0014 and ADR-0015 (scale-to-zero demo tier); starts when the ADR-0006 entry criteria have recorded evidence |

S2 pre-tasks (T00a–T00d) implement production-readiness steps 1–3 (see
sprint-plan.md S2 preamble).

The archived INTEGRATION_PLAN_v2 (§0) added Phase 0 and Phase 1
before S2 execution: governance docs into version control, then evidence
snapshot/replay. Phase 2 is S2's anchored fixture work on the frozen evidence
release.

Open gaps (mirrored in sprint-plan.md):

- **G10** — `app/` depends on `evaluation/` (`graph.py` imports `evaluation.grounding`). Resolve in S2-T00d by moving the runtime verifier to `app/verification/`.
- **G11** — CI does not invoke `scripts/ci/check_*.py`. Resolve in S2-T00a.
- **G12** — `add_error` mutates `state["errors"]` in place; unsafe under fan-out. Resolve in S2-T00c.
- **G13** — `/health` reports `degraded` on Ollama outage even when Bedrock is the configured provider. Resolve in S7.

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

## 14. Agent-tooling hooks

14.1 Governance rules that can be checked mechanically at the coding-agent
boundary are enforced by hooks in `.claude/settings.json` (Claude Code) with
scripts in `.claude/hooks/`. The same scripts are registered for Codex CLI
(six-event subset).

14.2 Blocking hooks exist only on PreToolUse, UserPromptSubmit, Stop. Others
observe or inject.

14.3 Hook coverage (id → rule): H1 session context; H2 plan-before-implement;
H3 governance docs read-only; H4 frozen fixtures immutable; H5 eval results need
lineage (§11.2); H6 single-axis benchmark discipline; H7 replay tests run without
network; H8 secrets; H9 destructive commands; H10 lint/type on write; H11 lessons
capture (owner corrections only: not stop-hook feedback, agent hand-backs or subagents;
revised 2026-10-22); H12 stop gate (tests + doc sync + clean tree); H13 quality gate
(added 2026-10-22, owner instruction: `ruff check .` + `mypy .` at Stop after a code change;
`ruff check .` + `mypy .` + `pytest` before every `git commit`, with a `test(<id>): red`
commit limited to tests/ and its test files excluded from mypy and pytest).

14.4 Hooks are the first line; `scripts/ci/check_*.py` in CI are the second. A
rule enforced by a hook without a CI counterpart is a defect (tracked as G11).

---

## 15. Correction Handling

Corrections are project data. When a user corrects the agent, record correction,
root cause, prevention rule, whether it was applied, and verification result in
`tasks/lessons.md`, then apply the correction.

---

## 16. Architecture Decisions

Accepted ADRs:

- `docs/adr/ADR-0003-qdrant-behind-interface.md` (local default; its 2026-10-08 managed-Qdrant addendum is superseded for deployment by ADR-0014 and kept as the fallback)
- `docs/adr/ADR-0004-edgartools-parser.md`
- `docs/adr/ADR-0005-unified-dataops-release-registry.md`
- `docs/adr/ADR-0006-production-cloud-deployment.md` (accepted 2026-10-08; implementation gated by its entry criteria)
- `docs/adr/ADR-0008-production-readiness-order.md`
- `docs/adr/ADR-0009-runtime-provider-fallback.md` (amendment 2026-10-08: production model ladder)
- `docs/adr/ADR-0010-identity-and-authorization.md`
- `docs/adr/ADR-0011-managed-embedding-and-rerank-models.md` (provisional on Gate C)
- `docs/adr/ADR-0012-ingestion-orchestration.md`
- `docs/adr/ADR-0013-durable-execution-and-caching.md`
- `docs/adr/ADR-0014-opensearch-serverless-vector-store.md` (accepted 2026-10-11; adoption gated by a paired retrieval comparison)
- `docs/adr/ADR-0015-scale-to-zero-serverless-deployment.md` (accepted 2026-10-11; supersedes in part ADR-0006, ADR-0012 and ADR-0013 for the demo tier; their always-on design is the production scale tier)

Proposed ADRs:

- ADR-0007: production interview scale design. Proposed; interview companion only. Superseded in part on 2026-10-08 and 2026-10-11 by ADR-0006 and ADR-0009 to ADR-0015.


## REST service contract clarification — 2026-09-21

Under §§3/5/8/11/12, the authorized focused service-readiness slice protects
analysis, ingestion, jobs, stats and metrics with a configured API key. Root and
health are public. Async acceptance is 202 with Location; terminal job/result
statuses agree and distinguish fatal failure, missing required evidence, failed
or absent verification, and verified completion in that precedence order.
Errors share a safe typed envelope and financial/job/error responses forbid
caching. Per-key submission limits and atomic async idempotency apply within the
existing single-process service; no distributed execution guarantee is claimed.
Health distinguishes model availability from configuration-only Bedrock checks;
no paid inference probe or successful-inference claim is made. Sequencing is
recorded in sprint-plan's authorized exception; broader S7 work remains gated.

## Run telemetry contract — 2026-10-07 (API-RUN-TELEMETRY)

Authorized by the user on 2026-10-07 as S7 service-readiness support for the S8 frontend. Additive and optional; existing fields and statuses are unchanged.

- `GET /jobs/{job_id}` returns `progress`: one entry per workflow node that has started, in start order, with `node`, `status` (`running`, `completed`, `degraded`), `started_at`, and, once finished, `ended_at` and `duration_ms`. A node whose update reports an error is `degraded`. Entries appear while the job is running and remain after it finishes. Jobs recorded before this contract return an empty list.
- `AnalysisResponse` adds `filing_chunk_count` (chunks retrieved for the run) and `usage` (`model`, `input_tokens`, `output_tokens` as reported by the provider that produced the memo; absent values are null, never estimated).
- `executive_summary` is the memo's Executive Summary paragraph; the "Analysis completed for …" text is only the fallback when the memo has no such section.
- `GET /jobs?limit=` (added 2026-10-07, RUNS-AVAILABLE) lists stored runs newest first as summaries (`job_id`, `ticker`, `status`, timestamps, grounding rates, latency) without memo or evidence; same API-key protection as other job routes. It is not user-scoped: per-user visibility arrives with the managed-auth decision. Compose persists the run store in a named volume so image rebuilds do not erase runs.
- Out of scope here: cost (needs a price table decision), stance/confidence (prompt change, S6), evidence snapshot id (S2-T00b/S7), per-citation source text.
