# SPEC — Alpha Financial Analyst Agent

> **Version:** 1.5 (draft; takes effect when ADR-0018 is accepted)
> **Date:** 2026-11-01
> **Status:** Canonical product specification.
> **Repo:** `nilesh-auradkar05/Financial-Analyst-Agent`
> **Supersedes:** v1.4 (2026-10-11), which superseded v1.3 (2026-09-12) and v1.2 (2026-07-04) from `SPEC-AMENDMENT-v1.3.md` (retired; archived 2026-10-11 to `old_artifacts/docs/`; application record is ADR-0008).
> **Amended 2026-10-31 (ADR recorded 2026-11-01; SPEC text recorded 2026-11-02):** ADR-0017 accepted — three agents (Researcher, Analyst, Reviewer) inside one run, three-tier execution memory (T1 working, T2 execution, T3 long-term), fenced checkpoints, an effect journal, handoff envelopes and crash resume at the failed agent (§§2, 3, 5, 8.6–8.8, 12 G14–G15, 16). No per-ticker episodic memory (owner decision).
> **v1.5 change (2026-11-01, owner-requested architecture revision; ADR-0018 proposed, renumbered from a colliding "ADR-0016" on 2026-11-02):** a bounded five-specialist decomposition under a deterministic supervisor, memory tiers with the "memory is context, never evidence" invariant, an MCP tool plane behind a gateway, asynchronous execution inside the worker, nine fingerprinted cache stages and agent-specific NFR contracts. Updated §§0, 2, 3, 5, 8.1, 11.1 (Gate F), 11.3, 12 (S11) and 16. Design companion: `docs/agent-architecture.md`. ADR-0018 is a proposed extension of the accepted ADR-0017; where they overlap ADR-0017 governs (three agents are the baseline, T1–T3 is the memory model, episodic memory is excluded unless the owner reverses ADR-0017 §1). Until ADR-0018 is accepted the §3.z scope lines are inactive.
> **Amended 2026-10-10:** S4 backend comparison retired; Gate B closes at S3 exit (§11.1, §12).
> **Amended 2026-11-07:** ADR-0019 (accepted) — news sentiment is a structured-output call to Claude Sonnet 5.5 on Microsoft Foundry; FinBERT and every local model leave the runtime (§§5, 7, 8.2); the model plane is cloud-only (owner decision).
> **Amended 2026-10-23:** ADR-0016 — Qdrant and Chroma retired; OpenSearch (local container for development and tests, Serverless in the cloud) is the only store; the data and model plane moves to managed services ahead of S7 (§§1, 3, 5, 6, 11.1, 12, 16).
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
6. `docs/svg/adr-0017/*.svg` (ADR-0017 diagrams, owner-authored) and `docs/System-design/agents/*.archify.json` (editable Archify sources for the ADR-0018 proposal) with `docs/System-design/agents/*.html`, `docs/png/agents/*.png` and `docs/svg/agents/*.svg` renders — diagram companions (the Excalidraw sets for the demo and scale tiers were retired on 2026-10-31; the Mermaid figures in `docs/cloud-deployment.md` and `docs/production-scale-architecture.md` are their companions now); `docs/cloud-deployment.md` — deployment design companion for ADR-0006, ADR-0015 and ADR-0017 (deployed demo tier, D20); `docs/production-scale-architecture.md` — production scale tier (interview design, not built); `docs/agent-architecture.md` — agent architecture revision companion for ADR-0018 (memory tiers, five specialists, MCP tool plane, async execution, cache stages, agent NFRs; proposed, not built).
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
an OpenSearch vector store behind one retrieval boundary (local container in development, Amazon OpenSearch Serverless in the cloud; ADR-0016).

### 1.1 Current State

The implementation already contains:

- LangGraph workflow with persistent run state.
- Request controls for filing analysis, news sentiment, and news count.
- Structured evidence packets and memo verification.
- Section-aware SEC ingestion with deterministic chunk identity.
- Chroma and Qdrant implementations behind a retrieval contract (retired by ADR-0016; removed in CM-7).
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

- Broad multi-agent or swarm orchestration. The three agents of ADR-0017 (Researcher, Analyst, Reviewer; S7.6) and the bounded five-specialist refinement of ADR-0018 (S11, proposed) are not this: the agent count is fixed, routing is code, and there is no LLM planner.
- Long-term memory beyond ADR-0017 §1: T3 holds only the snapshot store, index releases and fingerprint-keyed caches, written after verification and never cited as evidence. Per-ticker episodic memory is excluded by owner decision; ADR-0018's episodic tier (M1) is conditional on the owner reversing that decision and is otherwise out of scope.
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
- OpenSearch implementation behind those contracts (ADR-0016): a local container for development and tests, Amazon OpenSearch Serverless in the cloud. Chroma and Qdrant are retired and removed in CM-7.
- Managed data and model plane ahead of S7 (ADR-0016): OpenSearch Serverless, Bedrock and Foundry embeddings and models, hosted sentiment. The app, worker and ingestion keep running locally.
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

3.z In scope (added v1.5, ADR-0018 proposed 2026-11-01; effective on acceptance; built in S11 after S7.1–S7.12 (including the ADR-0017 three-agent split, S7.6) and Gate D; adopted only on Gate F):

- Exactly five specialists (`market`, `news`, `filings`, `analyst`, `verifier`) under a deterministic LangGraph supervisor with a fixed topology, as a refinement of ADR-0017's three agents: `market`, `news` and `filings` split the Researcher's failure domain into three budgeted, allowlisted specialists; `analyst` and `verifier` are the Analyst and Reviewer. Tool-using specialists make no model call and model-calling specialists use no tools.
- Memory tiers mapped onto ADR-0017: M0 working = T1; M2 semantic (issuer profile, C9) and M3 procedural (versions) live in T3 under its write-after-verification rule; the invariant "memory is context, never evidence" and replay pinning (`memory_ref`). M1 episodic (`app/memory/` episodic records, `DELETE /memory`) is **conditional** on an owner reversal of ADR-0017 §1 and is not built otherwise.
- A tool gateway (`app/tools/`) fronting in-process tools and allowlisted, read-only, schema-pinned MCP servers; the analyst exposed as an MCP server at `/mcp` over the existing auth and quotas.
- Asynchronous execution inside the worker: specialist fan-out, concurrent filing queries, per-target semaphores, provider token bucket, nested deadlines with cancellation; resume after a crash uses ADR-0017's fenced checkpoints and effect journal (§8.7–8.8), not a separate mechanism.
- Cache stages C0–C9 (§11.3) with per-stage metrics.
- Agent NFR contracts (§10 of the companion): bounded autonomy, degradation ladder, breakers, tenant isolation, deterministic replay of tool calls and memory, per-agent observability, Gate F.

Deferred, gated work:

- S5: hybrid dense+sparse retrieval, reranking, section-prior search.
- S6: answer-quality evaluation, eval registry, bounded repair loop, prompt optimization.
- S7: service-readiness hardening (guardrails, caching, queue/worker, auth).
- S8: frontend MVP.
- S9: portfolio polish.
- S10: cloud deployment (INTEGRATION_PLAN_v2 Phases 4–5) per ADR-0006 (accepted 2026-10-08), ADR-0010 to ADR-0013 and ADR-0015 (2026-10-11): AWS `us-east-1` serves a scale-to-zero demo tier (Lambda functions, DynamoDB, SQS, a Cloudflare Worker in front; guest access for recruiters, Cognito for the admin); Microsoft Foundry is the model plane only (primary draft host and fallbacks, no state; ADR-0009 amendment 2026-10-17); $100/month kill switch with $10 and $25 alerts; deployed index on Amazon OpenSearch Serverless with a 45-minute daily awake cap (ADR-0014); Terraform teardown; eval-gated deploy. S10 starts only when the ADR-0006 entry criteria have recorded evidence. Design companion: `docs/cloud-deployment.md`.

3.y Out of scope (ADR-0006 accepted 2026-10-08; ADR-0013):

- Cloud resources before the ADR-0006 entry criteria are met, except the data and model plane resources ADR-0016 authorizes (budget alarms, the OpenSearch Serverless collection, model and embedding endpoints), each created only with owner approval.
- Multi-region serving, cross-region data copies and recovery sites.
- Production scale-tier components (ECS Fargate services, Aurora PostgreSQL, Valkey, VPC with NAT) before their ADR-0015 graduation triggers are measured; they are designed in `docs/production-scale-architecture.md`, not built.
- Multi-agent decomposition beyond the five specialists and the single reviewer loop of §8.1 (ADR-0018): no LLM planner, no per-source summarizers, no additional critics without a further ADR.
- Semantic caching of memo *outputs* (prohibited; see §11.3).
- Memory used as citable evidence, cross-principal memory, vector-indexed episodic recall, runtime MCP tool discovery, and MCP tools with side effects (ADR-0018).

Out of scope unless this section is explicitly changed and backed by an ADR:

- Broad multi-agent or swarm architectures (the bounded decomposition of 3.z is the only admitted form).
- Long-term memory outside the tiers and invariants of 3.z.
- Cloud deployment outside the gated S10 item (see 3.y).
- Trading or brokerage execution.
- Personalized investment advice.

---

## 4. Architecture Overview

Runtime provider policy (authorized 2026-09-29, ADR-0009; configuration amended 2026-09-30; ladder amended 2026-10-17): one variable per provider tier. Draft ladder, local and deployed: Microsoft Foundry DeepSeek-V4-Flash (AZURE_OPENAI_ENDPOINT, AZURE_API_KEY, AZURE_FOUNDRY_MODEL = deployment name; used only when all three are set) → Bedrock (AWS_BEDROCK_MODEL, default `deepseek.v3.2`) → personal Anthropic (CLAUDE_LLM_MODEL with ANTHROPIC_API_KEY) → personal OpenAI (OPENAI_LLM_MODEL with OPENAI_API_KEY). Each attempt is bounded; a failure, timeout or rate-limit response moves to the next configured tier; an unconfigured tier is skipped. There is no generic provider selector and no Ollama path (removed in CM-6e). Embeddings: one tier, Cohere Embed v5 fast on Foundry (`EMBEDDING_PROVIDER=foundry_cohere`, 1024 d, ADR-0011 amendment 2026-11-09), paced per process at no more than 18 requests and 18,000 estimated tokens (⌈characters ÷ 4⌉, at most 6,000 per request) per minute, under the deployment's ceilings of 20 requests and 20,000 tokens per minute; on a throttle or provider failure it waits for `Retry-After`, retries once, then raises a typed error, and the filings step reports the gap (`evidence_missing`, §8.5); there is no embedding fallback model. Sentiment is a separate single-tier call (Foundry Claude Sonnet 5.5, `AZURE_FOUNDRY_SENTIMENT_MODEL`; ADR-0019) with no fallback: a failure is a sentiment gap, never a substitute model. Native traces identify actual attempted models. Judge ladder (evaluation only): GPT-5.6 Luna on Bedrock (`bedrock-mantle`) → GPT-5.6-luna on Foundry (AZURE_FOUNDRY_JUDGE_MODEL); the runtime gate stays the heuristic grounding verifier. Personal Anthropic and OpenAI keys are development-only.

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
- `app/agents/` — LangGraph graph, state, routing (fan-out allowed; nodes must be non-blocking). From S7.6 (ADR-0017): three agent subgraphs (Researcher, Analyst, Reviewer) handing off through `HandoffEnvelope`. From S11 (ADR-0018): `supervisor.py` (fixed topology, code routing), `context.py` (`RunContext`), `specialists/` (`SpecialistAgent` protocol and the five specialists); specialists import the tool gateway and the memory protocol, never a client.
- `app/execution/` (S7.3–S7.5, S7.8; ADR-0017) — `FencedCheckpointer` over the SQLite, DynamoDB and Postgres savers, `EffectJournal`, `HandoffEnvelope`, the error classifier and the fault-injection harness. Imports nothing from `app/agents/`.
- `app/tools/` (S11, ADR-0018) — `ToolGateway`, `ToolSpec` registry from Pydantic settings with pinned schema hashes, in-process adapters, and the only module that imports the MCP SDK client. Read-only tools only.
- `app/mcp/` (S11, ADR-0018) — the analyst exposed as an MCP server at `/mcp`; a thin adapter over the API routes' services, never over the graph.
- `app/memory/` (S11, ADR-0018) — `MemoryStore` protocol and `IssuerProfile`, adapters (in-memory, DynamoDB; Postgres at its scale-tier trigger) and `MEMORY` block rendering; `EpisodicRecord` and delta computation only if the owner reverses ADR-0017 §1. No store client escapes it.
- `app/components/retrieval/` — retrieval contracts, evidence packets, the OpenSearch store (local container and Serverless; ADR-0014, ADR-0016), retrieval methods. The Chroma and Qdrant adapters remain only until CM-7 removes them.
- `app/verification/` — runtime memo verifier (moved from `evaluation/grounding.py`). `evaluation/` MUST NOT be imported by `app/`.
- `app/guardrails/` — `input.py` (schema), `content.py` (untrusted-source framing), `output.py` (policy screen).
- `app/dataops/` — EvidenceSnapshot build/freeze/replay (INTEGRATION_PLAN_v2 §3.4). Runtime replay and the snapshot hash live here (S2-T00b); release contracts, registry and gates stay in repo-root `dataops/`.
- `app/cache/` — EvidenceCache (per-source TTL), QueryEmbeddingCache (exact, keyed by embedding fingerprint), RetrievalCache (keyed by index release and retrieval config), MemoCache (`snapshot_hash` + `prompt_version` + model and verifier/policy fingerprints). Key formats: ADR-0013.
- `app/services/` — tools, run store (`RunStore` protocol; DynamoDB adapter per ADR-0015), sentiment (Claude Sonnet 5.5 on Microsoft Foundry, deployment set by `AZURE_FOUNDRY_SENTIMENT_MODEL`, structured output, bounded timeout and retries; ADR-0019).
- `evaluation/` — offline harness + registry; imports from `app/` only, never the reverse.
- `dataops/` — local-first release contracts, registry, and quality gates. Must not import vector-store SDKs (OpenSearch, Qdrant, Chroma), provider SDKs, or cloud SDKs.
- `tests/` — behavior-first tests.
- `scripts/ci/` — executable governance checks.
- `.claude/hooks/` — governance hooks (§14); mirrored for Codex via the same scripts.

---

## 6. Retrieval Contract

Retrieval consumers use typed packets and filters, not backend clients. The
contract must support document upsert/count/search, filtering by canonical
metadata, idempotent re-ingest, and backend selection by config.

The local OpenSearch container and OpenSearch Serverless (ADR-0014, ADR-0016) may differ operationally, but they must remain substitutable
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

8.1b Three agents from S7.6 (ADR-0017 §2, recorded here 2026-11-02): the graph splits into Researcher (`research_news ‖ fetch_stock ‖ retrieve_filings → analyze_sentiment → freeze_snapshot`), Analyst (`guard_inputs → draft_memo`, or `revise_memo` on a returned envelope) and Reviewer (`verify_memo → decide → guard_output`), each a subgraph with its own retry policy and failure domain, running inside one graph execution. No agent starts until the handoff envelope before it is durably committed.

8.1a Supervisor topology from S11 (ADR-0018; refines 8.1b by splitting the Researcher into `market`, `news` and `filings`; default only after Gate F, until then behind a flag):

```text
START → guard_inputs → memory_read_issuer
      → [market ‖ news ‖ filings]                 (news includes sentiment; filing queries concurrent)
      → freeze_snapshot (manifest += memory_ref, tool_calls) → memo_cache_lookup
          hit  → verifier
          miss → [memory_read_episodic, only if ADR-0017 §1 is reversed] → analyst → verifier
      → (passed | revisions ≥ 2) ? guard_output → persist → memory_write → END
                                 : analyst(revise) → verifier
```

Rules: the supervisor is code; its only data-dependent branch is the verifier's bounded loop. `market`, `news` and `filings` make no model call; `analyst` and `verifier` call no tool. Memory items carry no citation index and the verifier rejects a claim supported only by memory. Memory is written only after the fenced terminal commit. Replay runs `memo_cache_lookup → analyst → verifier → guard_output` from the release manifest and reads no live tool or memory.

8.2 No node may block the event loop: CPU-bound work (tokenization) and
blocking provider calls (sentiment, ADR-0019) run via `asyncio.to_thread` or an
async client; store calls are awaited or threaded.

8.3 The compiled graph is a module-level singleton; LLM clients are constructed
once per process.

8.4 State reducers: `errors` uses an append reducer; nodes never mutate input
state.

8.5 Terminal statuses: `completed` | `degraded` (verification failed after max
attempts) | `evidence_missing` (a required evidence class absent) | `failed`.
`completed` requires `verification.passed == True` AND no evidence class missing.

8.6 Bounded revise loop with early stop (ADR-0017 §2, recorded here 2026-11-02; S6.4): the Reviewer returns only the failed claim ids to the Analyst; at most 2 revisions (3 drafts); the loop stops early when a revision does not raise the grounded-claim rate (Δgrounded ≤ 0). A run that ends without passing returns the best draft by grounded rate, with uncited claims removed, as `degraded`, never `completed`. The response reports the number of drafts, never job attempts.

8.7 Effect journal and evidence pinning (ADR-0017 §3, recorded here 2026-11-02; S7.4): every model and tool call inside a node runs through `EffectJournal.memoize(key, fn)` with `key = sha256(run_id, node, model_id, params, prompt_version, rendered_prompt_hash, snapshot_hash)`; a resumed node reuses a journaled result and never calls the provider again. A partial stream is never journaled; it is discarded and the call is repeated, counted as a duplicate provider call. After the snapshot is frozen, evidence is read from T3 by `snapshot_hash`, never fetched again.

8.8 Durable checkpoints, fencing and error classes (ADR-0017 §§3–5, recorded here 2026-11-02; S7.3, S7.5, S7.8): `thread_id = job_id`, `durability = "sync"`, one checkpoint per super-step, each carrying `graph_version`; every T2 write (checkpoint, pending writes, journal, envelope) succeeds only while `jobs.attempt` equals the writer's attempt, so a stale attempt can neither commit a result nor fork the checkpoint history. A redelivered job resumes at the agent that was running from the latest compatible checkpoint; an incompatible `graph_version` restarts from the pinned snapshot. Errors are classified transient (per-node retry with jitter, bounded by the run deadline), deterministic (typed outcome, no retry) or fatal (abort the attempt; SQS redelivers; the third receive goes to the DLQ). The model that first answers a run is pinned for that run; a fallback after the breaker opens is a new attempt on the fallback model, recorded in lineage and never eligible as a baseline. If a checkpoint or envelope write fails after its retry, the attempt aborts before the next agent starts.

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
made for the backend choice (ADR-0003 addendum). Redefined 2026-10-23 (ADR-0016):
Gate B is met when the OpenSearch store passes the test-plan §4 contract suite and
§5 checks against a local container (CM-2) and the CM-4 backend-only paired
comparison against the Gate A result stays within `retrieval-benchmark.md` §9
tolerance.

Gate C: retrieval-quality method changes are proven by paired benchmark
comparisons.

Gate D: answer-quality methodology is stable enough for prompt/model changes.

Gate E: S10 cloud work. ADR-0006 was accepted on 2026-10-08; no cloud resource exists until its entry criteria have recorded evidence, except the ADR-0016 data and model plane resources (2026-10-23).

Gate F (added v1.5, ADR-0018): the five-specialist supervisor replaces ADR-0017's three-agent graph (§8.1b) only after a paired comparison on the frozen evidence release (same model, temperature and prompt; single-axis rule) shows no regression of `grounded_claim_rate` or `citation_coverage` beyond the predeclared tolerance, and the S11 game days (specialist kill, tool-schema drift, memory outage, provider 429 storm, identical-request burst) each end in their documented terminal status.

Phase 1 evidence freeze is a prerequisite for S2/Gate A work in the v2 DataOps
plan. No backend, embedding or retrieval-method comparison may run on live evidence.

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
- Stages from S11 (v1.5, ADR-0018), additive to the four caches above and keyed with the same fingerprint discipline: C0 request idempotency (existing); C1 memo (= MemoCache, single-flight); C2/C3 evidence and tool results (`tool:{server}:{tool}:{schema_hash}:{args_hash}`, generalizing the `ev:` keys; same TTLs per source); C4 query embedding (= QueryEmbeddingCache, exact only); C5 retrieval result (= RetrievalCache); C6 sentiment score by `(sha256(text), model_version)`, permanent; C7 provider prompt-prefix caching (stable prefix first; saving reported from provider usage, never estimated); C8 verification result (`ver:{sha256(memo)}:{registry_hash}:{verifier_policy_fp}`); C9 issuer profile (`issuer:{ticker}`, until a new accession). Each stage reports `cache_events_total{stage, outcome ∈ hit|miss|bypass|wait}`. Memory (M2; and M1 only if ADR-0017 §1 is reversed) shares the cache store and key discipline; it is never a second system.
- T3 write policy (ADR-0017 §1 and scenario 15, recorded here 2026-11-02; S7.9): long-term caches accept only verified, structured, provenance-tagged artifacts, written after verification passes; raw tool or web text is never written to T3; a stored memo is never returned as evidence.

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
| S3 | Qdrant backend parity / Gate B — retargeted 2026-10-23 to OpenSearch (ADR-0016; S3-T04/T05 run in CM-2, the comparison in CM-4) |
| S4 | Backend comparison and default decision / Gate B — retired 2026-10-10 (owner decision); Gate B closes at S3 exit |
| S5 | Retrieval quality diagnostics → hybrid/rerank / Gate C (the embedding bake-off was pulled forward to CM-5 on 2026-10-23, ADR-0016) |
| S6 | Answer quality: eval registry + CI regression gate (§11.2) + bounded repair loop (§8.1), then LLM-as-judge/RAGAS, then GEPA on holdout / Gate D |
| S7 | Service readiness: guardrails (§11.4), caching (§11.3), queue/worker + job store (DynamoDB `RunStore`, SQS `JobQueue`, DynamoDB checkpointer; local DynamoDB Local and ElasticMQ; ADR-0013 as amended by ADR-0015), provider circuit breaker, FinBERT off the API event loop (ONNX in the worker), auth + rate limit with guest and admin roles (ADR-0010). Restructured 2026-11-01 into one-feature sprints S7.1–S7.12; S7.3–S7.6, S7.8, S7.10 and S7.12 implement ADR-0017 (fenced checkpoints, effect journal, error taxonomy, three agents, execution observability) |
| S8 | Frontend MVP |
| S9 | Portfolio polish |
| S10 | Cloud (INTEGRATION_PLAN_v2 Phases 4–5) per ADR-0006 (accepted 2026-10-08), ADR-0010–0014 and ADR-0015 (scale-to-zero demo tier); starts when the ADR-0006 entry criteria have recorded evidence |
| S11 | Agent architecture revision (ADR-0018, proposed; added v1.5 on 2026-11-01): five specialists refining ADR-0017's three agents, tool gateway and MCP tool plane, cache stages C7–C9, issuer-profile memory (episodic memory only if ADR-0017 §1 is reversed), `/mcp` exposure, agent NFR drills and Gate F. Opens after S7.12 and Gate D, when the owner accepts ADR-0018; may run in parallel with S10, since it changes only the inside of the worker |

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
- **G14** (2026-11-01, ADR-0017) — no checkpointer; jobs run in-process, so a crash restarts the run. Resolve in S7.3 and S7.4.
- **G15** (2026-11-01, ADR-0017) — the stock `DynamoDBSaver` writes are not fenced; two attempts of one job can fork the checkpoint history. Resolve in S7.8.
- **G13** — `/health` reports `degraded` on Ollama outage even when Bedrock is the configured provider. Resolved 2026-11-09 in CM-6e (with CM-6a): health reports the configured embedding arm and chat plane from configuration only, with no Ollama dependency.

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

- `docs/adr/ADR-0004-edgartools-parser.md`
- `docs/adr/ADR-0005-unified-dataops-release-registry.md`
- `docs/adr/ADR-0006-production-cloud-deployment.md` (accepted 2026-10-08; implementation gated by its entry criteria)
- `docs/adr/ADR-0008-production-readiness-order.md`
- `docs/adr/ADR-0009-runtime-provider-fallback.md` (amendment 2026-10-08: production model ladder)
- `docs/adr/ADR-0010-identity-and-authorization.md`
- `docs/adr/ADR-0011-managed-embedding-and-rerank-models.md` (provisional on Gate C)
- `docs/adr/ADR-0012-ingestion-orchestration.md`
- `docs/adr/ADR-0013-durable-execution-and-caching.md`
- `docs/adr/ADR-0014-opensearch-serverless-vector-store.md` (accepted 2026-10-11; amended 2026-10-23 by ADR-0016: only store, adoption gate is CM-4)
- `docs/adr/ADR-0015-scale-to-zero-serverless-deployment.md` (accepted 2026-10-11; supersedes in part ADR-0006, ADR-0012 and ADR-0013 for the demo tier; their always-on design is the production scale tier)
- `docs/adr/ADR-0016-cloud-data-and-model-plane-migration.md` (accepted 2026-10-23; supersedes ADR-0003; data and model plane first, Qdrant and Chroma retired)
- `docs/adr/ADR-0017-agent-topology-and-execution-memory.md` (accepted 2026-10-31, recorded 2026-11-01; three agents, three-tier execution memory, fenced checkpoints, effect journal, crash resume; amends ADR-0013 and ADR-0015; implemented in S6.4, S7.3–S7.8, S7.10, S7.12, S10-T11)

Proposed ADRs:

- ADR-0007: production interview scale design. Proposed; interview companion only. Superseded in part on 2026-10-08 and 2026-10-11 by ADR-0006 and ADR-0009 to ADR-0015.
- `docs/adr/ADR-0018-agent-architecture-revision.md`: agent architecture revision (five specialists refining ADR-0017's three agents, memory tiers mapped onto T1–T3 with episodic memory conditional, MCP tool plane, async execution, cache stages, agent NFRs, Gate F). Proposed 2026-11-01 at the owner's request as "ADR-0016"; renumbered 2026-11-02 because that number was already taken by the accepted cloud data and model plane record. Acceptance makes the §3.z scope lines effective. Companion: `docs/agent-architecture.md`.
- `docs/adr/ADR-0019-hosted-sentiment-foundry-claude.md` (accepted 2026-11-07 after the CM-6b label-parity run, 53.3% agreement with FinBERT; sentiment on Claude Sonnet 5.5 via Microsoft Foundry, FinBERT removed; amends ADR-0013 D13 and ADR-0015 §1)

Superseded ADRs:

- `docs/adr/ADR-0003-qdrant-behind-interface.md` (accepted 2026-06-04; superseded 2026-10-23 by ADR-0016)


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
