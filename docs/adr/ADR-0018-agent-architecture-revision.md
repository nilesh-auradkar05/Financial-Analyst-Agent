# ADR-0018 — Agent architecture revision: memory tiers, five specialists, MCP tool plane, staged caching, agent NFRs

**Status:** Proposed (2026-11-01) at the owner's request for a system architecture revision. **Renumbered 2026-11-02:** written as "ADR-0016", a number the accepted cloud data and model plane record (2026-10-23) already held; this file and every reference to it now use ADR-0018. **Extends ADR-0017** (accepted 2026-10-31): where the two overlap, ADR-0017 governs until the owner rules on the two open points under "Relation to ADR-0017". Acceptance by the owner records the SPEC §3.z scope change this record carries (SPEC v1.5). Implementation is gated: S7.1–S7.12 (including ADR-0017's three-agent split, S7.6) and Gate D before any phase beyond the behavior-preserving refactor; adoption only on Gate F (a paired comparison on the frozen release).
**Date:** 2026-11-01
**Related:** SPEC §§2, 3, 5, 8, 11.1, 11.3, 11.4, 12 (S11), 16; test-plan §23; ADR-0017 (agent topology and execution memory, the baseline this record refines); sprint-plan S11; ADR-0009 (provider ladder), ADR-0010 (identity, authorization as data), ADR-0012 (EDGAR limiter), ADR-0013 (fencing, caches), ADR-0015 (demo tier stores); design companion `docs/agent-architecture.md`; `docs/production-scale-architecture.md` (the worker this revision runs inside).

## Context

SPEC v1.4 §3 keeps "broad multi-agent or swarm orchestration" and "long-term memory" out of scope unless §3 is explicitly changed and backed by an ADR. The owner asked on 2026-11-01 for a revision that adds a memory architecture, a five-specialist decomposition of the analysis task, a Model Context Protocol (MCP) integration plan, asynchronous agents and tools for many concurrent requests, caching at several stages to reduce model and tool calls, and hardening of the non-functional requirements that are specific to agent systems.

At `HEAD` (`eee2fec`, the code this record was written against) the worker runs six graph nodes with one model call, sequential filing queries, no cache, no memory and no tool abstraction. ADR-0017 (accepted 2026-10-31, recorded in the repository on 2026-11-01 alongside this proposal) already decides the next step: three agents (Researcher, Analyst, Reviewer) in one run, three-tier execution memory (T1 working, T2 fenced checkpoints, effect journal and envelopes, T3 snapshots, index and caches) and crash resume at the failed agent. This record was drafted without that text and is reconciled to it below. The scale tier (ADR-0006, ADR-0013, `docs/production-scale-architecture.md`) already sizes capacity by provider tokens and defines acceptance, fencing and event replay. What it leaves open is the inside of a job: how work is split, remembered, reached and cached. That is what this record decides.

## Relation to ADR-0017

| Topic | ADR-0017 (accepted) | This record (proposed) | Status |
| --- | --- | --- | --- |
| Topology | Researcher, Analyst, Reviewer subgraphs | `market`, `news`, `filings` split the Researcher; `analyst` = Analyst; `verifier` = Reviewer | Compatible refinement; built after S7.6 lands the three agents |
| Resume after a crash | Fenced checkpoints, effect journal, handoff envelopes (T2) | Specialist checkpoints | ADR-0017's mechanism is used; this record adds none of its own |
| Working, semantic, procedural memory | T1 working; T3 snapshot store, index, caches | M0 = T1; M2 issuer profile and M3 versions live in T3 under its write-after-verification rule | Compatible |
| Episodic memory | Excluded by owner decision (§1: breaks replay unless pinned; no benefit at demo volume) | M1 per `customer_id` × ticker, uncited, pinned by `memory_ref` | **Conflict — owner decision.** M1, `DELETE /memory` and the episodic rows of test-plan §23.2 are not built unless the owner reverses ADR-0017 §1 |
| Revise loop | ≤ 2 revisions, early stop on Δgrounded ≤ 0, failed claims only | ≤ 2 revisions | ADR-0017's rule applies |
| Tool gateway, MCP, cache stages C7–C9, Gate F | Not covered | D3, D5, D6 | Additive |

## Decision

### D1 — Five specialists under a deterministic supervisor

The analysis is decomposed into exactly five specialists: `market`, `news` (news search plus FinBERT sentiment), `filings` (retrieval plus ingest on miss), `analyst` (drafting and revision) and `verifier` (heuristic grounding, output policy, revision requests; an LLM judge only after SPEC §11.2's κ ≥ 0.85). The supervisor is a code-routed LangGraph graph with a fixed topology; its only data-dependent branch is the bounded revise loop (≤ 2). No LLM planner. Each specialist has a typed contract, a tool allowlist, a budget (wall time, tool calls, tokens, cost), a cache policy, a failure row in the degradation ladder and an offline eval.

Containment rule: specialists that use tools have no model call; specialists that call a model have no tools.

This is not "broad multi-agent or swarm orchestration": the count is fixed at five, routing is deterministic, and one model call per draft (plus revisions) remains the cost envelope.

### D2 — Memory tiers with one invariant

Four tiers, mapped onto ADR-0017's T1–T3: M0 working (per job; ADR-0017's T1, checkpointed through its T2); M1 episodic **(conditional: only if the owner reverses ADR-0017 §1)** (per `customer_id` × ticker: prior run summaries); M2 semantic (per issuer, principal-free: issuer profile, FinBERT scores, snapshots and releases); M3 procedural (prompt, retrieval, budget and policy versions, changed offline only through the eval registry).

Invariant: **memory is context, never evidence.** Memory enters the analyst's prompt in an uncited, stale-framed `MEMORY` block; the verifier rejects any claim supported only by memory; a snapshot manifest records `memory_ref` so replay reuses the same memory items and never reads live memory; evaluation runs read no memory unless the fixture pins it. M1 is authorized as data on `customer_id` (ADR-0010), guests get 24 h episodic memory, `DELETE /memory` forgets a principal. Memory is written only after the fenced terminal commit, idempotently on `job_id`. A memory outage degrades (`memory_unavailable` guardrail event); it never fails a run.

Stores: `MemoryStore` protocol in `app/memory/`; DynamoDB adapter (demo tier, DynamoDB Local in compose), in-memory adapter (tests); a Postgres adapter only at its ADR-0015 trigger.

### D3 — MCP as the tool plane, behind a gateway

A `ToolGateway` in `app/tools/` is the only path from a specialist to a tool. In-process tools and MCP servers implement one `ToolSpec`/`ToolResult` contract. Servers and tools come from static Pydantic settings with pinned schema hashes; a drifted schema disables the tool until re-approved. Only read-only tools are registrable (structural field, not convention). Server-initiated sampling and elicitation are disabled. Every payload is validated against its output schema and framed as untrusted content before use. Transport is streamable HTTP with pooled sessions; stdio is development-only. Own wrappers (`sec-edgar`, `market-data`) run outside the worker's execution role; the vendor-hosted news-search server is first because it is lowest-risk. The evidence release gains a `tool_calls` log for replay.

The analyst is later exposed as an MCP server at `/mcp` (`analyze_ticker`, `get_job`, `list_runs`, `memo://{job_id}`), a thin adapter over the existing routes with the same issuer allowlist, quotas and error envelope.

### D4 — Asynchronous execution inside the worker

Evidence specialists fan out as graph branches; the four filing queries run concurrently; sentiment moves inside `news` so it overlaps retrieval. One semaphore per tool target per process; a provider token bucket checked before each model call; deadlines nest tool < specialist < job with propagated cancellation; blocking libraries stay behind `to_thread` with a bounded pool; specialist results are checkpointed so a re-claimed attempt skips completed, in-TTL specialists. Acceptance, fencing, queue and autoscaling stay as ADR-0013 and ADR-0015 define them.

### D5 — Nine cache stages, every key fingerprinted

C0 request idempotency, C1 memo (single-flight), C2/C3 evidence and tool results (`tool:{server}:{tool}:{schema_hash}:{args_hash}`, generalizing the `ev:` keys), C4 exact query embedding, C5 retrieval result, C6 sentiment by text hash, C7 provider prompt-prefix caching (stable prefix first; saving reported from provider usage only), C8 verification result, C9 issuer profile. SPEC §11.3's prohibition on semantic reuse of memo outputs is unchanged. Each stage reports `cache_events_total{stage, outcome}`.

### D6 — Agent NFRs as contracts, and Gate F

Bounded autonomy (fixed topology, revision cap, per-specialist tool-call and cost caps, per-job cost cap), deadline propagation, per-target breakers, a per-specialist degradation ladder, tenant-scoped memory, deterministic replay of memory and tool calls, per-agent spans and metrics, quality-drift sampling, offline-only model and prompt change control, and a human review path for policy-rewritten or degraded runs.

**Gate F:** the specialist graph replaces ADR-0017's three-agent graph only after a paired comparison on the frozen release (same model, temperature and prompt; single-axis rule) shows no regression of `grounded_claim_rate` or `citation_coverage` beyond the predeclared tolerance, and the game days (specialist kill, schema drift, memory outage, provider 429 storm, identical-request burst) each end in their documented status.

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| LLM supervisor (ReAct planner over tools) | Rejected | Reads tool output to choose actions (injection sink); non-replayable; unbounded cost |
| More than five specialists (per-source summarizers, meta-critic) | Rejected for now | Each adds a model call per request without a measured need; SPEC §3 forbids swarm orchestration |
| Vector-indexed episodic memory | Deferred | Lookup key is exact (`customer_id`, ticker); revisit for cross-ticker questions |
| Memory as citable evidence | Rejected | Breaks the snapshot invariant (SPEC §7.x) and replay; stale facts would gain citations |
| Runtime MCP tool discovery | Rejected | Tool poisoning and rug-pull risk; three read-only sources need no marketplace |
| Separate memory and cache stores | Rejected | Two systems that can disagree about what is known |
| Semantic memo-output caching | Prohibited | SPEC §11.3, unchanged |
| Exposing `/mcp` before consuming MCP | Rejected | The browser is the main audience; consumption is where the risk and the saving are |

## Consequences

- SPEC v1.5: §2 non-goals and §3 scope admit a bounded five-specialist decomposition and scoped memory tiers (swarm orchestration and uncontrolled long-term memory stay out); §5 adds `app/agents/specialists/`, `app/tools/`, `app/mcp/`, `app/memory/`; §8.1 is revised (sentiment inside `news`, memory reads, memo-cache lookup); §11.1 adds Gate F; §11.3 lists the nine stages; §12 adds S11; §16 lists this record.
- test-plan §23 defines the behavior cases (§22 belongs to ADR-0017); sprint-plan adds S11 as a gated horizon sprint after S7.12 and Gate D.
- ADR-0017's three-agent graph stays the default until Gate F; the specialist graph runs behind a flag.
- Expected effect (formula, not measurement): model calls per request fall from `1 + r` to `(1 − h)(1 + r)`; evidence-phase wall time becomes the slowest specialist rather than the slowest branch plus sentiment; tool calls collapse per TTL window during a burst.
- New dependency: the MCP Python SDK, confined to `app/tools/mcp_client.py` and `app/mcp/server.py`; added through `uv add` with an advisory check, in S11-T02 at the earliest.
- Cost: negligible for memory; about two small always-idle functions (demo) or two small tasks (scale) for own MCP wrappers; Gate F eval runs bounded by the frozen release.

## Verify before implementation

- MCP SDK: streamable-HTTP client with pooled sessions, the ability to disable sampling and elicitation callbacks, and a server mountable inside FastAPI; pin the version and check advisories.
- Provider prompt caching: which of the ladder's hosts report cached-token usage (Foundry OpenAI-compatible route, Bedrock) so C7 can be measured rather than assumed.
- The news-search vendor's MCP server: tool schema, rate limits, and whether results match the current HTTP tool within a TTL window (MCP-1 parity check).
- LangGraph checkpoint size with per-specialist results, against the 350 KB S3 offload threshold (ADR-0015).
- DynamoDB item size for episodic records and the issuer profile; TTL behavior for guest records.
