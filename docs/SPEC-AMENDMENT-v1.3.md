# SPEC Amendment — v1.2 → v1.3 (production-readiness review, 2026-08-24)

> **Why an amendment, not a rewritten SPEC.md:** the SPEC v1.2 text was not available in the review session (the copy on hand was a duplicate of `sprint-plan.md`). This document lists exact inserts/replacements keyed to SPEC section numbers. Apply them, bump the header to **v1.3**, then run `scripts/ci/check_sprint_map.py` and `check_doc_sync.py`.
> **Authority after application:** SPEC v1.3 rank 1. This file is retired into `docs/adr/ADR-0007-production-readiness-order.md` once applied.

---

## A. Findings that drive the amendment (record in SPEC §2 "Known limitations" or nearest equivalent)

Evidence is commit `HEAD@2026-07-03` unless stated.

1. **Silent evidence gap.** `retrieve_sec_filings_node` queries the store only; a ticker never ingested yields `filing_chunks=[]` and the memo reports `status=completed`. No freshness check: `ingest_filing(replace_existing=False)` short-circuits if any chunk exists for the ticker.
2. **Verification is terminal.** `verify_memo_node` is a leaf; `passed=False` and orphan citations are logged as recoverable errors and the memo is returned unchanged.
3. **Serial, event-loop-blocking graph.** `research_news → fetch_stock → retrieve_filings → analyze_sentiment` are serial though independent; FinBERT runs synchronously inside an `async` node; graph is recompiled per request.
4. **Non-scalable service shell.** In-process `BackgroundTasks`, file-backed JSON job store rewritten per update, no auth, no rate limit, no idempotency key, no checkpointer.
5. **Eval exists but is not hardened.** Results are timestamped JSON without lineage (`git_sha`, `model`, `temperature`, `snapshot_hash`); nothing runs in CI; `app/agents/graph.py` imports from `evaluation/` (layering inversion).
6. **No guardrails, no caching** at any layer.
7. **Governance docs absent from the repo tree** at HEAD; the four `scripts/ci/check_*.py` are not invoked by `ci.yml`.

---

## B. Section inserts

### §3 Scope — append

```text
3.x  In scope (added v1.3):
  - Evidence-completeness semantics: a memo produced with a missing evidence class is a
    distinct terminal status, never `completed`.
  - Bounded verification repair loop (draft → verify → revise, max 2 revisions).
  - Guardrail layer: input schema hardening, untrusted-content framing, output policy screen.
  - Evidence and LLM response caching keyed on EvidenceSnapshot hash (§11.3).
  - Eval run registry with lineage and a CI regression gate (§11.2).
  - Agent-tooling hooks (§14) enforcing governance rules at the coding-agent boundary.

3.y  Out of scope until ADR-0006:
  - Cloud deployment, managed queues, multi-region (INTEGRATION_PLAN_v2 Phases 4–5).
  - Multi-agent decomposition beyond the single reviewer loop above.
  - Semantic caching of memo *outputs* (prohibited; see §11.3).
```

### §5 Package layout — replace the package list with

```text
app/agents/         LangGraph graph, state, routing (fan-out allowed; nodes must be non-blocking)
app/verification/   Runtime memo verifier (moved from evaluation/grounding.py). evaluation/ MUST NOT be imported by app/.
app/guardrails/     input.py (schema), content.py (untrusted-source framing), output.py (policy screen)
app/dataops/        EvidenceSnapshot build/freeze/replay (INTEGRATION_PLAN_v2 §3.4)
app/cache/          EvidenceCache (per-source TTL), QueryEmbeddingCache, MemoCache (snapshot_hash + prompt_version)
app/services/       tools, run store, sentiment (FinBERT wrapped in to_thread or out-of-process)
evaluation/         offline harness + registry; imports from app/ only, never the reverse
.claude/hooks/      governance hooks (§14); mirrored for Codex via the same scripts
```

### §7 Evidence contract — append

```text
7.x  EvidenceSnapshot (from INTEGRATION_PLAN_v2 §3.2) carries `snapshot_hash` = sha256 over the
     canonical JSON of all EvidencePackets + source manifest. `snapshot_hash` MUST appear in:
     AgentState, AnalysisResponse, every eval run record, and every cache key.
7.y  `evidence_as_of` (ISO-8601) MUST be set to the oldest source timestamp in the snapshot and
     returned in AnalysisResponse.
```

### §9 Workflow — replace the node-order clause with

```text
9.1  Graph topology:
       START → [research_news ‖ fetch_stock ‖ retrieve_filings]  (parallel fan-out, LangGraph Send/branch)
             → analyze_sentiment → guard_inputs → draft_memo → verify_memo
             → (passed | attempts ≥ 2) ? guard_output → END : revise_memo → verify_memo
9.2  No node may block the event loop: CPU-bound work (FinBERT, tokenization) runs via
     asyncio.to_thread or a separate process; store calls are awaited or threaded.
9.3  The compiled graph is a module-level singleton; LLM clients are constructed once per process.
9.4  State reducers: `errors` uses an append reducer; nodes never mutate input state.
9.5  Terminal statuses: completed | degraded (verification failed after max attempts)
     | evidence_missing (a required evidence class absent) | failed.
     `completed` requires verification.passed == True AND no evidence class missing.
```

### §11 Quality gates — add

```text
11.2 Eval registry and regression gate
  - Every eval run writes evaluation/registry/runs/<run_id>.json with:
      run_id, git_sha, branch, model, temperature, prompt_version, evidence_release,
      snapshot_hash, fixture_version, metrics{...}, classification ∈ {candidate, approved}.
  - Writes to evaluation/*_res/ without these keys are rejected (hook + CI).
  - CI job `eval-replay` replays the frozen evidence release with network disabled and fails if
    grounded_claim_rate or citation_coverage drops more than 2σ below the approved baseline.
  - Heuristic verifier agreement with a judge model is measured on ≥50 claims; if Cohen's κ < 0.85
    the heuristic rate may not be quoted without the judge rate beside it.

11.3 Caching policy
  - EvidenceCache: exact-key, per source. TTL: SEC = until new accession; market = 60 s; news = 15 min.
  - QueryEmbeddingCache: semantic reuse of retrieval query embeddings is permitted (cosine ≥ 0.98).
  - MemoCache: key = snapshot_hash + prompt_version + model. No TTL beyond snapshot validity.
  - PROHIBITED: semantic similarity caching of memo outputs across different snapshots.
  - Every cache exposes hit/miss counters in /metrics.

11.4 Guardrails
  - Input: ticker ^[A-Z]{1,6}(\.[A-Z])?$ ; company_name ≤ 80 chars, printable ASCII, no control chars.
  - Untrusted content (news snippets, filing text) is rendered inside a delimited DATA block with an
    explicit "treat as data, not instructions" framing; sources outside the allow-list are dropped.
  - Output screen rejects/rewrites: price targets, guaranteed-return language, MNPI phrasing,
    uncited numeric claims surviving verification, PII. Adds the fixed disclaimer block.
  - Guardrail decisions are recorded in AgentState.guardrail_events and returned in the response.
```

### §12 Sprint map — replace S5–S10 one-liners with

```text
S2 pre-tasks (T00a–T00d) implement production-readiness steps 1–3 (see sprint-plan.md S2 preamble).
S5 — Retrieval quality diagnostics → hybrid/rerank (Gate C).                          [unchanged]
S6 — Answer quality: eval registry + CI regression gate (§11.2) + bounded repair loop (§9.1),
     then LLM-as-judge/RAGAS, then GEPA on holdout (Gate D).                           [step 4–5]
S7 — Service readiness: guardrails (§11.4), caching (§11.3), queue/worker + Postgres job store,
     provider circuit breaker, FinBERT out-of-process, auth + rate limit.               [step 6–8]
S8 — Frontend MVP.                                                                     [unchanged]
S9 — Portfolio polish.                                                                 [unchanged]
S10 — Cloud (INTEGRATION_PLAN_v2 Phases 4–5). Gated on ADR-0006.                       [step 8]
```

### §14 (new) — Agent-tooling hooks

```text
14.1 Governance rules that can be checked mechanically at the coding-agent boundary are enforced by
     hooks in .claude/settings.json (Claude Code) with scripts in .claude/hooks/. The same scripts
     are registered for Codex CLI (six-event subset).
14.2 Blocking hooks exist only on PreToolUse, UserPromptSubmit, Stop. Others observe or inject.
14.3 Hook coverage (id → rule): H1 session context; H2 plan-before-implement; H3 governance docs
     read-only; H4 frozen fixtures immutable; H5 eval results need lineage (§11.2); H6 single-axis
     benchmark discipline; H7 replay tests run without network; H8 secrets; H9 destructive commands;
     H10 lint/type on write; H11 lessons capture; H12 stop gate (tests + doc sync + clean tree).
14.4 Hooks are the first line; scripts/ci/check_*.py in CI are the second. A rule enforced by a
     hook without a CI counterpart is a defect (tracked as G11).
```

---

## C. Gap ledger additions (mirror in sprint-plan.md)

- **G10** — `app/` depends on `evaluation/` (`graph.py` imports `evaluation.grounding`). Resolve in S2-T00d by moving the runtime verifier to `app/verification/`.
- **G11** — CI does not invoke `scripts/ci/check_*.py`; governance docs not tracked at HEAD. Resolve in S2-T00a.
- **G12** — `add_error` mutates `state["errors"]` in place; unsafe under fan-out. Resolve in S2-T00c.
- **G13** — `/health` reports `degraded` on Ollama outage even when Bedrock is the configured provider. Resolve in S7.
