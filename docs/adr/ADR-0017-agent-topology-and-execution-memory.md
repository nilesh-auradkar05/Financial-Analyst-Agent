# ADR-0017 — Agent topology and three-tier execution memory

**Status:** Accepted (2026-10-31) by owner decision. Implementation pending: sprints S7.3 to S7.8, S7.10 and S7.12 in `docs/sprint-plan.md`, deployed drill S10-T11; the bounded revise loop it relies on is implemented and tested locally and lands as S6.4.
**Date:** 2026-10-31 (recorded 2026-11-01)
**Amends:** SPEC §§1, 2, 3, 5, 8, 11.3 and 12 (v1.5); ADR-0013 (the checkpointer must be fenced; see Decision §4); ADR-0015 (a worker crash resumes at the failed agent, not at the start of the run).
**Related:** ADR-0005 (evidence releases), ADR-0009 (provider ladder), ADR-0013 (acceptance, fencing, caches), ADR-0015 (demo tier), test-plan §22, `docs/cloud-deployment.md` D20, `docs/production-scale-architecture.md` §§2, 6.5, 7.4.

## Context

The owner's requirement (2026-10-31): if one agent fails mid-task, the run must not start again from the beginning. The same agent must pick up where it failed. Every other failure (provider outage, worker crash, duplicate delivery, deploy mid-run, non-converging review) must end in a defined state.

What the code does at HEAD `eee2fec` (2026-10-23):

| Fact | Where | Consequence |
| --- | --- | --- |
| One `StateGraph`: evidence fan-out, join, `draft_memo`, `verify_memo`, `END` | `app/agents/graph.py` | There are no agent boundaries to resume at or hand off across |
| `workflow.compile()` has no checkpointer and no `thread_id` | `app/agents/graph.py` | Any crash restarts the whole run and re-pays every model call |
| Jobs run in FastAPI `BackgroundTasks` with a JSON-file run store | `app/main.py`, `app/services/run_store.py` | A dead process leaves the job `running` forever |
| Fencing, leases, SQS redelivery and `DynamoDBSaver` are designed, not built | ADR-0013 (+ amendment), ADR-0015 | Half of the fault-tolerance contract exists on paper; memory must plug into it, not duplicate it |
| The bounded draft → verify → revise loop | Owner-reported 2026-10-31: implemented and tested locally, not pushed | It becomes the Reviewer → Analyst edge |

Two facts about the stock components shape the design:

- LangGraph keeps the writes of the nodes that finished in a super-step (pending writes), so a resume re-runs only the node that failed in that step. It does not stop a node that already received a provider response from calling the provider again.
- `DynamoDBSaver` (`langgraph-checkpoint-aws`) has no fencing. Two attempts of one job (a slow attempt that is not dead, and its redelivery) can both write checkpoints and fork the history.

## Decision

### 1. Fault tolerance is durable execution; memory tiers make the split explicit

Resuming a failed agent is a durable-execution problem (checkpoints, an effect journal, fencing), not an agent "remembering" something. Three tiers make that explicit, and each failure in §6 maps to exactly one tier.

| Tier | Holds | Lifetime | Store: local → demo tier → scale tier | Rule |
| --- | --- | --- | --- | --- |
| **T1 Working** | LangGraph state, scratchpad, the partial stream | One attempt | RAM | References only (`snapshot_hash`, chunk ids, `payload_ref`), never payloads. Expendable by design |
| **T2 Execution** | Checkpoints and pending writes, the effect journal, handoff envelopes, the attempt fence | One run; 14-day TTL | SQLite saver → `DynamoDBSaver` (fenced wrapper; S3 offload above 350 KB) → Postgres saver | Every write is fenced by `attempt`. No agent starts until the handoff before it is durably committed |
| **T3 Long-term** | EvidenceSnapshot store, index releases, memo / retrieval / query-embedding caches | Across runs | Local files and the OpenSearch container → S3 + OpenSearch Serverless + DynamoDB TTL items → S3 + OpenSearch + Valkey | Immutable or fingerprint-keyed. Written only after verification. Never cited as evidence |

T3 is deliberately limited to the snapshot store, the index and the caches (owner decision). There is no per-ticker episodic memory and no agent that learns from past memos: it would break frozen-evidence replay unless pinned into `snapshot_hash`, and at about one query per recruiter visit it has no measurable benefit. A memo is never evidence for another memo, because that would be circular grounding.

### 2. Topology: three agents inside one run

The single graph splits into three agents (owner decision), each a LangGraph subgraph with its own retry policy and failure domain. They run inside one graph execution in one worker invocation; a resume is a redelivery to a new attempt, not a planned split across invocations.

| Agent | Nodes | Reads | Writes | Failure domain |
| --- | --- | --- | --- | --- |
| **Researcher** | `research_news ‖ fetch_stock ‖ retrieve_filings` → `analyze_sentiment` → `freeze_snapshot` | Live tools, or an evidence release in replay | EvidenceSnapshot to T3 (write once); envelope R→A | Evidence sources, ingest on miss, sentiment |
| **Analyst** | `guard_inputs` → `draft_memo`, or `revise_memo` on a returned envelope | The snapshot by `snapshot_hash`; the V→A envelope | Draft to T3 by hash (`payload_ref`); envelope A→V | Draft model, prompt rendering |
| **Reviewer** | `verify_memo` → decision → `guard_output` | The draft by `payload_ref`; the snapshot | Verification report; envelope V→A with `failed_claim_ids`, or the terminal result | Verifier, output policy |

The revise loop is the Reviewer → Analyst edge: at most 2 revisions (at most 3 drafts), and it stops early when a revision does not raise the grounded-claim rate (Δgrounded ≤ 0). A run that ends without passing returns the best draft, with uncited claims removed, as `degraded`, never `completed`. The envelope carries only the failed claim ids forward, not the whole verification report.

Rejected: one agent per SQS hop (choreography). It isolates failures further but adds three queue round-trips and more state transitions per run. Revisit only when one agent's runtime outgrows the Lambda timeout.

### 3. Resume granularity: checkpoints plus an effect journal

- **Checkpoints** (owner decision): `thread_id = job_id`, `durability="sync"`, one commit per super-step. A new attempt loads the latest checkpoint and continues at the agent that was running. Sync writes cost about 10–30 ms per super-step × 6–8 super-steps, under 0.3 s of a ~38 s run.
- **Effect journal** (owner decision): every LLM and tool call inside a node goes through `EffectJournal.memoize(key, fn)`. A resumed node finds the completed result and does not call the provider again.

```text
key = sha256(run_id, node, model_id, params, prompt_version, rendered_prompt_hash, snapshot_hash)
```

The key includes everything that changes the result, so a changed prompt or snapshot can never reuse a stale entry. A partial stream is never journaled or resumed: it is discarded and the call is issued again (recorded as a duplicate provider call, which ADR-0013 already concedes).

### 4. Contracts

```python
class HandoffEnvelope(BaseModel):             # T2; committed before the next agent starts
    run_id: str
    attempt: int                              # fence of the attempt that wrote it
    from_agent: Literal["researcher", "analyst", "reviewer"]
    to_agent: Literal["researcher", "analyst", "reviewer", "end"]
    schema_version: int
    graph_version: str
    snapshot_hash: str                        # evidence pin (T3 pointer)
    payload_ref: str | None                   # findings or draft in T3, by content hash
    revision: int = 0                         # 0..2
    failed_claim_ids: list[str] = []          # set only on Reviewer → Analyst

class EffectJournal(Protocol):                # T2; inside-node idempotency
    async def memoize(self, key: str, fn: Callable[[], Awaitable[T]]) -> T: ...

class FencedCheckpointer(BaseCheckpointSaver):
    """Wraps SqliteSaver (local), DynamoDBSaver (demo tier) or the Postgres saver (scale tier).
    put() and put_writes() succeed only if jobs[run_id].attempt == self.attempt.
    DynamoDB: each put is a TransactWriteItems with a ConditionCheck on the jobs item.
    Postgres: the write and the fence check run in one transaction."""
```

This amends ADR-0013: the checkpointer named there (`DynamoDBSaver`) is used only through `FencedCheckpointer`, so a stale attempt can neither commit a result (already fenced) nor fork the checkpoint history (new).

### 5. Error taxonomy, model pinning, versioning

| Class | Examples | Handling |
| --- | --- | --- |
| **Transient** | 429, 5xx, timeouts, EDGAR 503 | Per-node `RetryPolicy`: exponential backoff with jitter, within the node, bounded by the run deadline |
| **Deterministic** | Parser failure on one filing, schema invalid after one repair, absent 10-K | No retry. A typed outcome: `evidence_missing`, `degraded` or a typed `failed` reason |
| **Fatal** | Retries exhausted, deadline passed, checkpoint store unavailable after retry | Abort the attempt. SQS redelivers; the third receive goes to the DLQ, whose handler bumps `attempt` and writes `failed` |

- **Model pinned per run.** The first model that answers a run is recorded in T2. A fallback after the breaker opens (ADR-0009) is a new attempt on the fallback model, recorded in lineage; such a run is never eligible as a baseline.
- **`graph_version`.** The job records the deployed version alias at acceptance; every checkpoint carries `graph_version`. A worker resumes only a compatible checkpoint; otherwise it restarts the run from the pinned snapshot, which is cheap because the evidence is in T3.
- **Never past a handoff without a durable commit.** If a checkpoint or envelope write fails after its retry, the attempt aborts; later effects would otherwise be unrecoverable.

### 6. Failure scenarios

| # | Scenario | Detection | Recovery | Tier |
| --- | --- | --- | --- | --- |
| 1 | One evidence source fails while the others succeed in the same super-step | Exception in the node | Node `RetryPolicy`; pending writes keep the successes, so only the failed node runs again. After retries: SEC → `evidence_missing`; news or sentiment → `degraded` | T2 |
| 2 | Worker dies mid-agent (OOM, Lambda timeout) | Lease expiry; SQS redelivery | New attempt (`attempt + 1`) loads the last checkpoint and runs only the interrupted agent | T2 |
| 3 | Crash between a provider response and the checkpoint | Journal hit on resume | Journaled result reused: no second bill, deterministic output | T2 |
| 4 | Crash mid-stream | No journal entry | Partial output discarded; the call is issued again and counted as a duplicate | T1 |
| 5 | Zombie worker: attempt 1 is slow, not dead, and both attempts write | Fence mismatch | `FencedCheckpointer` refuses the stale checkpoint; the fenced commit refuses the stale result | T2 |
| 6 | Resume after the 60 s quote TTL | — | The snapshot is read from T3 by `snapshot_hash`, never fetched again | T3 |
| 7 | Reviewer never passes the draft | Revision counter; Δgrounded | Stop at 2 revisions or on Δgrounded ≤ 0; best draft, uncited claims removed, `degraded` | T2 |
| 8 | Bedrock throttled or down | 429/5xx; breaker open | Backoff, then a new attempt on the fallback host (ADR-0009); recorded in lineage; not a baseline | T2 |
| 9 | Deterministic failure | Error classifier | Typed outcome, no retry; a poison message reaches the DLQ after 3 receives | T2 |
| 10 | Checkpoint store unavailable | Write error | Retry once, then abort the attempt; SQS redelivers; never continue past a handoff | T2 |
| 11 | Deploy mid-run changes the state schema | `graph_version` mismatch | Resume if compatible, else restart from the pinned snapshot | T2 |
| 12 | State too large (filing chunks) | Size check | State carries references only; no item approaches DynamoDB's 400 KB limit, so S3 offload is off the hot path | T1 |
| 13 | Duplicate submit, or identical concurrent requests | Idempotency key; single-flight | Same job, or one generation that all waiters read (ADR-0013) | T3 |
| 14 | Deadline passed | Checked between super-steps; lazy expiry | `failed` (`deadline_exceeded`); checkpoints expire by TTL | T2 |
| 15 | Memory poisoning: injected web text persisted and reused | T3 write policy | T3 accepts only verified, structured, provenance-tagged artifacts; never raw tool text | T3 |

### 7. Observability

Counters and histograms, exported with the existing OpenTelemetry and LangSmith setup (ADR-0006 D17): `runs_resumed_total`, `nodes_reexecuted_per_resume`, `journal_hits_on_resume_total`, `duplicate_provider_calls_total{reason}`, `tokens_wasted_total`, `stale_writes_rejected_total{kind=checkpoint|commit}`, `checkpoint_write_seconds`, `revisions_per_run`, `review_not_converged_total`.

## Alternatives considered

| Alternative | Why not now | When it would be preferable |
| --- | --- | --- |
| Checkpoints only, no effect journal | A crash between a provider response and the checkpoint pays twice and can produce a different draft | Never for paid model calls |
| Temporal or Step Functions for durable execution | A second runtime that replaces LangGraph's checkpoint semantics | Scale tier, if workflows outgrow one graph run or need human-in-the-loop waits measured in days |
| One agent per SQS hop | Three queue round-trips per run | When an agent's runtime outgrows the Lambda timeout |
| Keep one graph and call the nodes "stages" | No independent retry or failure domains; "multi-agent" would not be an honest claim | — |
| Cross-run episodic memory per ticker | Breaks replay unless pinned; no measurable benefit at demo volume | A product with repeat users per ticker and a replay-safe pinning design |

## Consequences

- Adds `app/execution/` (`FencedCheckpointer`, `EffectJournal`, `HandoffEnvelope`, error classifier, fault-injection harness) and three agent subgraphs under `app/agents/`. `app/` still never imports `evaluation/`.
- About 20 small T2 writes per run: fractions of a cent per month at demo volume. The cost lever is scenario 3 (no re-billing).
- The topology split is a single-axis change: on a frozen snapshot with the same model and prompts, `grounded_claim_rate` and `citation_coverage` must stay within the eval-replay tolerance (SPEC §11.2).
- Crash resume is proven by fault injection (`FAIL_AT=<agent>.<node>:<pre|post>-effect`), not asserted: the memo hash after a crash and resume equals the hash of an uninterrupted run on the same frozen snapshot.
- New test cases: test-plan §22. New sprints: S6.4, S7.3 to S7.8, S7.10 and S7.12; deployed drill S10-T11.
