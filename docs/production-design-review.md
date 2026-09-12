**Production design review — Financial Analyst Agent — 2026-09-06**

> Historical review of the then-canonical SPEC v1.2. SPEC v1.3 was applied on 2026-09-12 (`ADR-0008`). Do not treat this file as current source of truth.

**Director’s assessment: 6/10 for the proposed design; return for revision before production approval.** The decomposition and evidence-first approach are sensible. Several low-level contracts contradict the guarantees used to justify the design. The current code remains a prototype; missing target features are not, by themselves, defects in an explicitly future-facing design.

These grades are qualitative reviewer judgments, not benchmark results. A 5 means plausible components with unresolved contracts; an 8 means specified failure behavior backed by representative measurements; a 10 would require sustained production evidence. I would fund the next correctness milestone, but would not approve the stated capacity, latency, availability, or cost commitments yet.

**Scope and evidence**

Reviewed README.md, docs/SPEC.md v1.2, the active ledger and relevant sprint/test/benchmark rules, ADR-0003 and proposed ADR-0006, production-scale-topic.txt, the complete interview companion, all eight Excalidraw sources, and all four production PNGs. Applied the local Excalidraw skill to inspect flow and contract consistency. Used Mem0 for prior decisions and CodeGraph for runtime entry points, graph flow, response formatting, ingestion, and the Qdrant adapter. A second reviewer independently examined the distributed execution, caching, and evaluation contracts.

Vendor-specific claims were checked against primary documentation. Agent Reach routing instructions were read; its CLI and mcporter were unavailable, so the available web tool supplied official-source verification. No cloud deployment, live benchmark, load test, or runtime implementation change was performed. Estimates below are calculations from the document’s assumptions.

**Grades against your questions**

| Topic from production-scale-topic.txt | Grade /10 | Assessment |
|---|---:|---|
| Document volume and ingestion throughput | 6 | Decoupling is right; publication, amendment handling, and end-to-end throughput are incomplete. |
| Concurrent query volume | 5 | Queue and workers are appropriate; capacity units, fairness, and crash recovery need correction. |
| Sub-3-second p95 TTFT | 4 | Useful product distinction between first token and verified completion; the stream contract and capacity evidence do not establish the SLO. |
| Cost and economics | 4 | Correct levers, incorrect arithmetic and incomplete total cost. |
| Metadata and pre-retrieval filtering | 7 | Strong access-pattern reasoning; ticker identity, customer authorization, and sharding are conflated. |
| Vector database and HNSW memory | 4 | Quantization is sensible; custom-shard semantics and configured precision contradict the sizing. |
| Hybrid retrieval and reranking | 7 | Native sparse search and RRF are defensible; serving compatibility and acceptance rules require proof. |
| Multi-layer caching | 4 | Useful cache boundaries; identity and version omissions can silently corrupt results. |
| Monitoring, quality evaluation, and cost observability | 7 | Good lineage and metric direction; some current capabilities are understated and offline generation is underspecified. |
| Production checklist and recovery | 5 | Concrete intentions, but no complete recovery, authorization, overload, or rollout evidence. |

**Findings requiring correction**

1. **P1 — Custom sharding creates a radically different cluster than the diagrams claim.** Evidence: interview lines 98 and 119–129; production HLD element `qdrant_t`; LLD `schema_t`.

   In Qdrant custom sharding, `shard_number` is the number of shards per key. Twelve shards per ticker × 8,000 ticker keys × replication two means **192,000 physical shard replicas**, assuming those keys are provisioned with the documented defaults. It does not mean twelve total shards or one shard per query. The design also omits shard-key creation. This invalidates the operational and memory model. [Qdrant distributed deployment documentation](https://qdrant.tech/documentation/scaling/distributed_deployment/).

   Correction: start with a bounded number of automatic shards plus indexed issuer filters, or explicitly map issuers into a bounded set of custom shard groups. Reserve dedicated shards for measured large/hot partitions. Prove the physical shard count and search fan-out on the pinned server version before sizing hardware.

2. **P1 — The 15-minute market identity contradicts immutable evidence.** Evidence: interview lines 12–16 and 175–185; production LLD `cache_t` and `n_cache`.

   The proposal preserves a fresh 60-second quote in the memo while using a 15-minute bucket to preserve cache identity. If the actual quote is hashed, changing it changes the hash. If it is excluded, different evidence shares an identity. A memo generated at $100 can be displayed alongside a new $90 quote while claiming the same provenance. This also conflicts with SPEC §7 and the 60-second evidence TTL in test-plan §13.

   Correction: cache the exact memo with its exact immutable quote and source timestamp. A separately refreshed quote may appear outside that artifact, clearly distinguished from evidence used in the memo. Any changed hashed payload creates a new identity. Test two materially different quotes inside the same bucket.

3. **P1 — The stream exposes content before its safety gate and cannot deliver the terminal contract as written.** Evidence: interview lines 63–68 and 234; production critical-flow subtitle, SSE example, verify and guard-output path.

   “After the stream closes” cannot be followed by another event on that connection. “Can never upgrade” conflicts with moving an unverified draft to verified `completed`; the SSE example actually makes that transition. More seriously, output policy runs after tokens have already reached the client. Neither downgrading nor revising can retract delivered content. If guard-output changes the memo, the earlier verification does not necessarily describe the final bytes.

   Correction: distinguish model token-stream completion from transport closure. Keep drafts visibly provisional; emit a durable terminal event with memo version and content hash. Revisions replace a version instead of appending another answer. Apply mandatory blocking policy before exposing prohibited text; use buffered sections or progress events if necessary. Verify the final returned artifact after material policy edits. Test reconnects, rejected output, revision replacement, and a successful transition to completed.

4. **P1 — Postgres plus Redis Streams has no specified crash-consistent job protocol.** Evidence: interview lines 48–50 and 250; production HLD API, Redis, worker, and Postgres arrows.

   Creating the job row and publishing `XADD` are separate writes. A crash between them can strand an accepted job or leave work without its authoritative row. PEL and DLQ labels do not define lease expiry, redelivery, retry limits, or protection against an old worker overwriting a new result. An idempotent HTTP request does not prevent duplicate provider spend after uncertain execution.

   Correction: either use Postgres as the durable job queue initially, or retain Streams with a transactional job/outbox record and idempotent dispatch. Specify leases, attempt fencing, finite retry/dead-letter policy, cancellation, and acknowledgement only after durable result persistence. Scope idempotency to authenticated customer plus request hash; conflicting reuse must be rejected. Demonstrate recovery at each write/ack boundary.

5. **P1 — Durable jobs do not make Pub/Sub token delivery durable.** Evidence: interview line 48; production HLD `noteA`; critical-flow token-relay note.

   A fast cache hit can publish before the API subscribes. A disconnected API loses Pub/Sub messages, potentially including the only terminal verification event. Redis documents at-most-once Pub/Sub delivery; sticky sessions do not recover lost events. [Redis Pub/Sub documentation](https://redis.io/docs/latest/develop/pubsub/).

   Correction: specify the HTTP contract explicitly, such as POST returning 202 and an authenticated GET events endpoint. Use sequence IDs and replayable events, or make durable terminal polling/reconciliation mandatory when token replay is not required. Heartbeats, disconnect behavior, retention, and cancellation belong in this contract. A replica crash must not make the final answer undiscoverable.

6. **P1 — Cache keys omit versions that change the meaning of cached values.** Evidence: interview lines 175–181; production LLD `cache_t`.

   A permanent `content_sha` embedding key survives changing the embedding model or vector dimensions. Text-only query keys have the same defect. Retrieval keys omit sparse encoding, fusion, reranker and prior configuration. A stored verification decision can survive a stricter verifier or output policy. Semantic query-embedding reuse also needs an explicit lookup mechanism: computing the new embedding merely to discover similarity can erase the intended saving, and near-identical financial queries can differ in year or negation.

   Correction: use compact, versioned artifact fingerprints: embedding model/revision/dimension/preprocessing; index release and retrieval configuration; memo request/prompt/model/generation settings; verifier and policy version. Use exact query-embedding caching first. Test misses on each relevant version change and coalesce concurrent cache misses so earnings-day traffic does not generate the same memo hundreds of times.

7. **P1 — The concurrency formula has the wrong units; the latency budget omits major waits.** Evidence: interview lines 50, 53 and 63–66; production HLD `worker_t`; system-design saturation panel.

   TPM divided by tokens per memo gives memos per minute, not concurrent jobs. Under the document’s own example, 1,000 memos/minute at 25 seconds mean residence time implies roughly **417 in-flight jobs**, before reserve capacity. A fixed 15 ms queue/claim budget cannot represent arbitrary backlog. Cold news and market calls, sentiment, GPU queueing, revisions, and fallback are not included in the stated first-token path. Component p95 estimates do not establish overall p95.

   Correction: specify offered request rate, burst shape, service-time distribution, per-provider input/output quotas, revision frequency, and deadline-based admission. Report client-observed TTFT and verified-completion latency separately, with rejection and abandonment rates. Load-test cache-hot, cache-cold, hot-ticker, provider-failure, and backfill-concurrent cases. Reserve interactive GPU capacity; the HLD shares embedding service capacity with ingestion.

8. **P1 — Index upsert plus a version bump does not produce a coherent searchable release.** Evidence: interview lines 29 and 92; production HLD `gold_t` and `sync_t`.

   During multi-batch publication, a query can see only part of a filing. A sync job that crashes after upserts but before the version bump can leave changed data behind a still-valid retrieval cache. Re-parsing an accession into fewer chunks leaves obsolete trailing IDs unless explicitly removed. Deterministic IDs make repeated identical writes idempotent; they do not make an evolving dataset atomic.

   Correction: define staged release identity, chunk-manifest reconciliation, serving visibility, deletion/tombstones, and promotion only after completeness checks. Retain the previous approved release for rollback. Specify replica-read behavior during promotion; replication factor two and `write_consistency_factor: 1` alone do not establish the claimed snapshot view. Test interruption halfway through publication, reparse-with-fewer-chunks, and rollback.

9. **P1 — Fallback does not guarantee an answer and needs a separate quality contract.** Evidence: interview lines 221 and 230–234; LLD breaker and cached-verification notes.

   A snapshot may exist without any memo. The normal memo key cannot locate a result for a new snapshot when every provider is unavailable. Reusing an older memo needs eligibility and maximum-age rules; re-emitting its old successful verification must not clear its stale status. Switching model mid-stream can splice two incompatible drafts. “Same prompt_version” does not prove quality across Sonnet, GPT, and an unspecified open model. The Haiku verification dependency also needs failover when the primary provider is down.

   Correction: approve each concrete model/prompt/policy combination; reserve fallback capacity and use request deadlines. Restart under a new attempt/version when failover follows partial output. Stale fallback must preserve its old snapshot and timestamps and remain degraded. Otherwise return a durable typed failure. Test simultaneous primary-draft and verifier failure, no eligible cache, quota exhaustion, and failover after the first token.

10. **P1 — Financial correctness is stronger than citation presence or cosine similarity.** Evidence: interview lines 12 and 199–208; original LLD verifier panel; test-plan §11–14.

   A sentence can cite the right passage and share its numbers while reversing an increase into a decrease, confusing millions with billions, or assigning last year’s amount to this year. Heuristic/judge agreement does not establish correctness if both share the same blind spots. A no-answer example about a section that genuinely does not exist also needs to distinguish “not applicable” from an ingestion failure.

   Correction: define correctness against entity, period, unit, accounting concept, calculation, and entailment. Add expert-adjudicated adversarial financial cases, arithmetic/period checks, supported-opinion handling, and calibrated refusal coverage. Track false acceptance of material errors, not only overall agreement or average grounded rate. Keep this separate from retrieval recall.

11. **P1 — Ticker partitioning is not customer authorization, and ticker is an unstable issuer identifier.** Evidence: interview lines 17, 48, 92–98 and 207; production LLD payload.

   The design uses “tenant” for both AAPL-style partitions and paying customers’ token budgets. Those are different identities. OIDC establishes who made a request; it does not establish who may read a job, event stream, cached memo, or trace. Ticker renames and share classes also complicate historical lookup and can duplicate issuer evidence.

   Correction: make issuer identity (CIK for the covered SEC entities) distinct from customer identity. Resolve ticker aliases with effective dates, then authorize customer-owned job/event access independently. Public evidence may be shared, subject to entitlement rules, while private prompts/results remain scoped. Test cross-customer job reads, event subscriptions and idempotency collisions, plus renamed and dual-class tickers. The proposed customer boundary needs an explicit contract even if the initial corpus is public.

12. **P2 — “Within 2σ” is underspecified for retrieval; fully offline hosted generation is unresolved.** Evidence: interview lines 78, 163, 199 and 219; production system-design `gate_t`; retrieval-benchmark §8.

   The document does not define whether σ measures across-case variation, repeated generations, or uncertainty in paired deltas. A noisy benchmark can permit a large loss. The canonical retrieval policy requires paired deltas with 95% bootstrap confidence intervals and at least 10,000 resamples. Generation’s 2σ rule already exists in test-plan §14, so the issue is the undefined statistic and extension to retrieval, not that every use is newly unauthorized. Network-disabled evidence replay also cannot call Bedrock or Azure to evaluate a newly changed prompt.

   Correction: use the canonical paired retrieval policy, a predeclared acceptable loss, per-slice results and a held-out set. Separate hermetic evidence/retrieval replay from fixed-evidence hosted generation evaluation, or pin local model assets for a truly offline run. Replaying old outputs is useful for verifier changes but cannot establish new prompt quality.

13. **P2 — The configuration and cost arithmetic do not support the advertised resource budget.** Evidence: interview lines 78–82 and 106–129; production LLD RAM/schema and system-design cost row.

   The collection omits `datatype: "float16"`, so its originals default to float32: **163.84 GB** for 40M × 1024 values before replication, not the stated ~80 GB. Quantization does not change the original vector datatype. [Qdrant vector datatype documentation](https://qdrant.tech/documentation/manage-data/vectors/).

   Other corrections: 2560 → 1024 is **2.5×**, not 4× at equal precision; $0.10 with 40% free cache hits is **$0.06**, before infrastructure, not $0.05. The 37 GPU-hour embedding calculation is arithmetically sound if 300 chunks/s is achieved, but excludes fetch, parse, sparse encoding, transfer, index construction, and validation.

   Correction: measure resident memory with payload indexes, sparse indexes, page cache, WAL/optimizer work, shard overhead, and failure/rebuild headroom. Price the whole service: replicas, storage/IO, backups, Redis/RDS, Databricks, warm fallback GPUs, observability, networking, and operations. Distinguish marginal inference cost from fully loaded cost per successful verified memo. Cloud prices in the interview were not validated as current quotes.

14. **P2 — Reranker deployment compatibility and observability infrastructure are incomplete.** Evidence: interview lines 160, 198 and 244; production HLD `rerank_t` and `langfuse_t`.

   TEI support for Qwen3 embeddings does not prove that its rerank endpoint serves the chosen Qwen3 reranker. The official support page lists other reranker architectures, while the named Qwen artifact declares `Qwen3ForCausalLM`. Treat this combination as **unproven**, not as a ready deployment or an assertion that every TEI version is incapable of serving it. Pin the image and demonstrate the request format, scoring semantics, truncation, and measured performance. [TEI model support](https://huggingface.co/docs/text-embeddings-inference/supported_models), [Qwen reranker configuration](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/raw/main/config.json).

   “Langfuse on the same RDS” does not describe a complete modern deployment: v3 introduced worker infrastructure, ClickHouse, Redis, and blob storage alongside Postgres. Cost, failure isolation, and retention must include them. [Langfuse v2-to-v3 architecture changes](https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v2-to-v3). Test trace-backend outage without blocking memo delivery.

15. **P2 — The multi-cloud and operational rationale is incomplete, even if the placement is intentional.** Evidence: interview lines 37, 187, 216–224 and 253–255; production system-design `n_mc`.

   Azure LLM fallback is itself a cross-cloud request path, so “never on the request path” needs to be limited to the data-plane transfer. AWS edge, Redis, workers and RDS still share a serving-region failure domain. The egress estimate covers one transfer leg, not raw S3-to-ADLS mirroring and all storage/replay traffic. “A second consumer means Kafka” is not a sufficient criterion: consumer count alone does not decide broker choice.

   Correction: document the actual failure matrix and operational owners, restore-tested RPO/RTO, index/database backup reconciliation, retention, deployment rollback, and network/secret boundaries. Separate disposable cache eviction from durable queue/idempotency/rate-limit state by policy or deployment. Define freshness per source/form, including amendments and accepted/published/fetched times; generic two/five-day dbt freshness is not the consumer-facing 8-K SLO. Decide infrastructure from required retention, lag, recovery, throughput and existing team capability.

**Specific diagram revisions**

| Artifact | What works | What must change |
|---|---|---|
| production/hld.excalidraw | Clear serving/data separation; native retrieval; provider alternatives | Fix custom sharding, add durable execution/delivery contracts, show snapshot/verification ownership, isolate interactive embedding capacity, budget the complete observability stack. The summary places drafting before retrieval; reorder it to evidence → freeze → draft. |
| production/lld.excalidraw | Concrete schema, cache and state panels | Correct precision/shard semantics; version keys; define job ownership and customer boundaries; distinguish provisional and terminal status. The note that every key carries collection version or snapshot hash is contradicted by its own table. |
| production/critical-flow.excalidraw | Cache and failure branches make the intended behavior inspectable | Join all evidence branches before freeze; explicitly show the news/market return paths. Define final transport events, revision replacement and policy-before-exposure. Retain optional sentiment when requested, as in the earlier flow. |
| production/system-design.excalidraw | Useful comparison of scaling stages and proposed triggers | Correct HEAD descriptions, concurrency/cost math, cloud failure claims and eval gates. Replace absolute claims such as linear reranker scaling and corpus-independent latency with measured operating envelopes. |

The PNGs are readable at enlarged resolution and match the source labels examined. The main weaknesses are semantic, not artistic; adding more boxes will not resolve them.

**Corrections to the interview’s description of current code**

- The serious existing status bug is real: `app/main.py:368` marks any nonempty memo completed. `app/agents/graph.py:625` records failed verification as a recoverable error. The future design should fix this explicitly; describing a repair loop does not establish it at HEAD.
- Serial evidence routing is visible in `app/agents/graph.py:688`. The document correctly distinguishes target fan-out from current execution.
- The statement that current eval outputs lack lineage is too broad: `evaluation/quality_baseline.py:411` records git, provider/model, temperature, evaluator configuration, evidence release and snapshot IDs. `dataops/contracts.py:33` already defines source-level immutable evidence snapshots. The proposed per-run aggregate `snapshot_hash` is an additional contract, not the first evidence identity in the repository.
- `.github/workflows/ci.yml:29` already runs document governance; a generation-quality replay gate is still missing. “Nothing runs in CI” should be narrowed to the specific missing evaluation gate.
- README’s Chroma-default claim conflicts with accepted ADR-0003, which ratifies Qdrant as the operational default. The README also shows `api/` and `rag/` paths where canonical SPEC §5 names the actual `app/` layout.
- The existing hybrid search’s 512-candidate cap bounds its work; it does not necessarily scan millions of vectors per request. The stronger problem is incomplete sparse candidate coverage and rebuilding BM25 per query. The target native sparse approach is sensible, but the stated explanation exaggerates the current algorithm’s scaling behavior.
- SPEC v1.2 still defines S10 as optional multi-agent work, while the interview assigns production infrastructure to S10 and references SPEC §11.3, which does not exist in canonical v1.2. The amendment is a proposal. Surface this traceability mismatch; do not silently treat the overlay as an accepted scope change.

**What I would keep**

Keep the single-agent workflow, evidence registry before generation, immutable snapshots, typed retrieval boundary, offline ingestion, metadata-aware retrieval, measured adoption of sparse/reranked methods, bounded repair, and a durable queue when workload justifies it. Matryoshka reduction is supported by the chosen 4B embedding family; its quality and performance still need this project’s measurements. [Qwen3-Embedding-4B model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B).

Do not introduce a new “missing prefetch filter” finding merely because the example uses a top-level filter. Qdrant’s official guide says global filters propagate into prefetches; verify behavior against a pinned server version instead of assuming post-filtering. [Qdrant global and prefetch filters](https://qdrant.tech/course/essentials/day-5/universal-query-api/).

**Evidence I would require before approving the production target**

| Gate | Reviewable evidence |
|---|---|
| Trust contract | Failed verification never returns completed; exact quote identity; final bytes and verification agree; prohibited output is blocked before exposure. |
| Durable execution | Crash-boundary tests, fenced redelivery, idempotency conflicts, terminal result reconciliation, and authenticated event access. |
| Retrieval release | Bounded shard topology, valid pinned configuration, interrupted-publication recovery, version invalidation, and representative paired quality results. |
| Capacity | Offered-load benchmark with realistic distributions, cold caches, hot issuers, provider/GPU failure, queue deadlines and rejection rates. |
| Financial quality | Expert-labeled factual/period/unit/negation cases; calibrated false-acceptance and refusal metrics; approved fallback model combinations. |
| Operations and economics | Restore and rollback rehearsal, RPO/RTO, per-source freshness, full service cost, spending controls, and named operational ownership. |

A simpler initial deployment could retain one serving region, a bounded Qdrant deployment, one approved fallback provider, and Postgres-backed durable jobs. Existing Databricks investment may justify the chosen data plane, but the document should establish that organizational fact and its full cost. A warm third model fleet should earn its place through measured demand or a recovery requirement.

**Verification record and limits**

- Eight Excalidraw files parsed successfully; four production PNGs visually inspected.
- Reproduced shard, dimension, concurrency, cache-cost, vector-storage and embedding-duration arithmetic with Python.
- `python3 scripts/ci/check_no_scope_residue.py`: PASS.
- `python3 scripts/ci/check_sprint_map.py`: PASS.
- `python3 scripts/ci/check_doc_sync.py`: PASS.
- `python3 scripts/ci/check_test_hygiene.py`: FAIL on existing `tests/unit/test_llm.py:25,32` call-order assertions. These files were not changed.
- No runtime tests, deployment experiment or load measurements were added; this was a design/code review. The governance suite is not all green, and production readiness is not approved.
- Only this review and an appended task-ledger entry were written. The reviewed designs, interview, SPEC and implementation remain available unchanged for comparison.
