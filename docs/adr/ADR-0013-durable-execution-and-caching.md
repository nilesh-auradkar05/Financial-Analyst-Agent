# ADR-0013 — Durable execution, queue graduation and caching

**Status:** Accepted (2026-10-08) by owner decisions D06, D07, D08 and D13. **Amended 2026-10-11 by ADR-0015:** the contracts in this record (acceptance transaction, fencing, event replay, versioned cache keys, exact query-embedding reuse, single-flight) are unchanged, but the demo tier realizes them on DynamoDB, SQS and DynamoDB TTL items instead of Aurora, a Postgres queue and Valkey (see the amendment at the end). The Aurora / Valkey realization below is retained as the production scale tier. Built locally in S7; deployed in S10.
**Date:** 2026-10-08
**Related:** SPEC §§2.1, 5, 7, 8, 11.3; sprint-plan S7(c)–(e); ADR-0006; ADR-0007 (superseded in part); ADR-0010; ADR-0011; ADR-0015; `docs/production-scale-architecture.md`; production review findings 2, 4, 5, 6, 7.

## Context

The service runs in one process with in-process background tasks and a JSON file run store. ADR-0007 proposed Postgres jobs first and Redis Streams later. The production review found the job protocol, event delivery and cache identities underspecified (findings 2, 4, 5, 6). In the deployed design the cache is ElastiCache Serverless for Valkey, which evicts under memory pressure, so it cannot hold durable work.

## Decision

### Durable state (D06)

One Aurora PostgreSQL Serverless v2 cluster (Multi-AZ in the scale tier; the demo tier uses DynamoDB instead, see the amendment) holds:

- `jobs` (state, priority, `attempt` fencing token, `lease_until`, `customer_id`, `request_hash`, `snapshot_hash`, result reference), `job_events` (ordered per job, replayable after `Last-Event-ID`), `idempotency_keys` (unique per `customer_id` + key, with the normalized request hash), `customers` and quotas, run records, and the LangGraph checkpointer.
- Acceptance is one transaction: reserve the idempotency key, insert the job and its first event, return 202 with the events URL. A reused key with a different request hash is rejected.
- Claim: `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id, attempt` increments `attempt` and sets a lease renewed by heartbeat.
- Commit: the result and terminal event are written only `WHERE attempt = :fence`; zero rows means a stale worker and its result is discarded. Fencing does not give exactly-once model execution or billing; a possibly duplicated provider call is recorded as such.
- This replaces the concrete `FileBackedRunStore` (`app/services/run_store.py`) behind the `RunStore` protocol that S7(d) introduces.

### Queue graduation (D07)

- Postgres is the queue until measurements show it limits throughput: p95 claim latency above 200 ms, or sustained claims above about 500/second.
- Then a transactional outbox row is written with the job and dispatched to **Amazon SQS with a dead-letter queue**. Delivery is at-least-once, so workers stay fenced and idempotent.
- This supersedes sprint-plan S7(d) and ADR-0007's Redis Streams target. Kafka/MSK is considered only for stated retention or replay requirements.

### Caching (D08)

ElastiCache Serverless for Valkey (local: a Valkey container) holds only data that can be lost:

| Cache | Key | Lifetime |
| --- | --- | --- |
| Evidence | `ev:{source}:{issuer_cik}:{params_hash}` | market 60 s, news 15 min, SEC until a new accession |
| Query embedding | `qe:{embedding_fp}:{sha256(normalized_query)}` | release lifetime |
| Retrieval result | `rr:{index_release}:{retrieval_cfg_fp}:{query_hash}:{filter_hash}` | release lifetime |
| Memo | `memo:{snapshot_hash}:{prompt_version}:{model_fp}:{verifier_policy_fp}` | snapshot validity |
| Single-flight lock | `lock:memo:{same suffix}` (`SET NX PX`) | generation deadline |
| Rate-limit window | per customer | seconds |

- Every key carries the fingerprint of what produced the value, so a new embedding model, index release, prompt, model or verifier policy can never serve a stale value (review finding 6).
- Query-embedding reuse is **exact only**. SPEC §11.3's semantic reuse (cosine ≥ 0.98) is removed: it needs the new embedding to measure similarity, which erases the saving, and near-identical financial queries differ by fiscal year or negation.
- The memo cache hits only while the snapshot is identical, so its useful window is the 60 s market TTL, which is exactly an earnings-day burst. The single-flight lock lets one job generate while identical jobs wait for its artifact.
- Daily token and job quotas live in Postgres. Losing Valkey removes no correctness-critical limit: rate limits fall back to a conservative per-task limit and every request runs cold.

### Sentiment (D13)

FinBERT leaves the API event loop: a process pool in the worker at first, its own CPU service from the 100K-user tier (scale tier; the demo tier runs it as ONNX inside the worker Lambda, see the amendment). Scores are cached permanently by `(article sha256, model version)`, because sentiment is a pure function of text and model.

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| Redis Streams on Valkey (ADR-0007) | Rejected | Durable work on an evictable cache mixes failure domains |
| Kafka / MSK | Rejected for now | No retention or replay requirement |
| DynamoDB for jobs | Rejected on 2026-10-08; **adopted for the demo tier on 2026-10-11 (ADR-0015)** | Loses SQL joins for audits; acceptable because lineage lives in the eval registry and S3 run records, and nothing bills while idle |
| RDS PostgreSQL t4g | Alternative | Cheapest always-on; would change engine later |
| Upstash Redis | Alternative | Per-request pricing, outside AWS |
| Semantic memo-output caching | Prohibited | SPEC §11.3, unchanged |

## Consequences

- SPEC §11.3 and §5 change: exact query-embedding cache; memo key adds the verifier/policy fingerprint; retrieval keys are versioned.
- Scale tier: Aurora Serverless v2 takes about 15 s to resume from zero (30 s or more after a day paused), one reason the demo tier uses DynamoDB (ADR-0015).
- Tests (test-plan §20): crash after claim and after result persistence commits exactly once; a stale fenced worker's commit is rejected; idempotency conflicts are refused; each version change misses the cache; concurrent identical requests produce one generation; a Valkey outage loses no job.

## Amendment 2026-10-11 — demo-tier realization (ADR-0015)

The owner stated that the deployment serves occasional recruiter visits, one query each, with days of no traffic. ADR-0015 therefore realizes the same contracts on services that cost nothing while idle. The Aurora / Postgres queue / Valkey design above becomes the production scale tier (`docs/production-scale-architecture.md`), adopted per component on measured triggers.

| Contract | Scale tier (this record) | Demo tier (ADR-0015) |
| --- | --- | --- |
| Accept: key + job + event 1 atomically | One Postgres transaction | One DynamoDB `TransactWriteItems` |
| Enqueue without stranding a job | Transactional outbox row → SQS (after the D07 trigger; before it, the job row is the queue) | DynamoDB Streams on `jobs` → stream-triggered dispatcher Lambda → SQS |
| Claim with a fencing token | `SKIP LOCKED` + `attempt + 1` + lease | Conditional `UpdateItem` (`state = queued`, or `state = running` with an expired lease) + `attempt + 1`; SQS visibility timeout as the delivery lease |
| Fenced commit | `WHERE state = 'running' AND attempt = :fence` | `state = running AND attempt = :mine`; events in a `TransactWriteItems` with a `ConditionCheck` on the job; DLQ handler and lazy expiry bump `attempt` when they write `failed` |
| Event replay after `Last-Event-ID` | `job_events` table | `job_events` table (`job_id`, `seq`) |
| Deadline enforcement | Lease expiry and retry exhaustion | Same, plus lazy expiry on read (no scheduled sweeper) |
| Caches (same key strings) | Valkey, evictable | DynamoDB items with TTL; not evictable, so the "cache loss" case becomes "cache bypass on read error" |
| Single-flight | `SET NX PX` | Conditional put (`attribute_not_exists(pk) OR expires_at < :now`); TTL only cleans up, because it deletes late |
| Quotas and rate windows | Postgres quotas; Valkey windows | DynamoDB atomic counters with TTL |
| LangGraph checkpointer | Postgres saver | `DynamoDBSaver` (`langgraph-checkpoint-aws`), S3 offload above 350 KB |
| FinBERT | Process pool in the worker; own service at 100K users | ONNX model in the worker Lambda |
| Local development | Postgres and Valkey containers | DynamoDB Local and ElasticMQ containers |

D07's graduation trigger (p95 claim > 200 ms or > 500 claims/s) applies only once the job store is Postgres. In the demo tier SQS is the queue from the start, because it costs nothing while idle.

