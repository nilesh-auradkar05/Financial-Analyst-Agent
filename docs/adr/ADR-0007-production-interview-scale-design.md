# ADR-0007 — Production interview scale design

**Status:** Proposed — interview design only; user authorized documentation revision on 2026-09-06. Superseded in part on 2026-10-08 by ADR-0006 and ADR-0009 to ADR-0013 (see the amendment at the end).
**Related:** SPEC v1.3 §§0, 3, 7, 10–12; ADR-0003; proposed ADR-0006; accepted ADR-0008; production-design-review.md findings 1–15 (review archived 2026-10-11 to `old_artifacts/docs/`).

## Context

The production companion explains how this system could evolve at much larger corpus and request volumes. The owner's practical goal is to run affordable experiments and log results at achievable levels. Large-scale numbers are explicit projections, not claimed demonstrations or requirements to run million-user tests.

This proposal does not accept ADR-0006, select a paid deployment, or introduce runtime implementations. SPEC v1.3 (applied 2026-09-12, ADR-0008) already lists S10 as cloud work gated on ADR-0006 and keeps that work out of committed scope in §3.y. Existing canonical benchmark methodology remains authoritative.

## Proposed direction

- Preserve exact evidence identity, including the quote actually used; version embedding/retrieval/memo/policy caches with artifact fingerprints.
- Distinguish issuer CIK/ticker aliases from authenticated customer ownership.
- Begin durable execution with Postgres jobs, leases, fenced attempts and durable events. Add Redis Streams only with a transactional outbox and measured need. HTTP is POST acceptance plus authenticated GET event replay/polling.
- Default user-visible streaming carries progress, then the final verified and policy-approved memo. Optional provisional prose requires a pre-exposure policy contract, unverified state and version replacement. Model TTFT, visible-text latency and verified-completion latency are different metrics.
- Use a bounded Qdrant shard topology with issuer payload filtering; explicit float16 originals and independently evaluated quantization. Publish complete versioned index releases and retain rollback state. Dedicated custom shards are an optional hot-issuer optimization.
- Independently qualify primary and fallback draft/verifier models. Stale fallback preserves its old snapshot, age and degraded status. No eligible result means a typed durable failure.
- Keep the practical pipeline local and reproducible. Distributed workers, cloud serving and optional Databricks/Delta/dbt are larger-scale alternatives whose value depends on workload and organizational context.
- Reuse existing metrics/tracing initially. A self-hosted Langfuse option includes its actual worker, Postgres, ClickHouse, Redis and blob-store dependencies.
- Record affordable corpus/concurrency experiments with code, fixture, model and configuration lineage. State load, cache state, offered/completed/rejected requests, quality, latency, resources and costs.

## Evidence and alternatives

The accepted review records failure scenarios and links primary vendor sources. The revised interview contains the numerical model, follow-up answers, practical experiment ladder and the four matching SVG/Excalidraw diagrams.

Alternatives retained: a Postgres-only queue versus outbox-to-Streams; automatic bounded shards versus dedicated hot partitions; one qualified fallback versus a warm third model fleet; local batch ingestion versus an existing organizational lakehouse. A consumer count or large headline corpus alone does not justify infrastructure.

## Consequences and implementation gates

The diagrams are design proposals. Component support, failure behavior and representative workload measurements must be demonstrated before implementation claims or adoption. Retrieval changes use the paired frozen-fixture policy in retrieval-benchmark.md; this proposal introduces no new statistical oracle.

Any later deployment requires ADR-0006 acceptance. SPEC §3.y keeps cloud out of committed scope until then. Practical validation can stop at an affordable measured level and document the remaining uncertainty.

## Amendment 2026-10-08 — superseded in part by accepted decisions

ADR-0006 was accepted on 2026-10-08 together with ADR-0010 to ADR-0013. Where this proposal and those records differ, the accepted records govern. This file remains the interview companion for scale projections.

| Topic in this proposal | Now governed by | Change |
| --- | --- | --- |
| Postgres outbox → Redis Streams when Postgres claim throughput is a measured constraint | ADR-0013 | Graduation target is an outbox dispatching to SQS with a dead-letter queue. Valkey holds only disposable cache data, so a durable queue there would mix failure domains |
| Bedrock primary plus an "independently qualified Azure alternative" | ADR-0009 amendment | The same models (DeepSeek V3.2, gpt-oss-120b) on Bedrock and Foundry; qualification is a parity check |
| Embedding service, optional reranker service | ADR-0011 | Managed Bedrock embedding (Cohere Embed v4, 1024 dims) with a Foundry fallback behind a parity gate; managed Bedrock rerank only on Gate C lift |
| Optional Databricks/Delta/dbt data plane, Azure Databricks | ADR-0012 | Ingestion is the `dataops` CLI orchestrated by Airflow 3 on MWAA Serverless; no lakehouse |
| Self-hosted Langfuse option | ADR-0006 (D17) | Grafana Cloud through OpenTelemetry, plus LangSmith |
| Customer authorization contract | ADR-0010 | Cognito, a server-side session in the web BFF, an issuer allowlist and customer-scoped authorization with roles |
| AWS serving region with a separately planned restore/failover | ADR-0006 | One active region; region loss is a documented failure mode with listed mitigations, none built |

The diagrams under `docs/System-design/production/` were re-produced on 2026-10-08 to show the accepted deployment rather than this proposal.
