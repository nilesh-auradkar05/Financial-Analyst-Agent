# ADR-0007 — Production interview scale design

**Status:** Proposed — interview design only; user authorized documentation revision on 2026-09-06.
**Related:** SPEC v1.3 §§0, 3, 7, 10–12; ADR-0003; proposed ADR-0006; accepted ADR-0008; production-design-review.md findings 1–15.

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
