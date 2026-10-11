# ADR-0003 — Qdrant behind the retrieval interface

**Status:** Accepted (2026-06-04). Qdrant is the working default backend on operational grounds (below). The paired quality comparison planned for Gate B (S4-T04) was retired by owner decision on 2026-10-10 (addendum below); no quality win is claimed for the backend choice.
**Date:** 2026-06-03 (accepted 2026-06-04)
**Related:** SPEC §1, SPEC §6, SPEC §11, `retrieval-benchmark.md` §9, sprint-plan S3–S4.

## Context

The system retrieves SEC evidence through a typed `RetrievalStore` contract. Chroma was the original implementation; Qdrant is now implemented behind the same contract and is the configured default (`VECTOR_BACKEND` defaults to `qdrant`, selected via the `get_vector_store()` DI factory). We want a more production-shaped vector backend with first-class payload-index filtering and native sparse/hybrid support for S5, without churning code above `rag/`. The benchmark fixtures and method-matrix results to date were generated on Qdrant, so standardizing the working substrate on Qdrant keeps benchmarking embedding-matched and reproducible.

## Decision

Implement `QdrantVectorStore` as a `RetrievalStore` implementation behind the existing Protocol. Run Qdrant locally via Docker. Create payload indexes for `ticker`, `filing_type`, `section_key`, `filing_date`, and `accession_number`. **Adopt Qdrant as the working default backend** (`VECTOR_BACKEND=qdrant`); keep Chroma as a config-level fallback.

**Why Qdrant is the default — operational, not quality:** payload-index filtering and native sparse/hybrid support needed for S5, plus the fact that the benchmark fixtures and results already run on Qdrant. With identical embeddings and `top_k`, the two backends are expected to **tie** on retrieval quality within the confidence interval. This ADR claims **no** quality advantage.

**Gate B obligation (S4-T04):** once the Gate-A anchored oracle exists, run the strict, paired, embedding-matched Chroma-vs-Qdrant comparison and record the actual numbers here. Gate B confirms (or revisits) the default on evidence; until then the default rests solely on the operational rationale above. Do not manufacture a quality win. The world already has enough synthetic confidence.

## Consequences

- A single contract-parity test file must pass under both backends.
- Differences are fixed in adapters, never by forking tests.
- Ingestion must write with the same embedding model/version under both backends.
- No public signature above `rag/` may change.
- Adds local Docker dependency for Qdrant.
- Rollback is config-level: Chroma remains a working implementation.

## Alternatives considered

- Stay on Chroma: lowest effort, weaker payload filtering and harder S5 path.
- Managed vector DB: out of scope now.
- pgvector / other stores: viable, but Qdrant better fits payload filtering and hybrid-readiness for this roadmap.

## Addendum 2026-10-08 — managed deployment and release publication (owner decision D11, ADR-0006)

Applies to the deployed (S10) configuration; the local Docker setup above is unchanged.

- **Managed Qdrant Cloud on AWS `us-east-1`**, the same region as the workers. Free 1 GB node for the demo tier (it suspends after a week idle, so a weekly keep-alive query or a small paid node is required); a 3-node multi-AZ cluster with replication factor 2 and int8 quantization over float16 originals at scale.
- **One collection per index release** (`sec_chunks__<release_id>`). A release is built, validated (point count equals the manifest, filter smoke tests pass, removed chunk IDs are absent) and then promoted by an atomic alias swap of `sec_chunks_active`. The previous release is retained for rollback, which is one alias update. Queries never see a partially published filing (production review finding 8).
- **Automatic sharding with an issuer payload filter**, not per-ticker custom shards (8,000 keys × 12 shards × replication 2 would be 192,000 shard replicas, review finding 1).
- **Dense and sparse vectors in one collection**, so S5 can retire the per-query BM25 rebuild if Gate C confirms it.
- Vectors are rebuildable without re-embedding from the Parquet files in S3 (ADR-0011).

Why managed: the `RetrievalStore` contract and payload indexes stay unchanged above the retrieval boundary, and backups, upgrades and multi-AZ placement are not operated by one person. Alternatives: self-hosted Qdrant on ECS + EBS (cheaper at the 10M-user tier, owner runs upgrades and backups); Aurora pgvector (one fewer system, weaker sparse/hybrid support); OpenSearch Serverless (native hybrid, higher cost floor); S3 Vectors (very cheap, no sparse vectors; fits a cold archive tier).

Trigger to self-host: managed spend above about twice the equivalent EC2 cost plus operating time. Verify before implementation: Qdrant Cloud paid-tier availability in `us-east-1`.

## Addendum 2026-10-10 — paired backend comparison retired

Owner decision: the Chroma-vs-Qdrant paired comparison (sprint S4) will not be run. Qdrant remains the default for the operational reasons already recorded here (payload-index filtering, native sparse/hybrid support for S5, and the managed Qdrant Cloud target in the 2026-10-08 addendum). With identical embeddings and `top_k` the two backends were expected to tie on quality (`docs/retrieval-benchmark.md` §9), so the comparison would not have changed the decision. Consequences: Gate B closes at S3 exit; no retrieval-quality claim may be attributed to the backend; Chroma remains a supported local backend behind the same contract.
