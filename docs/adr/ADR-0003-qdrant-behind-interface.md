# ADR-0003 — Qdrant behind the retrieval interface

**Status:** Accepted (2026-06-04). Qdrant is the working default backend on operational grounds (below). The *quality-default* decision still passes through Gate B (S4-T04) once the trusted Gate-A oracle exists; Gate B may confirm or revisit this, but no quality win may be claimed without it.
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
