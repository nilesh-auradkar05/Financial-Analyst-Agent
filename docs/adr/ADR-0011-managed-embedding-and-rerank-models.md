# ADR-0011 — Managed embedding and rerank models

**Status:** Accepted (2026-10-08) by owner decisions D10 and D12, **provisional on Gate C**: the embedding model changes only if the paired bake-off passes; rerank is enabled only on measured lift.
**Date:** 2026-10-08
**Related:** SPEC §§6, 11.1 (Gate C), 11.3; `docs/retrieval-benchmark.md`; ADR-0003 (and its 2026-10-08 addendum); ADR-0006; ADR-0009 amendment; production review findings 6, 13, 14.

## Context

Embeddings come from Ollama `qwen3-embedding:4b` (2560 dimensions). A deployed system would need an always-on GPU endpoint for query embeddings. ADR-0006 keeps Azure as a fallback for models only, and the owner asked for the embedding fallback to be on Azure. An index belongs to exactly one embedding model and dimension, so a fallback for query embeddings must be the same model at the same dimension, or retrieval silently degrades. The reranker candidate in the interview design (TEI with Qwen3-Reranker) was unproven (review finding 14).

## Decision

### Embeddings (D10)

- **Candidate model: Cohere Embed v4 at 1024 dimensions**, primary on Amazon Bedrock, fallback on Microsoft Foundry ($0.12 per million input tokens on both).
- **Gate C bake-off before adoption:** Cohere Embed v4 (1024d) against the current `qwen3-embedding:4b` baseline, with Titan Text Embeddings V2 as a low-cost control, on the frozen fixture, with the embedding model as the only changed variable (`docs/retrieval-benchmark.md` paired policy). If Cohere loses beyond the predeclared tolerance, Qwen3 stays and the deployed system self-hosts it.
- **Cross-host parity gate before the fallback is enabled:** Bedrock lists dimensions up to 3072 and Foundry up to 1536, which suggests different serving revisions. A fixed 200-chunk sample is embedded on both hosts at 1024 dimensions; every pair must reach cosine similarity ≥ 0.999. The gate reruns whenever either host changes the model version. Until it passes, the fallback is off.
- **Embedding fingerprint** = model id + version + dimension + preprocessing version. It is part of every query-embedding cache key and every index release manifest (ADR-0013).
- **Vectors are persisted** to S3 Parquet keyed by `(chunk_id, embedding_fingerprint)`, so any new index (release, rebuild, vector-store migration) is a bulk load, not a re-embed.
- **Storage:** float16 originals with int8 quantization (about 2 KB on disk and 1 KB in RAM per vector; 40M chunks ≈ 82 GB of originals instead of 164 GB at float32, review finding 13).
- **Degradation ladder for query embedding:** exact cache → Bedrock → Foundry (only if parity passed) → sparse-only retrieval with the run marked `degraded` (`reduced_retrieval`).

### Rerank (D12)

- **Bedrock Rerank with Cohere Rerank 3.5, off by default.** It is enabled only if Gate C shows recall lift on the frozen fixture. Billed per query of up to 100 documents.
- With the cheaper generation models (ADR-0009 amendment), rerank costs about $0.008 per uncached memo against about $0.014 for generation, so the lift must be clear.

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| `qwen3-embedding:4b` self-hosted | Baseline / fallback outcome | Current, measured; needs an always-on GPU for queries; 2560 dims |
| Titan Text Embeddings V2 | Control only | Cheapest; AWS only, so no Azure fallback |
| OpenAI text-embedding-3-large | Rejected | Azure only |
| Voyage finance-2 | Alternative | Finance-tuned; marketplace deployment, no Azure twin |
| Self-hosted bge / Qwen3 reranker on GPU | Alternative | Lower marginal cost at the 10M-user tier |
| TEI + Qwen3-Reranker | Rejected | Unproven runtime support (review finding 14) |

## Consequences

- Changing the embedding model invalidates every retrieval baseline until the bake-off reruns; no retrieval quality claim carries across the change.
- Cohere costs well under $1/month for ingestion at demo scale, so cost does not decide the bake-off.
- Tests (test-plan §20): parity gate refuses to enable the fallback below threshold; a cache key built with one fingerprint never serves another; sparse-only fallback marks the run `degraded`.

## Verify before implementation

- Bedrock batch support for Cohere Embed v4; Foundry deployment of Cohere Embed v4 at 1024 dimensions.
- Current Bedrock rerank price per 1,000 queries.
