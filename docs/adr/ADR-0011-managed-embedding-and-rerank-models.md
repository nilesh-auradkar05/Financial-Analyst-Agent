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
- **Storage (Parquet originals and the local Qdrant index):** float16 originals with int8 quantization (about 2 KB on disk and 1 KB in RAM per vector; 40M chunks ≈ 82 GB of originals instead of 164 GB at float32, review finding 13). The deployed OpenSearch Serverless index chooses its own compression (ADR-0014), so its recall is measured separately.
- **Degradation ladder for query embedding:** exact cache → Bedrock → Foundry (only if parity passed) → keyword-only (BM25) retrieval in the deployed OpenSearch index (ADR-0014) with the run marked `degraded` (`reduced_retrieval`).

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
- Tests (test-plan §20; provider switch and per-arm indexes in §21): parity gate refuses to enable the fallback below threshold; a cache key built with one fingerprint never serves another; keyword-only fallback marks the run `degraded`.

## Amendment 2026-10-11 — four-arm bake-off (owner decision)

The Gate C bake-off compares four embedding models instead of one candidate against the baseline with a control.

| Arm | Host | Dimensions | Role |
| --- | --- | --- | --- |
| `qwen3-embedding:4b` | Ollama (self-hosted) | 2560 | Current baseline |
| Cohere Embed v4 | Amazon Bedrock | 1024 | Candidate (has a Microsoft Foundry twin) |
| Titan Text Embeddings V2 | Amazon Bedrock | 1024 | Candidate (was "control only") |
| OpenAI `text-embedding-3-small` | Azure OpenAI | 1024 (reduced from 1536) | Candidate (added) |

- **Dimensions:** the three cloud arms run at 1024 dimensions so that dimension is not a second variable among them. Qwen3 runs at 2560, its current configuration, because it is the baseline being replaced.
- **Input conventions:** each arm uses its model's documented query and document input convention. The convention is part of the embedding fingerprint's preprocessing version.
- **Two stages.** Stage 1 is the retrieval comparison on the frozen fixture and decides Gate C. Stage 2 measures the downstream memo: news and stock evidence replayed from a frozen evidence release, the same chat model and settings, filings retrieved again per arm, and repeated runs per ticker. Stage 2 is reported beside Stage 1 and does not override it.
- **Single-host winners:** Titan V2 exists only on AWS and `text-embedding-3-small` only on Azure. The cross-host fallback above needs the same model at the same dimension on both hosts, so it cannot be enabled for either. If one of them wins, adoption needs a follow-up decision on the query-embedding ladder (exact cache → single host → keyword-only) before the model changes.
- **Sequencing:** the provider switch, per-arm indexes and the embedding identifier in ingest metadata may be built before Gate A. No arm is measured, and no comparison is cited, until the Gate A fixture is frozen.
- The "Titan Text Embeddings V2 — Control only" row in the alternatives table is superseded by this amendment. `text-embedding-3-large` stays rejected.

## Verify before implementation

- Bedrock batch support for Cohere Embed v4; Foundry deployment of Cohere Embed v4 at 1024 dimensions.
- Current Bedrock rerank price per 1,000 queries.
- Added 2026-10-11: this account's Bedrock access to Titan Text Embeddings V2 and Cohere Embed v4, and an Azure deployment of `text-embedding-3-small`.
