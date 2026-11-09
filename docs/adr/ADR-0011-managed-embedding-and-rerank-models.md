# ADR-0011 — Managed embedding and rerank models

**Status:** Accepted (2026-10-08) — amended 2026-11-09 (CM-6e: Foundry Cohere default arm, 20 RPM constraint) by owner decisions D10 and D12, **provisional on Gate C**: the embedding model changes only if the paired bake-off passes; rerank is enabled only on measured lift. **Amended 2026-10-23 by ADR-0016:** the bake-off runs on OpenSearch Serverless (CM-5). The Qwen3 arm is the Gate A vectors copied unchanged (CM-4). If no cloud arm qualifies, the owner decides; Qwen3 is no longer the fallback because it is not served in the cloud.
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

## Amendment 2026-11-07 — rerank on Microsoft Foundry (owner decision, CM-6d)

- **D12 changes host and model:** Cohere Rerank v5 fast on Microsoft Foundry, deployment `cohere-rerank-v5-fast` (setting `AZURE_FOUNDRY_RERANK_MODEL`), replaces Bedrock Cohere Rerank 3.5. Called at `POST {AZURE_OPENAI_ENDPOINT}/providers/cohere/v2/rerank` with the Foundry key; verified live 2026-11-07 (1.1 s; `results[{index, relevance_score}]`, `meta.billed_units`).
- **The local cross-encoder is removed** (sentence-transformers, `Qwen/Qwen3-Reranker-0.6B`), and with it torch, transformers and every CUDA package (owner: cloud-only model plane).
- **Unchanged:** rerank stays off by default and is enabled for runtime only if Gate C shows lift; evaluation mode fails closed when the reranker is unavailable or unconfigured; there is no cross-host rerank fallback. Cohere's relevance scores are calibrated to [0, 1], unlike the old cross-encoder logits, so the section and keyword prior boosts (`section_prior_boost` 0.20, `keyword_prior_boost` 0.03) act on a different scale; no reranked result was recorded before this change, so nothing becomes non-comparable.
- **Tests:** test-plan §25.

## Prices for the CM-5b tie-break (recorded 2026-11-08, before any arm run)

On-demand input-token prices; the §9a tie rule compares these.

| Arm | Price per million input tokens | Source |
| --- | --- | --- |
| Bedrock `cohere.embed-v4:0` | $0.12 | AWS Price List API, `AmazonBedrockFoundationModels` us-east-1 (published 2026-10-12): "Cohere Embed 4 Model (Amazon Bedrock Edition)" |
| Bedrock `amazon.titan-embed-text-v2:0` | $0.02 | AWS Price List API, `AmazonBedrock` us-east-1 (published 2026-11-06): `TitanEmbeddingV2-Text-input-tokens`, $0.00002 per 1K |
| Azure `text-embedding-3-small` (deployment `embed-3-small`) | $0.022 | azure.microsoft.com/pricing/details/azure-openai, embedding models: $0.000022 per 1K |

## Outcome 2026-11-08 — Gate C embedding (CM-5b): no arm qualified; owner override adopts Cohere Embed v4

- **Run:** three cloud arms (1024 d) built from the CM-4 index (532 chunks, chunk ids unchanged, `embedding_id` verified) on OpenSearch Serverless at 1x, section_aware, top_k 10, anchored fixture v1; each paired with the Qwen3 arm (Gate B result), 75 shared cases, bootstrap seed 13, 10,000 samples. Results commit 8dd1d4a (`evaluation/results/gate_c_<arm>_opensearch_section_aware_retrieval_anchored_benchmark_v1.json`).
- **K=10 paired Δ vs Qwen3 [95% CI]:** Cohere v4 recall +0.000 [−0.060, +0.060] (4 wins / 67 ties / 4 losses), section_recall +0.000 [−0.040, +0.040], MRR +0.011; Titan V2 recall −0.053 [−0.127, +0.020], section_recall +0.013 [−0.027, +0.067]; Azure 3-small recall −0.033 [−0.107, +0.033], section_recall −0.013 [−0.053, +0.027]. Cloud pairs: Cohere − Titan recall +0.053 [−0.013, +0.120]; Cohere − Azure +0.033 [−0.027, +0.093]; Titan − Azure −0.020 [−0.100, +0.060].
- **Rule result:** no arm qualifies under retrieval-benchmark §9a (recall@10 CI lower bound below −0.05 for all three; Cohere −0.060). The interval width comes from n = 75 with half-credit scoring; Cohere's means equal Qwen3's. 21 of 75 cases have recall@10 = 0 on all four arms (a shared ceiling, not the embedding).
- **Decision (owner, 2026-11-08): override, not a qualification.** Cohere Embed v4 at 1024 d (`bedrock:cohere.embed-v4:0:1024`) is adopted as the embedding model; CM-6e makes it the default. Superseded the same day by the amended outcome below. Not gated and accepted with it: warm query latency +477 ms p50 vs local Qwen3 (cloud round trip), price $0.12 per million tokens.
- **Unchanged:** the Foundry fallback stays off until the cross-host parity gate above passes; rerank remains a separate, later axis on the Cohere arm.
- **Cost:** 0.92 OCU-h ≈ $0.22 (search 2 OCU 20:35–20:53Z, indexing 1 OCU 20:36–20:52Z) plus ≈ $0.02 tokens (estimate).

## Outcome 2026-11-08 (amended) — Gate C reopened: Cohere Embed v5 fast qualifies and is adopted

- **Run (owner request, after the override above):** a fourth arm, Cohere Embed v5 fast on Azure Foundry (deployment `cohere-embed-v5-fast`, 1024 d), built from the same CM-4 index under the same conditions (532 chunks, per-ticker counts equal, `embedding_id` verified; Serverless 1x, section_aware, top_k 10, anchored fixture v1; seed 13, 10,000 samples), run 22:12–22:28Z. Results commit 19c634a.
- **Route:** the deployment refuses the OpenAI-schema embeddings routes (404 `api_not_supported`) and serves only Foundry's Cohere-native `/providers/cohere/v2/embed`. The app's `azure_openai` arm cannot call it, so the run used a one-off harness adapter with the same input types as the Bedrock Cohere v4 arm (`search_document` for chunks, `search_query` for queries). Provenance therefore reads `azure_openai:cohere-embed-v5-fast:1024` although the route is Cohere-native.
- **K=10 paired Δ [95% CI]:** vs Qwen3 recall +0.067 [+0.007, +0.133], section_recall +0.000 [+0.000, +0.000]. vs Cohere v4 recall +0.067 [+0.007, +0.133] (7 wins / 66 ties / 2 losses), section_recall +0.000 [−0.040, +0.040], pass +0.080 [+0.027, +0.147].
- **Rule result:** qualifies under retrieval-benchmark §9a (both lower bounds ≥ −0.05). It is the only qualifying arm, so the tie-break does not apply.
- **Decision (owner, 2026-11-08):** Cohere Embed v5 fast at 1024 d replaces Cohere Embed v4 as the embedding default, superseding the override above. CM-6e adds the Cohere-native Foundry embedding arm, makes it the default and re-runs this benchmark through that arm to confirm.
- **Not measured:** query latency (the deployment's rate limit forced 114 HTTP 429 retries; recorded latencies include backoff waits); price not yet cited.
- **Cost:** 0.80 OCU-h ≈ $0.19 through 22:28Z plus the idle tail; tokens not estimated.

## Amendment 2026-11-09 (CM-6e) — the Foundry Cohere default arm and its quota

Recorded before the CM-6e confirmation run.

- **Price:** Cohere Embed v5 fast (Global) is $0.08 per million text tokens on Microsoft Foundry (azure.microsoft.com/pricing/details/ai-foundry-models/cohere, read 2026-11-09; Cohere's Embed 5 announcement lists the same price).
- **Arm:** provider `foundry_cohere`, deployment `cohere-embed-v5-fast`, 1024 d, identifier `foundry_cohere:cohere-embed-v5-fast:1024`. It calls Foundry's Cohere-native `/providers/cohere/v2/embed` route (`input_type` `search_document` / `search_query`, at most 96 texts per request). This replaces the CM-5b harness label `azure_openai:cohere-embed-v5-fast:1024`, which named the wrong route. Every arm, the default included, gets an index named from its identifier; `sec_filings` keeps its 2560-d Qwen3 vectors and is not reused. Existing tickers are ingested again into the new index on first use (ingest-on-miss). Nothing is deleted.
- **Quota as a design constraint (owner, 2026-11-08; corrected 2026-11-09):** the deployment allows 20 requests and 20,000 tokens per 60 s (`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`, read 2026-11-09). 20 RPM is the maximum and cannot be raised. Queries are request-bound: an analysis embeds 4–9 queries of about 15 tokens, so one deployment serves about 2–5 analyses per minute system-wide until the exact query-embedding cache lands (S7.9, unchanged). Ingestion is token-bound: a 96-chunk batch is about 20,000 tokens, so a ticker's filing (about 170–210 chunks) takes two to three minutes. The first CM-6e build (2026-11-09 08:50Z) hit HTTP 429 after two such batches; the text first recorded here said ingestion was not limited, which was wrong.
- **Client behaviour (owner, 2026-11-09):** the arm spaces its own requests at no more than 18 per minute and keeps the estimated tokens (⌈characters ÷ 4⌉) of the requests in any 60 s window at or below 18,000, with at most 6,000 estimated tokens per request (owner choice 2026-11-09, option 1: token-aware pacing). On HTTP 429, a 5xx reply or a transport error, it waits for `Retry-After` (at most 60 s) and retries once; if the retry fails too, it raises a typed `EmbeddingUnavailableError`; the analysis records `filing_retrieval_failed` and reports the filings gap (`evidence_missing`, SPEC §8.5). There is no fallback to another model: a query vector from another model is meaningless against this index, and the dimension guard cannot catch it because both are 1024 d. The cross-host parity fallback above stays off; v5 fast is served only on Foundry.
- **Ceiling (`ponytail`):** the pacing is per process. Several API or worker processes sharing one deployment can still exceed 20 RPM together; a shared limiter (for example in the cache store, S7.9) is the upgrade when more than one process embeds queries.
- **Health:** `/health` reports the embedding arm from configuration only, because a live probe would spend the quota. Ollama is removed from code, Compose and dependencies (ADR-0016 decision 7).

## Outcome 2026-11-09 (CM-6e confirmation) — the product arm confirms Gate C

- **Run:** product arm `foundry_cohere` (deployment `cohere-embed-v5-fast`, 1024 d) at git 1da05d4, built with `evaluation/build_embedding_arm.py` from the CM-4 index into `sec_filings_bench_alpha_evidence_0_1_0__foundry-cohere-cohere-embed-v5-fast-1024` (532 chunks: AAPL 125, MSFT 209, NVDA 198; every chunk `embedding_id` = the identifier), Serverless 1x, section_aware, top_k 10, anchored fixture v1, seed 13, 10,000 samples. Build 17:40:31–17:47:23Z, benchmark 17:47:27–18:01:30Z. Result: `evaluation/results/gate_c_foundry-cohere-embed-v5-fast_opensearch_section_aware_retrieval_anchored_benchmark_v1.json`. Index kept (owner, 2026-11-09).
- **Quality:** identical to the CM-5b v5 arm on every metric at K=5 and K=10, and item by item: 75/75 cases with the same 750 ranked chunk ids in the same order, max score difference 3.8e-4. vs Qwen3 (Gate B) at K=10: recall +0.067 [+0.007, +0.133], section_recall +0.000 [+0.000, +0.000], pass +0.067 [+0.000, +0.133] → qualifies under retrieval-benchmark §9a. This result is the R4 baseline from now on.
- **Quota behaviour:** a first build at 08:50Z (git 646979f, request pacing only) failed with HTTP 429 after two 96-chunk batches: the deployment's 20,000 tokens-per-minute limit (see the amendment above). With token pacing the build made 21 requests, all 200, no 429; estimated tokens 98,362 vs billed 76,571 (the ⌈characters ÷ 4⌉ estimate over-counts by about 28%, the safe side); lowest remaining-token header 4,170. The benchmark made 236 query requests, all 200, no 429; one transport `ConnectError` recovered on its single retry.
- **Latency (now measured):** per-request embedding latency p50 527 ms, p95 965 ms (HTTP, no 429). The runner's per-case warm latency (p50-paired mean 5,220 ms vs Qwen3 292 ms) is dominated by the client's own pacing: each case issues several queries back to back and waits 60/18 s between them (607 s of pacing over the run). It measures the 20 RPM quota, not the model.
- **Cost:** run 2 0.883 OCU-h (search 2 OCU 17:43–18:01Z, indexing 1 OCU 17:44–17:58Z) ≈ $0.21 plus ≈ 80k billed tokens ≈ $0.01; the failed first build 0.483 OCU-h ≈ $0.12. Total ≈ $0.34. Both pools 0 by 18:02Z.

## Verify before implementation

- Bedrock batch support for Cohere Embed v4; Foundry deployment of Cohere Embed v4 at 1024 dimensions.
- Current Bedrock rerank price per 1,000 queries.
- Added 2026-10-11: this account's Bedrock access to Titan Text Embeddings V2 and Cohere Embed v4, and an Azure deployment of `text-embedding-3-small`.
