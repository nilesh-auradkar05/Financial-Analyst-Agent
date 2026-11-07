# ADR-0016 — Cloud data and model plane first; Qdrant and Chroma retired

**Status:** Accepted (2026-10-23) by owner decision. Amended 2026-11-07: decision 6 resolved by ADR-0019 (Claude Sonnet 5.5 on Foundry); decision 3 corrected — copying the vectors avoids re-embedding the corpus, but every benchmark query is still embedded with `qwen3-embedding:4b`, so CM-4 and CM-5b need host Ollama; Ollama is removed after them (CM-6e). Supersedes ADR-0003. Amends ADR-0006 (entry criteria scope), ADR-0011 (bake-off substrate and fallback rule) and ADR-0014 (only store; adoption gate).
**Date:** 2026-10-23
**Related:** SPEC §§1, 3, 5, 6, 11.1, 12, 16; ADR-0003; ADR-0006; ADR-0011; ADR-0014; ADR-0015; `docs/retrieval-benchmark.md` §§8, 9, 9a; test-plan §§4, 5, 20, 21; sprint-plan "Authorized cloud data and model plane migration (2026-10-23)".

## Context

Gate A closed on 2026-10-22 (c1fcd21): the anchored fixture is frozen and has a baseline on local Qdrant with `qwen3-embedding:4b` served by Ollama. The documented order then ran Gate B on Qdrant, the embedding bake-off on Qdrant, S6 and all S7 seams before any cloud resource, and moved the index to OpenSearch Serverless only in S10.

On 2026-10-23 the owner set a new priority: move the project onto managed cloud services first, then benchmark there, without building indexes that are thrown away. Two facts make the old order wasteful:

- Ollama cannot run in the deployed tier (Lambda, ADR-0015), so `qwen3-embedding:4b` is not a deployable option. Any bake-off whose fallback is "Qwen3 stays" cannot produce a deployable result.
- Every quality number measured on local Qdrant would have to be measured again on OpenSearch Serverless before the deployment could cite it.

## Decision

1. **One vector store.** OpenSearch is the only `RetrievalStore` backend: a local OpenSearch container for development and tests, Amazon OpenSearch Serverless (next generation, ADR-0014) in the cloud. Qdrant and Chroma are retired. Qdrant is kept only as the source of the Gate A vectors copied in step 3 and is removed with Chroma in CM-7 (adapters, Compose services, `qdrant-client`, `chromadb`). Volumes are not deleted without owner approval.
2. **Data and model plane ahead of S7.** The data plane (vector store, embeddings) and model plane (draft, judge, sentiment) move to managed services now. The app, worker and ingestion keep running locally or in Compose and call those services. The S10 demo-tier runtime (Lambda, DynamoDB, SQS, Cloudflare Worker, Cognito, MWAA) and the S7 seams keep their ADR-0006 entry criteria.
3. **Backend comparison without re-embedding (CM-4).** The `qwen3-embedding:4b` vectors, chunk IDs, text and metadata of the Gate A collection (`sec_filings_bench_alpha_evidence_0_1_0`) are copied unchanged into an OpenSearch Serverless index. The anchored benchmark then runs on it and is compared with the Gate A result; only the backend differs. No Ollama call is needed.
4. **Embedding bake-off on OpenSearch Serverless (CM-5).** Titan Text Embeddings V2 and Cohere Embed v4 (Bedrock) and `text-embedding-3-small` (Foundry), each at 1024 dimensions, are built from the same chunks as one index each and compared with the copied Qwen3 index; only the embedding differs. The bake-off tolerance and winner rule are in `docs/retrieval-benchmark.md` §9a. If no cloud arm qualifies, work stops and the owner decides; Qwen3 is not a fallback because it is not served in the cloud.
5. **Cost controls before the first resource (CM-1).** AWS Budgets alerts at $10 and $25 and the $100 action, resource tags, and a written teardown command exist before any cloud resource is created. Every `apply` that creates billable resources is approved by the owner first.
6. **Sentiment moves to a hosted model (CM-6b).** The choice between a hosted FinBERT endpoint and an LLM classification call is made after a spike and recorded in its own ADR, which amends ADR-0015 §1 (ONNX FinBERT in the worker). Label parity against local FinBERT is recorded before the switch.
7. **Ollama.** No new code depends on Ollama. Whether it leaves the default development path is decided in CM-6.

## Ordered tasks

| Task | Scope | Done when |
| --- | --- | --- |
| CM-0 | This ADR; SPEC §§1, 3, 5, 6, 11.1, 12, 16; ADR-0003/0006/0011/0014 status notes; test-plan §§4, 5, 20; retrieval-benchmark §§8, 9, 9a; sprint-plan | Four governance checks pass |
| CM-1 | Budgets, tags, teardown command; official NEXTGEN prices and idle timeout recorded in ADR-0014 | Alarms visible; prices recorded |
| CM-2 | `OpenSearchVectorStore` behind `RetrievalStore`; contract suite and coverage/idempotency against a local OpenSearch container (S3-T04, S3-T05 retargeted); embedding identifier in ingest metadata | test-plan §§4, 5 pass; replay suite still offline |
| CM-3 | Terraform for the Serverless collection group (0–2 OCU) and a SigV4 data-access policy for one local principal; spike on filters and the hybrid pipeline | apply → smoke → destroy rehearsed once |
| CM-4 | Vector copy and backend-only comparison (decision 3) | Paired result recorded; new baseline, or stop and re-plan |
| CM-5 | Embedding provider switch and bake-off (decision 4); default changes only on a qualifying win | Paired results recorded; Gate C embedding outcome applied |
| CM-6 | Model-plane close-out: Azure draft live check, Bedrock Luna quota, G13 `/health` without Ollama | Checks recorded |
| CM-6b | Hosted sentiment (decision 6) | Label parity recorded; ADR accepted |
| CM-7 | Remove Qdrant and Chroma (decision 1); update CLAUDE.md/AGENTS.md §8 and `scripts/ci/check_doc_sync.py` together | Full suite and governance checks pass |

## Consequences

- Gate B is redefined (SPEC §11.1): OpenSearch passes the contract suite against a local container and the CM-4 comparison stays within tolerance.
- The ADR-0014 adoption gate is CM-4. Its fallback to Qdrant Cloud is withdrawn; a CM-4 loss beyond tolerance stops work for an owner decision.
- The ADR-0006 entry criteria still govern the S10 runtime, but data and model plane resources may exist before them. SPEC §3.y records the exception.
- Cloud spend starts at CM-3, months before S10. The CM-1 controls and per-apply approval bound it.
- The local development loop needs a running OpenSearch container instead of Qdrant.
- Invariants: chunk IDs are identical in every index (gold labels depend on them); backend and embedding never change in one comparison; tests and `eval-replay` stay offline; no secret enters logs, fixtures or `tasks/`.
