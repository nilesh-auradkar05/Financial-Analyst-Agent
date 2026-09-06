# Production-Scale Design — Interview Companion

**Project:** Alpha Financial Analyst Agent (ticker → SEC / news / market evidence → grounded, verified investment memo)
**Status of this document:** design decisions for the production-scale target, written as interview answers. Each answer states the decision, the number that justifies it, what was rejected, and what exists at HEAD today versus what is target. Diagrams: `docs/System-design/production/*.excalidraw`, rendered in `docs/png/production/`.

The questions come from `production-scale-topic.txt`. Decisions were taken 2026-09-05 (scale target, latency shape, hybrid backend, LLM serving, queue, observability, cloud placement). Nothing here changes SPEC scope by itself: cloud deployment is still gated on ADR-0006, and every mechanism below is a candidate for S7–S10, not a claim about HEAD.

---

## 0. The system in sixty seconds

A request is a ticker. Three evidence branches run in parallel (SEC filing sections from a vector store, news, market quote). The results are frozen into an `EvidenceSnapshot` whose `snapshot_hash` is a sha256 over the canonical packets and source manifest. A citation registry `[1]..[n]` is built from that snapshot *before* the LLM is called; the model may only cite registry entries. The draft is verified (claims → registry, tolerance number match, cosine ≥ 0.45), revised at most twice, screened by an output policy, and returned with a status that is *derived* (`completed` ⇔ verification passed ∧ no evidence class missing), never asserted by a node.

Two facts drive every scaling answer below:

1. **`snapshot_hash` is one primitive with three uses:** replay key (offline eval with the network off), cache key (memo cache), and eval lineage key (every run record carries it). Anything that would break that identity is rejected.
2. **Every query is ticker-scoped.** The corpus is tens of millions of chunks; the search space of any single query is one company's few thousand chunks. Most "millions of vectors" problems collapse once the vector store is told this (tenant index + shard key).

---

## 1. Document volume: thousands → millions. What changes architecturally?

**Target:** SEC-wide — ~8k filers × (10-K + 3×10-Q + ~8×8-K) × 5 years ≈ 40M chunks at 1000 chars / 200 overlap (range 20–50M depending on 8-K coverage).

**Decision — three things change, none of them is "a bigger Qdrant":**

| Concern | HEAD | Target |
| --- | --- | --- |
| How documents arrive | `POST /ingest` per ticker, inside the request; `replace_existing=False` short-circuits on any existing chunk, no freshness check | A data plane decoupled from requests: EDGAR poller (full-index daily, RSS every 10 min) → raw lake (S3, mirrored to ADLS) → **Databricks medallion**: `bronze.edgar_filings` (Auto Loader, Delta) → `silver.filing_sections` (**dbt**: edgartools parse, `section_key` normalisation, `fiscal_period`, `content_sha`, tests) → `gold.evidence_chunks` (dbt incremental, `chunk_id = f(accession, section, idx)`, embed only new `content_sha`, GPU job) → **Qdrant sync job** (upsert by `chunk_id`, `shard_key = ticker`, bump `collection_version[ticker]`) |
| Where vectors live | one Qdrant node, fp32, global HNSW | 4-node cluster, `shard_key = ticker`, replication 2, tenant HNSW (see Q6) |
| What the search sees | whole collection filtered | one tenant graph of ~3–5k points |

**Numbers.** Backfill: 40M chunks at ~300 chunks·s⁻¹ per GPU (qwen3-embedding via TEI) ≈ 37 GPU-hours → an overnight job on 4 GPUs. Steady state: ~400 filings/day ≈ 100k chunks/day, which is trivial. Cross-cloud egress from Azure Databricks to Qdrant on AWS: 40M × ~5 KB ≈ 200 GB once (~$17) and ~0.5 GB/day after — the multi-cloud hop is on the ingest path, never the request path.

**Why dbt.** The silver→gold step is relational (dedupe on `(accession_number, section_key)`, `not_null` on `cik/ticker/filing_date/content_sha`, `accepted_values` on `section_key`, source freshness `warn_after 2d / error_after 5d`). Those are dbt tests, and they fail the pipeline instead of poisoning the index. The parse step (edgartools) stays Python; dbt owns the contracts.

**Why Delta tables and not Kafka (yet).** The existing decision (system-design diagram, ADR logic) says Kafka enters when a *second consumer type* exists on the same event stream. At SEC scale there are four readers of the gold table — Qdrant sync, eval fixture builder, evidence-release capture, BM25 idf statistics — and Delta already serves them all as a multi-consumer, replayable log with time travel. Event Hubs / MSK stays **gated** on one measurable trigger: an 8-K freshness SLO below 5 minutes. Batch/micro-batch is honest for filings; nobody needs a 10-K in 200 ms.

**Rejected:** ingesting inside the request path at any scale (couples ingest latency to memo latency, no dedupe, no lineage); Kafka from day one (brokers, partitions, rebalance bugs, for one consumer type).

---

## 2. Query volume: one user → thousands of concurrent queries

**Decision:** everything left of the LLM scales horizontally by adding processes; the LLM is the deliberate, budgeted bottleneck.

- **Edge:** ALB + WAF, TLS, IP limit, sticky sessions for SSE.
- **API:** FastAPI × N on EKS (HPA on CPU/connections), stateless. Handlers validate, authenticate (OIDC), enforce `Idempotency-Key` → same `job_id`, `XADD` to Redis Streams, reply 202 + `job_id`, then `SUBSCRIBE job:{id}` and relay tokens as SSE. A replica can die; the job row and stream survive.
- **Queue:** Redis Streams `JobQueue` (consumer groups, PEL, DLQ). Queue depth is the backpressure signal and is surfaced as `429 Retry-After`, never hidden in a buffer.
- **Workers:** EKS pool, HPA on queue depth. `concurrency = provider TPM budget ÷ tokens per memo`. Adding workers beyond that only lengthens the queue.
- **LLM:** Bedrock provisioned throughput as the budget; per-tenant token buckets in Redis.

**Numbers.** A verified memo ≈ 8k input + 2k output tokens for the draft, plus a smaller verification pass. 1000 memos/min ≈ 10M TPM on the draft model — that is provisioned-throughput territory, not on-demand. Hence the router and the second cloud (Q4).

**Rejected:** self-hosted vLLM as primary (GPU fleet ops before the product has proven demand; kept as the tertiary cost floor); synchronous everything (a 30–90 s LLM path behind a load balancer timeout).

---

## 3. Latency target

**Decision: two modes on one path.**

- **Streaming mode:** p95 ≤ 3 s to first token. Budget: edge + auth 10 ms → `XADD` + claim 15 ms → snapshot frozen at ~60 ms when evidence caches hit → retrieval ≤ 300 ms p95 on a cache miss → LLM TTFT ≈ 1.5 s → first token ≈ 1.8 s p50, ≤ 3 s p95.
- **Verified mode:** the same job continues after the stream closes: verify (Haiku-class model) → revise at most twice → output policy → final status ≈ 25 s. The client receives a final `verification` SSE event that can *downgrade* the visible status to `degraded`; it can never upgrade it.

**Retrieval budget (300 ms p95, one tenant):** cache lookup 5 ms → query embedding 15 ms (TEI, embedding cache) → Qdrant Query API 40 ms (dense + sparse prefetch, RRF, tenant filter in-graph) → reranker 120 ms (TEI, batch 50 → top 8) → priors 5 ms → 185 ms, leaving 115 ms slack.

**Why not a hard 3 s for the whole memo.** Verification is what makes the memo trustworthy; collapsing it into the TTFT budget means either skipping it or serving only cached answers. Streaming the draft while verifying in the background is the shape that keeps both.

---

## 4. Cost: development cost vs. API and infrastructure spend

**Decision:** tokens dominate; measure cost per memo as a Prometheus counter and pull five levers in order.

1. **MemoCache on `snapshot_hash + prompt_version + model`** — bursts on hot tickers (earnings day) are served from cache. Expected 30–60 % hit on hot tickers, ~0 % on the long tail, which is fine.
2. **Model tiering** — draft on a Sonnet-class model, verification and claim extraction on a Haiku-class model (~5× cheaper).
3. **Matryoshka truncation 2560 → 1024 dims** — 4× less vector storage and RAM; adopted only if recall@5 and NDCG@10 stay within 2σ on the paired benchmark.
4. **int8 scalar quantization** in RAM, fp16 originals on disk — 4× less RAM, rescore on the top candidates.
5. **Spot GPUs** for the embedding and reranker services (stateless, batch-tolerant).

**Numbers.** ≈ $0.10 per verified memo without cache (Sonnet draft + Haiku verify); ≈ $0.05 effective after 40 % memo-cache hits. Vector RAM: 4 × r6i.2xlarge ≈ $1.2k/month. GPU fleet for embed + rerank ≈ $3/h on Spot. Egress ≈ $17 once.

**Rejected:** semantic caching of memo outputs across snapshots (Q8 — unsafe for time-sensitive financial content, and it would break the `snapshot_hash` identity).

---

## 5. Metadata architecture and pre-retrieval filtering

**Decision:** metadata is the primary index, vectors are secondary. Every query carries `ticker`; most carry `section_key` and a `filing_date` range.

**Payload (per point):** `chunk_id` (deterministic: `accession:section_key:index`), `ticker` (tenant + shard key), `cik`, `filing_type`, `fiscal_year`, `fiscal_period`, `section_key`, `filing_date`, `accession_number`, `chunk_index`, `content_sha`, `collection_version`, `text`.

**Indexes:** keyword on `ticker` (**`is_tenant: true`**), `filing_type`, `section_key`, `accession_number`, `content_sha`; integer on `fiscal_year`; datetime on `filing_date`.

**Mechanism.** Qdrant applies filters *during* HNSW traversal (filterable HNSW), not as a post-filter on an oversized top-k. With `ticker` as a tenant field and `hnsw_config: {m: 0, payload_m: 16}`, the global graph is disabled and one small graph is built per ticker. A query for AAPL walks ~4k points whether the collection holds 400k or 40M. This is the single most important answer in the set: the scaling problem is solved by the access pattern, not by hardware.

**Sharding.** `sharding_method: custom`, `shard_key = ticker`, 12 shards, replication factor 2. A query names its shard key, so it touches one shard (and its replica). Year-based sharding was rejected: queries span years within one company far more often than they span companies within one year.

**Where it exists today.** Payload indexes on `ticker, filing_type, section_key, filing_date, accession_number` are already created in `app/components/retrieval/qdrant_store.py`. Tenant index, custom sharding, `fiscal_year`, `content_sha`, `collection_version` are target.

---

## 6. Vector database architecture and HNSW RAM

**RAM budget for 40M chunks:**

| Layer | Size | Where |
| --- | --- | --- |
| raw fp32 × 2560 dims | 400 GB | never |
| Matryoshka 1024 × fp16 | 80 GB | disk (gp3), `on_disk: true` |
| int8 quantized copy | 40 GB | RAM, `always_ram: true` |
| HNSW links ≈ 150 B/vector | 6 GB | RAM |
| payload + keyword indexes | 8 GB | RAM |
| × replication factor 2 | **108 GB** | across the cluster |

→ 4 × r6i.2xlarge (64 GB) at ~27 GB each, 58 % headroom. Rescore the top-N with the fp16 originals to recover quantization loss.

**Collection config (the actual call):**

```json
PUT /collections/evidence
{
  "vectors": {
    "dense": {"size": 1024, "distance": "Cosine", "on_disk": true,
              "hnsw_config": {"m": 0, "payload_m": 16, "ef_construct": 128}}},
  "sparse_vectors": {"bm25": {"modifier": "idf", "index": {"on_disk": true}}},
  "quantization_config": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": true}},
  "shard_number": 12, "replication_factor": 2, "sharding_method": "custom"
}
```

**Next rung (100M+):** binary quantization (32× on the RAM copy) with rescore, hot/cold shard placement by query volume, `ef` tuned down (64 → 48) only after the benchmark says recall holds. Not built now; named so the interviewer sees the path.

**Rejected:** keeping 2560 dims fp32 (400 GB RAM is a $/month number nobody signs); disabling quantization "for accuracy" without measuring it.

---

## 7. Hybrid retrieval and re-ranking

**HEAD (honest):** `hybrid_retrieve.py` `scroll()`s up to 512 candidates out of Qdrant and builds a LangChain `BM25Retriever` per query; the cross-encoder (`Qwen3-Reranker-0.6B`) is loaded inside the worker process. That is correct for a benchmark and dies at millions of chunks: the scroll is O(candidate pool) per query and the reranker holds GPU memory in every worker.

**Decision:**

1. **Sparse moves into Qdrant** as a named sparse vector (`bm25`, `modifier: idf`; fastembed BM25 or SPLADE at ingest, computed in the gold step).
2. **One Query API call** does dense + sparse `prefetch` (40 each) and server-side **RRF** fusion to 50, with the tenant filter applied inside the graph:

```json
POST /collections/evidence/points/query
{ "prefetch": [
    {"query": [..1024 f32], "using": "dense", "limit": 40},
    {"query": {"indices": [..], "values": [..]}, "using": "bm25", "limit": 40}],
  "query": {"fusion": "rrf"}, "limit": 50,
  "filter": {"must": [{"key": "ticker", "match": {"value": "AAPL"}},
                      {"key": "section_key", "match": {"any": ["risk_factors", "mdna"]}},
                      {"key": "filing_date", "range": {"gte": "2024-01-01"}}]},
  "shard_key": "AAPL" }
```

3. **Reranker becomes a service** (TEI on a g5.xlarge, batch 32, HTTP, its own HPA). It is circuit-broken with **fail-open to the RRF order** — a slower reranker must never fail a memo.
4. The existing section prior and keyword prior boosts stay (they live in `reranked_hybrid_retrieve.py` and are measured).

**Governance.** Adoption goes through the existing paired benchmark on the shared fixture with the single-axis rule: backend *or* method changes in one comparison, never both. Decision rules from the README hold: keep the section-aware baseline if hybrid reduces recall or section coverage; never adopt on precision@5 alone.

**Rejected:** OpenSearch/Elasticsearch sidecar for BM25 (a second store with its own filter semantics to keep in parity); dropping BM25 (exact identifiers — accession numbers, item labels, product names — are where dense retrieval is weakest in filings).

---

## 8. Multi-layer caching: what to cache in a research workload

**Framing.** This is research, not FAQ. Two users rarely ask the same free-text question, so a semantic *answer* cache would have a low hit rate and — worse — would serve a stale memo when the evidence has changed. The parts that *do* repeat are the evidence and the section-templated retrieval queries.

| Cache | Key | TTL | Expected hit |
| --- | --- | --- | --- |
| RetrievalCache | `retr:{ticker}:{query_sha}:{filters_sha}:{collection_version}` | until new accession | 70–90 % (queries are section-templated) |
| EvidenceCache — SEC | `evid:sec:{ticker}:{accession}` | until new accession | ~100 % |
| EvidenceCache — news | `evid:news:{ticker}:{bucket_15m}` | 15 min | 50–70 % |
| EvidenceCache — market | `evid:market:{ticker}:{bucket_15m}` (raw quote 60 s) | 15 min | 50–70 % |
| MemoCache | `memo:{snapshot_hash}:{prompt_version}:{model}` | snapshot validity | 30–60 % hot tickers, ~0 % tail |
| ChunkEmbeddingCache | `emb:chunk:{content_sha}` | ∞ (ingest side) | ~95 % on amendments / re-ingest |
| QueryEmbeddingCache | `emb:q:{sha(text)}` + semantic reuse at cos ≥ 0.98 | 24 h | 40–60 % |

**Prohibited:** memo lookup by query similarity across snapshots. Financial content is time-sensitive; a memo is only reusable when the *evidence* is identical, which is exactly what `snapshot_hash` encodes.

**Why the 15-minute market bucket.** Without it the snapshot hash changes every 60 s and the MemoCache can never hit during an earnings-day burst — the one moment it matters. The raw 60 s quote still appears in the memo header; the bucket only governs cache identity. This is a deliberate, documented approximation.

**Where.** All caches in the same Redis cluster as the job stream and token buckets. Every cache exposes hit/miss counters in `/metrics`; alerts fire when a rate leaves its expected band for 15 minutes (production checklist item 3).

**HEAD:** no caching at any layer. S7 adds evidence / query-embedding / memo caches keyed on `snapshot_hash` (already in SPEC §11.3); RetrievalCache and ChunkEmbeddingCache are additions from this design.

---

## 9. Monitoring, quality evaluation, cost observability

**Three planes, one join key (`snapshot_hash`).**

- **Metrics (Prometheus + Grafana):** `alpha_ttft_seconds` p95, end-to-end p95, `alpha_retrieval_seconds` p95, queue depth, per-cache hit ratio, `alpha_provider_breaker_state`, tokens and `alpha_memo_cost_usd` per memo, GPU utilisation.
- **Traces (Langfuse, self-hosted on the same RDS):** every span tagged `snapshot_hash`, `git_sha`, `prompt_version`, `model`, `tenant`. Chosen over LangSmith to avoid a SaaS dependency for a system whose evidence is the product.
- **Quality (eval registry + CI `eval-replay`):** every run writes `evaluation/registry/runs/<run_id>.json` with lineage (`git_sha`, `model`, `temperature`, `prompt_version`, `evidence_release`, `snapshot_hash`, `fixture_version`, metrics, classification). CI replays the frozen evidence release with the network disabled and fails a PR when any of `recall@5`, `NDCG@10`, `section_recall`, `grounded_claim_rate`, `citation_coverage` drops more than 2σ below the approved baseline, or when any no-answer case returns `completed`. Heuristic verifier agreement with a judge model is measured (Cohen's κ ≥ 0.85 before the heuristic rate is quoted alone).

**Golden dataset — four classes, all replayed offline:**

| Class | Example | Expected |
| --- | --- | --- |
| easy · single-section | "AAPL FY25 risk factors about supply chain" | recall@5 = 1, first relevant rank 1, one accession |
| multi-hop · across filings | "how did MSFT's AI capex language change 2024 → 2025" | packets from two accessions, YoY delta cited from both |
| ambiguous · entity resolution | "FB 10-K 2021" (renamed META); GOOG vs GOOGL | CIK-based resolution, union filter or clarification — never the wrong tenant |
| no-answer · must refuse | never-ingested ticker; "Item 1A" for a filer that has none | `evidence_missing`, never `completed`, zero fabricated citations |

**Audit trail.** A production memo id resolves to a `snapshot_hash`, which resolves to the exact evidence and to the eval run that approved the `prompt_version` that produced it. Same record, two uses.

**HEAD:** Prometheus `/metrics` and LangSmith tracing exist; eval results are timestamped JSON without lineage and nothing runs in CI. Steps 1 and 4 of the README's implementation order close this.

---

## 10. Production checklist (with pass criteria)

1. **Throughput soak** — 100k chunks/day sustained for 72 h plus a 40M backfill rehearsal on one shard; p99 upsert latency flat; zero duplicate `chunk_id`s.
2. **HNSW + quantization tuning** — sweep `m ∈ {16, 32}`, `ef ∈ {48, 64, 128}`, int8 vs binary on the paired benchmark; adopt only within 2σ of recall@5.
3. **Cache validation** — replay one week of traffic; retrieval 70–90 %, memo 30–60 % on hot tickers; alert when a rate leaves its band for 15 min.
4. **Circuit breakers** — chaos test: kill the Bedrock endpoint, assert Azure OpenAI is serving within 30 s, zero 500s, `degraded` status visible in the SSE stream.
5. **Cost observability** — $ per memo panel, per-tenant budget alerts, GPU utilisation > 60 % or the fleet scales down.

Each gate requires the previous gate's commit evidence — the same discipline the README applies to the eight-step order.

---

## 11. Resilience: provider router and status semantics

**Breaker per provider, state in Redis:** `closed` → (5 failures / 30 s, or p95 > 8 s) → `open` → (30 s) → `half-open` (one probe) → `closed` on success, back to `open` on failure.

**Fallback order:** Bedrock (Claude Sonnet, provisioned throughput) → Azure OpenAI (GPT-4.1, same `prompt_version`) → vLLM warm pool on EKS GPUs (open-weights, cost floor) → MemoCache stale-flagged (`status: degraded`, `evidence_as_of` shown). Never a 500 while a snapshot exists.

**Status machine:** `queued → running → snapshot → drafting (streaming) → verifying → completed | degraded | evidence_missing | failed`. `completed` requires `verification.passed` and no missing evidence class. A cached memo re-emits its stored verification event, so the client contract is identical on hit and miss.

---

## 12. Tech-stack decisions (one line each)

| Choice | Decision | Trigger to revisit |
| --- | --- | --- |
| Vector store | Qdrant cluster (ratified ADR-0003), tenant index + custom sharding | recall regression vs Chroma fallback on the parity suite |
| Sparse retrieval | Qdrant native BM25 sparse vectors, server-side RRF | benchmark shows SPLADE > BM25 by > 2σ |
| Reranker | `Qwen3-Reranker-0.6B` on TEI as a service, fail-open | p95 > 150 ms at batch 32 → larger GPU or smaller model |
| Embeddings | `qwen3-embedding` on TEI, Matryoshka 1024 | benchmark shows > 2σ loss vs 2560 |
| LLM | Bedrock primary → Azure OpenAI → vLLM → stale cache | on-demand TPM ceiling hit → provisioned throughput; sustained > 60 % GPU utilisation → vLLM promoted |
| Jobs | Redis Streams `JobQueue` | second consumer type on the job stream → Kafka |
| Ingestion bus | Delta tables on Databricks (multi-consumer, replayable) | 8-K freshness SLO < 5 min → Event Hubs / MSK |
| Transformations | dbt (silver → gold, tests, freshness) | — |
| Job store | Postgres (RDS Multi-AZ) `RunStore`, read replica for polling | — |
| Cache / rate limit | same Redis cluster | memory pressure → separate cache cluster |
| Metrics / traces | Prometheus + Grafana; Langfuse self-hosted | — |
| Cloud placement | serving on AWS (one region); data plane on Azure Databricks; LLM across both | egress or on-call cost exceeds the outage-independence benefit |

**Why multi-cloud here and not everywhere.** Independence pays where an outage would otherwise be total (the LLM provider) and where the team already is (the data platform). It does not pay on the hot request path, which stays in one region with one runbook. The egress math (~$17 once, ~0.5 GB/day) is what makes the data-plane split defensible.

---

## 13. Productionization path (what ships in what order)

The README's eight-step order stands; this design fills in steps 6–8 and adds the scale rungs.

| Step | Sprint | Work | Exit evidence |
| --- | --- | --- | --- |
| 1 | S2-T00a | governance docs + CI governance job + hooks | deliberate break fails CI |
| 2 | S2-T00c | fan-out of evidence nodes, `to_thread` for FinBERT/store, graph singleton, `errors` reducer | 4 concurrent requests < 1.5× single |
| 3 | S2-T00b/d | `EvidenceSnapshot` freeze/replay, `degraded` / `evidence_missing` statuses | replay green under `unshare -n` |
| 4 | S6 | eval registry with lineage, `eval-replay` CI gate, verifier↔judge κ | a regressing PR fails CI |
| 5 | S6 | bounded draft → verify → revise loop (max 2) | paired comparison vs no-loop |
| 6 | S7 | guardrails (input, untrusted content, output policy), OIDC auth, rate limit | adversarial fixture in CI |
| 7 | S7 | EvidenceCache / QueryEmbeddingCache / MemoCache on `snapshot_hash`; **RetrievalCache on `collection_version`**; **15-min market bucket** | hit rates in `/metrics` within bands |
| 8 | S7 → S10 | Redis Streams queue + worker pool, Postgres `RunStore`, FinBERT out of process, provider breaker + **Azure OpenAI second provider**, **SSE streaming mode** | `JobQueue` / `RunStore` / `LLMProvider` swapped without app changes |
| 9 | S10 (ADR-0006) | Qdrant cluster: tenant index, custom sharding, int8 quantization, Matryoshka 1024; **Qdrant-native sparse + server-side RRF**; **reranker as TEI service** | paired benchmark within 2σ; retrieval p95 ≤ 300 ms |
| 10 | S10 (ADR-0006) | Databricks medallion + dbt ingestion, Qdrant sync job, Langfuse, production checklist gates 1–5 | soak, chaos and cost gates green |

Steps 9–10 require ADR-0006 to move from Proposed to Accepted (provider, budget ceiling, teardown path, secrets service, deployment gate tied to frozen-evidence evaluation).

---

## 14. Thirty-second answers

- **"How do you go from thousands to millions of documents?"** Move ingestion out of the request into a Databricks medallion pipeline with dbt contracts, make the vector store tenant-aware so a query walks one company's graph, and shard by ticker. The corpus grows; the per-query search space does not.
- **"Thousands of concurrent users?"** Stateless API and workers scale by replicas; the LLM is the budgeted bottleneck behind a provider router with per-tenant token buckets; queue depth becomes `429 Retry-After`.
- **"Sub-3-second latency for a 60-second task?"** Stream the draft at ≤ 3 s TTFT, verify after the stream closes, and let the final SSE event downgrade the status. Retrieval has a 300 ms budget with 115 ms slack.
- **"What about RAM for HNSW?"** 40M × 1024 dims int8 = 40 GB in RAM plus 6 GB of links, fp16 originals on disk, ×2 replication = 108 GB across four 64 GB nodes. Next rung is binary quantization.
- **"Hybrid retrieval?"** Dense + BM25 as named vectors in Qdrant, one Query API call with RRF, cross-encoder as a fail-open GPU service, adopted only through the paired benchmark.
- **"Caching?"** Cache evidence, retrieval results and memos keyed on content identity (`collection_version`, `snapshot_hash`), never on query similarity. Research workloads repeat evidence, not questions.
- **"How do you know it still works?"** Every run carries lineage; CI replays a frozen evidence release offline and fails on a 2σ regression or any no-answer case that returns `completed`.
- **"Kafka?"** Not until a second consumer type exists on the job stream, or the 8-K freshness SLO drops below five minutes. Delta tables already give the ingestion side a replayable multi-consumer log.
- **"Why multi-cloud?"** Only where independence pays: LLM failover (Bedrock ↔ Azure OpenAI) and the data plane where the data team lives. The request path stays in one region.

---

## 15. What is honest about this design

- Every mechanism above is a target; HEAD is an instrumented prototype with the gaps listed in the README (silent evidence omission, unenforced verification, serial nodes, no guardrails, no caching, in-process jobs, in-process BM25 via `scroll()`).
- The hit-rate bands, the 300 ms retrieval budget, the 37 GPU-hour backfill and the $0.10 → $0.05 cost are estimates from model and hardware specs. The production checklist exists to replace them with measurements before anyone quotes them as facts.
- The 15-minute market bucket trades quote freshness inside the cache key for cache hits during bursts. It is documented as an approximation, and the raw quote remains in the memo.
- Matryoshka truncation, int8 quantization, Qdrant-native BM25 and the TEI reranker each carry a recall risk. None is adopted without the paired benchmark and the single-axis rule.

---

## 16. Diagram index

| File | What it argues |
| --- | --- |
| `docs/System-design/production/hld.excalidraw` (`docs/png/production/hld.png`) | Five planes — serving (AWS), retrieval, LLM, data (Azure Databricks + S3), observability — with the Query API call, dbt tests and live metrics as evidence artifacts |
| `docs/System-design/production/lld.excalidraw` (`lld.png`) | Collection schema, payload, RAM budget, 300 ms retrieval timeline, cache-key table, provider breaker state machine, status machine with streaming, typed seams |
| `docs/System-design/production/critical-flow.excalidraw` (`critical-flow.png`) | One streaming `/analyze` across six lanes with the latency budget, cache short-circuits, evidence-missing and breaker-open paths, SSE event stream |
| `docs/System-design/production/system-design.excalidraw` (`system-design.png`) | The scaling ladder (thousands → millions → 100M+) per axis with triggers, where the system saturates, the multi-cloud decision, the production checklist, the golden dataset and the CI gate |

The earlier `docs/System-design/*.excalidraw` files remain the S7 single-region target; the `production/` set is the rung above it.
