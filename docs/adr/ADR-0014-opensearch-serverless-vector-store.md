# ADR-0014 — Deployed vector store: Amazon OpenSearch Serverless with a daily awake cap

**Status:** Accepted (2026-10-11) by owner decision. Supersedes the ADR-0003 addendum (managed Qdrant Cloud) for the deployed (S10) configuration. Adoption as the deployed default is gated by the paired comparison below. **Amended 2026-10-11 (ADR-0015):** the owner kept OpenSearch Serverless for the scale-to-zero demo tier; the wake ledger moves to DynamoDB, the cap becomes an abuse guard rather than the main spending control, and a landing-page warm-up hides the resume time.
**Date:** 2026-10-11
**Related:** SPEC §§3, 6, 11.1 (Gates B, C), 12 (S10); ADR-0003 (Qdrant behind the retrieval interface, local default unchanged); ADR-0006; ADR-0011; ADR-0012; ADR-0013; `docs/cloud-deployment.md`.

## Context

ADR-0006 placed the deployed index on Qdrant Cloud's free tier in AWS `us-east-1`. The owner chose to replace it with an AWS-native service. Two facts from 2026 decided the shape of this record:

- **OpenSearch Serverless, next generation** (generally available 2026-05-28) scales compute to zero when idle. In a published hands-on test, compute dropped to zero after about 10 minutes without activity and resumed in about 10 seconds. The previous generation kept 2 OCUs running at all times, which alone would have exceeded the $100/month demo budget.
- **Azure AI Search serverless** exists only as a preview Developer tier (no SLA, not recommended for production, billing since 2026-09-13). It does not support index aliases and limits each index to 1 GB. The dedicated tiers bill every hour and never scale to zero.

While awake, a next-generation collection runs about 2 OCUs (standby replicas are mandatory). The only rate found so far is about $0.334 per OCU-hour, implied by a third-party test rather than taken from AWS's pricing page. That is about $0.67 per awake hour, and every wake-up bills at least the 10-minute idle tail, about $0.11. An uncapped demo could cost $20–85/month depending on how requests spread across the day, so the owner kept the $100 ceiling and asked for a daily cap on awake time.

## Decision

1. **Deployed store:** an OpenSearch Serverless collection group with `generation = NEXTGEN`, `VECTORSEARCH` collection type, `us-east-1`, minimum capacity 0 and maximum 2 OCUs for both indexing and search.
2. **Contract:** a new `OpenSearchVectorStore` implements the existing `RetrievalStore` protocol. Client objects stay inside the retrieval boundary; nothing above `app/components/retrieval/` changes. Local development, the benchmark substrate and Gate B remain on Qdrant (ADR-0003).
3. **Release publication:** one index per release (`sec_chunks__<release_id>`) in the same collection, promoted by an atomic `POST _aliases` swap of `sec_chunks_active`, with the previous index kept for rollback. Index aliases are supported by OpenSearch Serverless. New documents become searchable asynchronously (about 18 s in the hands-on test), so the validate step polls `_count` until it equals the manifest before swapping.
4. **Hybrid retrieval:** dense k-NN plus native BM25 through a search pipeline (hybrid search has been supported on OpenSearch Serverless since August 2025). This replaces the per-query in-process BM25 rebuild, subject to S5/Gate C.
5. **Vector engine settings are not tunable.** The next generation sets the k-NN engine itself (on-disk mode with 32× compression in the hands-on test). ADR-0011's float16 + int8 plan applies to the Parquet originals, not to this index. Recall must be measured, not assumed.
6. **Access:** IAM data access policies for the worker and ingestion task roles (SigV4; no stored search credential). For the demo, the collection uses a public endpoint with IAM authorization only; a VPC endpoint is evaluated at scale, priced first.
7. **Infrastructure as code:** Terraform. If the `aws` provider does not yet expose the next-generation collection group, the `awscc` provider (generated from the CloudFormation registry, which documents `AWS::OpenSearchServerless::CollectionGroup` with `Generation`) creates it.

### Daily awake cap

- **Budget:** search may be awake at most **45 minutes per UTC day**, about $15/month at 2 OCUs. The nightly ingestion window of the always-on design was about 20 minutes, about $7/month, for a projection of about $95/month against the $100 ceiling with rerank off. The scale-to-zero demo tier (amendment below) projects about $3/month for search.
- **Ledger:** DynamoDB table `wake_ledger` keyed by `day`, holding `(window_start, window_end)` items (ADR-0015; it was the Postgres table `search_wake_ledger` in the always-on design). Every retrieval call extends the current window to `now + 10 min` (the idle tail) or opens a new one. Today's awake minutes are the union of the windows.
- **Admission:** a new job that needs retrieval is accepted only if today's awake minutes, plus the extension it would cause, stay within the cap. Otherwise:
  - an eligible exact memo-cache hit is still served;
  - other requests get `429` with `Retry-After` set to the next UTC day and reason `search_awake_cap`.
  Jobs already admitted always finish.
- **Reconciliation:** a daily job compares the ledger with the CloudWatch search OCU metrics. An alarm fires if actual awake time exceeds the ledger by more than 20%, because the 10-minute idle timeout is observed behavior, not a documented contract.
- **Cold start:** the first query after idle waits about 10 s. This adds to verified-completion latency only; the event stream reports a "warming search" progress event.
- **Backstops:** the AWS Budgets kill switch (ADR-0006) still refuses all new jobs at 100% of $100. Adopting rerank (about $6/month) requires lowering the cap to about 35 minutes.

### Azure AI Search (owner decision: documented mitigation only)

Azure AI Search is **not built**. It is listed in ADR-0006 as a possible Azure-side index in the region-loss mitigations, to be reconsidered only when its serverless tier is generally available with alias support. Building it now would add cross-cloud state that ADR-0006 rules out. It would also not be a same-engine fallback: a different engine scores keyword and vector matches differently, so failover would change which filings are retrieved and would need its own parity gate.

## Adoption gate

OpenSearch Serverless becomes the deployed default only after a paired comparison on the frozen fixture: Qdrant (current baseline) against OpenSearch Serverless, same embeddings, `top_k` and filters, so the backend is the only changed variable, using the `docs/retrieval-benchmark.md` paired policy. If recall@k or section coverage loses beyond the predeclared tolerance, the deployment falls back to Qdrant Cloud (the ADR-0003 addendum design) and this record is revisited.

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| Qdrant Cloud free tier (ADR-0003 addendum) | Superseded; fallback if the adoption gate fails | $0 at demo scale, but a third-party vendor and API key; idle suspension after a week |
| OpenSearch Serverless, classic generation | Rejected | 2 OCUs always on, above the demo budget |
| OpenSearch managed domain (provisioned) | Rejected for now | Always-on instance cost; revisit when steady load exceeds about 4 OCUs around the clock |
| Azure AI Search serverless (preview) | Documented mitigation only | Preview, no SLA, no aliases, 1 GB per index; cross-cloud state |
| Azure AI Search Basic or Standard | Rejected | Billed every hour, never scales to zero |
| S3 Vectors | Alternative | Very cheap and serverless, but no native BM25 or hybrid; fits a cold archive tier |
| Aurora pgvector | Alternative | One fewer system; weaker hybrid support, and it would keep Aurora from scaling to zero |

## Consequences

- One fewer third-party vendor and secret. The search index shares the `us-east-1` failure domain (unchanged: Qdrant Cloud ran there too) and is rebuildable from the S3 Parquet vectors.
- At scale, OpenSearch costs more than Qdrant Cloud (projected about $1k/month at 100K users, $3.9k at 1M, $15.6k at 10M, against $300, $2.5k and $9k), in exchange for AWS-native operation. The trigger to revisit is in `docs/production-scale-architecture.md` (scale tier).
- Demo users can be refused for the rest of a day once the cap is reached. Accepted for a portfolio deployment.
- Tests (test-plan §20): the adapter passes the shared contract suite against a local OpenSearch container; the cap refuses a cold wake beyond 45 minutes but serves a memo-cache hit; promotion waits for the refreshed count; the ledger reconciles with the metrics.

## Verify before implementation

- The official next-generation OCU-hour and storage prices in `us-east-1`; the idle timeout and cold-start time on the pinned configuration.
- Terraform `aws` provider support for `NEXTGEN` collection groups (else `awscc`).
- Hybrid search pipeline (normalization or RRF) on a `VECTORSEARCH` collection of the next generation, and payload filtering on `cik`, `section_key` and `filing_date`.

## Amendment 2026-10-11 — role of the cap in the scale-to-zero demo tier (ADR-0015)

- At recruiter-only traffic a session wakes the collection once: about 12 minutes at 2 OCUs (the work plus the 10-minute idle tail), about $0.13. Search is still the largest per-session cost, but the projected monthly total is about $3, not $22.
- The 45-minute daily cap stays, now as an abuse guard: with the guest quotas (20 live jobs a day, ADR-0010 amendment) it bounds the worst case at about $15/month for search.
- **Warm-up:** after Turnstile, the landing page triggers one `_count` on `sec_chunks_active` so the ~10 s resume overlaps with the visitor typing. A warm-up is admitted only within the cap, extends the ledger like any retrieval call, and is limited to 3 per IP hash per day.
- Ingestion runs (monthly or on demand) are accounted for in their own ledger window and limited to two per month; they do not consume the query cap.
- The ledger and reconciliation logic are unchanged apart from the store.
