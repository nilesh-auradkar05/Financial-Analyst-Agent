# ADR-0006 — Production Cloud Deployment

**Status:** Accepted (2026-10-08) by owner decision through the multi-cloud blueprint review. **Superseded in part on 2026-10-11 by ADR-0015** for the deployed demo tier: edge wiring (D03), runtime (D04), the secrets store and the budget projection (D16) now follow ADR-0015, which runs every component pay-per-use for recruiter-only traffic. The D03/D04 designs below are retained as the production scale tier (`docs/production-scale-architecture.md`). Posture, region, evidence storage, deploy gate, observability, the region-loss analysis and the entry criteria (as amended below) still govern. Implementation remains gated by the S10 entry criteria below.
**Date:** 2026-07-04 (proposed) / 2026-10-08 (accepted)
**Related:** SPEC §§2, 3, 11.1 (Gate E), 12 (S10); ADR-0003, ADR-0005, ADR-0007, ADR-0009, ADR-0010, ADR-0011, ADR-0012, ADR-0013, ADR-0014 (vector store, amended 2026-10-11), ADR-0015 (serverless demo tier, 2026-10-11); `docs/cloud-deployment.md`; `docs/production-scale-architecture.md`; `old_artifacts/docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md` Phases 4–5; production review 2026-09-06 finding 15.

## Context

The proposed version of this ADR required the owner to choose a cloud provider and runtime, a monthly budget ceiling, a teardown path, a secret-management service and a deployment gate tied to frozen-evidence evaluation. On 2026-10-07/08 the owner reviewed seventeen design decisions (D01–D17) in the interactive blueprint, changed four of them and accepted the rest. This record captures the deployment-level outcome. Decisions with their own contract surface are recorded in ADR-0010 (identity), ADR-0011 (embedding and rerank models), ADR-0012 (ingestion orchestration), ADR-0013 (durable execution and caching), and amendments to ADR-0003 (vector store) and ADR-0009 (model providers).

The project is a portfolio system operated by one person. Its design must show the failure analysis of a production system while building only what one owner can run, pay for and rehearse.

## Decision

### Posture (D01, D02)

- **AWS is the only serving cloud.** One active region, `us-east-1`, already configured in the repository. All state and compute live there.
- **Multi-region is limited to model routing.** Bedrock is called through US cross-region inference profiles, which spread model requests across US regions without any state outside `us-east-1`.
- **Azure is the model plane only.** Microsoft Foundry serves the same LLM and embedding models as fallbacks (ADR-0009 amendment, ADR-0011). No Azure service holds jobs, evidence, indexes or users. Amended 2026-10-17 by ADR-0009: Foundry is also the primary draft host (DeepSeek-V4-Flash, Bedrock V3.2 as fallback) and the judge fallback; it still holds no state.
- **Loss of all of `us-east-1` is a documented failure mode, not built infrastructure.** See *Region-loss failure mode* below.

Why: every cross-cloud call is a stateless model request, so nothing has to stay consistent across providers. Model capacity is where throttling and outages occur most often, so it is the one plane with a live fallback. A recovery site, warm standby or cross-region data copy only pays off with a stated availability SLO and rehearsed restores, which a portfolio deployment does not have. `us-east-1` keeps the existing configuration and has the broadest Bedrock catalog (DeepSeek V3.2 and gpt-oss-120b included).

### Edge (D03)

Cloudflare provides DNS, WAF, Turnstile and edge rate rules on the free plan. *Demo tier (ADR-0015):* a Cloudflare Worker signs requests (SigV4) to an IAM-protected Lambda function URL; there is no Tunnel. *Scale tier:* an ALB with authenticated origin pulls (the Tunnel was the original demo choice and needs an always-running connector). The event stream sends a heartbeat every 15 s because Cloudflare closes idle proxied connections after about 100 s. If the origin is unreachable, Cloudflare serves a static status page.

### Runtime (D04)

*Demo tier (ADR-0015, 2026-10-11):* three arm64 Lambda functions (web BFF, API, worker with ONNX FinBERT) with response streaming through the Lambda Web Adapter; reserved concurrency caps spend. The original reason for rejecting Lambda (25–60 s jobs and streamed events) no longer holds: response streaming and a 15-minute limit cover both, and nothing bills while idle.

*Scale tier (retained design):* ECS Fargate on Graviton (ARM64) runs four services: the Next.js web BFF, the FastAPI API, LangGraph workers and FinBERT (a process pool in the worker, its own service from the 100K-user tier). Asynchronous workers multiplex many I/O-bound jobs per task, which beats Lambda's per-job wall-clock billing under steady load. EKS is rejected because a cluster control plane and its operations are not justified (integration plan: no Kubernetes). Workers scale on a published "ready jobs per worker" metric, capped by the provider token budget.

### Evidence storage, backups and teardown (D14)

- S3 holds EvidenceSnapshots, manifests, chunk and embedding Parquet and the eval registry under the existing content-addressed `{type}:{sha256}` names. The bucket is versioned; approved releases are protected by Object Lock. `releases.jsonl` appends with S3 conditional writes until a database is justified (ADR-0005).
- DynamoDB point-in-time recovery on `jobs` and `job_events` in the demo tier (ADR-0015); Aurora point-in-time recovery (about 5 min RPO within the region) in the scale tier.
- **No cross-region copies** (owner decision, 2026-10-08): no S3 Cross-Region Replication, no cross-region snapshot copies, no image replication.

### Secrets, infrastructure as code, deploy gate and budget (D16)

- No stored cloud credentials. GitHub Actions reaches AWS and Azure through OIDC; Lambda and ECS tasks use execution or task roles; workers reach Foundry through AWS STS outbound identity federation trusted by an Entra federated credential (spike required). Third-party keys (Tavily, market data, LangSmith) live in SSM Parameter Store SecureString in the demo tier and in Secrets Manager with rotation in the scale tier (ADR-0015). One scoped exception: the Cloudflare Worker's invoke-only signing key, rotated every 30 days through OIDC.
- Terraform with the `aws` (plus `awscc` where `aws` lags), `azurerm` and `cloudflare` providers. One `apply` from a clean checkout creates the environment; no console-created resources.
- **Deploy gate:** deploy on tag. ARM images are pushed to ECR; the frozen-evidence `eval-replay` job (SPEC §11.2) must pass before the release; then new Lambda versions are smoke-tested on a `candidate` alias and the `live` alias is moved (demo tier), or an ECS blue/green deployment shifts traffic (scale tier). A release that regresses `grounded_claim_rate` or `citation_coverage` beyond the accepted tolerance against the active approved baseline cannot deploy.
- **Budget ceiling:** $100/month kill switch for the demo tier, with early-warning alerts at $10 and $25 (ADR-0015). Projected spend is about $4/month at 20 recruiter sessions; the worst case allowed by the guest quotas (20 live jobs/day) and the 45-minute daily search awake cap (ADR-0014) is about $30/month (see `docs/cloud-deployment.md`). An AWS Budgets action at 100% of the ceiling sets an SSM parameter that the API reads at admission: new jobs receive a typed 503, running jobs finish. Guest quotas are the first control. (The 2026-10-08 projection of about $95 at 25 memos/day applied to the always-on design, now the scale tier.)
- **Teardown:** `terraform destroy` removes every resource except the evidence bucket, which carries `prevent_destroy` and Object-Locked approved releases. The OpenSearch Serverless collection group and the Foundry deployments are destroyed with the rest of the stack.

### Observability (D17)

OpenTelemetry in every service, exported through an ADOT collector to Grafana Cloud (metrics, logs, traces). LangSmith remains the LLM trace store. Per-run cost comes from provider usage fields. A tracing outage never blocks memo delivery (bounded buffers with drop counters). Monitoring lives outside the cloud it watches.

### Region-loss failure mode (documented, not built)

If `us-east-1` is lost:

| Component | Effect | Data at risk |
| --- | --- | --- |
| Web, API, workers (Lambda) | Unavailable; Cloudflare shows the status page | none (stateless) |
| Cognito user pool | Nobody can sign in | User accounts, if the region's storage is permanently lost: the admin user is recreated (demo tier; guests have no accounts) or users re-register or are migrated and `customers.cognito_sub` mappings are rebuilt (scale tier) |
| DynamoDB (demo tier) / Aurora (scale tier) | Unavailable | Jobs, results and run records if the region's storage is permanently lost |
| S3 evidence bucket | Unavailable | Frozen evidence releases. SEC filings can be refetched; past news and quotes cannot, so approved baselines would no longer be reproducible |
| OpenSearch Serverless collection (ADR-0014) | Unavailable | Rebuildable from Parquet vectors in S3, which are in the same region |
| Bedrock, Foundry | Bedrock calls originate in the region, so they stop; Foundry stays up but nothing calls it | none |

Mitigations, ordered by cost, recorded so they can be adopted if an SLO ever requires them: S3 Cross-Region Replication of the evidence bucket (about $1–3/month at current size, removes the irreproducible-baseline risk); AWS Backup cross-region copies of DynamoDB (or, in the scale tier, Aurora) backups (RPO up to 24 h); ECR replication; DynamoDB global tables or, in the scale tier, Aurora Global Database (RPO about 1 s, a second always-on cluster); a warm standby stack applied from the same Terraform with `region` as a variable. With the serverless demo tier a standby costs almost nothing while idle, which makes it the cheapest it will ever be; it stays unbuilt by the D14 decision. Recovery would then be: apply the stack in the second region, restore the database, create an OpenSearch Serverless collection there and bulk-load vectors from Parquet (no re-embedding), point Cloudflare at the new origin. An Azure-side index on Azure AI Search is also listed, to be reconsidered only when its serverless tier is generally available with alias support (owner decision 2026-10-11, ADR-0014); it would need its own retrieval parity gate because it is a different search engine.

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| AWS active region + `us-west-2` data copies + Azure model plane (blueprint v2) | Rejected by owner | Copies cost little, but the owner prefers to explain the failure mode and its mitigation rather than run them |
| AWS primary + Azure pilot-light recovery site (blueprint v1) | Rejected | Only worth having with rehearsed restores |
| Active-active AWS + Azure | Rejected | Stateful replication of jobs, idempotency keys and index releases across providers; no SLO pays for it |
| Azure primary | Rejected | Re-platforms a working Bedrock integration |
| `us-east-2` | Rejected | Smaller blast radius, but regional failover is out of scope, so reconfiguring buys little |
| GCP Cloud Run (integration plan option) | Rejected | Bedrock and the existing AWS configuration are already in use |

## Consequences

- SPEC §2 and §3 move cloud deployment from out of scope to the gated S10 scope item (SPEC v1.4). Multi-region deployment and cross-region data copies remain out of scope.
- Infrastructure is reproducible from a clean checkout; secrets stay out of the repository.
- Release artifacts remain the traceability spine; the cloud changes only where they are stored and what triggers them.
- A region loss is an outage of unknown duration, and permanent loss of the region would make frozen evidence releases irrecoverable. Accepted.
- Cognito user pools are regional, so sign-in shares the region's failure domain (ADR-0010).

## S10 entry and exit criteria (implementation gate)

Each item needs recorded verification evidence.

**Before the first `terraform apply`:**

1. Gate A cleared on frozen evidence; S2-T00b and S2-T00d done.
2. S7 local seams built and tested: DynamoDB `RunStore` and SQS `JobQueue` against DynamoDB Local and ElasticMQ (ADR-0013 as amended by ADR-0015), caches, guardrails, guest and admin access (ADR-0010 amendment), FinBERT as ONNX with recorded label parity.
3. The `eval-replay` CI job exists and fails on regression (SPEC §11.2), because the deploy gate calls it.
4. DeepSeek V3.2 and gpt-oss-120b qualified on Bedrock against the frozen eval set (paid API calls, no infrastructure); the Gate C embedding bake-off done if the embedding model is to change (ADR-0011).

**Before the environment is left running:**

5. Foundry parity for both models and embedding parity, measured against the Foundry deployments the first apply creates. The fallbacks stay disabled until they pass (ADR-0009 amendment, ADR-0011).
6. Retrieval parity of OpenSearch Serverless against the Qdrant baseline on the frozen fixture (ADR-0014 adoption gate); if it fails, the deployment uses Qdrant Cloud.
7. Two rehearsals of `apply → smoke → destroy` from a clean checkout, each recording cold-start latency with and without the warm-up (ADR-0015).

## Verify before implementation

- OpenSearch Serverless next-generation pricing, idle timeout and Terraform support (ADR-0014).
- Entra accepts the issuer and subject of the AWS STS outbound identity token.
- Current prices for every line in the cost model (`docs/cloud-deployment.md`).
- ADR-0015 spikes: IAM-signed streaming through the function URL; Worker CPU time; stream-triggered dispatch to SQS.
