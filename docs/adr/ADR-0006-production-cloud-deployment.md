# ADR-0006 — Production Cloud Deployment

**Status:** Accepted (2026-10-08) by owner decision through the multi-cloud blueprint review. Implementation remains gated by the S10 entry criteria below.
**Date:** 2026-07-04 (proposed) / 2026-10-08 (accepted)
**Related:** SPEC §§2, 3, 11.1 (Gate E), 12 (S10); ADR-0003, ADR-0005, ADR-0007, ADR-0009, ADR-0010, ADR-0011, ADR-0012, ADR-0013; `docs/cloud-deployment.md`; `docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md` Phases 4–5; production review 2026-09-06 finding 15.

## Context

The proposed version of this ADR required the owner to choose a cloud provider and runtime, a monthly budget ceiling, a teardown path, a secret-management service and a deployment gate tied to frozen-evidence evaluation. On 2026-10-07/08 the owner reviewed seventeen design decisions (D01–D17) in the interactive blueprint, changed four of them and accepted the rest. This record captures the deployment-level outcome. Decisions with their own contract surface are recorded in ADR-0010 (identity), ADR-0011 (embedding and rerank models), ADR-0012 (ingestion orchestration), ADR-0013 (durable execution and caching), and amendments to ADR-0003 (vector store) and ADR-0009 (model providers).

The project is a portfolio system operated by one person. Its design must show the failure analysis of a production system while building only what one owner can run, pay for and rehearse.

## Decision

### Posture (D01, D02)

- **AWS is the only serving cloud.** One active region, `us-east-1`, already configured in the repository. All state and compute live there.
- **Multi-region is limited to model routing.** Bedrock is called through US cross-region inference profiles, which spread model requests across US regions without any state outside `us-east-1`.
- **Azure is the model plane only.** Microsoft Foundry serves the same LLM and embedding models as fallbacks (ADR-0009 amendment, ADR-0011). No Azure service holds jobs, evidence, indexes or users.
- **Loss of all of `us-east-1` is a documented failure mode, not built infrastructure.** See *Region-loss failure mode* below.

Why: every cross-cloud call is a stateless model request, so nothing has to stay consistent across providers. Model capacity is where throttling and outages occur most often, so it is the one plane with a live fallback. A recovery site, warm standby or cross-region data copy only pays off with a stated availability SLO and rehearsed restores, which a portfolio deployment does not have. `us-east-1` keeps the existing configuration and has the broadest Bedrock catalog (DeepSeek V3.2 and gpt-oss-120b included).

### Edge (D03)

Cloudflare provides DNS, WAF, Turnstile and edge rate rules on the free plan. The demo tier reaches ECS through Cloudflare Tunnel with no public ingress; a scale tier uses an ALB with authenticated origin pulls. The event stream sends a heartbeat every 15 s because Cloudflare closes idle proxied connections after about 100 s. If the origin is unreachable, Cloudflare serves a static status page.

### Runtime (D04)

ECS Fargate on Graviton (ARM64) runs four services: the Next.js web BFF, the FastAPI API, LangGraph workers and FinBERT (inside the worker as a process pool in the demo tier, its own service from the 100K-user tier). Lambda is rejected because jobs run 25–60 s and the UI streams events; EKS is rejected because a cluster control plane and its operations are not justified (integration plan: no Kubernetes). Workers scale on a published "ready jobs per worker" metric, capped by the provider token budget.

### Evidence storage, backups and teardown (D14)

- S3 holds EvidenceSnapshots, manifests, chunk and embedding Parquet and the eval registry under the existing content-addressed `{type}:{sha256}` names. The bucket is versioned; approved releases are protected by Object Lock. `releases.jsonl` appends with S3 conditional writes until a database is justified (ADR-0005).
- Aurora keeps in-region point-in-time recovery (about 5 min RPO within the region).
- **No cross-region copies** (owner decision, 2026-10-08): no S3 Cross-Region Replication, no cross-region snapshot copies, no image replication.

### Secrets, infrastructure as code, deploy gate and budget (D16)

- No stored cloud credentials. GitHub Actions reaches AWS and Azure through OIDC; ECS tasks use task roles; workers reach Foundry through AWS STS outbound identity federation trusted by an Entra federated credential (spike required). Third-party keys (Tavily, market data, Qdrant, LangSmith) live in AWS Secrets Manager with KMS.
- Terraform with the `aws`, `azurerm`, `cloudflare` and Qdrant Cloud providers. One `apply` from a clean checkout creates the environment; no console-created resources.
- **Deploy gate:** deploy on tag. ARM images are pushed to ECR; the frozen-evidence `eval-replay` job (SPEC §11.2) must pass before the release; ECS blue/green deployment, smoke test, then traffic shift. A release that regresses `grounded_claim_rate` or `citation_coverage` beyond the accepted tolerance against the active approved baseline cannot deploy.
- **Budget ceiling:** $100/month for the demo tier (projection about $73 at 25 memos/day with rerank off, the default; $79 if Gate C adopts rerank; see `docs/cloud-deployment.md`). An AWS Budgets action at 100% of the ceiling sets an SSM parameter that the API reads at admission: new jobs receive a typed 503, running jobs finish. A 25-memo/day admission cap is the first control.
- **Teardown:** `terraform destroy` removes every resource except the evidence bucket, which carries `prevent_destroy` and Object-Locked approved releases. The Qdrant Cloud cluster and Foundry deployments are destroyed through their providers.

### Observability (D17)

OpenTelemetry in every service, exported through an ADOT collector to Grafana Cloud (metrics, logs, traces). LangSmith remains the LLM trace store. Per-run cost comes from provider usage fields. A tracing outage never blocks memo delivery (bounded buffers with drop counters). Monitoring lives outside the cloud it watches.

### Region-loss failure mode (documented, not built)

If `us-east-1` is lost:

| Component | Effect | Data at risk |
| --- | --- | --- |
| Web, API, workers | Unavailable; Cloudflare shows the status page | none (stateless) |
| Cognito user pool | Nobody can sign in | User accounts, if the region's storage is permanently lost: users re-register or are migrated, and `customers.cognito_sub` mappings are rebuilt |
| Aurora | Unavailable | Jobs, results and run records if the region's storage is permanently lost |
| S3 evidence bucket | Unavailable | Frozen evidence releases. SEC filings can be refetched; past news and quotes cannot, so approved baselines would no longer be reproducible |
| Qdrant Cloud (AWS `us-east-1`) | Unavailable | Rebuildable from Parquet vectors in S3, which are in the same region |
| Bedrock, Foundry | Bedrock calls originate in the region, so they stop; Foundry stays up but nothing calls it | none |

Mitigations, ordered by cost, recorded so they can be adopted if an SLO ever requires them: S3 Cross-Region Replication of the evidence bucket (about $1–3/month at current size, removes the irreproducible-baseline risk); AWS Backup cross-region copies of Aurora snapshots (RPO up to 24 h); ECR replication; Aurora Global Database (RPO about 1 s, a second always-on cluster); a warm standby stack applied from the same Terraform with `region` as a variable. Recovery would then be: apply the stack in the second region, restore the database, create a Qdrant Cloud cluster there and bulk-load vectors from Parquet (no re-embedding), point Cloudflare at the new origin.

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
2. S7 local seams built and tested: Postgres RunStore and job queue (ADR-0013), caches, guardrails, FinBERT out of process.
3. The `eval-replay` CI job exists and fails on regression (SPEC §11.2), because the deploy gate calls it.
4. DeepSeek V3.2 and gpt-oss-120b qualified on Bedrock against the frozen eval set (paid API calls, no infrastructure); the Gate C embedding bake-off done if the embedding model is to change (ADR-0011).

**Before the environment is left running:**

5. Foundry parity for both models and embedding parity, measured against the Foundry deployments the first apply creates. The fallbacks stay disabled until they pass (ADR-0009 amendment, ADR-0011).
6. Two rehearsals of `apply → smoke → destroy` from a clean checkout.

## Verify before implementation

- Qdrant Cloud paid-tier availability in `us-east-1`.
- Entra accepts the issuer and subject of the AWS STS outbound identity token.
- Current prices for every line in the cost model (`docs/cloud-deployment.md`).
