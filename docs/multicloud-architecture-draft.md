# Multi-cloud architecture — FINAL (accepted 2026-10-08)

Status: decisions recorded in the repo on branch `docs/adr-0006-cloud-deployment` (uncommitted at time of writing). Nothing is deployed; S10 is gated by the ADR-0006 entry criteria.
Blueprint discussion page (owner verdicts in its `reviews` collection): https://claude.ai/artifact/96MXLotSPLkKFoBUDJseDX

## Final decisions
- D01 AWS is the only serving cloud; Azure = model plane only (same-model LLM + embedding fallback on Foundry). Region loss = documented failure mode, not built.
- D02 us-east-1; Bedrock US cross-region inference profiles are the only multi-region element.
- D03 Cloudflare (WAF, Turnstile, Tunnel, status page). D04 ECS Fargate Graviton.
- D05 Cognito Essentials + BFF server-side session + issuer allowlist + customer-scoped authz + analyst/admin roles (ADR-0010).
- D06–D08, D13 Aurora PG Serverless v2 jobs/events/fencing; Postgres queue → SQS on measured trigger; Valkey disposable caches with versioned keys, exact query-embedding cache; FinBERT out of process (ADR-0013).
- D09 DeepSeek V3.2 draft + gpt-oss-120b judge on Bedrock; same models on Foundry (ADR-0009 amendment). ~$0.014/uncached memo.
- D10/D12 Cohere Embed v4 1024d, Foundry fallback behind parity gate; Gate C bake-off; rerank off until lift (ADR-0011).
- D11 Qdrant Cloud us-east-1, collection per release + alias swap (ADR-0003 addendum).
- D14 S3 versioned + Object Lock, in-region backups only, NO cross-region copies (owner: explain the failure mode instead).
- D15 Airflow 3 on MWAA Serverless, ECS task per dataops step (ADR-0012).
- D16 OIDC, Secrets Manager, Terraform, eval-gated deploy, $100/month demo ceiling (projection ~$73) with Budgets kill switch.
- D17 OTel → Grafana Cloud + LangSmith.

## Repo changes
ADR-0006 accepted; new ADR-0010–0013; amendments to ADR-0003/0007/0009; SPEC v1.4; sprint-plan S7(d)/S10; test-plan §20; integration plan Phase 4; interview §§12–13; README cloud section; new docs/cloud-deployment.md; re-produced docs/System-design/production/* (+ png/svg); patched base hld/lld/system-design labels; scope-residue check allows Terraform in SPEC/test-plan/ADRs.
