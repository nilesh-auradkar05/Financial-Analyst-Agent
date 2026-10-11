# ADR-0012 — Ingestion orchestration and external data sources

**Status:** Accepted (2026-10-08) by owner decision D15, after the owner challenged the earlier rejection of Airflow.
**Date:** 2026-10-08
**Related:** SPEC §9; ADR-0004 (edgartools); ADR-0005 (release registry); ADR-0006; ADR-0011; `docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md` Phase 4 ("Airflow explicitly rejected", amended by this record).

## Context

Ingestion is a linear nightly pipeline: fetch changed SEC accessions, parse with edgartools, chunk with deterministic IDs, embed new chunks, write Parquet and manifests, build a staged Qdrant collection, validate it and promote it by alias swap (ADR-0003 addendum). The integration plan rejected Airflow because of its always-on cost at portfolio scale. The owner challenged that: Airflow is an in-demand, open-source skill.

The challenge was right about the skill and partly wrong about cost. Open source removes the license fee, not the runtime: a provisioned MWAA small environment is $0.49/hour (about $358/month, more than three times the $100 budget), and a self-hosted Airflow 3 on Fargate costs about $40–80/month while its scheduler polls the metadata database continuously. On the shared Aurora Serverless v2 cluster that polling would stop the database from ever pausing (ADR-0013). Amazon MWAA Serverless, launched in November 2025, changes this: Airflow 3 workflows billed only for task run time, with no idle environment.

## Decision

1. **Airflow 3 on Amazon MWAA Serverless** in `us-east-1`. The workflow is defined in YAML (DAG-factory format). Each task is an `EcsRunTaskOperator` running one `dataops` subcommand in the application image: `fetch → parse → chunk → embed → stage → validate → promote`. The `dataops` CLI is the unit of work; the DAG holds only ordering, retries and parameters. These subcommands do not exist yet (today `dataops/` exposes only live evidence capture); they are built in S10-T05 from the existing ingestion and release code.
2. **Schedule:** nightly. Per-issuer backfills run the same workflow with task mapping.
3. **Local parity:** the same YAML runs locally through the open-source `dag-factory` package under docker-compose, so the DAG is tested before deployment.
4. **SEC EDGAR:** one shared token-bucket limiter at ≤ 8 requests/second for all ingestion, because the fair-access limit (10 requests/second) applies per declared user agent.
5. **Market data:** yfinance is replaced by a licensed market-data API behind the existing `stock_data_tool` seam before any memo is published from the deployed system. Vendor choice is decided by redistribution terms for quotes shown inside memos, then price.
6. **News:** Tavily is kept; results are cached 15 minutes per issuer and query (ADR-0013).

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| EventBridge Scheduler + one ECS task (blueprint v1) | Alternative | Cheapest and simplest; no per-step retries, dependencies or backfill semantics |
| MWAA provisioned | Rejected | About $358/month for the small environment |
| Self-hosted Airflow 3 on ECS | Rejected | Idle cost, and it keeps Aurora from scaling to zero |
| GitHub Actions cron (integration plan) | Rejected | Gives CI credentials on the production data plane |
| Step Functions | Alternative | Native AWS; adopt if per-issuer fan-out outgrows the DAG |

## Consequences

- MWAA Serverless has no Airflow UI (monitor through logs and the console), supports only listed operators (all logic lives in the container, so custom operators are not needed) and starts compute per task (acceptable for a nightly batch).
- YAML DAG-factory definitions read less like the Python DAGs common in interviews; Python definitions can be authored and converted.
- Orchestration cost is expected to be a dollar or two a month at nightly cadence; the per-task rate must be verified.
- The integration plan's Phase 4 orchestration line is amended accordingly.
- Tests (test-plan §20): a failed task retries alone; promotion never happens after a failed validation; the EDGAR limiter never exceeds 8 requests/second across concurrent tasks.

## Verify before implementation

- MWAA Serverless per-task rate in `us-east-1`; `EcsRunTaskOperator` in the serverless supported-operator list.
- Market-data vendor redistribution terms.
