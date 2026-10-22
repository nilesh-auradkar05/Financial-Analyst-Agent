# ADR-0015 — Scale-to-zero serverless deployment for the demo tier

**Status:** Accepted (2026-10-11) by owner decision. Supersedes in part ADR-0006 (D03 edge wiring, D04 runtime, D16 secrets store and budget projection), ADR-0012 (D15 schedule) and ADR-0013 (D06 store, D07 queue, D08 cache store, D13 sentiment placement) **for the deployed demo tier**, and amends ADR-0010 (guest access) and ADR-0014 (ledger store, role of the awake cap). The superseded ECS / Aurora / Valkey design is retained, unbuilt, as the **production scale tier** in `docs/production-scale-architecture.md`, with a measured trigger for each component.
**Date:** 2026-10-11
**Related:** SPEC §§3, 5, 11.3, 12 (S7, S10); ADR-0006; ADR-0009 amendment; ADR-0010; ADR-0011; ADR-0012; ADR-0013; ADR-0014; `docs/cloud-deployment.md`; `docs/production-scale-architecture.md`.

## Context

The owner stated the usage pattern on 2026-10-11: this is a portfolio project, used only when a recruiter wants to try it, typically one query per visit, with days of no traffic in between. No service needs to run every day.

The accepted deployment (ADR-0006, 2026-10-08) was sized for steady traffic. At this usage it pays mostly for idle capacity:

| Idle cost in the accepted design | Why it never reaches zero |
| --- | --- |
| ECS Fargate services (web BFF, API, workers) | At least one task per service stays running |
| ElastiCache Serverless for Valkey | Minimum billed data storage, every hour |
| Cloudflare Tunnel | Needs a `cloudflared` connector running next to the services |
| NAT gateway | Needed for private-subnet egress (SEC, Tavily, Foundry); about $33/month before traffic, and missing from the earlier cost model |
| Aurora Serverless v2 with auto-pause | Reaches 0 ACU, but resumes in about 15 s, and **30 s or more after a pause longer than 24 h**; it also needs a VPC |

Under this usage nearly every visit follows more than 24 hours of idleness, so Aurora's slow-resume case is the normal case.

Facts verified on 2026-10-11 for this decision:

- Lambda response streaming works through function URLs, up to 200 MB per response (about 2 MB/s after the first 6 MB). It is **not available for function URLs inside a VPC**, and a streaming function keeps running and billing after the client disconnects. Python streams through the Lambda Web Adapter (`AWS_LWA_INVOKE_MODE=response_stream`).
- `langgraph-checkpoint-aws` provides `DynamoDBSaver`: on-demand table with `PK`/`SK`, TTL through `ttl_seconds`, and checkpoints over 350 KB offloaded to S3.
- MWAA Serverless bills only for workflow run time, with no environment cost, a VPC is optional, and `EcsRunTaskOperator` is in its supported-operator list.
- Cloudflare Workers, free plan: 10 ms of CPU per request; time spent waiting on network I/O does not count; no wall-clock limit while the client is connected; 100,000 requests per day; 50 subrequests per request.
- Lambda SnapStart supports container images (Python 3.12+), but for runtimes other than Java it bills a snapshot cache charge of $0.0000015046 per GB-second for as long as the version is active (minimum 3 hours), plus $0.0001398 per GB restored.

## Decision

Every component of the demo tier is pay-per-use or free when idle. Contracts above the storage seams (acceptance, fencing, event replay, versioned cache keys, index releases, issuer allowlist, status precedence) are unchanged; only the services that realize them change.

### 1. Compute: three Lambda functions (arm64)

| Function | Role | Memory | Concurrency |
| --- | --- | --- | --- |
| `web` | Next.js BFF on the Lambda Web Adapter, response streaming | 1 GB | reserved 10 |
| `api` | FastAPI on the Lambda Web Adapter, response streaming | 1 GB | reserved 10 |
| `worker` | LangGraph pipeline; FinBERT exported to ONNX and loaded in process | 3 GB | reserved 3; SQS event source mapping, batch size 1, maximum concurrency 2, so one slot stays free for the warm-up and the queue is never throttled |

Reserved concurrency is a hard ceiling on how fast money can be spent, independent of the budget alarm. Each open event stream holds one `web` and one `api` execution for up to 5 minutes, so `web` and `api` get 10 each: about ten concurrent visitors, far above recruiter traffic, at no idle cost. Fargate is kept only for ingestion tasks (§6). FinBERT moves into the worker because every invocation already has its own isolated execution environment; the out-of-process requirement of ADR-0013 was about event-loop and memory contention between concurrent jobs, which one-job-per-environment removes.

### 2. Edge: Cloudflare Worker signing requests to an IAM-protected function URL

- The `web` function URL uses `AuthType = AWS_IAM`. A Cloudflare Worker (free plan) validates Turnstile on the landing page, applies the free WAF rules, signs each request with SigV4 and passes the response stream through untouched.
- Why IAM rather than a public URL with a shared header: unsigned requests are refused **before** Lambda is invoked, so traffic that bypasses Cloudflare cannot create cost.
- The Worker holds the only stored AWS credential in the system: an IAM user whose single permission is `lambda:InvokeFunctionUrl` on the `web` alias, rotated every 30 days by a GitHub Actions job authenticated through OIDC. This is an accepted, scoped exception to ADR-0006's "no stored cloud credentials".
- The Worker route fails closed. The 15 s SSE heartbeat stays (Cloudflare closes idle proxied connections after about 100 s). Each event-stream response is capped at 5 minutes and the client reconnects with `Last-Event-ID`; the handler also stops on disconnect, because a streaming function bills after the client leaves.
- `web` calls `api` through its function URL, also `AWS_IAM`, signed with the `web` execution role.

### 3. Durable state: DynamoDB on-demand

| Table | Key | Purpose | Expiry |
| --- | --- | --- | --- |
| `jobs` | `job_id` | state, `attempt` (fence), `lease_until`, `deadline`, `customer_id`, `request_hash`, `snapshot_hash`, result reference | none |
| `job_events` | `job_id`, `seq` | ordered, replayable events | 30 days |
| `idempotency` | `customer_id#key` | request hash and job | 7 days |
| `sessions` | `session_id` | BFF session (encrypted token set or guest identity) | 24 h guest, 8 h admin |
| `quotas` | `scope#window` | atomic counters for guest, IP and global quotas | window end |
| `cache` | the ADR-0013 key strings, unchanged | evidence, query-embedding, retrieval, memo and FinBERT caches; single-flight locks | per key type |
| `wake_ledger` | `day` | search awake windows (ADR-0014) | 35 days |
| `checkpoints` | `PK`, `SK` (`DynamoDBSaver`) | LangGraph checkpoints, S3 offload above 350 KB | 7 days |

- **Accept:** one `TransactWriteItems`: put the idempotency item if absent, put the job, put event 1. A reused key with a different request hash returns 409.
- **Enqueue (outbox without an outbox table):** DynamoDB Streams on `jobs` captures the insert; a dispatcher Lambda triggered by the stream (filter: `INSERT`) sends the job ID to SQS. A crash between commit and enqueue cannot strand a job, and the web path never calls SQS directly.
- **Claim:** `UpdateItem SET attempt = attempt + 1, state = running, lease_until = :t` with condition `state = queued OR (state = running AND lease_until < :now)`.
- **Commit:** the job update carries `state = running AND attempt = :mine`. An event is written in a `TransactWriteItems` with a `ConditionCheck` on the job (`state = running AND attempt = :mine`) and a put of `job_events` with `attribute_not_exists(seq)`, so a stale attempt can write neither events nor results. As before, this does not make model calls exactly-once; a possibly duplicated call is recorded.
- **Lease:** worker timeout 5 minutes, SQS visibility timeout 6 minutes, job deadline 10 minutes. AWS's advice to set the visibility timeout to six times the function timeout exists to absorb throttled invocations; the event source mapping's maximum concurrency (2) sits below the worker's reserved concurrency (3), so the mapping does not throttle and a short timeout is safe. Inside one invocation, transient provider errors retry from the last LangGraph checkpoint. On an error the worker cannot recover from, it sets the message's visibility to 10 seconds before failing, so the retry runs at once. Only a hard timeout or crash waits for the 6-minute redelivery, which still leaves time before the deadline. After three receives the message moves to a dead-letter queue.
- **Writes that end someone else's attempt:** the DLQ handler and lazy expiry write `failed` with `SET state = failed, attempt = attempt + 1` under the condition `state = running AND lease_until < :now` (or `state = queued`). Bumping `attempt` fences out the attempt they replace, so a late worker can no longer commit over `failed`. A redelivered message whose job is already past its deadline writes `failed` (`deadline_exceeded`) the same way instead of running.
- **Deadline without a sweeper:** no scheduled job sweeps job state. When the API reads a job whose deadline has passed and whose lease has expired, it writes `failed` (`deadline_exceeded`) with the conditional write above. Expiry is resolved lazily, on read. The only scheduled work in the demo tier is the monthly ingestion run and a daily wake-ledger reconciliation (ADR-0014): a Lambda that runs for seconds a day through EventBridge Scheduler, effectively free.
- **Single-flight:** a lock item written with `attribute_not_exists(pk) OR expires_at < :now`; identical jobs wait for the winner's artifact. The condition checks `expires_at` itself because DynamoDB TTL deletes expired items late (up to a day or two), so TTL alone cannot release a crashed holder's lock.
- **Event stream:** the API polls `job_events` after `Last-Event-ID` every 500 ms while a stream is open.
- **Local development:** DynamoDB Local and ElasticMQ (SQS-compatible) containers in docker-compose behind the same `RunStore` and `JobQueue` protocols. One adapter per protocol; no Postgres adapter is built in S7.

### 4. Access: guest mode for recruiters, Cognito for the admin (amends ADR-0010)

- **Guest:** a visitor who passes Turnstile gets a server-side guest session (`customer_id = guest:<uuid>`, role `guest`). The BFF mints a 15-minute ES256 token for the API under its own issuer, which joins the API's issuer allowlist; the private key is a Parameter Store SecureString.
- **Guest quotas:** 1 live job per session, enforced by a conditional update of the session's `active_job` (allowed only if absent or if that job is terminal, so no release step exists); 3 per IP hash and 20 across all guests per UTC day, as counters in `quotas` incremented with `ADD n 1` under `attribute_not_exists(n) OR n < :limit`. Example memos replay without limit, because they are memo-cache hits that wake nothing. Guests see only their own jobs.
- **Admin:** Cognito Essentials, self-sign-up disabled, MFA required, group `admin`. The `analyst` role stays defined for named users in the scale tier.

### 5. Cold start: warm-up on the landing page, no SnapStart

- After Turnstile succeeds, the landing page calls `POST /warm`. It sends an asynchronous no-op to `worker` (loads ONNX FinBERT) and a `_count` on `sec_chunks_active` to OpenSearch (about 10 s to wake). Both overlap with the recruiter reading the page and typing a ticker.
- A warm-up counts against the search awake cap and the guest's IP quota for warm-ups (3 per day), so it cannot be used to keep the collection awake.
- Expected extra latency on the first query: about 1–3 s when the warm-up ran at least 10 s earlier, otherwise about 12–15 s (OpenSearch resume plus function initialization). The event stream reports a "warming" progress event.
- **SnapStart is rejected.** Its cache charge bills while idle: about $11.70/month for the 3 GB worker and $3.90/month for each 1 GB function, more than the whole idle floor of this design.

### 6. Ingestion: monthly and on demand (amends ADR-0012 D15)

The MWAA Serverless DAG, the `EcsRunTaskOperator` steps and the release flow are unchanged. The schedule changes from nightly to **monthly, plus an admin trigger** (`POST /ingest`), because the corpus changes quarterly and only market and news data must be fresh, and those are fetched at query time. Fargate tasks run in public subnets of the default VPC with a public IP and no inbound rules, so no NAT gateway exists. An ingestion run wakes OpenSearch for about 20 minutes; it is accounted for separately from the query awake cap and limited to two runs a month.

### 7. Network, secrets and keys

- **No VPC for Lambda.** Function URL streaming does not work inside a VPC, and every dependency (DynamoDB, SQS, S3, Bedrock, OpenSearch Serverless with IAM data access, Cognito) has a public, IAM-authenticated endpoint. Removing the VPC removes the NAT gateway.
- **Secrets:** SSM Parameter Store SecureString (standard tier, free) with the AWS-managed key, replacing Secrets Manager ($0.40 per secret per month). Third-party keys, the guest-token signing key and the Turnstile secret live there.
- **Encryption:** AWS-managed keys instead of customer-managed keys ($1 per key per month).
- Workers reach Foundry through AWS STS outbound identity federation, as before (ADR-0009 amendment); no Azure key is stored.

### 8. Deploy, budget and teardown

- **Deploy on tag:** OIDC, arm64 images to ECR, the `eval-replay` gate, publish new function versions, smoke-test the `candidate` alias, then move the `live` alias (the function URLs point at `live`). Rollback is one alias update.
- **Budget:** the $100/month kill switch stays (AWS Budgets → SSM admission flag → typed 503). Early-warning alerts at $10 and $25. Projected spend is about $4/month at 20 recruiter sessions and one ingestion run; the worst case allowed by the guest quotas and the 45-minute awake cap is about $30/month (`docs/cloud-deployment.md` §7).
- **Teardown:** `terraform destroy` removes everything except the evidence bucket (`prevent_destroy`, Object Lock), as before.

## Graduation to the production scale tier

The design superseded here is not discarded. It becomes the documented target when traffic is steady. Each component moves only when its trigger is measured, and each move is an adapter swap behind an existing protocol (`RunStore`, `JobQueue`, cache, `RetrievalStore`) or a runtime change for the same application image. The full design, capacity model and migration plan are in `docs/production-scale-architecture.md`.

| Demo tier (this ADR) | Scale tier | Trigger |
| --- | --- | --- |
| Lambda `worker` | ECS Fargate async workers, many jobs per task | Lambda worker spend above twice an equivalent Fargate fleet (about 1k memos/day; break-even ≈ 540/day), or a job class longer than 15 minutes |
| Lambda `web`/`api` behind a Cloudflare Worker | Fargate services behind an ALB with Cloudflare authenticated origin pulls | Sustained traffic near the Workers free limits, or a cold-start latency SLO |
| DynamoDB job store | Aurora PostgreSQL (Multi-AZ): `SKIP LOCKED` claims, transactional outbox to SQS | Relational queries in the product (audit, entitlements, lineage joins) or transactions above DynamoDB's 100-item limit |
| DynamoDB caches | ElastiCache Serverless for Valkey | Sustained cache reads where on-demand read cost exceeds Valkey, or a sub-millisecond rate-limit budget |
| API polling `job_events` | Pub/sub fan-out with durable replay | More than about 1,000 concurrent streams |
| Guest mode + admin | Cognito self-sign-up analysts, per-customer quotas and API keys | Named users or a paid plan |
| Monthly ingestion | Nightly, with per-issuer backfill fan-out | Issuer universe beyond the fixture set, or a filing-freshness SLO under 24 h |
| OpenSearch awake cap | Cap removed; minimum OCUs above zero; managed domain above about 4 OCUs steady | Cap reached on 3+ days a week, or a first-query latency SLO |
| No VPC, Parameter Store, AWS-managed keys | VPC with endpoints and NAT, Secrets Manager with rotation, customer-managed keys | Aurora or Valkey introduced (both need a VPC), or a compliance requirement |

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| Keep ADR-0006 as accepted | Rejected for the demo; kept as the scale tier | About $95/month (plus an omitted NAT gateway) spent mostly on idle capacity |
| Aurora Serverless v2 with auto-pause + Data API | Rejected | 30 s+ resume after a day idle, which is the normal case here; still needs a VPC |
| Lambda async invoke instead of SQS | Rejected | At most two retries and no visibility-timeout lease; SQS costs nothing idle and is also the scale-tier queue |
| CloudFront with origin access control in front of the function URL | Alternative | Viable; for `POST` bodies the client must send a payload hash, and Turnstile and the WAF rules already live in Cloudflare |
| Public function URL with a shared secret header | Rejected | Requests that bypass Cloudflare still invoke and bill the function |
| Lambda SnapStart | Rejected | Snapshot cache charge accrues while idle (about $11.70/month for the worker) |
| Provisioned concurrency | Rejected | Billed continuously |
| EventBridge Pipes from the stream to SQS | Alternative | Works and is billed per request; the stream-triggered dispatcher has the same effect, and stream reads by Lambda triggers should be free (to verify) |
| S3 Vectors instead of OpenSearch | Rejected | No BM25 or hybrid retrieval; owner kept OpenSearch Serverless |
| Requiring Cognito sign-up for recruiters | Rejected by owner | Sign-up friction for a one-query visit |

## Consequences

- Idle cost falls from about $95/month to about $0.45/month; a recruiter session costs about $0.15, most of it the OpenSearch idle tail.
- The first query after idle is slower, mostly hidden by the warm-up. A recruiter who submits within 10 s of landing waits up to about 15 s longer.
- The worker is billed for wall-clock time while it waits for model tokens. That is cheap at demo volume and is exactly why the worker is the first component to graduate (see triggers).
- Ad-hoc SQL over jobs is gone in the demo tier; audits read the eval registry and the S3 run records, which are the lineage source of truth anyway (ADR-0005).
- One stored, narrowly scoped AWS credential exists (the Worker's signing key), rotated automatically.
- S7(d) builds DynamoDB and SQS adapters (local: DynamoDB Local, ElasticMQ) instead of Postgres ones; test-plan §20 cases are re-expressed for conditional writes and visibility timeouts.
- Interview framing: the demo tier shows cost discipline for a bursty, low-volume workload; the scale tier shows how the same contracts run under steady load, and what measurement moves each component.

## Verify before implementation

- Lambda function URL `AWS_IAM` auth combined with `RESPONSE_STREAM`, signed by the Worker (spike); Worker CPU time per request stays under 10 ms.
- That a stream-triggered Lambda reads DynamoDB Streams without read charges (DynamoDB pricing: reads by Lambda triggers are free).
- MWAA Serverless per-run rate in `us-east-1`; Fargate tasks with a public IP in the default VPC under MWAA Serverless.
- ONNX FinBERT parity: label agreement with the PyTorch model on the sentiment fixture, and initialization time at 3 GB.
- Cloudflare free-plan rate-limiting rule for `POST /analyze`.
- Measured cold-start latency with and without the warm-up, recorded in the run registry.
