# ADR-0010 — Identity and authorization

**Status:** Accepted (2026-10-08) by owner decision D05. **Amended 2026-10-11 (ADR-0015):** recruiters use a guest mode; Cognito is kept for the admin (see the amendment at the end). Implementation in S7 (local, with a development issuer) and S10 (Cognito).
**Date:** 2026-10-08
**Related:** SPEC §3, REST service contract clarification (2026-09-21), run telemetry contract (2026-10-07, `GET /jobs` "not user-scoped"); ADR-0006; ADR-0013; ADR-0015; production review finding 11.

## Context

The API is protected by one configured API key. `GET /jobs` lists every stored run and the SPEC records that per-user visibility "arrives with the managed-auth decision". Commit 6c5da85 (2026-10-07) records a planned split of the frontend into user and admin pages with role-based access control; the admin page does not exist yet. The browser never calls the API directly: Next.js route handlers call it server-side. Browser `EventSource` cannot set an `Authorization` header, so a bearer token cannot authenticate the event stream from the browser. Production review finding 11 requires issuer identity (CIK, ticker aliases) to stay separate from customer identity.

## Decision

1. **Identity provider:** Amazon Cognito user pool, Essentials tier (managed login, passkeys, email MFA), in `us-east-1`.
2. **Browser session:** the Next.js web BFF runs the authorization-code flow with PKCE server-side and keeps tokens in an encrypted, HttpOnly, `SameSite=Lax`, same-origin session cookie. The browser holds no bearer token. The BFF forwards the access token to the API, including for `GET /jobs/{id}/events`, so server-sent events work through same-origin cookies.
3. **API authentication:** FastAPI validates JWTs against an **issuer allowlist** (issuer, audience, signature through JWKS, expiry). Changing the identity provider is a configuration change. Programmatic clients use per-customer API keys, stored hashed in the job store; the single `API_KEY` becomes the local-development issuer.
4. **Authorization is data, not identity.** Every job, event stream, result and idempotency key carries `customer_id`; reads and writes are checked against it in the job store (DynamoDB in the demo tier, Postgres in the scale tier; ADR-0013). Ticker and CIK never determine access.
5. **Roles (RBAC):** Cognito groups `analyst` and `admin` appear in the token's `cognito:groups` claim and map to API roles. `analyst` reads and cancels only its own jobs; `admin` can list all runs (`GET /jobs`), view operational endpoints (`/stats`, `/metrics` through the BFF) and trigger ingestion. The role check is enforced in the API, never only in the UI.
6. **Quotas:** per-customer daily token and job quotas live in Postgres; short-window rate limits live in Valkey and fail to a conservative per-task limit (ADR-0013).

## Alternatives considered

| Option | Verdict | Reason |
| --- | --- | --- |
| Auth0 | Alternative | Cloud-neutral and survives an AWS outage; much higher MAU pricing past 1M users |
| Microsoft Entra External ID | Rejected | Would put sign-in on Azure, which ADR-0006 keeps to model fallback only |
| WorkOS | Alternative | If the audience becomes teams that need SSO and SCIM |
| Self-hosted Keycloak | Rejected | Operating burden for one owner |
| Bearer tokens in the browser | Rejected | Token exposure to scripts; no auth header on `EventSource` |

## Consequences

- `GET /jobs` becomes user-scoped for `analyst` and global for `admin`, closing the gap noted in the run telemetry contract.
- Cognito pools are regional with no native cross-region replication, so a `us-east-1` loss blocks sign-in (accepted with ADR-0006).
- Cost: free below 10,000 MAU; Essentials is $0.015 per MAU above that, about $7k/month at 1M users and $75k/month at 10M. The Lite tier or negotiated pricing is the lever at that scale.
- Tests (test-plan §20): cross-customer reads of jobs, events and results are refused; idempotency keys are scoped per customer; an `analyst` cannot list other customers' runs; tokens from an issuer outside the allowlist are refused; the browser never receives an access token.

## Trigger to revisit

A sign-in recovery objective under one hour, or enterprise SSO becoming the main audience.

## Amendment 2026-10-11 — guest mode for recruiters, Cognito for the admin (owner decision, ADR-0015)

The deployed demo is used by recruiters trying one query. Requiring a Cognito sign-up for that visit is friction most would abandon, so:

1. **Guest role.** A visitor who passes Cloudflare Turnstile receives a server-side guest session in the BFF (`customer_id = guest:<uuid>`, role `guest`, 24 h). The BFF mints a 15-minute ES256 access token for the API under its own issuer (`https://<site>/guest`), and that issuer joins the API's **issuer allowlist** next to Cognito. The API therefore keeps a single authentication path: allowlisted issuer, audience, signature, expiry, role claim. The signing key is a Parameter Store SecureString; the public key is in API configuration.
2. **Guest quotas** (DynamoDB atomic counters with TTL, ADR-0015): 1 live job per guest session, 3 live jobs and 3 warm-ups per IP hash per UTC day, 20 live jobs across all guests per UTC day. Example memos replay without limit (memo-cache hits). Guests read, stream and cancel only their own jobs; `GET /jobs` returns only theirs. The IP is stored only as a salted hash with a one-day TTL.
3. **Admin.** Cognito Essentials, self-sign-up disabled, MFA required, group `admin`, same server-side PKCE session in the BFF. The admin lists all runs, sees `/stats` and triggers ingestion (`POST /ingest`).
4. **`analyst`** stays defined but unused in the demo tier; it becomes the self-sign-up role in the production scale tier (`docs/production-scale-architecture.md`).
5. Quotas and rate windows move from Postgres/Valkey to DynamoDB in the demo tier (ADR-0013 amendment).

Tests (test-plan §20): a guest token cannot read another guest's job; the fourth live job from one IP hash in a day is refused with 429; the 21st guest job of the day is refused while memo replays still succeed; a guest token is refused on admin routes; an expired or foreign-issuer token is refused.
