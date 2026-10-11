# ADR-0010 — Identity and authorization

**Status:** Accepted (2026-10-08) by owner decision D05. Implementation in S7 (local, with a development issuer) and S10 (Cognito).
**Date:** 2026-10-08
**Related:** SPEC §3, REST service contract clarification (2026-09-21), run telemetry contract (2026-10-07, `GET /jobs` "not user-scoped"); ADR-0006; ADR-0013; production review finding 11.

## Context

The API is protected by one configured API key. `GET /jobs` lists every stored run and the SPEC records that per-user visibility "arrives with the managed-auth decision". Commit 6c5da85 (2026-10-07) records a planned split of the frontend into user and admin pages with role-based access control; the admin page does not exist yet. The browser never calls the API directly: Next.js route handlers call it server-side. Browser `EventSource` cannot set an `Authorization` header, so a bearer token cannot authenticate the event stream from the browser. Production review finding 11 requires issuer identity (CIK, ticker aliases) to stay separate from customer identity.

## Decision

1. **Identity provider:** Amazon Cognito user pool, Essentials tier (managed login, passkeys, email MFA), in `us-east-1`.
2. **Browser session:** the Next.js web BFF runs the authorization-code flow with PKCE server-side and keeps tokens in an encrypted, HttpOnly, `SameSite=Lax`, same-origin session cookie. The browser holds no bearer token. The BFF forwards the access token to the API, including for `GET /jobs/{id}/events`, so server-sent events work through same-origin cookies.
3. **API authentication:** FastAPI validates JWTs against an **issuer allowlist** (issuer, audience, signature through JWKS, expiry). Changing the identity provider is a configuration change. Programmatic clients use per-customer API keys, stored hashed in Postgres; the single `API_KEY` becomes the local-development issuer.
4. **Authorization is data, not identity.** Every job, event stream, result and idempotency key carries `customer_id`; reads and writes are checked in Postgres against it (ADR-0013). Ticker and CIK never determine access.
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
