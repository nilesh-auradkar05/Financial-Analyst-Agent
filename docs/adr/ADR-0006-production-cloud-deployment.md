# ADR-0006 — Production Cloud Deployment

**Status:** Proposed
**Date:** 2026-07-04
**Related:** SPEC §2, SPEC §3, SPEC §11, `docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md`

## Context

The v2 integration plan proposes a production-grade cloud-deployed end state.
The current SPEC still keeps production cloud deployment out of scope for the
committed local-first arc. That guard stays in force until this ADR is accepted
and SPEC §3 is amended.

Phase 1 must first prove evidence snapshot replay and frozen-evidence quality
baselines. Cloud work before that would add operational surface before the
evaluation substrate is trustworthy.

## Proposed Decision

Adopt a single-environment, cost-bounded cloud deployment only after Phase 1
exit criteria are green and the owner chooses:

- cloud provider and runtime,
- monthly budget ceiling,
- teardown path,
- secret-management service,
- deployment gate tied to frozen-evidence evaluation.

The deployed service must serve `/analyze` from approved active releases and
must block deployment when grounding quality regresses beyond the accepted
tolerance against the active approved baseline.

## Consequences If Accepted

- SPEC §3 must move production cloud deployment from out-of-scope to a gated
  Phase 4 scope item.
- Infrastructure must be reproducible from a clean checkout.
- Secrets remain out of the repo.
- Release artifacts remain the traceability spine; cloud only changes storage
  and execution location.

## Non-Decisions While Proposed

- No provider selection.
- No budget selection.
- No cloud resources.
- No infrastructure files.
- No deployment pipeline changes.
- No multi-agent or long-term-memory scope change.
