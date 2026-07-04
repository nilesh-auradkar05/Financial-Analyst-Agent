# ADR-0005 — Unified DataOps Release Registry

**Status:** Accepted (2026-07-04)
**Date:** 2026-07-04
**Related:** SPEC §7, SPEC §10, SPEC §11, `docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md`

## Context

The repo already produces version-shaped artifacts: retrieval fixtures carry
`fixture_version`, benchmark runs emit manifests, and memo-grounding baselines
record model/config/git provenance. These were not governed by one release
lifecycle, and answer-quality baselines still depended on live-changing evidence
feeds.

The immediate need is local-first evidence freeze and replay, not a service or
database. The registry must make every quoted metric traceable to evidence,
code, config, model, and quality-report versions without adding infrastructure
before the local gates are proven.

## Decision

Adopt one DataOps release model:

- `EvidenceSnapshot` records immutable multi-source evidence payloads.
- `DatasetReleaseManifest` records release metadata for evidence snapshots,
  canonical sections, chunks, vector indexes, retrieval benchmarks, answer evals,
  and quality baselines.
- `artifacts/dataops/releases.jsonl` is the append-only registry.
- `artifacts/dataops/active/*.yaml` stores active pointers.
- Parent release IDs on manifests are the lineage mechanism.

Existing fixture versions, benchmark manifests, and quality baseline result files
become release types within this system instead of parallel versioning schemes.

## Consequences

- A release can be approved only with a passing quality report.
- A quality baseline produced against live evidence can be registered as
  `candidate` at most.
- Runtime code does not need to depend on registry internals; it may read active
  pointers through config when the relevant phase reaches that step.
- No database is introduced for Phase 1.
- Registry writes are append-only; bad releases are rejected or deprecated, not
  deleted.

## Non-Decisions

- No provider-specific cloud storage.
- No orchestration framework.
- No public release API.
- No change to retrieval benchmark methodology; that remains in
  `docs/retrieval-benchmark.md`.
