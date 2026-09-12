# ADR-0008 — Production-readiness order (SPEC v1.3 application)

**Status:** Accepted (2026-09-12)
**Date:** 2026-08-24 (review) / 2026-09-12 (applied)
**Related:** SPEC §§0, 2.1, 3, 5, 7.x–8.5, 11.2–11.4, 12, 14; sprint-plan S2-T00a–T00d, S6–S10; test-plan §§10–15; proposed ADR-0006

## Context

The 2026-08-24 production-readiness review found the eval/grounding harness to be
the strongest asset and the service shell the weakest. The review session did not
have SPEC v1.2, so the inserts were recorded as `docs/SPEC-AMENDMENT-v1.3.md`
keyed to guessed section numbers.

User authorized applying that amendment and bumping SPEC to v1.3 on 2026-09-12.
The amendment named this retirement target `ADR-0007-production-readiness-order.md`.
ADR-0007 was already used for the production interview-scale design, so this
record is ADR-0008.

## Decision

SPEC v1.3 is canonical. Inserts from the amendment are applied to the v1.2
section homes:

| Amendment label | Landed in SPEC v1.3 |
| --- | --- |
| Findings | §2.1 Known limitations |
| §3.x / §3.y | §3 |
| Package list | §5 (merged with existing retrieval/`dataops/` layout) |
| Evidence 7.x / 7.y | §7 |
| Workflow 9.1–9.5 | §8.1–8.5 (v1.2 §9 remains SEC ingestion) |
| Quality 11.2–11.4 | §11.2–11.4; existing gates are §11.1 |
| Sprint S5–S10 | §12 table |
| Hooks §14 | §14; former §14/§15 become §15 Correction Handling and §16 ADRs |
| Gaps G10–G13 | §12 and sprint-plan.md |

S10 is cloud work gated on ADR-0006. Cloud remains out of committed scope (SPEC
§3.y) until that ADR is accepted. Multi-agent work beyond the single reviewer
loop stays out of scope.

Repo-root `dataops/` remains the implementation until S2-T00b; §5 names
`app/dataops/` as the target layout.

## Consequences

- `docs/SPEC-AMENDMENT-v1.3.md` is a pointer, not an alternate SPEC.
- Sprint-plan traces that cited amendment §9.1–9.5 retarget to SPEC §8.1–8.5.
- S1 traces to SPEC §9 (SEC ingestion) and §16 (ADRs).
- Applying v1.3 does not accept ADR-0006, authorize cloud spend, or implement
  the new packages.
