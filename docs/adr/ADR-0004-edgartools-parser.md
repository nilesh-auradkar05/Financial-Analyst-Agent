# ADR-0004 — edgartools as the SEC parser strategy

**Status:** Accepted; regression-locked at S1-T03.
**Date:** 2026-06-03
**Related:** SPEC §9, test-plan §3, sprint-plan S1.

## Context

An earlier parser produced **zero sections and zero chunks** for some filings, notably MSFT and NVDA. Retrieval cannot recover evidence ingestion never indexed. We need reliable extraction of critical 10-K sections: `business`, `risk_factors`, `md&a`, and `market_risk`, with metadata needed for citations and filtering.

## Decision

Use `edgartools` as the SEC parser strategy. Normalize extracted sections to canonical `section_key`s. Attach citation/filter metadata at ingestion and mint deterministic `chunk_id = f(accession_number, section_key, chunk_index)`.

Require section-coverage tests that assert non-zero chunks for each present critical section **through the retrieval store interface**, not parser internals. Missing optional sections must be explicit skip/xfail with reason, never silent pass.

The same `edgartools` parse builds the committed source-section cache that benchmark validation checks anchors against.

## Consequences

- Adds `edgartools` as a dependency.
- Section coverage becomes a CI gate.
- The prior MSFT/NVDA zero-section failure cannot recur undetected.
- Parser changes require updated coverage tests, rebuilt source-section cache, fixture re-validation, and an ADR/update note.

## Fallback trigger

If `edgartools` produces a measured section-coverage failure on representative tickers that cannot be resolved inside the adapter, revisit via a new ADR. Candidate fallback: paid `sec-api.io` or another parser. Do not silently swap parser strategy.

## Alternatives considered

- Raw EDGAR HTML parsing: brittle; source of the original failure class.
- Paid `sec-api.io`: reliable but adds cost and external dependency; held as fallback.
- Heavier document toolkits: more machinery than the current section-extraction job needs.
