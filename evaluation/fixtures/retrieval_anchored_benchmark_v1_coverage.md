# Coverage report — retrieval_anchored_benchmark_v1

Source: `evaluation/fixtures/retrieval_anchored_benchmark_v1.json`, validated against `evaluation/fixtures/source_sections/` (release `alpha-evidence:0.1.0`). Chunks for the straddle check come from the ingestion chunker (`build_index_documents`: chunk_size 1000, overlap 200) run over the same release; no retrieval output was used.

- Cases: 75 (AAPL 24, MSFT 24, NVDA 27)
- Anchors: 138 (relevance 2: 93, relevance 1: 45); anchors per case: 1: 16 cases, 2: 55 cases, 3: 4 cases
- answer_intent: comparison 6, factual 23, financial_performance 9, risk 29, section_specific 8
- Multi-section cases: 6; cases with a section filter: 8
- Query wording shared with its relevance-2 anchors (content words, company names excluded): median 0.20, max 0.60

## Cases per ticker × section (a multi-section case counts in each of its sections)

| Ticker | business | risk_factors | md&a | market_risk |
| --- | --- | --- | --- | --- |
| AAPL | 7 (11 anchors) | 8 (15 anchors) | 7 (11 anchors) | 4 (7 anchors) |
| MSFT | 7 (10 anchors) | 8 (14 anchors) | 7 (13 anchors) | 4 (6 anchors) |
| NVDA | 9 (16 anchors) | 8 (13 anchors) | 8 (15 anchors) | 4 (7 anchors) |

**Thin cells (< 5 cases):** AAPL market_risk, MSFT market_risk, NVDA market_risk. Market-risk sections are 2–4k characters, so few distinct questions exist; per-cell differences there have very low power.

## Recall ceiling

Anchors not wholly inside any single benchmark chunk (always a miss under the §3 straddle rule): 0 of 138.

## Notes

- Anchors were chosen from the cached source text only, before any retrieval run. Queries paraphrase the question; the 12 queries that share 40% or more of their content words with a relevance-2 anchor share common domain words (revenue, data center, tax, currency), not lifted phrases.
- Relevance 2 = the anchor answers the question; relevance 1 = supporting context in the same section.
- Risk Factors text in the frozen release is cut at 50,000 characters for all three filings (extractor cap), so later risk factors are not in the corpus and not labelled.
- Status: frozen 2026-10-22 after owner review (D-A4). Any change to cases or labels requires a `fixture_version` bump, re-validation and a regenerated baseline (retrieval-benchmark §8, §10).
