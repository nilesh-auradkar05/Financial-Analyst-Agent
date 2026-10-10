# Observations on live runs

Item-level notes that explain rows in `live_runs.jsonl`. One section per observation, newest last.

## 2026-10-10 — NVDA run was degraded because the memo was badly cited

- **Run:** NVDA, label `ingest-on-miss`, LangSmith trace `01a1271a-f1fd-7e90-8668-91c5f7ac44b9`, model `global.anthropic.claude-sonnet-4-6`.
- **Result:** verification failed. Citation coverage 0.69 (46 of 67 claims cited); grounded 0.26 (12 of 46 cited claims).
- **Verdict:** the verifier was right. This was not an ingestion fault: ingest-on-miss ran (21.5 s) and five filing chunks reached the memo.
- **Control:** AAPL on the same build and day had 64 of 66 cited claims grounded (trace `01a1271f-99de-7161-a33f-69a0ad14bfc2`), so the checker is not broadly failing.

### Why the 34 cited claims failed

| Reason given by the verifier | Claims |
| --- | --- |
| Claim numbers not found in cited evidence | 25 |
| Claim does not align with cited evidence | 9 |

Three causes, read from the per-claim output:

1. **Wrong citation index.** The number is in the evidence, but in a different source than the one cited.
   - Market cap $5.54T cited to source 8 (news); it is in source 11 (market data).
   - $238.70 cited to source 9; it is in source 10.
   - Moving averages $221, $212, $211 and $197 cited to source 10; they are in source 3.
2. **Figures in no listed source.** $130.4, 60.4%, $120.1, $62.5, $72.9, $29.8, $167.6 and the $321.89 price target appear in none of the 17 registry entries given to the model.
3. **Thin filing evidence.** The five retrieved chunks were four from Business and one from Market Risk. None came from Risk Factors or the financial statements. 14 of 15 claims cited to filings failed.

### By cited source type

| Source type | Supported | Failed |
| --- | --- | --- |
| News | 10 | 19 |
| SEC filing | 1 | 14 |
| Market data | 2 | 5 |

A claim that cites two source types is counted under both.

### Not done

No fix was applied. Candidates:

- The bounded draft, verify, revise loop planned for S6, which would give the model a chance to correct wrong indices and drop unsupported figures.
- Filing retrieval that reaches Risk Factors and the financial sections, so numeric claims have filing evidence to cite.

### How to re-check

Compare the NVDA row's `verification` and `evidence.filing_sections` fields in `live_runs.jsonl` against a later NVDA run. A fix should raise `grounded_claim_rate` and add Risk Factors or financial sections to `filing_sections`.
