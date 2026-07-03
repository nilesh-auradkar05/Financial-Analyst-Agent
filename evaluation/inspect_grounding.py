"""Validate + calibrate the semantic grounding oracle against your hand labels.

Runs ONE memo, then for every claim shows the token-overlap verdict next to the
semantic cosine similarity, and sweeps the similarity threshold. Use it like this:
  1. Read each claim and compare the SEM_SIM column to your own fair/unfair judgment.
  2. Find the threshold that best separates the claims you judged supported from the
     ones you judged genuinely unsupported.
  3. Set that value as min_similarity in grounding_semantic.py.

Numeric claims still require a number match (NUM column); market-metric claims will
stay ungrounded here until the registry gap is fixed - that's expected.

    TICKER=AAPL python -m evaluation.inspect_grounding
"""

from __future__ import annotations

import asyncio
import os

from app.agents.graph import create_agent
from app.agents.state import create_initial_state
from app.components.retrieval.embeddings import embed_texts
from evaluation.grounding import (
    _build_evidence_map,
    _extract_citations,
    _extract_claim_sentences,
    _numbers_supported_any,
    _strip_citations,
    evaluate_memo_grounding,
)
from evaluation.semantic_grounding import _cosine


async def main(ticker: str) -> None:
    agent = create_agent()
    state = create_initial_state(
        ticker, None, include_filing_analysis=True, include_news_sentiment=True, max_news_articles=10
    )
    final = await agent.ainvoke(state, config={})
    memo = final.get("investment_memo", "") or ""
    registry = final.get("citation_evidence", []) or []

    token = evaluate_memo_grounding(memo, registry)
    claims = _extract_claim_sentences(memo)
    cleaned = [_strip_citations(s) for s in claims]
    cites = [_extract_citations(s) for s in claims]
    emap = _build_evidence_map(registry)

    claim_vecs = embed_texts(cleaned) if cleaned else []
    ev_idx = list(emap.keys())
    ev_vecs = embed_texts([emap[i].text for i in ev_idx]) if ev_idx else []
    evmap = dict(zip(ev_idx, ev_vecs))

    print(f"TOKEN checker: coverage={token.citation_coverage_rate:.2f} grounded={token.grounded_claim_rate:.2f}\n")
    print(f"{'#':>3} {'TOKEN':<11} {'SEM_SIM':>7} {'NUM':>4}  claim (cites)")

    cited_scores: list[tuple[float, bool]] = []
    for i, (clean, cvec, cite, ta) in enumerate(zip(cleaned, claim_vecs, cites, token.claims), 1):
        best_sim, num_ok = 0.0, True
        unmatched: list[str] = []
        if cite:
            best_sim = 0.0
            cited_texts = [emap[idx].text for idx in cite if idx in emap]
            for idx in cite:
                ev = evmap.get(idx)
                if ev is None:
                    continue
                s = _cosine(cvec, ev)
                if s > best_sim:
                    best_sim = s
            # Number gate is decoupled from best-similarity evidence: supported
            # if ANY cited evidence (not just the highest-similarity one) has
            # the number.
            num_ok, unmatched = _numbers_supported_any(clean, cited_texts)
            cited_scores.append((best_sim, num_ok))
        tok_flag = "UNCITED" if ta.missing_citation else ("OK" if ta.supported else "UNGROUNDED")
        # Unmatched numbers are printed verbatim so a hand audit can jump straight
        # to the offending figure instead of re-deriving it from the claim text.
        num_note = f"  MISSING: {', '.join(unmatched)}" if unmatched else ""
        print(f"{i:>3} {tok_flag:<11} {best_sim:>7.2f} {('y' if num_ok else 'n'):>4}  {clean[:88]} {cite}{num_note}")

    print("\nsemantic grounded_claim_rate by threshold (over cited claims):")
    for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        if cited_scores:
            g = sum(1 for sim, num_ok in cited_scores if sim >= t and num_ok) / len(cited_scores)
        else:
            g = 0.0
        print(f"  thr {t:.2f} -> grounded {g:.2f}")

    print("\n----- REGISTRY -----")
    for e in registry:
        print(f"[{e['index']}] {e.get('title','')} :: {(e.get('text') or '')[:150]}")


if __name__ == "__main__":
    asyncio.run(main(os.environ.get("TICKER", "AAPL")))
