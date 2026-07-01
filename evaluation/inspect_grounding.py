"""Print one memo's claim-by-claim grounding verdicts for manual validation.

You are about to optimize grounded_claim_rate, so first confirm it's honest:
run this, read every UNGROUNDED line, and decide whether the cited evidence
genuinely fails to support the claim, or whether it's a valid paraphrase the
token-overlap heuristic simply missed. If most UNGROUNDED verdicts are unfair,
the fix is a better grounding check (semantic / LLM-judge), not more prompting.

    TICKER=AAPL python -m evaluation.inspect_grounding
"""

from __future__ import annotations

import asyncio
import json
import os

from app.agents.graph import create_agent
from app.agents.state import create_initial_state
from evaluation.grounding import evaluate_memo_grounding


async def main(ticker: str) -> None:
    agent = create_agent()
    state = create_initial_state(
        ticker, None, include_filing_analysis=True, include_news_sentiment=True, max_news_articles=10
    )
    final = await agent.ainvoke(state, config={})
    memo = final.get("investment_memo", "") or ""
    registry = final.get("citation_evidence", []) or []

    result = evaluate_memo_grounding(memo, registry).to_dict()
    print(f"coverage={result.get('citation_coverage_rate', 0):.2f}  "
          f"grounded={result.get('grounded_claim_rate', 0):.2f}  "
          f"claims={result.get('total_claims', 0)}\n")

    for i, c in enumerate(result.get("claims", []), 1):
        grounded = c.get("grounded", c.get("is_grounded"))
        cited = c.get("citations", c.get("cited"))
        text = (c.get("text") or "").strip()
        if not cited:
            flag = "UNCITED   "
        elif grounded:
            flag = "OK        "
        else:
            flag = "UNGROUNDED"
        print(f"[{i:>2}] {flag} cites={cited}")
        print(f"     {text}")
        # dump anything else the claim carries, so you can see what it was checked against
        extra = {k: v for k, v in c.items() if k not in {"text", "grounded", "is_grounded", "citations", "cited"}}
        if extra:
            print(f"     ~ {json.dumps(extra, default=str)[:300]}")
        print()

    print("\n----- REGISTRY (cited evidence) -----")
    for e in registry:
        print(f"[{e['index']}] {e.get('title','')} :: {(e.get('text') or '')[:160]}")


if __name__ == "__main__":
    asyncio.run(main(os.environ.get("TICKER", "AAPL")))
