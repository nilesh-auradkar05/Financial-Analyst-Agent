"""Shared helpers for the S7.1 guardrail tests (docs/test-plan.md section 12, S7.1 rows).

The output screen is fed by the verifier, so these helpers run the real verifier on a memo and
hand the screen exactly what the graph hands it: the sentences the verifier reports as not
supported and as missing a citation. Nothing here types a verifier answer in by hand.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.agents.state import create_initial_state
from app.verification.grounding import evaluate_memo_grounding


def registry_of(*texts: str) -> list[dict[str, Any]]:
    """A citation registry with one news entry per text, numbered from 1."""
    return [
        {
            "index": number,
            "source_type": "news",
            "title": "Company posts results",
            "text": text,
            "url": None,
            "date": None,
        }
        for number, text in enumerate(texts, start=1)
    ]


def verifier_flags(memo: str, registry: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """(unsupported sentences, sentences missing a citation) exactly as the verifier reports them."""
    claims = evaluate_memo_grounding(memo, registry).to_dict()["claims"]
    unsupported = {claim["sentence"] for claim in claims if not claim["supported"]}
    uncited = {claim["sentence"] for claim in claims if claim["missing_citation"]}
    return unsupported, uncited


def flagged_uncited(memo: str, registry: list[dict[str, Any]], fragment: str) -> bool:
    """True when the verifier marks a sentence containing `fragment` as missing a citation."""
    return any(fragment in sentence for sentence in verifier_flags(memo, registry)[1])


def flagged_unsupported(memo: str, registry: list[dict[str, Any]], fragment: str) -> bool:
    """True when the verifier does not mark a sentence containing `fragment` as supported."""
    return any(fragment in sentence for sentence in verifier_flags(memo, registry)[0])


def screened(memo: str, registry: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Run the output screen on `memo` with the real verifier's flags for it."""
    from app.guardrails.output import screen_memo

    unsupported, uncited = verifier_flags(memo, registry)
    result: tuple[str, list[dict[str, Any]]] = screen_memo(memo, unsupported, uncited)
    return result


def codes(events: list[dict[str, Any]]) -> list[str]:
    return sorted(event["code"] for event in events)


def data_spans(prompt: str) -> list[str]:
    """The text between each DATA_OPEN and the DATA_CLOSE that follows it."""
    from app.guardrails.content import DATA_CLOSE, DATA_OPEN

    spans = []
    position = 0
    while (start := prompt.find(DATA_OPEN, position)) != -1:
        end = prompt.find(DATA_CLOSE, start + len(DATA_OPEN))
        assert end != -1, "a DATA block was opened and never closed"
        spans.append(prompt[start + len(DATA_OPEN) : end])
        position = end + len(DATA_CLOSE)
    return spans


def outside_data(prompt: str) -> str:
    """The prompt with every DATA block (delimiters included) removed."""
    from app.guardrails.content import DATA_CLOSE, DATA_OPEN

    pieces = []
    position = 0
    while (start := prompt.find(DATA_OPEN, position)) != -1:
        end = prompt.find(DATA_CLOSE, start + len(DATA_OPEN))
        assert end != -1, "a DATA block was opened and never closed"
        pieces.append(prompt[position:start])
        position = end + len(DATA_CLOSE)
    pieces.append(prompt[position:])
    return "".join(pieces)


class CapturingLLM:
    """Records every prompt it is given and returns a harmless memo."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1]["content"])
        return SimpleNamespace(content="## Executive Summary\nNothing further to add.")


def evidence_state(
    *,
    title="Apple posts results",
    source="Reuters",
    snippet="Apple revenue rose.",
    section="Business",
    filing_text="Apple designs and sells devices.",
):
    state = create_initial_state("AAPL", "Apple Inc.")
    state["news_articles"] = [
        {
            "title": title,
            "source": source,
            "snippet": snippet,
            "published_date": "2026-03-15",
            "url": "https://www.reuters.com/apple-results",
        }
    ]
    state["filing_chunks"] = [
        {
            "chunk_id": "chunk-1",
            "section": section,
            "filing_type": "10-K",
            "filing_date": "2025-09-28",
            "text": filing_text,
        }
    ]
    return state
