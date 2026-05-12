"""Section intent inference for SEC filing retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from app.components.retrieval.sections import canonical_section_key

MAX_CONFIDENCE = 0.95
MIN_CONFIDENCE = 0.0
MAX_DEFAULT_SECTIONS = 3


class SectionIntent(BaseModel):
    """Inferred SEC section targets for a retrieval query."""

    query: str = Field(min_length=1)
    target_sections: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    section_scores: dict[str, float] = Field(default_factory=dict)


SECTION_SIGNAL_TERMS: dict[str, tuple[str, ...]] = {
    "business": (
        "business",
        "products",
        "product",
        "services",
        "customers",
        "customer",
        "markets",
        "distribution",
        "competition",
        "competitive",
        "research and development",
        "r&d",
        "segments",
        "platforms",
        "offerings",
        "operations",
    ),
    "risk_factors": (
        "risk factors",
        "risk factor",
        "supply chain risk",
        "supply chain",
        "supplier risk",
        "suppliers",
        "regulatory risk",
        "legal risk",
        "privacy risk",
        "cybersecurity risk",
        "data security",
        "geopolitical",
        "trade restrictions",
        "export controls",
        "manufacturing risk",
        "defects",
        "third-party",
        "environmental goals",
        "uncertainty",
        "volatility",
        "dependency",
    ),
    "md&a": (
        "md&a",
        "management discussion",
        "revenue",
        "net sales",
        "sales",
        "growth",
        "demand",
        "gross margin",
        "operating income",
        "operating expenses",
        "expenses",
        "cash flow",
        "liquidity",
        "capital resources",
        "capital expenditures",
        "capex",
        "share repurchases",
        "dividends",
        "results of operations",
        "performance",
    ),
    "market_risk": (
        "market risk",
        "foreign currency",
        "foreign exchange",
        "exchange rate",
        "exchange rates",
        "interest rate",
        "interest rates",
        "hedging",
        "hedge",
        "derivative",
        "derivatives",
        "investment portfolio",
        "investments",
        "securities",
        "fair value",
        "sensitivity analysis",
        "credit risk",
        "counterparties",
        "counterparty",
        "cash equivalents",
        "short-term investments",
    ),
}


def normalize_query_text(value: str) -> str:
    """Normalize query text for deterministic phrase matching."""

    token = value.strip().lower()
    token = token.replace("&", " and ")
    token = token.replace("'", "").replace("\u2019", "")
    token = re.sub(r"[^a-z0-9]+", " ", token)
    return re.sub(r"\s+", " ", token).strip()


def _term_weight(term: str) -> float:
    """Give multi-token phrases more weight than generic single words."""

    normalized = normalize_query_text(term)
    token_count = len(normalized.split())
    if token_count >= 3:
        return 2.5
    if token_count == 2:
        return 1.75
    return 1.0


def _matched_terms(query_token: str, terms: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for term in terms:
        term_token = normalize_query_text(term)
        if term_token and term_token in query_token:
            matches.append(term)
    return matches


def infer_section_intent(query: str, *, max_sections: int = MAX_DEFAULT_SECTIONS) -> SectionIntent:
    """Infer likely SEC filing sections for a query.

    Args:
        query: Natural-language retrieval query.
        max_sections: Maximum section keys to return.

    Returns:
        A deterministic Pydantic model with target section keys, confidence,
        matched terms, and raw section scores.
    """

    query_token = normalize_query_text(query)
    section_scores: dict[str, float] = {}
    all_matches: list[str] = []

    for raw_section_key, terms in SECTION_SIGNAL_TERMS.items():
        section_key = canonical_section_key(raw_section_key) or raw_section_key
        matches = _matched_terms(query_token, terms)
        if not matches:
            continue
        score = sum(_term_weight(match) for match in matches)
        section_scores[section_key] = score
        all_matches.extend(matches)

    ranked_sections = [
        section
        for section, _score in sorted(
            section_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ][:max_sections]

    best_score = max(section_scores.values(), default=0.0)
    confidence = (
        min(MAX_CONFIDENCE, best_score / (best_score + 2.0))
        if best_score > 0
        else MIN_CONFIDENCE
    )

    return SectionIntent(
        query=query,
        target_sections=ranked_sections,
        confidence=round(confidence, 4),
        matched_terms=sorted(set(all_matches), key=lambda item: normalize_query_text(item)),
        section_scores={key: round(value, 4) for key, value in section_scores.items()},
    )
