"""Semantic grounding evaluator.

Drop-in alternative to evaluate_memo_grounding that replaces token-overlap with
embedding cosine similarity for the text-grounding decision, fixing the paraphrase
false-positives you found by hand (faithful 10-K rewordings scored as ungrounded).

It REUSES the token-overlap checker's claim extraction, citation parsing, evidence
map, and number check, so the claim set is identical and old-vs-new verdicts are
directly comparable (that comparison is how you validate this new oracle).

Unchanged on purpose:
  - the number check still gates numeric claims (a "$3.5T market cap" claim must have
    that number in its cited source). Market-metric claims will therefore STILL flag
    ungrounded until the registry gap is fixed — that's correct, and it's the residual
    this checker is supposed to leave visible.

Known limitation: cosine similarity is not entailment. A claim and its evidence can be
lexically close but contradictory ("margins improved" vs "margins under pressure").
This is a big improvement over token overlap, not the final oracle — an LLM/NLI judge is
the eventual rigorous Gate-D checker. Validate this one by hand before trusting it.

Place at: evaluation/grounding_semantic.py
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from app.components.retrieval.embeddings import embed_texts
from evaluation.grounding import (
    ClaimAssessment,
    GroundingCheckResult,
    _build_evidence_map,
    _extract_citations,
    _extract_claim_sentences,
    _extract_numbers,
    _numbers_supported,
    _strip_citations,
)

EmbedFn = Callable[[list[str]], list[list[float]]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def evaluate_memo_grounding_semantic(
    memo: str,
    evidence: Sequence[object],
    *,
    embed_fn: EmbedFn = embed_texts,
    min_citation_coverage: float = 0.80,
    min_grounded_claim_rate: float = 0.75,
    min_similarity: float = 0.45,
) -> GroundingCheckResult:
    """Grounding via embedding cosine similarity between each claim and its cited evidence.

    `min_similarity` is the one free parameter and MUST be calibrated against your own
    hand-labeled memo (see inspect_grounding threshold sweep) before you trust the number.
    """
    claims = _extract_claim_sentences(memo)
    evidence_map = _build_evidence_map(evidence)

    cleaned = [_strip_citations(s) for s in claims]
    citations_per = [_extract_citations(s) for s in claims]

    # Batch-embed all claims once and all cited evidence once.
    claim_vecs = embed_fn(cleaned) if cleaned else []
    ev_indices = list(evidence_map.keys())
    ev_vecs_list = embed_fn([evidence_map[i].text for i in ev_indices]) if ev_indices else []
    ev_vec_map = dict(zip(ev_indices, ev_vecs_list))

    assessments: list[ClaimAssessment] = []
    cited_claims = 0
    grounded_claims = 0

    for cleaned_sentence, citations, cvec in zip(cleaned, citations_per, claim_vecs):
        if not citations:
            assessments.append(
                ClaimAssessment(
                    sentence=cleaned_sentence,
                    citations=[],
                    supported=False,
                    overlap_score=0.0,
                    missing_citation=True,
                    reason="claim is missing a citation",
                )
            )
            continue

        cited_claims += 1
        best_sim = 0.0
        best_number_match = False

        for idx in citations:
            evec = ev_vec_map.get(idx)
            if evec is None:
                continue
            sim = _cosine(cvec, evec)
            number_match = _numbers_supported(cleaned_sentence, evidence_map[idx].text)
            if sim > best_sim:
                best_sim = sim
                best_number_match = number_match

        supported = best_sim >= min_similarity and best_number_match
        if supported:
            grounded_claims += 1
            assessments.append(
                ClaimAssessment(
                    sentence=cleaned_sentence,
                    citations=citations,
                    supported=True,
                    overlap_score=best_sim,
                )
            )
        else:
            reason = "claim not semantically supported by cited evidence"
            if _extract_numbers(cleaned_sentence) and not best_number_match:
                reason = "claim numbers do not match cited evidence"
            assessments.append(
                ClaimAssessment(
                    sentence=cleaned_sentence,
                    citations=citations,
                    supported=False,
                    overlap_score=best_sim,
                    reason=reason,
                )
            )

    total_claims = len(assessments)
    citation_coverage_rate = cited_claims / total_claims if total_claims > 0 else 0.0
    grounded_claim_rate = grounded_claims / cited_claims if cited_claims > 0 else 0.0
    passed = (
        total_claims > 0
        and citation_coverage_rate >= min_citation_coverage
        and grounded_claim_rate >= min_grounded_claim_rate
    )

    return GroundingCheckResult(
        passed=passed,
        total_claims=total_claims,
        cited_claims=cited_claims,
        grounded_claims=grounded_claims,
        citation_coverage_rate=citation_coverage_rate,
        grounded_claim_rate=grounded_claim_rate,
        claims=assessments,
    )


__all__ = ["evaluate_memo_grounding_semantic"]
