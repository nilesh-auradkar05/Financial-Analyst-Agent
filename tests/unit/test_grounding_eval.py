from __future__ import annotations

from evaluation.grounding import evaluate_memo_grounding


class StubEvidence:
    def __init__(self, text: str, chunk_id: str) -> None:
        self.text = text
        self.chunk_id = chunk_id
        self.metadata = {"chunk_id": chunk_id}

def test_grounded_claims_pass_with_full_citation_coverage():
    memo = (
        "Apple reported revenue of $394 billion [1]. "
        "Supply chain concentration in China remains a risk [2]."
    )
    evidence = [
        StubEvidence("Apple reported net sales of $394 billion in fiscal 2024.", "chunk-1"),
        StubEvidence(
            "Risks include supply chain concentration in China and dependence on manufacturing partners.",
            "chunk-2",
        ),
    ]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.passed is True
    assert result.total_claims == 2
    assert result.citation_coverage_rate == 1.0
    assert result.grounded_claim_rate == 1.0

def test_uncited_claim_reduces_citation_coverage():
    memo = (
        "Apple reported revenue of $394 billion. "
        "Gross margin was 44% [1]."
    )
    evidence = [StubEvidence("Gross margin was 44% in fiscal 2024.", "chunk-1")]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.passed is False
    assert result.total_claims == 2
    assert result.citation_coverage_rate == 0.5
    assert any(claim.missing_citation for claim in result.claims)

def test_unsupported_claim_is_flagged_even_when_cited():
    memo = "Apple faces major supply chain risk in China [1]."
    evidence = [
        StubEvidence(
            "Apple launched new products and expanded services revenue during the year.",
            "chunk-1",
        )
    ]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.passed is False
    assert result.grounded_claim_rate == 0.0
    assert result.claims[0].supported is False
    assert "align" in (result.claims[0].reason or "")

def test_numeric_mismatch_is_not_treated_as_grounded():
    memo = "Apple reported revenue of $500 billion [1]."
    evidence = [StubEvidence("Apple reported revenue of $394 billion in fiscal 2024.", "chunk-1")]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.passed is False
    assert result.claims[0].supported is False
    assert "numbers" in (result.claims[0].reason or "")
