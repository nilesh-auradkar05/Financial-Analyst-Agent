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

def test_markdown_header_lines_are_not_assessed_as_claims():
    # Section headers are structure, not analytical claims.  A header that
    # contains a period ("## 3. Financial Analysis: Revenue Detail") must not be
    # fragmented into a bogus uncited claim ("Financial Analysis: Revenue
    # Detail") that deflates citation coverage.  Test-plan §7: only grounded
    # claims tied to evidence are counted.
    memo = (
        "# Investment Memo: Apple Inc. (AAPL)\n\n"
        "## 3. Financial Analysis: Revenue Detail\n\n"
        "Apple reported revenue of $394 billion [1].\n\n"
        "## Risks and Outlook\n\n"
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

    assert result.total_claims == 2
    assert result.citation_coverage_rate == 1.0
    assert all(not claim.missing_citation for claim in result.claims)
    assert "Financial Analysis" not in " ".join(claim.sentence for claim in result.claims)

def test_numeric_mismatch_is_not_treated_as_grounded():
    memo = "Apple reported revenue of $500 billion [1]."
    evidence = [StubEvidence("Apple reported revenue of $394 billion in fiscal 2024.", "chunk-1")]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.passed is False
    assert result.claims[0].supported is False
    assert "numbers" in (result.claims[0].reason or "")

def test_rounded_value_is_treated_as_grounded():
    # The analyst rounds ("$37.32"); the source is exact ("37.319225"). Rounding
    # to the claim's own stated precision must count as grounded.
    memo = "Apple trades at $37.32 per share [1]."
    evidence = [StubEvidence("Apple current price is 37.319225 per share.", "chunk-1")]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.claims[0].supported is True
    assert result.claims[0].numbers_ok is True

def test_hallucinated_number_is_rejected_and_named_in_reason():
    memo = "Analysts set a $300 price target with 12.5% upside potential [1]."
    evidence = [StubEvidence("Nothing in this evidence matches those figures.", "chunk-1")]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.claims[0].supported is False
    assert result.claims[0].numbers_ok is False
    assert result.claims[0].reason == "claim numbers not found in cited evidence: $300, 12.5%"

def test_number_found_in_second_cited_evidence_counts_as_supported():
    # Regression for the number-support/best-similarity coupling bug: [1] is the
    # topically closer (higher token-overlap) evidence but lacks the figure;
    # [2] is a weaker topical match yet carries the exact number. The claim's
    # numbers must be judged supported because the number is present in ANY
    # cited evidence, not just the evidence with the highest overlap score.
    memo = "Apple gross margin improved to 46.2% during the quarter [1][2]."
    evidence = [
        StubEvidence(
            "Apple gross margin improved due to strong pricing during the quarter.",
            "chunk-1",
        ),
        StubEvidence("Reported figure: 46.2%.", "chunk-2"),
    ]

    result = evaluate_memo_grounding(memo, evidence)

    assert result.claims[0].numbers_ok is True
    assert result.claims[0].supported is True
