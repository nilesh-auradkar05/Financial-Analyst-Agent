from app.models import (
    AnalysisResponse,
    CitationResponse,
    ErrorDetail,
    JobStatus,
    NewsArticleResponse,
    VerificationClaimResponse,
)


def test_analysis_response_collection_defaults_are_not_shared():
    first = AnalysisResponse(
        ticker="AAPL",
        company_name="Apple Inc.",
        status=JobStatus.COMPLETED,
    )
    second = AnalysisResponse(
        ticker="MSFT",
        company_name="Microsoft Corp.",
        status=JobStatus.COMPLETED,
    )

    first.news_articles.append(
        NewsArticleResponse(
            title="stub",
            url="https://example.com/stub",
            source="stub",
            snippet="stub",
        )
    )
    first.citations.append(
        CitationResponse(index=1, source_type="news", title="stub")
    )
    first.errors.append(
        ErrorDetail(step="stub", message="stub", timestamp="now")
    )

    assert second.news_articles == []
    assert second.citations == []
    assert second.errors == []


def test_verification_claim_citation_default_is_not_shared():
    first = VerificationClaimResponse(
        sentence="Revenue grew.",
        supported=True,
        overlap_score=0.9,
    )
    second = VerificationClaimResponse(
        sentence="Margins compressed.",
        supported=False,
        overlap_score=0.1,
    )

    first.citations.append(1)

    assert second.citations == []
