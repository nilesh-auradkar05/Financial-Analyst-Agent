"""S5.2: hybrid dense + BM25 search on OpenSearch (test-plan §5 "Hybrid search" rows).

Runs against the local OpenSearch test container. Vectors come from the 16-d hashed bag-of-words
fake embedder (it computes a vector from the text, so similarity is a function of the words);
BM25 and the search pipeline are the container's own. The pipeline is read back over plain HTTP.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from tests.rag.helpers import FakeEmbedder, raw_request

PIPELINE_PATH = "/_search/pipeline/hybrid-minmax-mean-v1"
FILTER_QUERY = "supply chain disruption risk"


def _document(
    chunk_id: str,
    text: str,
    *,
    ticker: str = "AAPL",
    filing_type: str = "10-K",
    section_key: str = "risk_factors",
) -> IndexDocument:
    accession = f"{ticker}-{filing_type}"
    return IndexDocument(
        id=chunk_id,
        text=text,
        metadata={
            "ticker": ticker,
            "company_name": ticker,
            "filing_type": filing_type,
            "filing_date": "2025-01-31",
            "accession_number": accession,
            "source_url": f"https://www.sec.gov/Archives/{accession}",
            "section": section_key,
            "section_name": section_key,
            "section_key": section_key,
            "chunk_id": chunk_id,
            "chunk_index": 0,
        },
    )


def _colliding_words(embedder: FakeEmbedder, term: str, count: int) -> list[str]:
    """Words the fake embeds exactly as it embeds `term` (same vector), but that are other words."""
    target = embedder.embed_query(term)
    words = (f"note{number}" for number in range(10_000))
    return [word for word in words if embedder.embed_query(word) == target][:count]


def _keyword_reach_corpus(embedder: FakeEmbedder, term: str) -> list[IndexDocument]:
    """Twelve chunks packed with words that embed like `term` (dense-near to a query for it) and one
    long chunk that holds `term` itself among unrelated words (dense-far, BM25's only match)."""
    near = _colliding_words(embedder, term, 4)
    unrelated = (
        "designs manufactures markets smartphones personal computers tablets wearables services "
        "subscriptions retail stores channel partners net sales increased gross margin operating "
        "expenses liquidity capital resources share repurchases dividends fiscal year comparison"
    )
    filler = ("supply chain", "currency exposure", "regulation litigation", "component shortage")
    documents = [
        _document(
            f"near-{number:02d}",
            f"{filler[number % 4]} {' '.join(near)} {' '.join(unrelated.split()[number : number + 3])}",
        )
        for number in range(12)
    ]
    documents.append(_document("rare-term", f"{unrelated} {term}", section_key="business"))
    return documents


def test_pipeline_id_is_the_declared_constant() -> None:
    """test-plan §5 Hybrid search: pipeline (S5.2) - the search pipeline has the id declared in
    retrieval-benchmark §7b's implementation (hybrid-minmax-mean-v1)."""
    from app.components.retrieval.opensearch_store import HYBRID_SEARCH_PIPELINE

    assert HYBRID_SEARCH_PIPELINE == "hybrid-minmax-mean-v1"


def test_hybrid_returns_the_exact_term_chunk_that_dense_ranks_below_n_results(
    index_name, make_store, embedder
) -> None:
    """test-plan §5 Hybrid search: keyword reach (S5.2) - a chunk that contains the query's exact
    rare term but that dense search() ranks below n_results is returned within n_results by
    search_hybrid()."""
    term, n_results = "zyxquantum", 5
    store = make_store(index_name)
    store.add_documents(_keyword_reach_corpus(embedder, term))

    dense_ids = [chunk.id for chunk in store.search(term, n_results=n_results).chunks]
    hybrid_ids = [chunk.id for chunk in store.search_hybrid(term, n_results=n_results).chunks]

    assert len(dense_ids) == n_results
    assert "rare-term" not in dense_ids, "precondition: dense search must miss the exact-term chunk"
    assert "rare-term" in hybrid_ids
    assert len(hybrid_ids) == n_results


_FILTER_CASES = [
    pytest.param({"ticker": "AAPL"}, id="ticker"),
    pytest.param({"ticker": "AAPL", "filing_type": "10-Q"}, id="ticker+filing_type"),
    pytest.param({"ticker": "MSFT", "section_key": "business"}, id="ticker+section_key"),
]


def _filter_corpus() -> list[IndexDocument]:
    """For every ticker x filing type x section: one chunk that repeats the query word for word
    (MSFT: nearest in vector space and top in BM25; AAPL: the query plus one extra word, so just
    behind) and three related chunks."""
    documents: list[IndexDocument] = []
    for ticker in ("AAPL", "MSFT"):
        for filing_type in ("10-K", "10-Q"):
            for section_key in ("risk_factors", "business"):
                prefix = f"{ticker}-{filing_type}-{section_key}"
                documents.append(
                    _document(
                        f"{prefix}-exact",
                        FILTER_QUERY if ticker == "MSFT" else f"{FILTER_QUERY} annual",
                        ticker=ticker,
                        filing_type=filing_type,
                        section_key=section_key,
                    )
                )
                for number, text in enumerate(
                    ("supply chain exposure", "disruption risk outlook", "chain risk factors")
                ):
                    documents.append(
                        _document(
                            f"{prefix}-{number}",
                            text,
                            ticker=ticker,
                            filing_type=filing_type,
                            section_key=section_key,
                        )
                    )
    return documents



@pytest.mark.parametrize("filter_fields", _FILTER_CASES)
def test_filters_apply_to_both_subqueries(index_name, make_store, filter_fields) -> None:
    """test-plan §5 Hybrid search: filters on both subqueries (S5.2) - with a ticker (and
    filing-type or section) filter, search_hybrid() never returns a chunk outside the filter, even
    one that matches the query word for word and is nearest in vector space."""
    store = make_store(index_name)
    store.add_documents(_filter_corpus())
    filters = SearchFilters(**filter_fields)

    unfiltered = store.search_hybrid(FILTER_QUERY, n_results=8).chunks
    filtered = store.search_hybrid(FILTER_QUERY, filters=filters, n_results=3).chunks

    def inside(metadata: dict[str, Any]) -> bool:
        return all(
            metadata[name] == value.upper() if name == "ticker" else metadata[name] == value
            for name, value in filter_fields.items()
        )

    assert any(not inside(chunk.metadata) for chunk in unfiltered), (
        "precondition: without the filter, word-for-word chunks outside it rank in the top 8"
    )
    assert len(filtered) == 3
    assert all(inside(chunk.metadata) for chunk in filtered)


def test_hybrid_results_are_ordered_by_fused_score_and_match_dense_text_and_metadata(
    index_name, make_store, corpus
) -> None:
    """test-plan §5 Hybrid search: scores (S5.2) - results come in non-increasing fused score order;
    each chunk's distance is 1 - fused score with the fused score in [0, 1]; chunks carry the same
    text and metadata as search() returns for them."""
    store = make_store(index_name)
    store.add_documents(corpus)
    query = "supply chain disruption competition regulation"
    dense = {chunk.id: chunk for chunk in store.search(query, n_results=len(corpus)).chunks}

    result = store.search_hybrid(query, n_results=6)

    assert result.query == query
    assert len(result.chunks) == 6
    fused = [1.0 - chunk.distance for chunk in result.chunks]
    assert fused == sorted(fused, reverse=True)
    assert all(-1e-6 <= score <= 1.0 + 1e-6 for score in fused)
    assert len({chunk.id for chunk in result.chunks}) == 6
    for chunk in result.chunks:
        assert chunk.text == dense[chunk.id].text
        assert chunk.metadata == dense[chunk.id].metadata


def test_a_query_equal_to_a_chunk_ranks_it_first_with_full_fused_score(
    index_name, make_store, corpus
) -> None:
    """test-plan §5 Hybrid search: scores (S5.2) - a chunk that tops both subqueries is normalised
    to 1.0 in each, so its fused score is 1 and its distance 0 (min_max, arithmetic_mean)."""
    store = make_store(index_name)
    store.add_documents(corpus)
    target = corpus[4]

    result = store.search_hybrid(target.text, n_results=5)

    assert result.chunks[0].id == target.id
    assert result.chunks[0].distance == pytest.approx(0.0, abs=1e-4)


def test_hybrid_search_on_a_missing_index_returns_an_empty_result(
    opensearch_url, new_index, make_store
) -> None:
    """test-plan §5 Hybrid search: missing index (S5.2) - search_hybrid() on an index that does not
    exist returns an empty result, as search() does."""
    dense_index, hybrid_index = new_index(), new_index()
    dense_store, hybrid_store = make_store(dense_index), make_store(hybrid_index)

    dense = dense_store.search("supply chain", n_results=5)
    hybrid = hybrid_store.search_hybrid("supply chain", n_results=5)
    filtered = hybrid_store.search_hybrid(
        "supply chain", filters=SearchFilters(ticker="AAPL"), n_results=5
    )

    assert dense.chunks == []
    assert hybrid.chunks == []
    assert hybrid.total_results == dense.total_results == 0
    assert hybrid.query == "supply chain"
    assert filtered.chunks == []
    created_by_dense = raw_request(opensearch_url, "HEAD", f"/{dense_index}")[0] == 200
    created_by_hybrid = raw_request(opensearch_url, "HEAD", f"/{hybrid_index}")[0] == 200
    assert created_by_hybrid == created_by_dense


def _read_pipeline(opensearch_url: str) -> tuple[int, dict[str, Any]]:
    return raw_request(opensearch_url, "GET", PIPELINE_PATH)


def test_first_hybrid_search_creates_the_pipeline_and_repeats_are_not_errors(
    opensearch_url, index_name, make_store, corpus
) -> None:
    """test-plan §5 Hybrid search: pipeline (S5.2) - the first hybrid search creates the search
    pipeline if absent; creating it again (a second store, a second call) is not an error; the
    pipeline read back is normalization-processor, min_max, arithmetic_mean, weights [0.5, 0.5]."""
    store = make_store(index_name)
    store.add_documents(corpus)
    raw_request(opensearch_url, "DELETE", PIPELINE_PATH)
    status_before, _ = _read_pipeline(opensearch_url)

    first = store.search_hybrid("supply chain disruption", n_results=3)
    status_after_first, created = _read_pipeline(opensearch_url)
    second = store.search_hybrid("supply chain disruption", n_results=3)
    another = make_store(index_name).search_hybrid("supply chain disruption", n_results=3)
    status_after_all, final = _read_pipeline(opensearch_url)

    assert status_before == 404
    assert status_after_first == status_after_all == 200
    assert [chunk.id for chunk in first.chunks] == [chunk.id for chunk in second.chunks]
    assert [chunk.id for chunk in first.chunks] == [chunk.id for chunk in another.chunks]
    assert final == created

    processors = created["hybrid-minmax-mean-v1"]["phase_results_processors"]
    normalization = [entry["normalization-processor"] for entry in processors]
    assert len(normalization) == 1
    assert normalization[0]["normalization"]["technique"] == "min_max"
    assert normalization[0]["combination"]["technique"] == "arithmetic_mean"
    assert normalization[0]["combination"]["parameters"]["weights"] == pytest.approx([0.5, 0.5])
