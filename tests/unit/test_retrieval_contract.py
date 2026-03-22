from rag.evidence import EvidencePacket
from rag.vector_store import RetrievedChunk, SearchFilters, SearchResult


def test_search_filters_map_section_name_to_existing_collection_key():
    filters = SearchFilters(
        ticker="aapl",
        filing_type="10-K",
        section_name="Risk Factors",
        filing_date="2025-11-01",
        extra={"custom": "value"},
    )

    where = filters.to_backend_filter()

    assert where == {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section": "Risk Factors",
        "filing_date": "2025-11-01",
        "custom": "value",
    }


def test_search_result_converts_chunks_to_evidence_packets():
    result = SearchResult(
        query="apple risks",
        chunks=[
            RetrievedChunk(
                id="chunk-1",
                text="Risk factor text",
                metadata={
                    "ticker": "AAPL",
                    "section": "Risk Factors",
                    "filing_type": "10-K",
                    "chunk_id": "chunk-1",
                },
                distance=0.1,
            ),
            RetrievedChunk(
                id="chunk-2",
                text="Business overview text",
                metadata={
                    "ticker": "AAPL",
                    "section": "Business",
                    "filing_type": "10-K",
                    "chunk_id": "chunk-2",
                },
                distance=0.2,
            ),
        ],
        total_results=2,
    )

    packets = result.to_evidence_packets(retrieval_method="dense")

    assert len(packets) == 2
    assert all(isinstance(packet, EvidencePacket) for packet in packets)
    assert packets[0].rank == 1
    assert packets[1].rank == 2
    assert packets[0].section_name == "Risk Factors"
    assert packets[1].section_name == "Business"