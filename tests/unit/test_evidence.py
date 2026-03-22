from dataclasses import dataclass

from rag.evidence import EvidencePacket


@dataclass
class StubChunk:
    id: str
    text: str
    metadata: dict
    distance: float = 0.25

    @property
    def relevance_score(self) -> float:
        return 1 / (1 + self.distance)


def test_evidence_packet_maps_chunk_metadata():
    chunk = StubChunk(
        id="AAPL_10K_Risk_001",
        text="Apple faces supply chain concentration risks.",
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "filing_type": "10-K",
            "filing_date": "2025-11-01",
            "accession_number": "0000320193-25-000010",
            "section": "Risk Factors",
            "source_url": "https://www.sec.gov/example",
            "chunk_id": "AAPL_10K_Risk_001",
        },
    )

    packet = EvidencePacket.from_chunk(chunk, retrieval_method="dense", rank=2)

    assert packet.ticker == "AAPL"
    assert packet.company_name == "Apple Inc."
    assert packet.section_name == "Risk Factors"
    assert packet.chunk_id == "AAPL_10K_Risk_001"
    assert packet.retrieval_method == "dense"
    assert packet.rank == 2
    assert packet.retrieval_score > 0


def test_evidence_packet_context_is_human_readable():
    packet = EvidencePacket(
        ticker="MSFT",
        company_name="Microsoft",
        filing_type="10-K",
        filing_date="2025-08-12",
        section_name="Business",
        text="Microsoft reports strong Azure demand.",
        chunk_id="msft_business_001",
        retrieval_score=0.87,
        rank=1,
    )

    context = packet.to_context()

    assert "Ticker: MSFT" in context
    assert "Section: Business" in context
    assert "Microsoft reports strong Azure demand." in context
