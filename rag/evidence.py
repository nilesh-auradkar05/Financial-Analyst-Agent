from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """Atomic retrieval unit used for grounding, verification, and citations."""

    ticker: str
    text: str
    chunk_id: str
    company_name: Optional[str] = None
    filing_type: Optional[str] = None
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    section_name: Optional[str] = None
    source_url: Optional[str] = None
    retrieval_score: float = 0.0
    retrieval_method: str = "dense"
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: Any,
        *,
        retrieval_method: str = "dense",
        rank: int = 0,
    ) -> "EvidencePacket":
        metadata = dict(getattr(chunk, "metadata", {}) or {})

        return cls(
            ticker=(metadata.get("ticker") or "").upper(),
            company_name=metadata.get("company_name"),
            filing_type=metadata.get("filing_type"),
            filing_date=metadata.get("filing_date"),
            accession_number=metadata.get("accession_number"),
            section_name=metadata.get("section_name") or metadata.get("section"),
            source_url=metadata.get("source_url"),
            text=getattr(chunk, "text", ""),
            chunk_id=metadata.get("chunk_id") or getattr(chunk, "id", ""),
            retrieval_score=float(getattr(chunk, "relevance_score", 0.0)),
            retrieval_method=retrieval_method,
            rank=rank,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "section_name": self.section_name,
            "source_url": self.source_url,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "retrieval_score": self.retrieval_score,
            "retrieval_method": self.retrieval_method,
            "rank": self.rank,
            "metadata": self.metadata,
        }

    def to_context(self, max_chars: int = 700) -> str:
        header_parts: list[str] = []
        if self.ticker:
            header_parts.append(f"Ticker: {self.ticker}")
        if self.section_name:
            header_parts.append(f"Section: {self.section_name}")
        if self.filing_date:
            header_parts.append(f"Filing: {self.filing_type}")
        if self.filing_date:
            header_parts.append(f"Filed: {self.filing_date}")
        if self.rank:
            header_parts.append(f"Rank: {self.rank}")
        if self.retrieval_score:
            header_parts.append(f"Score: {self.retrieval_score:.3f}")

        header = " | ".join(header_parts) if header_parts else "Evidence Packet"
        body = self.text[:max_chars].strip()
        return f"[{header}]\n{body}"
