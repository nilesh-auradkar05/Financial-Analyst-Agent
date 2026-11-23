"""Filing result cache: the chunks one filings step returned, kept for an hour (test-plan §13)."""

from __future__ import annotations

import time

from app.components.retrieval.vector_store import RetrievedChunk

FILING_CACHE_TTL_SECONDS = 3600.0

_Key = tuple[str, str, tuple[object, ...]]


# ponytail: per-process, so a second process or a restart starts cold. S7.9 moves this into the
# shared cache store (SPEC §11.3 RetrievalCache).
class FilingResultCache:
    """Final filing chunks by ticker, company name and retrieval configuration."""

    def __init__(self) -> None:
        self._entries: dict[_Key, tuple[float, list[RetrievedChunk]]] = {}

    def get(
        self, ticker: str, company_name: str, config: tuple[object, ...]
    ) -> list[RetrievedChunk] | None:
        key = (ticker.upper(), company_name, config)
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, chunks = entry
        if time.monotonic() - stored_at >= FILING_CACHE_TTL_SECONDS:
            del self._entries[key]
            return None
        return chunks

    def put(
        self,
        ticker: str,
        company_name: str,
        config: tuple[object, ...],
        chunks: list[RetrievedChunk],
    ) -> None:
        self._entries[(ticker.upper(), company_name, config)] = (time.monotonic(), chunks)

    def invalidate(self, ticker: str) -> None:
        """Drop every entry of the ticker."""
        ticker = ticker.upper()
        for key in [key for key in self._entries if key[0] == ticker]:
            del self._entries[key]


FILING_CACHE = FilingResultCache()
