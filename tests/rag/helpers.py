"""Shared helpers for the OpenSearch store tests (CM-2).

Raw index state is read over plain HTTP so the tests never import ``opensearchpy``:
the store under test is the only code that touches the backend client library.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from langchain_core.embeddings import Embeddings

from app.components.retrieval.vector_store import IndexDocument, RetrievalStore

REQUIRED_METADATA_KEYS = (
    "ticker",
    "filing_type",
    "filing_date",
    "accession_number",
    "section_key",
)

_TOKEN = re.compile(r"\w+")


class FakeEmbedder(Embeddings):
    """Deterministic provider-boundary fake: hashed bag-of-words plus a constant offset."""

    model = "fake-hash-embed"
    dimension = 16

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.05] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        return vector


def raw_request(base_url: str, method: str, path: str) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def raw_mapping(base_url: str, index: str) -> tuple[int, dict[str, Any]]:
    status, body = raw_request(base_url, "GET", f"/{index}/_mapping")
    properties = body.get(index, {}).get("mappings", {}).get("properties", {})
    return status, properties


def raw_document(base_url: str, index: str, document_id: str) -> tuple[int, dict[str, Any]]:
    return raw_request(base_url, "GET", f"/{index}/_doc/{quote(document_id, safe='')}")


def field_types(properties: dict[str, Any]) -> dict[str, set[str]]:
    """Map every (possibly nested) property's leaf name to the set of types it carries."""
    found: dict[str, set[str]] = {}
    for name, spec in properties.items():
        if "type" in spec:
            found.setdefault(name, set()).add(spec["type"])
        for child, types in field_types(spec.get("properties", {})).items():
            found.setdefault(child, set()).update(types)
    return found


def knn_dimensions(properties: dict[str, Any]) -> list[int]:
    dimensions: list[int] = []
    for spec in properties.values():
        if spec.get("type") == "knn_vector":
            dimensions.append(int(spec["dimension"]))
        dimensions.extend(knn_dimensions(spec.get("properties", {})))
    return dimensions


def all_chunk_ids(store: RetrievalStore) -> list[str]:
    return [chunk.id for chunk in store.search("all chunks", n_results=1000).chunks]


def find_backend_objects(value: Any, path: str = "value", *, _seen: set[int] | None = None) -> list[str]:
    """Locations holding an ``opensearchpy`` object, found by module name (public surface only)."""
    seen = _seen if _seen is not None else set()
    if id(value) in seen or isinstance(value, str | bytes | int | float | bool | type(None)):
        return []
    seen.add(id(value))
    if type(value).__module__.startswith("opensearchpy"):
        return [f"{path}: {type(value).__module__}.{type(value).__qualname__}"]

    children: dict[str, Any]
    if isinstance(value, dict):
        children = {f"{path}[{key!r}]": item for key, item in value.items()}
    elif isinstance(value, list | tuple | set | frozenset):
        children = {f"{path}[{index}]": item for index, item in enumerate(value)}
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        children = {
            f"{path}.{field.name}": getattr(value, field.name) for field in dataclasses.fields(value)
        }
    else:
        attributes = vars(value) if hasattr(value, "__dict__") else {}
        children = {
            f"{path}.{name}": item
            for name, item in attributes.items()
            if not name.startswith("_") and not callable(item)
        }

    found: list[str] = []
    for child_path, child in children.items():
        found.extend(find_backend_objects(child, child_path, _seen=seen))
    return found


def public_attribute_values(obj: Any) -> dict[str, Any]:
    """Values of every public non-method attribute (properties included) of ``obj``."""
    values: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        value = getattr(obj, name)
        if not callable(value):
            values[name] = value
    return values


def build_corpus() -> list[IndexDocument]:
    """Ten chunks across two tickers, a 10-K and a 10-Q, and four sections."""
    filings = [
        ("AAPL", "Apple Inc.", "10-K", "2025-09-28", "0000320193-25-000001"),
        ("AAPL", "Apple Inc.", "10-Q", "2025-05-02", "0000320193-25-000050"),
        ("MSFT", "Microsoft Corporation", "10-K", "2025-06-30", "0000789019-25-000001"),
    ]
    sections = {
        "business": "Item 1. Business",
        "risk_factors": "Item 1A. Risk Factors",
        "md&a": "Item 7. Management's Discussion and Analysis",
        "market_risk": "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
    }
    layout = {
        ("AAPL", "10-K"): {"business": 2, "risk_factors": 2, "md&a": 1},
        ("AAPL", "10-Q"): {"risk_factors": 1},
        ("MSFT", "10-K"): {"business": 1, "risk_factors": 2, "market_risk": 1},
    }
    texts = {
        "business": "designs manufactures markets smartphones personal computers tablets wearables "
        "services subscriptions retail stores channel partners",
        "risk_factors": "supply chain disruption competition regulation litigation cybersecurity "
        "currency exposure macroeconomic downturn component shortage",
        "md&a": "net sales increased gross margin operating expenses liquidity capital resources "
        "share repurchases dividends fiscal year comparison",
        "market_risk": "interest rate sensitivity foreign exchange hedging investment portfolio "
        "duration value at risk equity price",
    }

    documents: list[IndexDocument] = []
    for ticker, company, filing_type, filing_date, accession in filings:
        for section_key, chunk_count in layout[(ticker, filing_type)].items():
            for index in range(chunk_count):
                chunk_id = f"{accession}_{section_key}_{index:03d}"
                documents.append(
                    IndexDocument(
                        id=chunk_id,
                        text=(
                            f"{company} {filing_type} {section_key} passage {index}: "
                            f"{texts[section_key]} ref {accession} part{index}"
                        ),
                        metadata={
                            "ticker": ticker,
                            "company_name": company,
                            "filing_type": filing_type,
                            "filing_date": filing_date,
                            "accession_number": accession,
                            "source_url": f"https://www.sec.gov/Archives/{accession}",
                            "section": sections[section_key],
                            "section_name": sections[section_key],
                            "section_key": section_key,
                            "chunk_id": chunk_id,
                            "chunk_index": index,
                        },
                    )
                )
    return documents
