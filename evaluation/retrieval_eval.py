from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from rag.evidence import EvidencePacket
from rag.vector_store import SearchFilters, SearchResult, canonical_section_key, get_vector_store

# FIXTURE / RESULT MODELS

@dataclass(frozen=True, slots=True)
class RetrievalEvalCase:
    id: str
    ticker: str
    query: str
    filing_type: Optional[str] = None
    expected_sections: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    top_k: int = 5
    mode: str = "query" # "query" or "sections"
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrievalEvalCase:
        return cls(
            id=str(payload.get("id")),
            ticker=str(payload.get("ticker")).upper(),
            query=str(payload.get("query")),
            filing_type=str(payload.get("filing_type")),
            expected_sections=list(payload.get("expected_sections", [])),
            expected_keywords=list(payload.get("expected_keywords", [])),
            top_k=int(payload.get("top_k", 5)),
            mode=str(payload.get("mode", "query")).lower(),
            notes=payload.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    retrieved_count: int
    latency_ms: float
    section_hit_at_k: bool
    section_recall_at_k: float
    keyword_hit_rate: float
    first_relevant_rank: Optional[int]
    pass_at_k: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class RetrievalEvalResult:
    case_id: str
    ticker: str
    query: str
    mode: str
    top_k: int
    passed: bool
    metrics: RetrievalMetrics
    issues: list[str] = field(default_factory=list)
    expected_sections: list[str] = field(default_factory=list)
    retrieved_sections: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    evidence_packets: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "ticker": self.ticker,
            "query": self.query,
            "mode": self.mode,
            "top_k": self.top_k,
            "passed": self.passed,
            "metrics": self.metrics.to_dict(),
            "issues": self.issues,
            "expected_sections": self.expected_sections,
            "retrieved_sections": self.retrieved_sections,
            "expected_keywords": self.expected_keywords,
            "matched_keywords": self.matched_keywords,
            "evidence_packets": self.evidence_packets,
            "error": self.error,
        }

# HELPER Functions

def load_retrieval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Retrieval fixture file must contain a JSON list")
    return [RetrievalEvalCase.from_dict(item) for item in payload]

def _normalize_token(value: Optional[str]) -> str:
    return (value or "").strip().lower().replace("&", "and")

def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = _normalize_token(value)
        if not key or key in seen:
            continue

        seen.add(key)
        out.append(value)

    return out

def _search(store: Any, case: RetrievalEvalCase) -> SearchResult:
    if case.mode == "sections":
        try:
            return store.search_sections(
                ticker=case.ticker,
                sections=case.expected_sections,
                top_k=case.top_k,
                query=case.query,
                filing_type=case.filing_type,
            )
        except TypeError:
            return store.search_sections(
                ticker=case.ticker,
                sections=case.expected_sections,
                n_results=case.top_k,
                query=case.query,
                filing_type=case.filing_type,
            )

    filters = SearchFilters(ticker=case.ticker, filing_type=case.filing_type)
    try:
        return store.search(case.query, filters=filters, top_k=case.top_k)
    except TypeError:
        return store.search(case.query, filters=filters, n_results=case.top_k)

def _packet_sections(packets: list[EvidencePacket]) -> list[str]:
    return _dedupe_keep_order(
        [packet.section_name for packet in packets if packet.section_name]
    )

def _matched_keywords(expected_keywords: list[str], packets: list[EvidencePacket]) -> list[str]:
    matched: list[str] = []
    corpus = [packet.text.lower() for packet in packets]
    for keyword in expected_keywords:
        keyword_token = _normalize_token(keyword)
        if keyword_token and any(keyword_token in text for text in corpus):
            matched.append(keyword)

    return matched

def _first_relevant_rank(
    packets: list[EvidencePacket],
    expected_sections: list[str],
    matched_keywords: list[str],
) -> Optional[int]:
    expected_section_set = {_normalize_token(section) for section in expected_sections}
    keyword_set = {k.lower() for k in matched_keywords}

    for packet in packets:
        section_ok = _normalize_token(packet.section_name) in expected_section_set if expected_section_set else False
        keyword_ok = any(keyword in packet.text.lower() for keyword in keyword_set) if keyword_set else False
        if section_ok or keyword_ok:
            return packet.rank

    return None

def evaluate_retrieval_case(
    case: RetrievalEvalCase,
    *,
    store: Any | None = None,
) -> RetrievalEvalResult:
    store = store or get_vector_store()

    try:
        result = _search(store, case)
        packets = result.to_evidence_packets(retrieval_method=f"eval:{case.mode}")

        retrieved_sections = _packet_sections(packets)
        matched_keywords = _matched_keywords(case.expected_keywords, packets)

        expected_section_set = {_normalize_token(section) for section in case.expected_sections}

        expected_section_keys = {
            canonical_section_key(section) or section.lower().strip()
            for section in case.expected_sections
        }

        retrieved_section_keys = {
            canonical_section_key(section) or section.lower().strip()
            for section in retrieved_sections
            if section
        }

        matched_section_count = len(expected_section_keys & retrieved_section_keys)
        section_hit_at_k = matched_section_count > 0 if expected_section_keys else len(packets) > 0
        section_recall_at_k = (
            matched_section_count / len(expected_section_keys)
            if expected_section_keys
            else 1.0
        )
        keyword_hit_rate = (
            len(matched_keywords) / len(case.expected_keywords)
            if case.expected_keywords
            else 1.0
        )
        first_rank = _first_relevant_rank(packets, case.expected_sections, matched_keywords)

        passed = True
        issues: list[str] = []

        if expected_section_set and not section_hit_at_k:
            passed = False
            issues.append("No expected sections found in top-k")
        if case.expected_keywords and keyword_hit_rate < 0.0:
            passed = False
            issues.append("No Expected keywords found in retrieved chunks")
        if not packets:
            passed = False
            issues.append("No chunks retrieved")

        metrics = RetrievalMetrics(
            retrieved_count=len(packets),
            latency_ms=float(result.search_time_ms),
            section_hit_at_k=section_hit_at_k,
            section_recall_at_k=section_recall_at_k,
            keyword_hit_rate=keyword_hit_rate,
            first_relevant_rank=first_rank,
            pass_at_k=passed,
        )

        return RetrievalEvalResult(
            case_id=case.id,
            ticker=case.ticker,
            query=case.query,
            mode=case.mode,
            top_k=case.top_k,
            passed=passed,
            metrics=metrics,
            issues=issues,
            expected_sections=case.expected_sections,
            retrieved_sections=retrieved_sections,
            expected_keywords=case.expected_keywords,
            matched_keywords=matched_keywords,
            evidence_packets=[packet.to_dict() for packet in packets],
        )

    except Exception as exc:
        logger.error(f"Retrieval eval failed for case {case.id}")
        return RetrievalEvalResult(
            case_id=case.id,
            ticker=case.ticker,
            query=case.query,
            mode=case.mode,
            top_k=case.top_k,
            passed=False,
            metrics=RetrievalMetrics(
                retrieved_count=0,
                latency_ms=0.0,
                section_hit_at_k=False,
                section_recall_at_k=0.0,
                keyword_hit_rate=0.0,
                first_relevant_rank=None,
                pass_at_k=False,
            ),
            issues=["evaluation crashed"],
            expected_sections=case.expected_sections,
            expected_keywords=case.expected_keywords,
            error=f"{type(exc).__name__}: {exc}",
        )

def evaluate_retrieval_cases(
    cases: list[RetrievalEvalCase],
    *,
    store: Any | None = None,
) -> list[RetrievalEvalResult]:
    store = store or get_vector_store()
    return [evaluate_retrieval_case(case, store=store) for case in cases]

def summarize_retrieval_results(results: list[RetrievalEvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_latency_ms = (
        sum(r.metrics.latency_ms for r in results) / total if total else 0.0
    )
    avg_section_recall = (
        sum(result.metrics.section_recall_at_k for result in results) / total if total else 0.0
    )
    avg_keyword_hit_rate = (
        sum(result.metrics.keyword_hit_rate for result in results) / total if total else 0.0
    )

    return {
        "total": total,
        "passed": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "avg_latency_ms": avg_latency_ms,
        "avg_section_recall_at_k": avg_section_recall,
        "avg_keyword_hit_rate": avg_keyword_hit_rate,
    }

def save_retrieval_results(results: list[RetrievalEvalResult], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_retrieval_results(results),
        "results": [result.to_dict() for result in results],
    }

    output.write_text(json.dumps(payload, indent=2))
    return output

