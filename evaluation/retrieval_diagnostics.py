"""Miss diagnostics over one anchored retrieval result (retrieval-benchmark section 7a)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict

from app.components.retrieval.ingestion import build_index_documents
from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.section_intent import infer_section_intent
from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from app.services.tools.sec_filings_tool import Filing, FilingMetaData, FilingSection
from app.verification.grounding import _token_overlap_score
from dataops.replay import load_evidence_release_records
from evaluation.run_shared_retrieval_benchmark import DEFAULT_SOURCE_DIR
from evaluation.validate_retrieval_fixture import (
    AnchoredCase,
    AnchoredFixture,
    load_anchored_fixture,
    normalize_text,
)

MissClass = Literal["boundary_split", "section_miss", "vocabulary_gap", "ranking"]
OVERLAP_THRESHOLD = 0.5
DENSE_POOL_DEPTH = 50
TOP_K = 10
MISS_CLASSES: tuple[str, ...] = get_args(MissClass)
CANDIDATES = ("section_prior", "hybrid_bm25", "rerank", "chunking", "query_rewrite")


class RebuildMismatchError(ValueError):
    """The rebuilt chunks do not reproduce what the result file retrieved."""


class AnchorMiss(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    ticker: str
    anchor_index: int
    accession_number: str
    section_key: str
    miss_class: MissClass
    anchor_chunk_ids: list[str]
    section_reached: bool
    overlap: float | None
    dense_rank: int | None
    intent_sections: list[str]
    intent_targeted_gold_section: bool


class CandidateCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    misses: int


class DiagnosticsReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_result: str
    fixture_version: str
    evidence_release: str
    embedding_model: str
    overlap_threshold: float
    dense_pool_depth: int
    n_cases_with_misses: int
    n_misses: int
    class_counts: dict[str, int]
    candidates: list[CandidateCount]
    misses: list[AnchorMiss]


def classify_miss(
    *, anchor_chunk_ids: Sequence[str], section_reached: bool, overlap: float | None
) -> MissClass:
    if not anchor_chunk_ids:
        return "boundary_split"
    if not section_reached:
        return "section_miss"
    if overlap is not None and overlap < OVERLAP_THRESHOLD:
        return "vocabulary_gap"
    return "ranking"


def rebuild_chunks(
    release_id: str, *, registry_root: Path = Path("artifacts/dataops")
) -> dict[str, IndexDocument]:
    chunks: dict[str, IndexDocument] = {}
    bundle = load_evidence_release_records(release_id, registry_root=registry_root)
    for record in bundle.records:
        if record.snapshot.source_type != "sec_filing":
            continue
        payload = record.payload
        filing = Filing(
            metadata=FilingMetaData(
                cik=payload["cik"],
                accession_number=payload["accession_number"],
                filing_type=payload["filing_type"],
                filing_date=payload["filing_date"],
                report_date=payload["report_date"],
                primary_document=payload["primary_document"],
                company_name=payload["company_name"],
                ticker=payload["ticker"],
            ),
            sections={
                name: FilingSection(name=name, content=section["content"])
                for name, section in payload["sections"].items()
            },
        )
        for document in build_index_documents(filing).documents:
            chunks[document.id] = document
    return chunks


def _holds_anchor(case: AnchoredCase, anchor_index: int, chunk: IndexDocument) -> bool:
    """The section 3 matcher for one gold anchor."""
    gold = case.gold_evidence[anchor_index]
    return (
        chunk.metadata["accession_number"] == gold.accession_number
        and chunk.metadata["section_key"] == gold.section_key
        and normalize_text(gold.anchor_text) in normalize_text(chunk.text)
    )


def _top_chunks(
    result_case: Mapping[str, Any], case: AnchoredCase, chunks: Mapping[str, IndexDocument]
) -> list[IndexDocument]:
    """Top-10 retrieved chunks, refusing a rebuild that disagrees with the stored gold_hit."""
    top: list[IndexDocument] = []
    for row in result_case["retrieved"][:TOP_K]:
        chunk = chunks.get(row["chunk_id"])
        if chunk is None:
            raise RebuildMismatchError(
                f"case {case.case_id}: chunk {row['chunk_id']} is missing from the rebuild"
            )
        rebuilt_hit = any(_holds_anchor(case, i, chunk) for i in range(len(case.gold_evidence)))
        if rebuilt_hit != row["gold_hit"]:
            raise RebuildMismatchError(
                f"case {case.case_id}: chunk {row['chunk_id']} stored gold_hit "
                f"{row['gold_hit']} but the rebuild gives {rebuilt_hit}"
            )
        top.append(chunk)
    return top


def _dense_rank(anchor_chunk_ids: Sequence[str], pool: Sequence[str]) -> int | None:
    positions = [pool.index(chunk_id) + 1 for chunk_id in anchor_chunk_ids if chunk_id in pool]
    return min(positions) if positions else None


def _case_misses(
    result_case: Mapping[str, Any],
    case: AnchoredCase,
    chunks: Mapping[str, IndexDocument],
    pool: Sequence[str],
) -> list[AnchorMiss]:
    top = _top_chunks(result_case, case, chunks)
    intent_sections = list(infer_section_intent(case.query).target_sections)
    misses: list[AnchorMiss] = []
    for index, gold in enumerate(case.gold_evidence):
        if any(_holds_anchor(case, index, chunk) for chunk in top):
            continue
        anchor_chunk_ids = sorted(
            chunk.id for chunk in chunks.values() if _holds_anchor(case, index, chunk)
        )
        section_reached = any(
            chunk.metadata["accession_number"] == gold.accession_number
            and chunk.metadata["section_key"] == gold.section_key
            for chunk in top
        )
        overlap = (
            max(_token_overlap_score(case.query, chunks[i].text) for i in anchor_chunk_ids)
            if anchor_chunk_ids
            else None
        )
        misses.append(
            AnchorMiss(
                case_id=case.case_id,
                ticker=case.ticker,
                anchor_index=index,
                accession_number=gold.accession_number,
                section_key=gold.section_key,
                miss_class=classify_miss(
                    anchor_chunk_ids=anchor_chunk_ids,
                    section_reached=section_reached,
                    overlap=overlap,
                ),
                anchor_chunk_ids=anchor_chunk_ids,
                section_reached=section_reached,
                overlap=overlap,
                dense_rank=_dense_rank(anchor_chunk_ids, pool),
                intent_sections=intent_sections,
                intent_targeted_gold_section=gold.section_key in intent_sections,
            )
        )
    return misses


def _candidate_counts(misses: Sequence[AnchorMiss]) -> list[CandidateCount]:
    def addressed(name: str, miss: AnchorMiss) -> bool:
        match name:
            case "section_prior":
                return miss.miss_class == "section_miss"
            case "hybrid_bm25":
                return miss.miss_class == "ranking"
            case "rerank":
                return miss.miss_class != "boundary_split" and miss.dense_rank is not None
            case "chunking":
                return miss.miss_class == "boundary_split"
            case _:
                return miss.miss_class == "vocabulary_gap"

    counts = [
        CandidateCount(name=name, misses=sum(addressed(name, m) for m in misses))
        for name in CANDIDATES
    ]
    return sorted(counts, key=lambda c: (-c.misses, c.name))


def build_diagnostics_report(
    result: Mapping[str, Any],
    fixture: AnchoredFixture,
    chunks: Mapping[str, IndexDocument],
    *,
    source_result: str,
    dense_pool: Mapping[str, Sequence[str]] | None = None,
) -> DiagnosticsReport:
    cases = {case.case_id: case for case in fixture.cases}
    pools = dense_pool or {}
    depth = DENSE_POOL_DEPTH
    traces = {c["case_id"]: c["trace"] for c in result["cases"] if "trace" in c}
    if dense_pool is None and traces:
        pools = {
            case_id: [
                c["chunk_id"] for c in sorted(trace["candidates"], key=lambda c: c["dense_rank"])
            ]
            for case_id, trace in traces.items()
        }
        depth = max(trace["candidate_pool"] for trace in traces.values())
    misses: list[AnchorMiss] = []
    for result_case in result["cases"]:
        case_id = result_case["case_id"]
        misses += _case_misses(result_case, cases[case_id], chunks, pools.get(case_id, []))
    provenance = result["provenance"]
    return DiagnosticsReport(
        source_result=source_result,
        fixture_version=result["fixture_version"],
        evidence_release=provenance["evidence_release"],
        embedding_model=provenance["embedding_model"],
        overlap_threshold=OVERLAP_THRESHOLD,
        dense_pool_depth=depth,
        n_cases_with_misses=len({m.case_id for m in misses}),
        n_misses=len(misses),
        class_counts={name: sum(m.miss_class == name for m in misses) for name in MISS_CLASSES},
        candidates=_candidate_counts(misses),
        misses=misses,
    )


def collect_dense_pool(
    store: Any, fixture: AnchoredFixture, case_ids: Sequence[str], *, depth: int = DENSE_POOL_DEPTH
) -> dict[str, list[str]]:
    """Dense top-`depth` chunk ids per case, ordered like the dense branch of the benchmark."""
    cases = {case.case_id: case for case in fixture.cases}
    pool: dict[str, list[str]] = {}
    for case_id in case_ids:
        case = cases[case_id]
        found = store.search(
            case.query,
            filters=SearchFilters(ticker=case.ticker, filing_type=case.filters.get("filing_type")),
            n_results=depth,
        )
        ranked = sorted(found.chunks, key=lambda c: (-c.relevance_score, c.id))
        pool[case_id] = [chunk.id for chunk in ranked]
    return pool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify the misses of an anchored result file.")
    parser.add_argument("result", help="Anchored benchmark result JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Report JSON to write.")
    parser.add_argument("--dense-pool", type=Path, help="Dense top-50 pool JSON.")
    parser.add_argument("--collect-dense-pool", action="store_true")
    parser.add_argument("--collection", help="Index to search with --collect-dense-pool.")
    args = parser.parse_args(argv)
    if args.collect_dense_pool and not (args.collection and args.dense_pool):
        parser.error("--collect-dense-pool requires --collection and --dense-pool")

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    provenance = result["provenance"]
    fixture = load_anchored_fixture(Path(provenance["fixture_path"]), source_dir=DEFAULT_SOURCE_DIR)
    chunks = rebuild_chunks(provenance["evidence_release"])

    pool: dict[str, list[str]] | None = None
    if args.collect_dense_pool:
        offline = build_diagnostics_report(result, fixture, chunks, source_result=args.result)
        case_ids = list(dict.fromkeys(miss.case_id for miss in offline.misses))
        store = OpenSearchVectorStore(index_name=args.collection)
        pool = collect_dense_pool(store, fixture, case_ids)
        args.dense_pool.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")
    elif args.dense_pool:
        pool = json.loads(args.dense_pool.read_text(encoding="utf-8"))

    report = build_diagnostics_report(
        result, fixture, chunks, source_result=args.result, dense_pool=pool
    )
    args.out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
