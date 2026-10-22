"""Run the shared retrieval benchmark across configured retrieval methods."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.components.retrieval.ingestion import CHUNK_OVERLAP, CHUNK_SIZE
from app.components.retrieval.qdrant_store import QdrantVectorStore
from app.components.retrieval.section_aware_search import search_section_aware
from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters
from app.config import settings
from dataops.git_state import current_code_version
from evaluation.validate_retrieval_fixture import (
    AnchoredCase,
    AnchoredFixture,
    FixtureValidationError,
    load_anchored_fixture,
    load_fixture,
    normalize_text,
)

BackendName = Literal["chroma", "qdrant"]
RetrievalMode = Literal["query", "sections", "section_aware"]

DEFAULT_FIXTURE = Path("evaluation/fixtures/retrieval_shared_benchmark_v1.json")
DEFAULT_OUTPUT_DIR = Path("evaluation/results/shared_benchmark")
DEFAULT_SOURCE_DIR = Path("evaluation/fixtures/source_sections")

RETRIEVAL_METHODS: tuple[str, ...] = ("dense", "section_aware")
BENCHMARK_KS: tuple[int, ...] = (5, 10)
_RANKED_METRICS = ("pass", "recall", "precision", "mrr", "ndcg", "section_recall", "keyword_hit_rate")


class MethodSpec(BaseModel):
    """A retrieval method run configuration."""

    model_config = ConfigDict(frozen=True)

    name: str
    backend: BackendName
    mode: RetrievalMode
    env: dict[str, str] = Field(default_factory=dict)


DEFAULT_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(name="qdrant_query", backend="qdrant", mode="query"),
    MethodSpec(name="qdrant_section_aware", backend="qdrant", mode="section_aware"),
    MethodSpec(name="chroma_query", backend="chroma", mode="query"),
)


def _selected_methods(names: list[str] | None) -> list[MethodSpec]:
    methods_by_name = {method.name: method for method in DEFAULT_METHODS}
    if not names:
        return list(DEFAULT_METHODS)

    unknown = sorted(name for name in names if name not in methods_by_name)
    if unknown:
        valid = ", ".join(sorted(methods_by_name))
        raise SystemExit(f"Unknown method(s): {unknown}. Valid methods: {valid}")

    return [methods_by_name[name] for name in names]


def _load_case_payloads(fixture_path: Path) -> list[dict[str, object]]:
    # Validate first. Then use the original JSON shape so evaluation.retrieval_main
    # stays the single source of execution behavior.
    load_fixture(fixture_path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("validated fixture unexpectedly did not load as a list")
    return payload


def _write_mode_fixture(
    cases: list[dict[str, object]],
    *,
    mode: RetrievalMode,
    directory: Path,
) -> Path:
    mode_cases = [{**case, "mode": mode} for case in cases]
    output_path = directory / f"fixture_{mode}.json"
    output_path.write_text(json.dumps(mode_cases, indent=2) + "\n", encoding="utf-8")
    return output_path


def _run_method(
    method: MethodSpec,
    *,
    fixture_path: Path,
    output_path: Path,
    continue_on_error: bool,
) -> int:
    env = os.environ.copy()
    env["VECTOR_BACKEND"] = method.backend
    env.update(method.env)

    command = [
        sys.executable,
        "-m",
        "evaluation.retrieval_main",
        str(fixture_path),
        "--output",
        str(output_path),
    ]
    print(f"\n==> Running {method.name}")
    print(f"    backend={method.backend} mode={method.mode} output={output_path}")

    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0 and not continue_on_error:
        raise SystemExit(completed.returncode)
    return completed.returncode


def _gold_hits(case: AnchoredCase, chunk: RetrievedChunk) -> list[int]:
    """Indices of the gold anchors wholly contained in this chunk (retrieval-benchmark §3)."""
    text = normalize_text(chunk.text)
    return [
        i
        for i, gold in enumerate(case.gold_evidence)
        if chunk.metadata.get("accession_number") == gold.accession_number
        and chunk.metadata.get("section_key") == gold.section_key
        and normalize_text(gold.anchor_text) in text
    ]


def score_retrieval(
    case: AnchoredCase, chunks: Sequence[RetrievedChunk], k: int
) -> dict[str, float | int | None]:
    """Score the first k chunks against the case's gold anchors (retrieval-benchmark §3-§4)."""
    top = list(chunks[:k])
    gold = case.gold_evidence
    hits = [_gold_hits(case, chunk) for chunk in top]

    covered: set[int] = set()
    credited: set[int] = set()
    dcg = 0.0
    for rank, chunk_hits in enumerate(hits, start=1):
        covered.update(chunk_hits)
        uncredited = [i for i in chunk_hits if i not in credited]
        if uncredited:
            best = max(uncredited, key=lambda i: (gold[i].relevance, -i))
            credited.add(best)
            dcg += gold[best].relevance / math.log2(rank + 1)
    ideal = sorted((g.relevance for g in gold), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))

    first_relevant_rank = next((rank for rank, h in enumerate(hits, start=1) if h), None)
    gold_sections = {(g.accession_number, g.section_key) for g in gold}
    seen_sections = {
        (chunk.metadata.get("accession_number"), chunk.metadata.get("section_key")) for chunk in top
    }
    joined_text = normalize_text(" ".join(chunk.text for chunk in top))
    keywords = case.expected_keywords

    return {
        "pass": int(bool(covered)),
        "recall": len(covered) / len(gold),
        "precision": sum(1 for h in hits if h) / k,
        "mrr": 1 / first_relevant_rank if first_relevant_rank else 0,
        "ndcg": dcg / idcg if idcg else 0,
        "section_recall": len(gold_sections & seen_sections) / len(gold_sections),
        "keyword_hit_rate": (
            sum(1 for kw in keywords if normalize_text(kw) in joined_text) / len(keywords)
            if keywords
            else None
        ),
        "first_relevant_rank": first_relevant_rank,
    }


def _search(store: Any, case: AnchoredCase, method: str) -> list[RetrievedChunk]:
    top_k = max(BENCHMARK_KS)
    filing_type = case.filters.get("filing_type")
    if method == "dense":
        result = store.search(
            case.query,
            filters=SearchFilters(ticker=case.ticker, filing_type=filing_type),
            n_results=top_k,
        )
        return sorted(result.chunks, key=lambda c: (-c.relevance_score, c.id))
    return search_section_aware(
        store, query=case.query, ticker=case.ticker, filing_type=filing_type, top_k=top_k
    ).chunks


def _timed_search(
    store: Any, case: AnchoredCase, method: str
) -> tuple[list[RetrievedChunk], float]:
    started = time.perf_counter()
    chunks = _search(store, case, method)
    return chunks, (time.perf_counter() - started) * 1000


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def run_anchored_benchmark(
    fixture: AnchoredFixture,
    store: Any,
    *,
    backend: str,
    retrieval_method: str,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Run every fixture case through one retrieval method and score it (retrieval-benchmark §7)."""
    if retrieval_method not in RETRIEVAL_METHODS:
        raise ValueError(f"retrieval_method must be one of {RETRIEVAL_METHODS}: {retrieval_method!r}")
    if not backend:
        raise ValueError("backend must not be empty")

    cases: list[dict[str, Any]] = []
    for case in fixture.cases:
        chunks, cold_ms = _timed_search(store, case, retrieval_method)
        _, warm_ms = _timed_search(store, case, retrieval_method)
        metrics = {str(k): score_retrieval(case, chunks, k) for k in BENCHMARK_KS}
        cases.append(
            {
                "case_id": case.case_id,
                "ticker": case.ticker,
                "answer_intent": case.answer_intent,
                "first_relevant_rank": metrics[str(max(BENCHMARK_KS))]["first_relevant_rank"],
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": chunk.id,
                        "accession_number": chunk.metadata.get("accession_number"),
                        "section_key": chunk.metadata.get("section_key"),
                        "score": chunk.relevance_score,
                        "gold_hit": bool(_gold_hits(case, chunk)),
                    }
                    for rank, chunk in enumerate(chunks, start=1)
                ],
                "metrics": metrics,
                "latency_ms": {"cold": cold_ms, "warm": warm_ms},
            }
        )

    summary_metrics = {
        str(k): {
            name: _mean(
                [
                    value
                    for case in cases
                    if (value := case["metrics"][str(k)][name]) is not None
                ]
            )
            for name in _RANKED_METRICS
        }
        for k in BENCHMARK_KS
    }
    return {
        "fixture_version": fixture.fixture_version,
        "backend": backend,
        "retrieval_method": retrieval_method,
        "mode": retrieval_method,
        "ks": list(BENCHMARK_KS),
        "top_k": max(BENCHMARK_KS),
        "provenance": dict(provenance or {}),
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "metrics": summary_metrics,
            "latency_ms": {
                "cold_p50": statistics.median(c["latency_ms"]["cold"] for c in cases),
                "warm_p50": statistics.median(c["latency_ms"]["warm"] for c in cases),
            },
        },
    }


def _run_anchored_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.collection or not args.collection.strip():
        parser.error("--collection is required for an anchored fixture")
    if args.output is None:
        parser.error("--output is required for an anchored fixture")
    if not args.methods or len(args.methods) != 1 or args.methods[0] not in RETRIEVAL_METHODS:
        parser.error(f"--method must be passed once, one of {', '.join(RETRIEVAL_METHODS)}")

    try:
        fixture = load_anchored_fixture(args.fixture, source_dir=args.source_dir)
    except FixtureValidationError as error:
        raise SystemExit(f"Invalid anchored fixture {args.fixture}:\n{error}") from error

    store = QdrantVectorStore(collection_name=args.collection)
    if store.count == 0:
        raise SystemExit(f"Collection {args.collection!r} is empty; build the index first.")

    result = run_anchored_benchmark(
        fixture,
        store,
        backend=args.backend,
        retrieval_method=args.methods[0],
        provenance={
            "fixture_path": str(args.fixture),
            "collection": args.collection,
            "git_sha": current_code_version(),
            "embedding_model": settings.ollama.embed_model,
            "evidence_release": args.evidence_release,
            "chunker": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved anchored benchmark result: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared retrieval benchmark matrix.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help=(
            "Method name to run. May be passed multiple times. Defaults to all configured methods. "
            "For an anchored fixture, pass exactly one of: dense, section_aware."
        ),
    )
    parser.add_argument("--backend", choices=["qdrant"], default="qdrant")
    parser.add_argument("--collection", help="Qdrant collection to search (anchored fixture).")
    parser.add_argument("--output", type=Path, help="Result file (anchored fixture).")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--evidence-release", help="Evidence release id, recorded as provenance.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if isinstance(json.loads(args.fixture.read_text(encoding="utf-8")), dict):
        _run_anchored_cli(args, parser)
        return

    methods = _selected_methods(args.methods)
    cases = _load_case_payloads(args.fixture)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "fixture": str(args.fixture),
        "case_count": len(cases),
        "methods": [method.model_dump() for method in methods],
        "outputs": {},
    }

    with tempfile.TemporaryDirectory(prefix="retrieval_shared_fixture_") as tmp:
        tmp_dir = Path(tmp)
        for method in methods:
            method_fixture = _write_mode_fixture(cases, mode=method.mode, directory=tmp_dir)
            output_path = run_dir / f"{method.name}.json"
            manifest["outputs"][method.name] = str(output_path)  # type: ignore[index]

            if args.dry_run:
                print(f"DRY RUN: would run {method.name} with fixture {method_fixture}")
                continue

            return_code = _run_method(
                method,
                fixture_path=method_fixture,
                output_path=output_path,
                continue_on_error=args.continue_on_error,
            )
            if return_code != 0:
                manifest.setdefault("failed_methods", {})  # type: ignore[call-overload]
                manifest["failed_methods"][method.name] = return_code  # type: ignore[index]

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark manifest: {manifest_path}")


if __name__ == "__main__":
    main()
