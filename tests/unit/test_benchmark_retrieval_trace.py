"""Trace: docs/test-plan.md §25, "Retrieval trace in benchmark results" (FR.0, 2026-11-24);
docs/retrieval-benchmark.md §7 (rerank methods record per-case `trace`) and §7a ("Pool from the
trace").

A benchmark result for a rerank method carries per case `trace`, the cold search's retrieval
trace; other methods carry none. Miss diagnostics given no dense-pool file take each case's pool
from its trace and report `dense_pool_depth` as the trace's candidate pool; a dense-pool file,
when given, wins; a result without traces is diagnosed as before. The runner runs over the
in-memory stores of tests/unit/test_anchored_benchmark_runner.py and the loopback rerank fake; the
diagnostics run over that module's hand-built world (tests/unit/test_retrieval_diagnostics.py).
Expected traces are computed from the store's own output (tests/unit/trace_support.py).

Regression guards (they hold before FR.0 and keep holding): the diagnostics case "a given pool wins"
and "a result without traces is diagnosed as before".
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from evaluation.retrieval_diagnostics import DENSE_POOL_DEPTH, build_diagnostics_report
from evaluation.run_shared_retrieval_benchmark import run_anchored_benchmark
from tests.unit.fake_foundry import relevance
from tests.unit.test_anchored_benchmark_runner import (
    POOL_OF,
    RERANK_METHODS,
    PseudoEmbeddingStore,
    _crowded_corpus,
    _fixture,
)
from tests.unit.test_retrieval_diagnostics import SOURCE, World, _build_world
from tests.unit.trace_support import assert_trace_equals, expected_trace

K = 10
TRACE_FIELDS = {
    "query",
    "ticker",
    "filing_type",
    "candidate_source",
    "candidate_pool",
    "candidates",
    "dense_score_gap",
    "pool_sections",
    "kept_dense_ranks",
}


def _run(store: Any, method: str) -> dict[str, dict[str, Any]]:
    result = run_anchored_benchmark(_fixture(), store, backend="qdrant", retrieval_method=method)
    return {case["case_id"]: case for case in result["cases"]}


@pytest.mark.parametrize(("method", "source"), list(RERANK_METHODS.items()))
def test_a_rerank_method_records_the_cold_search_trace_of_every_case(
    foundry_rerank, method: str, source: str
) -> None:
    """test-plan §25: Retrieval trace in benchmark results (per case `trace`, the cold search's
    retrieval trace: candidate pool in store order with dense and rerank rank and score, derived
    gap, pool sections and kept dense ranks, at the method's pool)."""
    store = PseudoEmbeddingStore(_crowded_corpus())
    fixture = _fixture()
    pool = POOL_OF[method]

    cases = _run(store, method)

    for case in fixture.cases:
        expected = expected_trace(
            store,
            case.query,
            relevance,
            ticker=case.ticker,
            filing_type="10-K",
            source=source,
            pool=pool,
            top_k=K,
        )
        trace = cases[case.case_id]["trace"]
        assert_trace_equals(trace, expected)
        assert [r["chunk_id"] for r in cases[case.case_id]["retrieved"]] == expected["kept_ids"]
    crowded = cases["aapl-risk-001"]["trace"]
    assert len(crowded["candidates"]) == pool
    assert crowded["candidate_pool"] == pool


def test_the_recorded_trace_is_plain_json_without_chunk_text(foundry_rerank) -> None:
    """test-plan §25: Retrieval trace in benchmark results (the result file is JSON; the trace
    holds the documented fields and no chunk text)."""
    store = PseudoEmbeddingStore(_crowded_corpus())

    cases = _run(store, "dense_rerank_15")

    texts = {point[1] for point in _crowded_corpus()}
    for case in cases.values():
        assert set(case["trace"]) == TRACE_FIELDS
        text = json.dumps(case["trace"])
        assert not any(chunk_text in text for chunk_text in texts)


@pytest.mark.parametrize("method", ["dense", "section_aware", "hybrid"])
def test_other_methods_carry_no_trace_while_a_rerank_method_does(
    foundry_rerank, method: str
) -> None:
    """test-plan §25: Retrieval trace in benchmark results (other methods carry no `trace`)."""
    store = PseudoEmbeddingStore(_crowded_corpus())

    plain = _run(store, method)
    reranked = _run(store, "dense_rerank_15")

    assert reranked and all("trace" in case for case in reranked.values())
    assert plain and all("trace" not in case for case in plain.values())


# ---- diagnostics: pool from the trace ---------------------------------------------------


def _trace_of(ids: list[str], depth: int) -> dict[str, Any]:
    """The trace a rerank result file holds for a case whose dense candidates were `ids`, taken to
    `depth` (the requested pool)."""
    kept = ids[:depth]
    return {
        "query": "q",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "candidate_source": "dense",
        "candidate_pool": depth,
        "candidates": [
            {
                "chunk_id": chunk_id,
                "section_key": "risk_factors",
                "dense_rank": rank,
                "dense_score": round(0.9 - 0.01 * rank, 4),
                "rerank_rank": rank,
                "rerank_score": round(0.5 - 0.01 * rank, 4),
            }
            for rank, chunk_id in enumerate(kept, start=1)
        ],
        "dense_score_gap": round(0.01 * (min(15, len(kept)) - 1), 4) if len(kept) > 1 else None,
        "pool_sections": {"risk_factors": len(kept)},
        "kept_dense_ranks": [1] if kept else [],
    }


def _with_traces(world: World, depth: int) -> dict[str, Any]:
    result = copy.deepcopy(world.result)
    for case in result["cases"]:
        if case["case_id"] in world.pool:
            case["trace"] = _trace_of(world.pool[case["case_id"]], depth)
    return result


def _ranks(world: World, result: dict[str, Any], pool: dict[str, list[str]] | None) -> Any:
    report = build_diagnostics_report(
        result, world.fixture, world.chunks, source_result=SOURCE, dense_pool=pool
    )
    return report, {(m.case_id, m.anchor_index): m.dense_rank for m in report.misses}


@pytest.mark.parametrize("depth", [15, 60])
def test_diagnostics_take_the_pool_from_the_trace_and_report_its_depth(depth: int) -> None:
    """test-plan §25: Retrieval trace in benchmark results (given no dense-pool file, each case's
    pool is its trace's candidate chunk ids in dense-rank order and `dense_pool_depth` is the
    trace's candidate pool): a pool of 15 reports ranks only to 15."""
    world = _build_world()
    _, from_file = _ranks(world, world.result, world.pool)
    assert any(rank is not None and rank > 15 for rank in from_file.values()), "precondition"
    assert any(rank is not None and rank <= 15 for rank in from_file.values()), "precondition"

    report, from_trace = _ranks(world, _with_traces(world, depth), None)

    assert report.dense_pool_depth == depth
    assert from_trace == {
        key: (rank if rank is not None and rank <= depth else None)
        for key, rank in from_file.items()
    }
    assert any(rank is not None for rank in from_trace.values())
    rerank = next(c for c in report.candidates if c.name == "rerank")
    assert rerank.misses == sum(
        1 for m in report.misses if m.miss_class != "boundary_split" and m.dense_rank is not None
    )
    assert rerank.misses > 0


def test_a_given_dense_pool_is_used_instead_of_the_trace() -> None:
    """test-plan §25: Retrieval trace in benchmark results (a dense-pool file, when given, is used
    instead). Regression guard: the file's ranks and the declared depth, not the trace's."""
    world = _build_world()
    _, from_file = _ranks(world, world.result, world.pool)
    shifted = {case_id: ["shift-a", "shift-b", *ids] for case_id, ids in world.pool.items()}

    report, with_both = _ranks(world, _with_traces(world, 15), shifted)

    assert report.dense_pool_depth == DENSE_POOL_DEPTH
    assert with_both == {
        key: (rank + 2 if rank is not None else None) for key, rank in from_file.items()
    }


def test_a_result_without_traces_and_without_a_pool_is_diagnosed_as_before() -> None:
    """test-plan §25: Retrieval trace in benchmark results (a result without traces is diagnosed
    as before). Regression guard: no dense ranks, the declared depth."""
    world = _build_world()

    report, ranks = _ranks(world, world.result, None)

    assert report.misses
    assert set(ranks.values()) == {None}
    assert report.dense_pool_depth == DENSE_POOL_DEPTH
