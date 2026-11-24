"""Trace: docs/test-plan.md §25, "App filings workload report" (FR.0, 2026-11-24);
docs/sprint-plan.md "FR.0" (d) and "Comparative analysis" (App workload).

`evaluation/app_filings_workload.py` runs the filings step's four topic queries for each given
ticker through `search_reranked(candidates="dense", top_k=3)` at pool 15 and pool 50 on one index,
and reports per (ticker, topic) both pools' top 3 and traces, the top-3 agreement between the
pools, and whether each top-3 chunk is in the topic's declared section; the summary gives the mean
agreement and the section hit rate per pool and records the index, embedding and reranker identity.

The store is the in-memory fake; the reranker is an in-test scorer (word overlap, as the loopback
Foundry fake computes it) or the loopback Foundry fake itself. Every expected value is computed
here from the store's own output and the scorer. The module under test is loaded at run time, so
its absence fails each test, not the collection.
"""

from __future__ import annotations

import importlib
import json
from types import ModuleType
from typing import Any

import pytest

from app.agents import graph
from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from tests.unit.fake_foundry import RERANK_DEPLOYMENT, relevance
from tests.unit.fake_retrieval_store import FakeRetrievalStore
from tests.unit.trace_support import assert_trace_equals, expected_trace

COLLECTION = "fr0_unit_collection"
TOPICS = [
    ("overview", "business"),
    ("risk", "risk_factors"),
    ("financial_performance", "md&a"),
    ("competition", "business"),
]
POOLS = ("15", "50")
TEMPLATES = (
    "{company} business description overview",
    "{company} risk factors challenges",
    "{company} financial performance revenue",
    "{company} competition market position",
)
TICKERS = ["AAPL", "NVDA", "SMLL"]


class OverlapReranker:
    identity = "test:word-overlap"

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [relevance(query, document) for document in documents]


def _scorer(query: str, text: str) -> float:
    return relevance(query, text)


def _corpus(ticker: str, name: str, *, company_meta: str | None, deep: bool) -> list[IndexDocument]:
    """Dense order is the order below. The in-pool passages answer each topic only in part, and
    sit in a section other than the topic's for risk and competition; with `deep`, a perfect match
    for each topic in the topic's own section sits at dense rank 20 or later, outside pool 15."""
    points: list[IndexDocument] = []

    def add(text: str, section: str) -> None:
        metadata = {"ticker": ticker, "filing_type": "10-K", "section_key": section}
        if company_meta:
            metadata["company_name"] = company_meta
        points.append(IndexDocument(id=f"{ticker}-{len(points):02d}", text=text, metadata=metadata))

    for number in range(3):
        add(f"{ticker} plain passage {number}", "business")
    add(f"{name} business overview of segments", "business")
    add(f"{name} risk challenges in manufacturing", "business")
    add(f"{name} financial revenue in the quarter", "md&a")
    add(f"{name} competition and market share", "business")
    for number in range(12 if deep else 5):
        add(f"Widgets are catalogued under note {number}", "business")
    if deep:
        add(f"{name} business description overview", "business")
        add(f"{name} risk factors challenges", "risk_factors")
        add(f"{name} financial performance revenue", "md&a")
        add(f"{name} competition market position", "business")
    return points


@pytest.fixture
def store() -> FakeRetrievalStore:
    points = (
        _corpus("AAPL", "Acme", company_meta="Acme", deep=True)
        + _corpus("NVDA", "NVDA", company_meta=None, deep=True)
        + _corpus("SMLL", "SMLL", company_meta=None, deep=False)
    )
    return FakeRetrievalStore(points)


def _workload() -> ModuleType:
    return importlib.import_module("evaluation.app_filings_workload")


def _expected_cases(store: FakeRetrievalStore) -> list[dict[str, Any]]:
    """Per (ticker, topic): the query the app sends and, per pool, the traces and top 3 the store
    and scorer give (the company name is the indexed one of the ticker's top dense hit)."""
    cases: list[dict[str, Any]] = []
    for ticker in TICKERS:
        hit = store.search(ticker, filters=SearchFilters(ticker=ticker), n_results=1).chunks[0]
        name = hit.metadata.get("company_name") or ticker
        for template, (topic, section) in zip(TEMPLATES, TOPICS, strict=True):
            query = template.format(company=name)
            case: dict[str, Any] = {
                "ticker": ticker,
                "topic": topic,
                "query": query,
                "expected_section": section,
            }
            for pool in POOLS:
                trace = expected_trace(
                    store,
                    query,
                    _scorer,
                    ticker=ticker,
                    filing_type=None,
                    source="dense",
                    pool=int(pool),
                    top_k=3,
                )
                sections = {c["chunk_id"]: c["section_key"] for c in trace["candidates"]}
                scores = {c["chunk_id"]: c["rerank_score"] for c in trace["candidates"]}
                case[pool] = {
                    "trace": trace,
                    "top3": [
                        {
                            "chunk_id": chunk_id,
                            "section_key": sections[chunk_id],
                            "rerank_score": scores[chunk_id],
                        }
                        for chunk_id in trace["kept_ids"]
                    ],
                }
            ids = [{row["chunk_id"] for row in case[pool]["top3"]} for pool in POOLS]
            case["top3_agreement"] = len(ids[0] & ids[1]) / 3
            cases.append(case)
    return cases


def test_the_template_constant_is_what_the_report_is_built_from() -> None:
    """test-plan §25: App filings workload report (the app's four topic queries, in query order)."""
    assert getattr(graph, "FILING_TOPIC_QUERIES") == TEMPLATES


def test_each_ticker_and_topic_has_a_case_with_the_apps_query(store: FakeRetrievalStore) -> None:
    """test-plan §25: App filings workload report (the app's four topic queries for each given
    ticker, company name resolved as the app does: the indexed name of the ticker's top dense hit,
    else the ticker)."""
    report = _workload().run_app_filings_workload(
        store, TICKERS, reranker=OverlapReranker(), collection=COLLECTION
    )

    expected = _expected_cases(store)
    assert [(c["ticker"], c["topic"]) for c in report["cases"]] == [
        (c["ticker"], c["topic"]) for c in expected
    ]
    assert [c["query"] for c in report["cases"]] == [c["query"] for c in expected]
    assert [c["expected_section"] for c in report["cases"]] == [
        c["expected_section"] for c in expected
    ]
    assert report["cases"][0]["query"].startswith("Acme ")
    assert report["cases"][4]["query"].startswith("NVDA ")


def test_each_pool_reports_its_top_three_with_section_and_rerank_score(
    store: FakeRetrievalStore,
) -> None:
    """test-plan §25: App filings workload report (per pool, the top 3: chunk id, section, rerank
    score, from `search_reranked(candidates="dense", top_k=3)` at pool 15 and pool 50)."""
    report = _workload().run_app_filings_workload(
        store, TICKERS, reranker=OverlapReranker(), collection=COLLECTION
    )

    for got, want in zip(report["cases"], _expected_cases(store), strict=True):
        for pool in POOLS:
            rows = got[pool]["top3"]
            assert [set(row) for row in rows] == [
                {"chunk_id", "section_key", "rerank_score"}
            ] * len(want[pool]["top3"])
            assert [(r["chunk_id"], r["section_key"]) for r in rows] == [
                (r["chunk_id"], r["section_key"]) for r in want[pool]["top3"]
            ]
            assert [r["rerank_score"] for r in rows] == pytest.approx(
                [r["rerank_score"] for r in want[pool]["top3"]]
            )
    assert any(
        [r["chunk_id"] for r in case["15"]["top3"]] != [r["chunk_id"] for r in case["50"]["top3"]]
        for case in report["cases"]
    ), "precondition: the two pools disagree somewhere"


def test_each_pool_carries_the_trace_of_its_search(store: FakeRetrievalStore) -> None:
    """test-plan §25: App filings workload report (both pools' traces: the candidates at that
    pool, and the pool requested)."""
    report = _workload().run_app_filings_workload(
        store, TICKERS, reranker=OverlapReranker(), collection=COLLECTION
    )

    for got, want in zip(report["cases"], _expected_cases(store), strict=True):
        for pool in POOLS:
            trace = got[pool]["trace"]
            assert trace["candidate_pool"] == int(pool)
            assert_trace_equals(trace, want[pool]["trace"], filing_type=False)
    aapl = report["cases"][0]
    assert len(aapl["15"]["trace"]["candidates"]) == 15
    assert len(aapl["50"]["trace"]["candidates"]) == 23


def test_top3_agreement_is_the_shared_chunk_ids_over_three(store: FakeRetrievalStore) -> None:
    """test-plan §25: App filings workload report (top-3 agreement = shared chunk ids / 3)."""
    report = _workload().run_app_filings_workload(
        store, TICKERS, reranker=OverlapReranker(), collection=COLLECTION
    )

    expected = [c["top3_agreement"] for c in _expected_cases(store)]
    assert [c["top3_agreement"] for c in report["cases"]] == pytest.approx(expected)
    assert min(expected) < 1.0
    assert max(expected) == 1.0


def test_the_summary_gives_the_mean_agreement_and_the_section_hit_rate_per_pool(
    store: FakeRetrievalStore,
) -> None:
    """test-plan §25: App filings workload report (the summary: mean agreement; per pool the share
    of top-3 chunks in the topic's declared section: overview and competition -> business, risk ->
    risk_factors, financial performance -> md&a)."""
    report = _workload().run_app_filings_workload(
        store, TICKERS, reranker=OverlapReranker(), collection=COLLECTION
    )

    expected = _expected_cases(store)
    hit_rate = {}
    for pool in POOLS:
        rows = [(case["expected_section"], row) for case in expected for row in case[pool]["top3"]]
        hit_rate[pool] = sum(row["section_key"] == section for section, row in rows) / len(rows)
    assert hit_rate["15"] != hit_rate["50"], "precondition: the pools differ in section hits"
    assert 0 < min(hit_rate.values()) and max(hit_rate.values()) < 1
    summary = report["summary"]
    assert summary["mean_top3_agreement"] == pytest.approx(
        sum(c["top3_agreement"] for c in expected) / len(expected)
    )
    assert summary["section_hit_rate"]["15"] == pytest.approx(hit_rate["15"])
    assert summary["section_hit_rate"]["50"] == pytest.approx(hit_rate["50"])


def test_the_provenance_records_the_index_embedding_and_reranker(store: FakeRetrievalStore) -> None:
    """test-plan §25: App filings workload report (records the index, embedding and reranker
    identity)."""
    report = _workload().run_app_filings_workload(
        store, TICKERS, reranker=OverlapReranker(), collection=COLLECTION
    )

    provenance = report["provenance"]
    assert provenance["collection"] == COLLECTION
    assert OverlapReranker.identity in provenance.values()
    assert any(
        "embedding" in key and isinstance(value, str) and value.strip()
        for key, value in provenance.items()
    )


def test_the_report_is_json_without_chunk_text(store: FakeRetrievalStore) -> None:
    """test-plan §25: App filings workload report (chunk ids, sections, scores and traces; no
    chunk text)."""
    report = _workload().run_app_filings_workload(
        store, TICKERS, reranker=OverlapReranker(), collection=COLLECTION
    )

    text = json.dumps(report)
    assert report["cases"]
    # The deep passages equal the app's topic queries, which the row requires in the report.
    queries = {case["query"] for case in report["cases"]}
    assert not any(point.text in text for point in store.points if point.text not in queries)


def test_without_a_reranker_the_foundry_deployment_scores_and_is_recorded(
    store: FakeRetrievalStore, foundry_rerank: Any
) -> None:
    """test-plan §25: App filings workload report (the reranker is the configured deployment when
    none is given): one rerank request per ticker, topic and pool, each over that pool's
    candidates."""
    report = _workload().run_app_filings_workload(store, TICKERS, collection=COLLECTION)

    wanted = sorted(
        len(case[pool]["trace"]["candidates"]) for case in _expected_cases(store) for pool in POOLS
    )
    assert sorted(len(r.body["documents"]) for r in foundry_rerank.requests) == wanted
    assert f"foundry:{RERANK_DEPLOYMENT}" in report["provenance"].values()
    expected = _expected_cases(store)
    for got, want in zip(report["cases"], expected, strict=True):
        for pool in POOLS:
            assert [r["chunk_id"] for r in got[pool]["top3"]] == [
                r["chunk_id"] for r in want[pool]["top3"]
            ]
