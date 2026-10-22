"""S2-T04: anchored retrieval benchmark runner and scoring.

Traces to docs/test-plan.md §6 "Benchmark runner" and docs/retrieval-benchmark.md
§3 (matcher), §4 (metrics, amendment 2026-10-22) and §7 (runner).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any

import pytest

from app.components.retrieval.vector_store import (
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)
from evaluation.run_shared_retrieval_benchmark import (
    BENCHMARK_KS,
    RETRIEVAL_METHODS,
    run_anchored_benchmark,
    score_retrieval,
)
from evaluation.validate_retrieval_fixture import AnchoredCase, AnchoredFixture

AAPL_ACC = "0000320193-23-000106"
AAPL_10Q_ACC = "0000320193-23-000077"
MSFT_ACC = "0000789019-23-000014"

RISK_ANCHOR = (
    "Supply chain concentration in a few outsourcing partners could materially "
    "harm future operating results"
)
MDA_ANCHOR = (
    "Net sales increased during the fiscal year driven primarily by higher "
    "services revenue and iPhone demand"
)
MSFT_BUSINESS_ANCHOR = (
    "The company designs manufactures and markets smartphones personal computers "
    "tablets and wearables"
)


# --------------------------------------------------------------------------
# Faithful in-memory store at the retrieval boundary (computes, never canned)
# --------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9&]+", text.lower()))


class InMemoryStore:
    """Scores chunks by query-token overlap; filters by metadata; returns top n."""

    def __init__(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        self._rows = rows

    def _rank(
        self, query: str | None, rows: list[tuple[str, str, dict[str, Any]]], n: int
    ) -> list[RetrievedChunk]:
        q = _tokens(query or "")
        scored = []
        for cid, text, meta in rows:
            overlap = len(q & _tokens(text)) / len(q) if q else 0.0
            scored.append(
                RetrievedChunk(id=cid, text=text, metadata=dict(meta), distance=1.0 - overlap)
            )
        scored.sort(key=lambda c: (c.distance, c.id))
        return scored[:n]

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        n_results: int = 5,
    ) -> SearchResult:
        rows = [
            r
            for r in self._rows
            if filters is None
            or (
                (filters.ticker is None or r[2]["ticker"] == filters.ticker)
                and (filters.filing_type is None or r[2]["filing_type"] == filters.filing_type)
                and (filters.section_key is None or r[2]["section_key"] == filters.section_key)
            )
        ]
        chunks = self._rank(query, rows, n_results)
        return SearchResult(query=query, chunks=chunks, total_results=len(chunks))

    def search_sections(
        self,
        ticker: str,
        sections: list[str],
        query: str | None = None,
        filing_type: str | None = None,
        n_results: int = 5,
    ) -> SearchResult:
        rows = [
            r
            for r in self._rows
            if r[2]["ticker"] == ticker
            and r[2]["section_key"] in sections
            and (filing_type is None or r[2]["filing_type"] == filing_type)
        ]
        chunks = self._rank(query, rows, n_results)
        return SearchResult(query=query or "", chunks=chunks, total_results=len(chunks))


def _row(
    cid: str, text: str, ticker: str, acc: str, section: str, filing_type: str = "10-K"
) -> tuple[str, str, dict[str, Any]]:
    return (
        cid,
        text,
        {
            "ticker": ticker,
            "filing_type": filing_type,
            "accession_number": acc,
            "section_key": section,
        },
    )


CORPUS = [
    _row(
        f"{AAPL_ACC}_risk_factors_000",
        "Macroeconomic conditions and currency fluctuations could affect results",
        "AAPL",
        AAPL_ACC,
        "risk_factors",
    ),
    _row(f"{AAPL_ACC}_risk_factors_001", RISK_ANCHOR, "AAPL", AAPL_ACC, "risk_factors"),
    _row(f"{AAPL_ACC}_md&a_000", MDA_ANCHOR, "AAPL", AAPL_ACC, "md&a"),
    _row(
        f"{AAPL_ACC}_business_000",
        "The company sells mobile devices wearables and related services worldwide",
        "AAPL",
        AAPL_ACC,
        "business",
    ),
    # Same text as the risk anchor but a 10-Q: must be excluded by the filing_type filter.
    _row(
        f"{AAPL_10Q_ACC}_risk_factors_000",
        RISK_ANCHOR,
        "AAPL",
        AAPL_10Q_ACC,
        "risk_factors",
        "10-Q",
    ),
    _row(
        f"{MSFT_ACC}_business_000",
        "Cloud and productivity software are the core of what the company sells "
        "to enterprise customers",
        "MSFT",
        MSFT_ACC,
        "business",
    ),
    _row(
        f"{MSFT_ACC}_risk_factors_000",
        "Cybersecurity incidents and regulatory scrutiny could harm our reputation",
        "MSFT",
        MSFT_ACC,
        "risk_factors",
    ),
]


def _case_dict(
    case_id: str,
    ticker: str,
    acc: str,
    section: str,
    anchor: str,
    query: str,
    keywords: list[str],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "ticker": ticker,
        "query": query,
        "filters": {
            "ticker": ticker,
            "filing_type": "10-K",
            "section_key": section,
            "filing_date_from": None,
            "filing_date_to": None,
            "accession_number": None,
        },
        "expected_sections": [section],
        "answer_intent": "section_specific",
        "gold_evidence": [
            {
                "accession_number": acc,
                "section_key": section,
                "anchor_text": anchor,
                "relevance": 2,
            }
        ],
        "expected_keywords": keywords,
        "notes": None,
    }


def _fixture() -> AnchoredFixture:
    return AnchoredFixture.model_validate(
        {
            "fixture_version": "retrieval_shared_benchmark_v1",
            "generated_at": "2026-06-03T00:00:00Z",
            "embedding_model": "test-embedding@1",
            "cases": [
                _case_dict(
                    "aapl-risk-001",
                    "AAPL",
                    AAPL_ACC,
                    "risk_factors",
                    RISK_ANCHOR,
                    "supply chain concentration outsourcing partners harm operating results",
                    ["supply", "outsourcing"],
                ),
                _case_dict(
                    "aapl-mda-001",
                    "AAPL",
                    AAPL_ACC,
                    "md&a",
                    MDA_ANCHOR,
                    "net sales increased fiscal year services revenue iphone demand",
                    [],
                ),
                # The MSFT corpus never contains this anchor: a guaranteed miss.
                _case_dict(
                    "msft-business-001",
                    "MSFT",
                    MSFT_ACC,
                    "business",
                    MSFT_BUSINESS_ANCHOR,
                    "enterprise software cloud revenue",
                    ["cloud", "azure"],
                ),
            ],
        }
    )


def _run(method: str, **kwargs: Any) -> dict[str, Any]:
    return run_anchored_benchmark(
        _fixture(),
        InMemoryStore(CORPUS),
        backend="qdrant",
        retrieval_method=method,
        **kwargs,
    )


def _by_id(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["case_id"]: c for c in result["cases"]}


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Run on fixture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["dense", "section_aware"])
def test_result_contains_every_case_id(method: str) -> None:
    """test-plan §6 Benchmark runner: Run on fixture."""
    fixture = _fixture()
    result = _run(method)
    assert [c["case_id"] for c in result["cases"]] == [c.case_id for c in fixture.cases]
    assert result["fixture_version"] == fixture.fixture_version
    assert result["backend"] == "qdrant"
    assert result["ks"] == [5, 10]
    assert result["summary"]["n_cases"] == len(fixture.cases)
    for case in result["cases"]:
        assert set(case["metrics"]) == {"5", "10"}


def test_dense_search_is_scoped_to_ticker_and_filing_type_only() -> None:
    """test-plan §6 Benchmark runner: Run on fixture (case filters per retrieval-benchmark §7)."""
    case = _by_id(_run("dense"))["aapl-risk-001"]
    retrieved = case["retrieved"]
    assert {r["accession_number"] for r in retrieved} == {AAPL_ACC}
    # section_key from the case filters is not applied: every AAPL 10-K section is searched.
    assert {r["section_key"] for r in retrieved} == {"risk_factors", "md&a", "business"}
    assert len(retrieved) == 4


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Method/mode labels
# --------------------------------------------------------------------------


def test_method_constants() -> None:
    """test-plan §6 Benchmark runner: Method/mode labels."""
    assert tuple(RETRIEVAL_METHODS) == ("dense", "section_aware")
    assert tuple(BENCHMARK_KS) == (5, 10)


@pytest.mark.parametrize("method", ["dense", "section_aware"])
def test_result_carries_method_and_mode_labels(method: str) -> None:
    """test-plan §6 Benchmark runner: Method/mode labels (present)."""
    result = _run(method)
    assert result["retrieval_method"] == method
    assert result["mode"] == method


@pytest.mark.parametrize("method", ["bm25", "", "Dense", "hybrid"])
def test_unknown_retrieval_method_is_rejected(method: str) -> None:
    """test-plan §6 Benchmark runner: Method/mode labels (validated)."""
    with pytest.raises(ValueError):
        _run(method)


def test_empty_backend_is_rejected() -> None:
    """test-plan §6 Benchmark runner: Method/mode labels (validated)."""
    with pytest.raises(ValueError):
        run_anchored_benchmark(
            _fixture(),
            InMemoryStore(CORPUS),
            backend="",
            retrieval_method="dense",
        )


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Null first relevant rank
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["dense", "section_aware"])
def test_miss_emits_null_first_relevant_rank_in_json(method: str) -> None:
    """test-plan §6 Benchmark runner: Null first relevant rank."""
    result = _run(method)
    miss = _by_id(result)["msft-business-001"]
    assert miss["first_relevant_rank"] is None
    for k in ("5", "10"):
        assert miss["metrics"][k]["first_relevant_rank"] is None
        assert miss["metrics"][k]["mrr"] == 0
        assert miss["metrics"][k]["pass"] == 0
    assert all(not r["gold_hit"] for r in miss["retrieved"])

    parsed = json.loads(json.dumps(result))
    parsed_miss = {c["case_id"]: c for c in parsed["cases"]}["msft-business-001"]
    assert parsed_miss["first_relevant_rank"] is None
    assert '"first_relevant_rank": null' in json.dumps(result)


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Provenance
# --------------------------------------------------------------------------


def test_dense_provenance_preserves_chunk_ids_sections_and_scores() -> None:
    """test-plan §6 Benchmark runner: Provenance."""
    store = InMemoryStore(CORPUS)
    fixture = _fixture()
    result = _by_id(_run("dense"))
    for case in fixture.cases:
        expected = store.search(
            case.query,
            filters=SearchFilters(ticker=case.ticker, filing_type="10-K"),
            n_results=10,
        ).chunks
        got = result[case.case_id]["retrieved"]
        assert [r["rank"] for r in got] == list(range(1, len(expected) + 1))
        assert [r["chunk_id"] for r in got] == [c.id for c in expected]
        assert [r["accession_number"] for r in got] == [
            c.metadata["accession_number"] for c in expected
        ]
        assert [r["section_key"] for r in got] == [c.metadata["section_key"] for c in expected]
        assert [r["score"] for r in got] == pytest.approx([c.relevance_score for c in expected])


@pytest.mark.parametrize("method", ["dense", "section_aware"])
def test_retrieved_entries_trace_back_to_the_store(method: str) -> None:
    """test-plan §6 Benchmark runner: Provenance."""
    by_id = {cid: meta for cid, _text, meta in CORPUS}
    for case in _run(method)["cases"]:
        assert [r["rank"] for r in case["retrieved"]] == list(range(1, len(case["retrieved"]) + 1))
        for r in case["retrieved"]:
            meta = by_id[r["chunk_id"]]
            assert r["accession_number"] == meta["accession_number"]
            assert r["section_key"] == meta["section_key"]
            assert isinstance(r["score"], float)
            assert 0.0 < r["score"] <= 1.0
            assert isinstance(r["gold_hit"], bool)


def test_gold_hit_flags_agree_with_first_relevant_rank() -> None:
    """test-plan §6 Benchmark runner: Provenance."""
    case = _by_id(_run("dense"))["aapl-risk-001"]
    assert case["retrieved"][0]["chunk_id"] == f"{AAPL_ACC}_risk_factors_001"
    assert case["retrieved"][0]["score"] == pytest.approx(1.0)
    assert case["first_relevant_rank"] == 1
    assert [r["rank"] for r in case["retrieved"] if r["gold_hit"]] == [1]


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [({"fixture_path": "f.json", "collection": "bench_x"}, None), (None, {})],
)
def test_run_provenance_is_recorded(
    provenance: dict[str, str] | None, expected: dict[str, str] | None
) -> None:
    """test-plan §6 Benchmark runner: Provenance."""
    result = _run("dense", provenance=provenance)
    assert result["provenance"] == (provenance if expected is None else expected)


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Cold/warm latency
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["dense", "section_aware"])
def test_cold_and_warm_latency_are_captured_separately(method: str) -> None:
    """test-plan §6 Benchmark runner: Cold/warm latency."""
    result = _run(method)
    for case in result["cases"]:
        assert set(case["latency_ms"]) == {"cold", "warm"}
        assert case["latency_ms"]["cold"] >= 0.0
        assert case["latency_ms"]["warm"] >= 0.0
    cold = [c["latency_ms"]["cold"] for c in result["cases"]]
    warm = [c["latency_ms"]["warm"] for c in result["cases"]]
    assert result["summary"]["latency_ms"]["cold_p50"] == pytest.approx(statistics.median(cold))
    assert result["summary"]["latency_ms"]["warm_p50"] == pytest.approx(statistics.median(warm))


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["dense", "section_aware"])
def test_same_fixture_backend_method_gives_identical_metrics(method: str) -> None:
    """test-plan §6 Benchmark runner: Determinism."""
    first, second = _run(method), _run(method)
    for a, b in zip(first["cases"], second["cases"], strict=True):
        assert a["case_id"] == b["case_id"]
        assert a["metrics"] == b["metrics"]
        assert a["first_relevant_rank"] == b["first_relevant_rank"]
        assert [r["chunk_id"] for r in a["retrieved"]] == [r["chunk_id"] for r in b["retrieved"]]
    assert first["summary"]["metrics"] == second["summary"]["metrics"]


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4
# (end-to-end, dense: ranking is fully determined by the fake's token overlap)
# --------------------------------------------------------------------------


def test_dense_run_metrics_and_summary_means() -> None:
    """test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4."""
    result = _run("dense")
    cases = _by_id(result)

    # aapl-risk-001: the anchor chunk is rank 1 of 4 retrieved.
    for k, precision in (("5", 1 / 5), ("10", 1 / 10)):  # divides by K, not by 4
        m = cases["aapl-risk-001"]["metrics"][k]
        assert m["pass"] == 1
        assert m["recall"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(precision)
        assert m["mrr"] == pytest.approx(1.0)
        assert m["ndcg"] == pytest.approx(1.0)
        assert m["section_recall"] == pytest.approx(1.0)
        assert m["keyword_hit_rate"] == pytest.approx(1.0)
        assert m["first_relevant_rank"] == 1

    # aapl-mda-001: no expected keywords -> null keyword_hit_rate.
    mda = cases["aapl-mda-001"]["metrics"]["5"]
    assert mda["keyword_hit_rate"] is None
    assert mda["pass"] == 1

    # msft-business-001: anchor absent, but the gold (accession, section) is retrieved:
    # section_recall credits it while recall does not. keywords: cloud yes, azure no.
    miss = cases["msft-business-001"]["metrics"]["5"]
    assert miss["recall"] == 0
    assert miss["precision"] == 0
    assert miss["ndcg"] == 0
    assert miss["section_recall"] == pytest.approx(1.0)
    assert miss["keyword_hit_rate"] == pytest.approx(0.5)

    # Summary: mean over cases, null values left out, first_relevant_rank not averaged.
    s5 = result["summary"]["metrics"]["5"]
    assert s5["pass"] == pytest.approx(2 / 3)
    assert s5["recall"] == pytest.approx(2 / 3)
    assert s5["keyword_hit_rate"] == pytest.approx((1.0 + 0.5) / 2)
    assert "first_relevant_rank" not in s5


# --------------------------------------------------------------------------
# test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4
# (score_retrieval on a hand-computed case)
# --------------------------------------------------------------------------

G1_ANCHOR = (
    "The company's supply chain depends on a limited number of outsourcing "
    "partners in Asia"
)  # relevance 2
G2_ANCHOR = (
    "Component shortages or pricing changes from single source suppliers could "
    "delay product shipments significantly"
)  # relevance 1
G3_HEAD = "Net sales increased during the fiscal year driven"
G3_TAIL = "primarily by higher services revenue and iPhone demand"
G3_ANCHOR = f"{G3_HEAD} {G3_TAIL}"  # relevance 1, split across two chunks below


def _gold_case(keywords: list[str]) -> AnchoredCase:
    def gold(section: str, anchor: str, rel: int) -> dict[str, Any]:
        return {
            "accession_number": AAPL_ACC,
            "section_key": section,
            "anchor_text": anchor,
            "relevance": rel,
        }

    return AnchoredCase.model_validate(
        {
            "case_id": "aapl-hand-001",
            "ticker": "AAPL",
            "query": "supply chain risks and sales growth",
            "filters": {
                "ticker": "AAPL",
                "filing_type": "10-K",
                "section_key": None,
                "filing_date_from": None,
                "filing_date_to": None,
                "accession_number": None,
            },
            "expected_sections": ["md&a", "risk_factors"],
            "answer_intent": "comparison",
            "gold_evidence": [
                gold("risk_factors", G1_ANCHOR, 2),
                gold("risk_factors", G2_ANCHOR, 1),
                gold("md&a", G3_ANCHOR, 1),
            ],
            "expected_keywords": keywords,
            "notes": None,
        }
    )


def _chunk(
    n: int,
    text: str,
    section: str = "risk_factors",
    acc: str = AAPL_ACC,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{acc}_{section}_{n:03d}",
        text=text,
        metadata={"accession_number": acc, "section_key": section},
        distance=0.1 * n,
    )


HAND_CHUNKS = [
    # rank 1: right section, no anchor -> section hit only. Keyword "supply".
    _chunk(1, "Macroeconomic conditions could affect the supply of components."),
    # rank 2: anchor G1 with curly apostrophe, odd casing and newline/space runs.
    _chunk(
        2,
        "Item 1A.  Risk\nThe company’s   supply chain depends on a limited "
        "number of OUTSOURCING partners in Asia.",
    ),
    # rank 3: contains G1 (already credited) and G2 (uncredited).
    _chunk(3, f"{G1_ANCHOR}. Separately: {G2_ANCHOR}."),
    # rank 4: G1 again -> a relevant chunk for precision, but gain 0.
    _chunk(4, f"Repeated: {G1_ANCHOR}."),
    # rank 5/6: G3 split across two chunks -> neither hits (straddle rule).
    _chunk(5, G3_HEAD, section="md&a"),
    _chunk(6, G3_TAIL, section="md&a"),
]

L3 = math.log2(3)  # discount denominator at rank 2

# Gold relevances are [2, 1, 1]; discount at rank i is 1 / log2(i + 1).
# Hit chunks: r2 (G1), r3 (G1 credited -> G2), r4 (G1 credited -> nothing left, gain 0).
# DCG@K (K >= 3) = 2/log2(3) + 1/log2(4)                     = 1.7618595
# IDCG@K (K >= 3) = 2/log2(2) + 1/log2(3) + 1/log2(4)        = 3.1309298
# NDCG@K (K >= 3) = 1.7618595 / 3.1309298                     = 0.5627273
HAND_EXPECTED = [
    pytest.param(
        1,
        # [r1]: no hit. Section pair (AAPL, risk_factors) seen -> 1 of 2 pairs.
        # Keywords [supply, Outsourcing, dividend]: only "supply" -> 1/3.
        {
            "pass": 0,
            "recall": 0,
            "precision": 0,
            "mrr": 0,
            "ndcg": 0,
            "section_recall": 1 / 2,
            "keyword_hit_rate": 1 / 3,
            "first_relevant_rank": None,
        },
        id="k1-miss",
    ),
    pytest.param(
        2,
        # [r1, r2]: G1 covered at rank 2 -> recall 1/3, precision 1/2 (1 hit chunk / K=2),
        # mrr 1/2. DCG = 2/log2(3). IDCG@2 = 2/log2(2) + 1/log2(3) = 2 + 1/log2(3).
        # NDCG = 0.4796249. Keywords: supply, outsourcing (case-folded) -> 2/3.
        {
            "pass": 1,
            "recall": 1 / 3,
            "precision": 1 / 2,
            "mrr": 1 / 2,
            "ndcg": (2 / L3) / (2 + 1 / L3),
            "section_recall": 1 / 2,
            "keyword_hit_rate": 2 / 3,
            "first_relevant_rank": 2,
        },
        id="k2-curly-quote-whitespace-case-hit",
    ),
    pytest.param(
        5,
        # [r1..r5]: G1 and G2 covered, G3 not (head only in r5) -> recall 2/3.
        # Hit chunks r2, r3, r4 -> precision 3/5. mrr 1/2.
        # r5 (md&a, no anchor) is a section hit: section_recall 2/2 while recall stays 2/3.
        # NDCG = 1.7618595 / 3.1309298 = 0.5627273 (r4 earns no extra credit).
        {
            "pass": 1,
            "recall": 2 / 3,
            "precision": 3 / 5,
            "mrr": 1 / 2,
            "ndcg": (2 / L3 + 1 / 2) / (2 + 1 / L3 + 1 / 2),
            "section_recall": 1.0,
            "keyword_hit_rate": 2 / 3,
            "first_relevant_rank": 2,
        },
        id="k5-credit-once-and-section-only",
    ),
    pytest.param(
        6,
        # r6 completes the G3 sentence only jointly with r5: still a miss (straddle).
        # precision 3/6; recall and NDCG unchanged (IDCG has only 3 gold items).
        {
            "pass": 1,
            "recall": 2 / 3,
            "precision": 3 / 6,
            "mrr": 1 / 2,
            "ndcg": (2 / L3 + 1 / 2) / (2 + 1 / L3 + 1 / 2),
            "section_recall": 1.0,
            "keyword_hit_rate": 2 / 3,
            "first_relevant_rank": 2,
        },
        id="k6-straddle-is-a-miss",
    ),
    pytest.param(
        10,
        # Only 6 chunks exist: precision still divides by K = 10 -> 3/10.
        {
            "pass": 1,
            "recall": 2 / 3,
            "precision": 3 / 10,
            "mrr": 1 / 2,
            "ndcg": (2 / L3 + 1 / 2) / (2 + 1 / L3 + 1 / 2),
            "section_recall": 1.0,
            "keyword_hit_rate": 2 / 3,
            "first_relevant_rank": 2,
        },
        id="k10-precision-divides-by-k",
    ),
]


@pytest.mark.parametrize(("k", "expected"), HAND_EXPECTED)
def test_score_retrieval_matches_hand_computed_values(k: int, expected: dict[str, Any]) -> None:
    """test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4."""
    case = _gold_case(["supply", "Outsourcing", "dividend"])
    got = score_retrieval(case, HAND_CHUNKS, k)
    assert set(got) == set(expected)
    for name, value in expected.items():
        if value is None:
            assert got[name] is None, name
        else:
            assert got[name] == pytest.approx(value), name


def test_best_uncredited_anchor_is_credited_one_at_a_time() -> None:
    """test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4 (NDCG)."""
    # A single chunk satisfying G1 (rel 2) and G2 (rel 1) earns gain 2 and credits only G1.
    # The next chunk, holding only G2, then earns gain 1.
    # DCG@5 = 2/log2(2) + 1/log2(3); IDCG@5 also counts G3 (rel 1) at rank 3:
    # 2/log2(2) + 1/log2(3) + 1/log2(4) -> NDCG = 2.6309298 / 3.1309298 = 0.8403011.
    case = _gold_case([])
    chunks = [
        _chunk(1, f"{G1_ANCHOR}. {G2_ANCHOR}."),
        _chunk(2, f"Only this: {G2_ANCHOR}."),
    ]
    got = score_retrieval(case, chunks, 5)
    assert got["ndcg"] == pytest.approx((2 + 1 / L3) / (2 + 1 / L3 + 1 / 2))
    assert got["recall"] == pytest.approx(2 / 3)
    assert got["precision"] == pytest.approx(2 / 5)
    assert got["first_relevant_rank"] == 1
    assert got["mrr"] == pytest.approx(1.0)


def test_keyword_hit_rate_is_null_without_expected_keywords() -> None:
    """test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4 (keywords)."""
    got = score_retrieval(_gold_case([]), HAND_CHUNKS, 5)
    assert got["keyword_hit_rate"] is None
    assert got["recall"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("section", "acc"),
    [
        pytest.param("business", AAPL_ACC, id="right-accession-wrong-section"),
        pytest.param("risk_factors", AAPL_10Q_ACC, id="right-section-wrong-accession"),
    ],
)
def test_anchor_in_the_wrong_section_or_filing_is_not_a_hit(section: str, acc: str) -> None:
    """test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4 (matcher)."""
    got = score_retrieval(_gold_case([]), [_chunk(1, G1_ANCHOR, section=section, acc=acc)], 5)
    assert got["pass"] == 0
    assert got["recall"] == 0
    assert got["precision"] == 0
    assert got["mrr"] == 0
    assert got["ndcg"] == 0
    assert got["first_relevant_rank"] is None


def test_chunks_beyond_k_are_ignored() -> None:
    """test-plan §6 Benchmark runner: Metrics match retrieval-benchmark §3-§4 (ranks 1-based)."""
    chunks = [_chunk(1, "No anchor here."), _chunk(2, G1_ANCHOR)]
    case = _gold_case([])
    assert score_retrieval(case, chunks, 1)["first_relevant_rank"] is None
    at2 = score_retrieval(case, chunks, 2)
    assert at2["first_relevant_rank"] == 2
    assert at2["mrr"] == pytest.approx(1 / 2)
