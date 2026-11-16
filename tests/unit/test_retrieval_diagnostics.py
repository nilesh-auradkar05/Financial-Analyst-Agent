"""S5.1: retrieval miss diagnostics and failure taxonomy.

Oracle: docs/test-plan.md section 6 "Retrieval diagnostics: S5.1"; docs/retrieval-benchmark.md
sections 3 (matcher, straddle rule) and 7a (miss diagnostics).

The fixture, result and chunk inputs are built here with the shapes the benchmark runner and the
ingestion chunker emit. `gold_hit` flags are derived with the public section 3 matcher
(`normalize_text`) so each hand-built result is consistent with its chunks. Overlap is tested
through `classify_miss` with plain floats; report-level tests only use extremes (a chunk that
holds every query word, a chunk that holds none).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from app.components.retrieval.section_intent import infer_section_intent
from app.components.retrieval.vector_store import (
    IndexDocument,
    RetrievedChunk,
    SearchFilters,
    SearchResult,
)
from evaluation.retrieval_diagnostics import (
    DENSE_POOL_DEPTH,
    OVERLAP_THRESHOLD,
    AnchorMiss,
    DiagnosticsReport,
    RebuildMismatchError,
    build_diagnostics_report,
    classify_miss,
    collect_dense_pool,
)
from evaluation.validate_retrieval_fixture import AnchoredFixture, normalize_text

ACC = "0000320193-25-000079"
ACC2 = "0000320193-24-000123"
Q_RISK = "What does Apple say about supply chain risks?"
Q_FX = "How does Apple describe foreign currency exchange risk?"
SOURCE = "evaluation/results/unit_result.json"
CLASSES = ("boundary_split", "section_miss", "vocabulary_gap", "ranking")
CANDIDATES = ("section_prior", "hybrid_bm25", "rerank", "chunking", "query_rewrite")


# --------------------------------------------------------------------------- builders


def _anchor(tag: str) -> str:
    """An anchor of 8+ words that shares no word with either query."""
    return f"{tag} lanterns drift slowly over quiet harbors at midnight"


def _chunk(acc: str, section: str, idx: int, text: str) -> IndexDocument:
    return IndexDocument(
        id=f"{acc}_{section}_{idx:03d}",
        text=text,
        metadata={"accession_number": acc, "section_key": section, "ticker": "AAPL"},
    )


def _filler(acc: str, section: str, idx: int) -> IndexDocument:
    return _chunk(acc, section, idx, f"Unrelated filler about zebras and glaciers number {idx}.")


def _gold(acc: str, section: str, anchor: str, relevance: int = 2) -> dict[str, Any]:
    return {
        "accession_number": acc,
        "section_key": section,
        "anchor_text": anchor,
        "relevance": relevance,
    }


def _case(
    case_id: str, query: str, golds: list[dict[str, Any]], ticker: str = "AAPL"
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "ticker": ticker,
        "query": query,
        "filters": {"ticker": ticker, "filing_type": "10-K"},
        "expected_sections": sorted({g["section_key"] for g in golds}),
        "answer_intent": "risk",
        "gold_evidence": golds,
    }


def _fixture(cases: list[dict[str, Any]]) -> AnchoredFixture:
    return AnchoredFixture.model_validate(
        {
            "fixture_version": "unit_fixture_v1",
            "generated_at": "2026-11-14T00:00:00Z",
            "embedding_model": "fixture:model:4",
            "cases": cases,
        }
    )


def _gold_hit(case: Any, chunk: IndexDocument) -> bool:
    """The retrieval-benchmark section 3 matcher."""
    text = normalize_text(chunk.text)
    return any(
        g.accession_number == chunk.metadata["accession_number"]
        and g.section_key == chunk.metadata["section_key"]
        and normalize_text(g.anchor_text) in text
        for g in case.gold_evidence
    )


def _result(
    fixture: AnchoredFixture,
    retrieved: dict[str, list[IndexDocument]],
    order: Sequence[str],
) -> dict[str, Any]:
    by_id = {case.case_id: case for case in fixture.cases}
    cases = []
    for case_id in order:
        case = by_id[case_id]
        rows = [
            {
                "rank": rank,
                "chunk_id": chunk.id,
                "accession_number": chunk.metadata["accession_number"],
                "section_key": chunk.metadata["section_key"],
                "score": round(1.0 - 0.01 * rank, 4),
                "gold_hit": _gold_hit(case, chunk),
            }
            for rank, chunk in enumerate(retrieved[case_id], start=1)
        ]
        covered = sum(
            any(
                g.accession_number == c.metadata["accession_number"]
                and g.section_key == c.metadata["section_key"]
                and normalize_text(g.anchor_text) in normalize_text(c.text)
                for c in retrieved[case_id][:10]
            )
            for g in case.gold_evidence
        )
        recall = covered / len(case.gold_evidence)
        cases.append(
            {
                "case_id": case_id,
                "ticker": case.ticker,
                "retrieved": rows,
                "metrics": {"10": {"pass": int(covered > 0), "recall": recall}},
            }
        )
    return {
        "fixture_version": fixture.fixture_version,
        "provenance": {
            "evidence_release": "unit-release:9.9.9",
            "embedding_model": "fake:model:8",
            "fixture_path": "unused.json",
        },
        "cases": cases,
    }


@dataclass
class World:
    fixture: AnchoredFixture
    result: dict[str, Any]
    chunks: dict[str, IndexDocument]
    pool: dict[str, list[str]]
    order: list[str]


def _build_world() -> World:
    """Eleven cases that between them exercise every class, the precedence order and the
    covered / partial / duplicate-chunk / cutoff edge cases."""
    chunks: list[IndexDocument] = []
    cases: list[dict[str, Any]] = []
    retrieved: dict[str, list[IndexDocument]] = {}

    other_rf = _filler(ACC, "risk_factors", 90)
    other_biz = _filler(ACC, "business", 91)
    chunks += [other_rf, other_biz]

    # 1. covered: not a miss.
    a1 = _anchor("covered")
    rf_c0 = _chunk(ACC, "risk_factors", 1, f"Preface words. {a1} Closing words.")
    chunks.append(rf_c0)
    cases.append(_case("aapl_covered", Q_RISK, [_gold(ACC, "risk_factors", a1)]))
    retrieved["aapl_covered"] = [rf_c0]

    # 2. partial: anchor 0 covered, anchor 1 (business, not retrieved) is a section_miss.
    a2a, a2b = _anchor("partial_a"), _anchor("partial_b")
    mr_p0 = _chunk(ACC, "market_risk", 2, f"Preface words. {a2a} Closing words.")
    biz_p0 = _chunk(ACC, "business", 2, f"Preface words. {a2b} Closing words.")
    chunks += [mr_p0, biz_p0]
    cases.append(
        _case(
            "aapl_partial",
            Q_FX,
            [_gold(ACC, "market_risk", a2a), _gold(ACC, "business", a2b, relevance=1)],
        )
    )
    retrieved["aapl_partial"] = [mr_p0, other_rf]

    # 3. section_miss; gold section is the one the query's intent names; chunk not in the pool.
    a3 = _anchor("fxmiss")
    mr_f0 = _chunk(ACC, "market_risk", 3, f"Preface words. {a3} Closing words.")
    chunks.append(mr_f0)
    cases.append(_case("aapl_fx_miss", Q_FX, [_gold(ACC, "market_risk", a3)]))
    retrieved["aapl_fx_miss"] = [other_biz, other_rf]

    # 4. vocabulary_gap: section reached, anchor chunk shares no query word.
    a4 = _anchor("vocab")
    rf_v0 = _chunk(ACC, "risk_factors", 4, f"Preface words. {a4} Closing words.")
    chunks.append(rf_v0)
    cases.append(_case("aapl_vocab", Q_RISK, [_gold(ACC, "risk_factors", a4)]))
    retrieved["aapl_vocab"] = [other_rf]

    # 5. ranking: section reached, anchor chunk holds every query word.
    a5 = _anchor("rank")
    rf_r0 = _chunk(ACC, "risk_factors", 5, f"{Q_RISK} {a5} Trailing note.")
    chunks.append(rf_r0)
    cases.append(_case("aapl_rank", Q_RISK, [_gold(ACC, "risk_factors", a5)]))
    retrieved["aapl_rank"] = [other_rf]

    # 6/7. straddle: the anchor is split over two chunks of the gold section.
    a6 = _anchor("straddle")
    head, tail = a6.split(" over ", 1)
    rf_t0 = _chunk(ACC, "risk_factors", 6, f"Preface words. {head}")
    rf_t1 = _chunk(ACC, "risk_factors", 7, f"over {tail} Closing words.")
    chunks += [rf_t0, rf_t1]
    cases.append(_case("aapl_straddle_reached", Q_RISK, [_gold(ACC, "risk_factors", a6)]))
    retrieved["aapl_straddle_reached"] = [other_rf]
    cases.append(_case("aapl_straddle_unreached", Q_RISK, [_gold(ACC, "risk_factors", a6)]))
    retrieved["aapl_straddle_unreached"] = [other_biz]

    # 8. gold accession differs from the retrieved chunk's accession: section not reached.
    a8 = _anchor("otheracc")
    rf_x0 = _chunk(ACC2, "risk_factors", 8, f"Preface words. {a8} Closing words.")
    chunks.append(rf_x0)
    cases.append(_case("aapl_other_accession", Q_RISK, [_gold(ACC2, "risk_factors", a8)]))
    retrieved["aapl_other_accession"] = [other_rf]

    # 9. anchor wholly inside two overlapping chunks, plus decoys with the same anchor text in
    #    another section and another accession.
    a9 = _anchor("dup")
    rf_d1 = _chunk(ACC, "risk_factors", 11, f"Preface words. {a9} Closing words.")
    rf_d2 = _chunk(ACC, "risk_factors", 12, f"{Q_RISK} {a9} Trailing note.")
    decoy_section = _chunk(ACC, "business", 13, f"Preface words. {a9} Closing words.")
    decoy_accession = _chunk(ACC2, "risk_factors", 14, f"Preface words. {a9} Closing words.")
    chunks += [rf_d1, rf_d2, decoy_section, decoy_accession]
    cases.append(_case("aapl_dup_chunks", Q_RISK, [_gold(ACC, "risk_factors", a9)]))
    retrieved["aapl_dup_chunks"] = [other_rf]

    # 10. curly apostrophe in the anchor, straight one in the chunk: same text under norm().
    a10 = "The Company’s lanterns drift slowly over quiet harbors at midnight"
    rf_n0 = _chunk(
        ACC,
        "risk_factors",
        15,
        "Preface words. The Company's lanterns drift slowly "
        "over quiet harbors at midnight Closing words.",
    )
    chunks.append(rf_n0)
    cases.append(_case("aapl_curly", Q_RISK, [_gold(ACC, "risk_factors", a10)]))
    retrieved["aapl_curly"] = [other_rf]

    # 11. top-10 cutoff: the only gold-section row sits at rank 11 and does not count.
    a11 = _anchor("cutoff")
    rf_k0 = _chunk(ACC, "risk_factors", 16, f"Preface words. {a11} Closing words.")
    biz_rows = [_filler(ACC, "business", 100 + i) for i in range(10)]
    chunks += [rf_k0, *biz_rows]
    cases.append(_case("aapl_cutoff", Q_RISK, [_gold(ACC, "risk_factors", a11)]))
    retrieved["aapl_cutoff"] = [*biz_rows, rf_k0]

    fixture = _fixture(cases)
    order = [
        "aapl_vocab",
        "aapl_straddle_unreached",
        "aapl_partial",
        "aapl_cutoff",
        "aapl_covered",
        "aapl_rank",
        "aapl_dup_chunks",
        "aapl_other_accession",
        "aapl_fx_miss",
        "aapl_curly",
        "aapl_straddle_reached",
    ]
    filler_ids = [f"pool_filler_{i:02d}" for i in range(49)]
    pool = {
        "aapl_partial": [*filler_ids[:11], biz_p0.id],  # position 12
        "aapl_fx_miss": ["unrelated_a", "unrelated_b"],  # anchor chunk absent
        "aapl_vocab": [*filler_ids[:4], rf_v0.id],  # position 5
        "aapl_rank": [rf_r0.id],  # position 1
        "aapl_straddle_reached": [rf_t0.id, rf_t1.id],  # no anchor chunk exists
        "aapl_other_accession": [*filler_ids[:29], rf_x0.id],  # position 30
        "aapl_dup_chunks": [decoy_section.id, rf_d2.id, rf_d1.id],  # first anchor chunk: 2
        "aapl_curly": [*filler_ids[:6], rf_n0.id],  # position 7
        "aapl_cutoff": [*filler_ids, rf_k0.id],  # position 50
    }
    return World(
        fixture=fixture,
        result=_result(fixture, retrieved, order),
        chunks={chunk.id: chunk for chunk in chunks},
        pool=pool,
        order=order,
    )


@pytest.fixture
def world() -> World:
    return _build_world()


def _report(world: World, *, with_pool: bool = True) -> DiagnosticsReport:
    return build_diagnostics_report(
        world.result,
        world.fixture,
        world.chunks,
        source_result=SOURCE,
        dense_pool=world.pool if with_pool else None,
    )


def _by_key(report: DiagnosticsReport) -> dict[tuple[str, int], AnchorMiss]:
    return {(m.case_id, m.anchor_index): m for m in report.misses}


# --------------------------------------------------------------------------- constants and models


def test_declared_constants() -> None:
    """test-plan section 6: threshold 0.5 and pool depth 50 are declared up front (benchmark 7a)."""
    assert OVERLAP_THRESHOLD == 0.5
    assert DENSE_POOL_DEPTH == 50


def test_rebuild_mismatch_error_is_a_value_error() -> None:
    """test-plan section 6 'Rebuild guard': the refusal is a typed ValueError."""
    assert issubclass(RebuildMismatchError, ValueError)


def test_report_roundtrips_and_forbids_extra_fields(world: World) -> None:
    """Contract: report models are strict Pydantic models and survive a JSON round trip."""
    report = _report(world)
    assert DiagnosticsReport.model_validate_json(report.model_dump_json()) == report
    miss = report.misses[0]
    with pytest.raises(ValidationError):
        AnchorMiss.model_validate({**miss.model_dump(), "unexpected": 1})


# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("anchor_chunk_ids", "section_reached", "overlap", "expected"),
    [
        # Anchor in no single chunk: boundary_split, whatever else is true.
        ([], True, None, "boundary_split"),
        ([], False, None, "boundary_split"),
        ([], True, 0.0, "boundary_split"),
        ([], False, 0.9, "boundary_split"),
        # Section not reached beats the overlap classes.
        (["c1"], False, 0.0, "section_miss"),
        (["c1"], False, 0.49, "section_miss"),
        (["c1"], False, 1.0, "section_miss"),
        # Section reached: the 0.5 threshold, 0.5 exactly is ranking.
        (["c1"], True, 0.0, "vocabulary_gap"),
        (["c1"], True, 0.49, "vocabulary_gap"),
        (["c1"], True, 0.4999, "vocabulary_gap"),
        (["c1"], True, 0.5, "ranking"),
        (["c1"], True, 0.51, "ranking"),
        (["c1", "c2"], True, 1.0, "ranking"),
    ],
)
def test_classify_miss_precedence_and_threshold(
    anchor_chunk_ids: list[str], section_reached: bool, overlap: float | None, expected: str
) -> None:
    """test-plan section 6 rows 'Anchor in no single chunk', 'No top-10 chunk', 'overlap < 0.5',
    'overlap >= 0.5 (0.5 exactly is ranking)' and 'Precedence'."""
    assert (
        classify_miss(
            anchor_chunk_ids=anchor_chunk_ids, section_reached=section_reached, overlap=overlap
        )
        == expected
    )


# --------------------------------------------------------------------------- per-miss behavior

EXPECTED = {
    # (case_id, anchor_index): class, reached, anchor chunk suffixes, dense rank, intent hit
    ("aapl_partial", 1): ("section_miss", False, ["business_002"], 12, False),
    ("aapl_fx_miss", 0): ("section_miss", False, ["market_risk_003"], None, True),
    ("aapl_vocab", 0): ("vocabulary_gap", True, ["risk_factors_004"], 5, True),
    ("aapl_rank", 0): ("ranking", True, ["risk_factors_005"], 1, True),
    ("aapl_straddle_reached", 0): ("boundary_split", True, [], None, True),
    ("aapl_straddle_unreached", 0): ("boundary_split", False, [], None, True),
    ("aapl_other_accession", 0): ("section_miss", False, ["risk_factors_008"], 30, True),
    ("aapl_dup_chunks", 0): ("ranking", True, ["risk_factors_011", "risk_factors_012"], 2, True),
    ("aapl_curly", 0): ("vocabulary_gap", True, ["risk_factors_015"], 7, True),
    ("aapl_cutoff", 0): ("section_miss", False, ["risk_factors_016"], 50, True),
}


def _expected_ids(key: tuple[str, int]) -> list[str]:
    acc = ACC2 if key[0] == "aapl_other_accession" else ACC
    return [f"{acc}_{suffix}" for suffix in EXPECTED[key][2]]


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_each_miss_is_classified_with_its_evidence(world: World, key: tuple[str, int]) -> None:
    """test-plan section 6 rows 1-5 and 'Dense rank': class, section_reached, anchor chunks and
    dense rank of each miss in a report built over the hand-made world."""
    miss = _by_key(_report(world))[key]
    expected_class, reached, _, dense_rank, intent_hit = EXPECTED[key]
    assert miss.miss_class == expected_class
    assert miss.section_reached is reached
    assert miss.anchor_chunk_ids == _expected_ids(key)
    assert miss.dense_rank == dense_rank
    assert miss.intent_targeted_gold_section is intent_hit


def test_miss_carries_case_and_gold_identity(world: World) -> None:
    """Contract: ticker, accession, section and anchor index identify the missed gold anchor."""
    miss = _by_key(_report(world))[("aapl_partial", 1)]
    assert (miss.ticker, miss.accession_number, miss.section_key) == ("AAPL", ACC, "business")
    assert miss.anchor_index == 1


def test_covered_anchors_are_not_misses(world: World) -> None:
    """test-plan section 6 'Covered anchor': a covered case, and the covered half of a partial
    case, are absent; only the uncovered anchor is reported, with its own index."""
    keys = set(_by_key(_report(world)))
    assert not any(case_id == "aapl_covered" for case_id, _ in keys)
    assert ("aapl_partial", 0) not in keys
    assert ("aapl_partial", 1) in keys


def test_curly_quote_anchor_is_found_in_straight_quote_chunk(world: World) -> None:
    """benchmark section 3 norm(): quote style never turns an anchor into a boundary_split."""
    miss = _by_key(_report(world))[("aapl_curly", 0)]
    assert miss.miss_class != "boundary_split"
    assert miss.anchor_chunk_ids == [f"{ACC}_risk_factors_015"]


def test_straddling_anchor_has_no_anchor_chunk_overlap_or_dense_rank(world: World) -> None:
    """test-plan section 6 'Anchor in no single chunk' and 'Dense rank': nothing to rank or
    overlap when no chunk holds the anchor, even though the pool lists neighbouring chunks."""
    miss = _by_key(_report(world))[("aapl_straddle_reached", 0)]
    assert miss.anchor_chunk_ids == []
    assert miss.overlap is None
    assert miss.dense_rank is None


def test_anchor_chunks_are_limited_to_the_gold_accession_and_section(world: World) -> None:
    """test-plan section 6 'Anchor in no single chunk' (section 3 scope): copies of the anchor in
    another section or another accession are not the gold anchor's chunks; the first anchor chunk
    in pool order sets the dense rank, not the first by id."""
    miss = _by_key(_report(world))[("aapl_dup_chunks", 0)]
    assert miss.anchor_chunk_ids == [f"{ACC}_risk_factors_011", f"{ACC}_risk_factors_012"]
    assert miss.dense_rank == 2


def test_top_ten_rows_only(world: World) -> None:
    """benchmark 7a 'no top-10 chunk': a gold-section row at rank 11 is not reached, and the
    anchor stays a miss."""
    miss = _by_key(_report(world))[("aapl_cutoff", 0)]
    assert miss.section_reached is False
    assert miss.miss_class == "section_miss"


def test_gold_accession_must_match_for_section_reached(world: World) -> None:
    """test-plan section 6 'No top-10 chunk from the gold (accession, section)': a retrieved
    chunk from the same section of another accession does not reach the gold section."""
    miss = _by_key(_report(world))[("aapl_other_accession", 0)]
    assert miss.section_reached is False
    assert miss.miss_class == "section_miss"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (("aapl_vocab", 0), 0.0),
        (("aapl_curly", 0), 0.0),
        (("aapl_other_accession", 0), 0.0),
    ],
)
def test_overlap_is_zero_when_the_anchor_chunk_shares_no_query_word(
    world: World, key: tuple[str, int], expected: float
) -> None:
    """test-plan section 6 'overlap < 0.5': a chunk with none of the query's words overlaps 0."""
    miss = _by_key(_report(world))[key]
    assert miss.overlap == expected
    assert miss.overlap < OVERLAP_THRESHOLD


@pytest.mark.parametrize("key", [("aapl_rank", 0), ("aapl_dup_chunks", 0)])
def test_overlap_is_high_when_the_anchor_chunk_holds_the_query_words(
    world: World, key: tuple[str, int]
) -> None:
    """test-plan section 6 'overlap >= 0.5': for the duplicate case the highest overlap over the
    anchor chunks counts, so the low-overlap copy cannot make it a vocabulary_gap."""
    miss = _by_key(_report(world))[key]
    assert miss.overlap is not None
    assert miss.overlap >= OVERLAP_THRESHOLD


@pytest.mark.parametrize(
    ("case_id", "query", "gold_section"),
    [
        ("aapl_vocab", Q_RISK, "risk_factors"),
        ("aapl_fx_miss", Q_FX, "market_risk"),
        ("aapl_partial", Q_FX, "business"),
    ],
)
def test_intent_fields_follow_infer_section_intent(
    world: World, case_id: str, query: str, gold_section: str
) -> None:
    """test-plan section 6 'No top-10 chunk': intent_sections is what infer_section_intent names
    for the query; intent_targeted_gold_section says whether it named the gold section."""
    anchor_index = 1 if case_id == "aapl_partial" else 0
    miss = _by_key(_report(world))[(case_id, anchor_index)]
    named = list(infer_section_intent(query).target_sections)
    assert miss.intent_sections == named
    assert miss.intent_targeted_gold_section is (gold_section in named)


def test_intent_targeting_differs_between_a_named_and_an_unnamed_gold_section(
    world: World,
) -> None:
    """The same report holds one miss whose gold section the intent named and one it did not."""
    flags = {m.case_id: m.intent_targeted_gold_section for m in _report(world).misses}
    assert flags["aapl_fx_miss"] is True
    assert flags["aapl_partial"] is False


# --------------------------------------------------------------------------- dense rank


def test_dense_rank_is_null_without_a_pool(world: World) -> None:
    """test-plan section 6 'Dense rank': null when no pool is given."""
    report = _report(world, with_pool=False)
    assert report.misses
    assert all(m.dense_rank is None for m in report.misses)
    assert report.dense_pool_depth == DENSE_POOL_DEPTH


@pytest.mark.parametrize(
    ("before_anchor", "expected"),
    [([], 1), (["a", "b", "c"], 4)],
    ids=["first", "after_three_others"],
)
def test_dense_rank_is_the_one_based_position_of_the_anchor_chunk(
    world: World, before_anchor: list[str], expected: int
) -> None:
    """test-plan section 6 'Dense rank': 1-based position of the anchor chunk in the pool."""
    world.pool["aapl_rank"] = [*before_anchor, f"{ACC}_risk_factors_005", "z"]
    assert _by_key(_report(world))[("aapl_rank", 0)].dense_rank == expected


def test_dense_rank_is_null_when_the_anchor_chunk_is_not_in_the_pool(world: World) -> None:
    """test-plan section 6 'Dense rank': null when the anchor chunk is absent from the pool."""
    world.pool["aapl_rank"] = ["x", "y", "z"]
    assert _by_key(_report(world))[("aapl_rank", 0)].dense_rank is None


def test_dense_rank_is_null_when_the_case_has_no_pool_entry(world: World) -> None:
    """test-plan section 6 'Dense rank': a pool without this case's entry gives null."""
    del world.pool["aapl_rank"]
    assert _by_key(_report(world))[("aapl_rank", 0)].dense_rank is None


# --------------------------------------------------------------------------- report totals


def test_report_header_comes_from_inputs_and_result_provenance(world: World) -> None:
    """Contract: source_result as given; release and embedding model from the result's
    provenance, not the fixture; threshold and depth echoed."""
    report = _report(world)
    assert report.source_result == SOURCE
    assert report.fixture_version == "unit_fixture_v1"
    assert report.evidence_release == "unit-release:9.9.9"
    assert report.embedding_model == "fake:model:8"
    assert report.overlap_threshold == OVERLAP_THRESHOLD
    assert report.dense_pool_depth == DENSE_POOL_DEPTH


def test_class_counts_and_totals(world: World) -> None:
    """test-plan section 6 'Report over the baseline' (shape): every miss classified, counts sum
    to n_misses, partial cases count once in n_cases_with_misses."""
    report = _report(world)
    assert report.class_counts == {
        "boundary_split": 2,
        "section_miss": 4,
        "vocabulary_gap": 2,
        "ranking": 2,
    }
    assert sum(report.class_counts.values()) == report.n_misses == len(report.misses) == 10
    assert report.n_cases_with_misses == 10
    assert {m.miss_class for m in report.misses} <= set(CLASSES)


def test_misses_follow_result_case_order_then_anchor_index(world: World) -> None:
    """Contract: misses are ordered by the result file's case order, then anchor_index."""
    report = _report(world)
    result_position = {case_id: i for i, case_id in enumerate(world.order)}
    keys = [(result_position[m.case_id], m.anchor_index) for m in report.misses]
    assert keys == sorted(keys)
    assert [m.case_id for m in report.misses][:3] == [
        "aapl_vocab",
        "aapl_straddle_unreached",
        "aapl_partial",
    ]


def test_candidates_count_misses_addressed_and_sort_by_count_then_name(world: World) -> None:
    """test-plan section 6 'Candidates': section_prior=section_miss, hybrid_bm25=ranking,
    rerank=non-boundary misses with a dense rank, chunking=boundary_split,
    query_rewrite=vocabulary_gap; sorted by count desc then name (three-way tie here)."""
    report = _report(world)
    assert [(c.name, c.misses) for c in report.candidates] == [
        ("rerank", 7),
        ("section_prior", 4),
        ("chunking", 2),
        ("hybrid_bm25", 2),
        ("query_rewrite", 2),
    ]


def test_rerank_candidate_is_empty_without_a_dense_pool(world: World) -> None:
    """test-plan section 6 'Dense rank' + 'Candidates': no pool, no dense ranks, rerank 0; the
    other candidates still count and all five are listed."""
    report = _report(world, with_pool=False)
    assert [(c.name, c.misses) for c in report.candidates] == [
        ("section_prior", 4),
        ("chunking", 2),
        ("hybrid_bm25", 2),
        ("query_rewrite", 2),
        ("rerank", 0),
    ]


def test_rerank_excludes_boundary_splits_even_when_the_pool_lists_their_chunks(
    world: World,
) -> None:
    """benchmark 7a: rerank addresses any non-boundary_split miss with a dense rank; a
    boundary_split miss never counts for it."""
    report = _report(world)
    rerank = next(c for c in report.candidates if c.name == "rerank")
    non_boundary_ranked = [
        m for m in report.misses if m.miss_class != "boundary_split" and m.dense_rank is not None
    ]
    assert rerank.misses == len(non_boundary_ranked)
    assert any(m.miss_class == "boundary_split" for m in report.misses)


def test_all_covered_result_reports_no_misses() -> None:
    """test-plan section 6 'Covered anchor': nothing to report. Every class and candidate is still
    listed, with zero counts, candidates sorted by name."""
    anchor = _anchor("covered")
    chunk = _chunk(ACC, "risk_factors", 1, f"Preface words. {anchor} Closing words.")
    fixture = _fixture([_case("aapl_covered", Q_RISK, [_gold(ACC, "risk_factors", anchor)])])
    result = _result(fixture, {"aapl_covered": [chunk]}, ["aapl_covered"])
    report = build_diagnostics_report(
        result, fixture, {chunk.id: chunk}, source_result=SOURCE, dense_pool=None
    )
    assert report.misses == []
    assert report.n_misses == 0
    assert report.n_cases_with_misses == 0
    assert report.class_counts == dict.fromkeys(CLASSES, 0)
    assert [(c.name, c.misses) for c in report.candidates] == [
        (name, 0) for name in sorted(CANDIDATES)
    ]


# --------------------------------------------------------------------------- rebuild guard


def _drop_retrieved_chunk(world: World) -> None:
    del world.chunks[f"{ACC}_risk_factors_090"]


def _flip_covered_hit_to_false(world: World) -> None:
    row = next(c for c in world.result["cases"] if c["case_id"] == "aapl_covered")["retrieved"][0]
    assert row["gold_hit"] is True
    row["gold_hit"] = False


def _flip_miss_hit_to_true(world: World) -> None:
    row = next(c for c in world.result["cases"] if c["case_id"] == "aapl_vocab")["retrieved"][0]
    assert row["gold_hit"] is False
    row["gold_hit"] = True


def _rewrite_covered_chunk_text(world: World) -> None:
    chunk_id = f"{ACC}_risk_factors_001"
    old = world.chunks[chunk_id]
    world.chunks[chunk_id] = IndexDocument(
        id=old.id, text="Rebuilt text with a different chunker.", metadata=old.metadata
    )


@pytest.mark.parametrize(
    "tamper",
    [
        _drop_retrieved_chunk,
        _flip_covered_hit_to_false,
        _flip_miss_hit_to_true,
        _rewrite_covered_chunk_text,
    ],
    ids=["chunk_id_missing", "stored_hit_false", "stored_hit_true", "rebuilt_text_differs"],
)
def test_rebuild_guard_refuses_when_chunks_or_hits_disagree(
    world: World, tamper: Callable[[World], None]
) -> None:
    """test-plan section 6 'Rebuild guard': a retrieved chunk_id missing from the rebuilt chunks,
    or a recomputed gold_hit that differs from the stored one, refuses the report."""
    assert isinstance(_report(world), DiagnosticsReport)  # the untampered world is accepted
    tamper(world)
    with pytest.raises(RebuildMismatchError):
        _report(world)


# --------------------------------------------------------------------------- collect_dense_pool


class _ScoringStore:
    """A vector store that scores its own documents against the query.

    Distance is 1 / (1 + number of query words found in the document), so more shared words mean
    higher relevance. Ties come back in insertion order, which the tests make differ from id
    order. The ticker and filing_type filters and n_results are honoured.
    """

    def __init__(self, documents: list[IndexDocument]) -> None:
        self._documents = documents

    def search(
        self, query: str, filters: SearchFilters | None = None, n_results: int = 5
    ) -> SearchResult:
        words = {w.strip("?,.").lower() for w in query.split()}
        scored: list[tuple[float, IndexDocument]] = []
        for doc in self._documents:
            if filters is not None:
                if filters.ticker is not None and doc.metadata["ticker"] != filters.ticker:
                    continue
                if (
                    filters.filing_type is not None
                    and doc.metadata["filing_type"] != filters.filing_type
                ):
                    continue
            shared = len(words & set(doc.text.lower().split()))
            scored.append((1.0 / (1 + shared), doc))
        scored.sort(key=lambda pair: pair[0])
        chunks = [
            RetrievedChunk(id=d.id, text=d.text, metadata=dict(d.metadata), distance=dist)
            for dist, d in scored[:n_results]
        ]
        return SearchResult(query=query, chunks=chunks, total_results=len(chunks))


def _doc(doc_id: str, ticker: str, filing_type: str, text: str) -> IndexDocument:
    return IndexDocument(
        id=doc_id, text=text, metadata={"ticker": ticker, "filing_type": filing_type}
    )


def _pool_fixture() -> AnchoredFixture:
    def case(case_id: str, ticker: str, filing_type: str | None) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "ticker": ticker,
            "query": Q_RISK,
            "filters": {"ticker": ticker, "filing_type": filing_type},
            "expected_sections": [],
            "answer_intent": "risk",
        }

    return _fixture(
        [
            case("p_aapl", "AAPL", "10-K"),
            case("p_msft", "MSFT", "10-K"),
            case("p_aapl_any", "AAPL", None),
        ]
    )


def _pool_store() -> _ScoringStore:
    return _ScoringStore(
        [
            _doc("m1", "MSFT", "10-K", "what does apple say about apple supply chain risks"),
            _doc("q1", "AAPL", "10-Q", "apple supply chain risks"),
            _doc("a4", "AAPL", "10-K", "weather patterns"),
            _doc("a3", "AAPL", "10-K", "supply chain delays"),
            _doc("a2", "AAPL", "10-K", "supply chain disruption"),
            _doc("a1", "AAPL", "10-K", "apple supply chain risks"),
        ]
    )


def test_collect_dense_pool_filters_orders_and_keys_by_case() -> None:
    """benchmark 7a 'Dense rank': per case, ticker and filing-type filters apply, chunk ids come
    back ordered by relevance then id, and a null filing type matches every filing type."""
    pool = collect_dense_pool(_pool_store(), _pool_fixture(), ["p_aapl", "p_msft", "p_aapl_any"])
    assert pool == {
        "p_aapl": ["a1", "a2", "a3", "a4"],
        "p_msft": ["m1"],
        "p_aapl_any": ["a1", "q1", "a2", "a3", "a4"],
    }


def test_collect_dense_pool_collects_only_the_requested_cases() -> None:
    """Contract: the pool has one entry per requested case id."""
    pool = collect_dense_pool(_pool_store(), _pool_fixture(), ["p_msft"])
    assert list(pool) == ["p_msft"]


@pytest.mark.parametrize(("depth", "expected"), [(1, ["a1"]), (3, ["a1", "a2", "a3"])])
def test_collect_dense_pool_honours_depth(depth: int, expected: list[str]) -> None:
    """benchmark 7a: the pool is the store's top-`depth` result for the case."""
    pool = collect_dense_pool(_pool_store(), _pool_fixture(), ["p_aapl"], depth=depth)
    assert pool["p_aapl"] == expected


def test_collect_dense_pool_defaults_to_the_declared_depth() -> None:
    """benchmark 7a: default pool depth is 50 even when more documents match."""
    ids = (f"d{i:02d}" for i in itertools.count())
    docs = [_doc(next(ids), "AAPL", "10-K", "apple supply chain risks") for _ in range(60)]
    pool = collect_dense_pool(_ScoringStore(docs), _pool_fixture(), ["p_aapl"])
    assert len(pool["p_aapl"]) == DENSE_POOL_DEPTH == 50
    assert pool["p_aapl"] == sorted(pool["p_aapl"])
