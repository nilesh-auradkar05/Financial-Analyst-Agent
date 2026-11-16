"""S5.1: the miss diagnostics report over the frozen CM-6e baseline (offline).

Oracle: docs/test-plan.md section 6 "Retrieval diagnostics: S5.1", row "Report over the CM-6e
baseline"; docs/retrieval-benchmark.md section 7a.

Inputs are the committed artifacts: the evidence release `alpha-evidence:0.1.0` in the dataops
registry, the Foundry Cohere result file, its anchored fixture and source sections, and the
dense top-50 pool collected once and committed next to the report. Nothing here touches the
network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from evaluation.retrieval_diagnostics import (
    DENSE_POOL_DEPTH,
    OVERLAP_THRESHOLD,
    DiagnosticsReport,
    build_diagnostics_report,
    main,
    rebuild_chunks,
)
from evaluation.validate_retrieval_fixture import load_anchored_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = REPO_ROOT / "artifacts" / "dataops"
SOURCE_DIR = REPO_ROOT / "evaluation" / "fixtures" / "source_sections"
RESULT_REL = (
    "evaluation/results/"
    "gate_c_foundry-cohere-embed-v5-fast_opensearch_section_aware_retrieval_anchored_benchmark_v1.json"
)
POOL_REL = "evaluation/results/s5_1_dense_pool_top50.json"
REPORT_REL = "evaluation/results/s5_1_diagnostics_foundry-cohere-embed-v5-fast.json"
CLASSES = {"boundary_split", "section_miss", "vocabulary_gap", "ranking"}
CHUNK_ID = re.compile(r"^\d{10}-\d{2}-\d{6}_[a-z&_]+_\d{3}$")


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    payload = json.loads((REPO_ROOT / RESULT_REL).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def chunks(result: dict[str, Any]) -> dict[str, Any]:
    return rebuild_chunks(result["provenance"]["evidence_release"], registry_root=REGISTRY_ROOT)


@pytest.fixture(scope="module")
def report(result: dict[str, Any], chunks: dict[str, Any]) -> DiagnosticsReport:
    fixture = load_anchored_fixture(
        REPO_ROOT / result["provenance"]["fixture_path"], source_dir=SOURCE_DIR
    )
    pool = json.loads((REPO_ROOT / POOL_REL).read_text(encoding="utf-8"))
    return build_diagnostics_report(
        result, fixture, chunks, source_result=RESULT_REL, dense_pool=pool
    )


def test_rebuild_reproduces_every_retrieved_chunk(
    result: dict[str, Any], chunks: dict[str, Any]
) -> None:
    """test-plan section 6 'Rebuild guard' (offline half): the rebuild of the evidence release
    holds every chunk the CM-6e run retrieved, under the same accession and section."""
    assert result["provenance"]["evidence_release"] == "alpha-evidence:0.1.0"
    retrieved = [row for case in result["cases"] for row in case["retrieved"]]
    assert retrieved
    for row in retrieved:
        chunk = chunks[row["chunk_id"]]
        assert chunk.metadata["accession_number"] == row["accession_number"]
        assert chunk.metadata["section_key"] == row["section_key"]
    assert all(CHUNK_ID.match(chunk_id) for chunk_id in chunks)


def test_report_over_the_cm6e_baseline_counts(report: DiagnosticsReport) -> None:
    """test-plan section 6 'Report over the CM-6e baseline': 45 misses over 30 cases, every miss
    classified, class counts sum to 45."""
    assert report.n_misses == 45
    assert len(report.misses) == 45
    assert report.n_cases_with_misses == 30
    assert len({miss.case_id for miss in report.misses}) == 30
    assert set(report.class_counts) == CLASSES
    assert sum(report.class_counts.values()) == 45
    assert all(miss.miss_class in CLASSES for miss in report.misses)
    for name in CLASSES:
        assert report.class_counts[name] == sum(m.miss_class == name for m in report.misses)


def test_report_header_names_the_baseline(report: DiagnosticsReport) -> None:
    """Contract: the report records its source, release, embedding model and declared constants."""
    assert report.source_result == RESULT_REL
    assert report.evidence_release == "alpha-evidence:0.1.0"
    assert report.embedding_model == "foundry_cohere:cohere-embed-v5-fast:1024"
    assert report.overlap_threshold == OVERLAP_THRESHOLD
    assert report.dense_pool_depth == DENSE_POOL_DEPTH


def test_report_dense_ranks_come_from_the_committed_pool(report: DiagnosticsReport) -> None:
    """test-plan section 6 'Dense rank': ranks lie within the top-50 pool; boundary splits have
    no anchor chunk and so no rank."""
    ranks = [m.dense_rank for m in report.misses]
    assert all(r is None or 1 <= r <= DENSE_POOL_DEPTH for r in ranks)
    assert any(r is not None for r in ranks)
    assert all(m.dense_rank is None for m in report.misses if m.miss_class == "boundary_split")


def test_cli_reproduces_the_committed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: DiagnosticsReport
) -> None:
    """test-plan section 6 'Report over the CM-6e baseline': the CLI over the committed result and
    pool writes the committed report byte for byte (repo-relative paths, run from the repo root)."""
    monkeypatch.chdir(REPO_ROOT)
    out = tmp_path / "report.json"
    exit_code = main([RESULT_REL, "--out", str(out), "--dense-pool", POOL_REL])
    assert exit_code == 0
    written = out.read_bytes()
    assert written.endswith(b"\n")
    assert written == (REPO_ROOT / REPORT_REL).read_bytes()
    assert DiagnosticsReport.model_validate_json(written) == report
