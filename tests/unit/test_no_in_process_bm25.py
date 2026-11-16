"""Trace: docs/test-plan.md §25, "No in-process BM25 (S5.3)".

`pyproject.toml` declares neither `rank-bm25` nor `langchain-community`; no module under `app/` or
`evaluation/` imports `BM25Retriever` or `search_langchain_hybrid`; the legacy evaluator
(`evaluation/retrieval_eval.py`, `retrieval_main.py`, `run_retrieval_method_matrix.py`) treats
`hybrid`, `langchain_hybrid`, `lc_hybrid`, `reranked_hybrid` and their aliases as unknown modes.

Files are read as data and imports run in a fresh interpreter, so nothing this test session loaded
can mask a leak (the style of tests/unit/test_no_retired_stores.py).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.components.retrieval.vector_store import RetrievedChunk, SearchFilters, SearchResult
from evaluation.retrieval_eval import RetrievalEvalCase, _search
from evaluation.retrieval_main import main as retrieval_main
from evaluation.run_retrieval_method_matrix import DEFAULT_MODES, normalize_mode

ROOT = Path(__file__).resolve().parents[2]
RETIRED_PACKAGES = {"rank-bm25", "langchain-community"}
RETIRED_MODULES = ("rank_bm25", "langchain_community")
RETIRED_SYMBOLS = {"BM25Retriever", "search_langchain_hybrid"}
REMOVED_MODES = [
    "hybrid",
    "langchain_hybrid",
    "lc_hybrid",
    "reranked_hybrid",
    "rerank_hybrid",
    "cross_encoder_hybrid",
]
# Aliases the matrix runner used to resolve to the removed modes.
REMOVED_MATRIX_ALIASES = ["hybrid_rerank", "dense_bm25_rerank"]
IMPORTED_MODULES = (
    "app.main",
    "app.components.retrieval.reranked_hybrid_retrieve",
    "evaluation.run_shared_retrieval_benchmark",
    "evaluation.retrieval_eval",
    "evaluation.retrieval_main",
    "evaluation.run_retrieval_method_matrix",
)


def _python_files() -> Iterator[Path]:
    for folder in ("app", "evaluation"):
        yield from sorted((ROOT / folder).rglob("*.py"))


def test_the_project_declares_no_in_process_bm25_package() -> None:
    """test-plan §25: No in-process BM25 (`pyproject.toml` declares neither `rank-bm25` nor
    `langchain-community` as a direct dependency; edgartools and ragas may still pull them in)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        declared += group
    names = {
        re.split(r"[\s<>=!~\[;]", requirement, maxsplit=1)[0].lower() for requirement in declared
    }

    assert names & RETIRED_PACKAGES == set()


def _offences(path: Path) -> list[str]:
    """Imports of a retired module and uses of a retired symbol in one file (docstrings and
    comments are not code, so the file is parsed rather than searched)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in RETIRED_MODULES]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in RETIRED_MODULES:
                found.append(node.module or "")
            found += [a.name for a in node.names if a.name in RETIRED_SYMBOLS]
        elif isinstance(node, ast.Name) and node.id in RETIRED_SYMBOLS:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in RETIRED_SYMBOLS:
            found.append(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in RETIRED_SYMBOLS:
            found.append(node.name)
    return found


def test_no_module_under_app_or_evaluation_uses_the_in_process_bm25_path() -> None:
    """test-plan §25: No in-process BM25 (no module under `app/` or `evaluation/` imports
    `BM25Retriever` or `search_langchain_hybrid`, or the packages behind them)."""
    offenders = {
        str(path.relative_to(ROOT)): found for path in _python_files() if (found := _offences(path))
    }

    assert offenders == {}


def test_importing_app_and_evaluation_loads_no_in_process_bm25_package() -> None:
    """test-plan §25: No in-process BM25 (importing the app, retrieval and evaluation modules
    loads neither `rank_bm25` nor `langchain_community`)."""
    code = (
        "import importlib, json, sys\n"
        f"for name in {list(IMPORTED_MODULES)!r}:\n"
        "    importlib.import_module(name)\n"
        f"banned = {sorted(RETIRED_MODULES)!r}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in banned)))\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


class DenseOnlyStore:
    """A faithful dense store: one filtered chunk per search, as the evaluator's modes expect."""

    def search(
        self, query: str, filters: SearchFilters | None = None, n_results: int = 5
    ) -> SearchResult:
        chunks = [
            RetrievedChunk(
                id="risk-1",
                text="Cybersecurity and data security incidents could harm operations.",
                metadata={"ticker": "MSFT", "filing_type": "10-K", "section_key": "risk_factors"},
                distance=0.1,
            )
        ]
        return SearchResult(
            query=query,
            chunks=chunks[:n_results],
            total_results=len(chunks),
            search_time_ms=1.0,
            filter_used={"ticker": "MSFT"},
        )


def _outcome(mode: str) -> tuple[Any, ...]:
    """What the legacy evaluator's search does for `mode`: the chunk ids and filter record it
    returns, or the exception type it raises."""
    case = RetrievalEvalCase(
        id=f"legacy-{mode}",
        ticker="MSFT",
        query="cybersecurity data security risks",
        filing_type="10-K",
        expected_sections=["risk_factors"],
        expected_keywords=["cybersecurity"],
        mode=mode,
    )
    try:
        result = _search(DenseOnlyStore(), case)
    except Exception as error:  # noqa: BLE001 - the type is the observation
        return ("raised", type(error).__name__)
    return ("returned", [chunk.id for chunk in result.chunks], result.filter_used)


@pytest.mark.parametrize("mode", REMOVED_MODES)
def test_the_legacy_evaluator_treats_a_removed_mode_as_an_unknown_one(mode: str) -> None:
    """test-plan §25: No in-process BM25 (the legacy evaluator treats `hybrid`, `langchain_hybrid`,
    `lc_hybrid`, `reranked_hybrid` and their aliases as unknown modes: whatever it does for a mode
    it has never heard of, it does for these)."""
    assert _outcome(mode) == _outcome("no-such-mode")


@pytest.mark.parametrize("mode", ["hybrid", "reranked_hybrid"])
def test_the_evaluate_cli_no_longer_offers_a_removed_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, mode: str
) -> None:
    """test-plan §25: No in-process BM25 (`retrieval_main --mode` no longer lists `hybrid` or
    `reranked_hybrid`)."""
    monkeypatch.setattr(sys, "argv", ["retrieval_main", "--mode", mode, str(tmp_path / "f.json")])

    with pytest.raises(SystemExit) as raised:
        retrieval_main()

    assert raised.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_the_matrix_runner_no_longer_runs_a_removed_mode_by_default() -> None:
    """test-plan §25: No in-process BM25 (`run_retrieval_method_matrix.py` no longer lists the
    removed modes)."""
    assert set(DEFAULT_MODES) & (set(REMOVED_MODES) | set(REMOVED_MATRIX_ALIASES)) == set()


@pytest.mark.parametrize("name", REMOVED_MODES + REMOVED_MATRIX_ALIASES)
def test_the_matrix_runner_no_longer_resolves_a_removed_name_to_a_hybrid_mode(name: str) -> None:
    """test-plan §25: No in-process BM25 (aliases of the removed modes resolve to nothing: the name
    comes back as it went in, as an unknown mode does)."""
    assert normalize_mode(name) == name
