"""CM-4: the anchored benchmark runner against an OpenSearch index.

Traces to docs/test-plan.md §20 "Anchored runner on OpenSearch (CM-4, added 2026-11-08)" and
docs/retrieval-benchmark.md §9.1 ("Arms").

The index is populated through the store with the 16-d fake embedder, the store is found through
OPENSEARCH_URL, and the runner's query embeddings come from the same fake, substituted where the
store asks the settings for its embedding client. Runs against the local OpenSearch test container.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import app.components.retrieval.embeddings as embeddings_module
import app.components.retrieval.opensearch_store as opensearch_store
from app.components.retrieval.embeddings import embedding_identifier
from app.components.retrieval.vector_store import SearchFilters
from evaluation.run_shared_retrieval_benchmark import main
from tests.rag.gate_support import build_gate_chunks, write_anchored_fixture
from tests.rag.helpers import FakeEmbedder

StoreFactory = Callable[..., Any]


@pytest.mark.parametrize(
    ("method", "level"),
    [("section_aware", "1x"), ("dense", "1x"), ("section_aware", "32x")],
)
def test_anchored_run_on_an_opensearch_index_records_backend_index_level_and_embedding_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opensearch_url: str,
    index_name: str,
    make_store: StoreFactory,
    method: str,
    level: str,
) -> None:
    """test-plan §20 Anchored runner on OpenSearch: `--backend opensearch --collection <index>` runs
    the fixture against that index; the result records the backend, the index name, its compression
    level and the `embedding_model` identifier a Qdrant run records under the same settings."""
    monkeypatch.setenv("OPENSEARCH_COMPRESSION_LEVEL", level)
    monkeypatch.setenv("OPENSEARCH_URL", opensearch_url)
    chunks = build_gate_chunks()
    make_store(index_name).add_documents(chunks)
    fixture_path, source_dir = write_anchored_fixture(tmp_path)
    for module in (embeddings_module, opensearch_store):
        monkeypatch.setattr(module, "get_embeddings", FakeEmbedder)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_shared_retrieval_benchmark",
            "--fixture", str(fixture_path),
            "--source-dir", str(source_dir),
            "--backend", "opensearch",
            "--collection", index_name,
            "--method", method,
            "--output", str(output),
        ],
    )  # fmt: skip

    main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["backend"] == "opensearch"
    assert result["retrieval_method"] == method
    provenance = result["provenance"]
    assert provenance["collection"] == index_name
    assert provenance["compression_level"] == level
    assert provenance["embedding_model"] == embedding_identifier()
    assert provenance["embedding_model"] == "bedrock:fake-hash-embed:16"
    indexed_ids = {chunk.id for chunk in chunks}
    retrieved = {row["chunk_id"] for case in result["cases"] for row in case["retrieved"]}
    assert result["summary"]["n_cases"] == len(result["cases"]) > 50
    assert retrieved and retrieved <= indexed_ids


def test_hybrid_anchored_run_returns_what_the_stores_hybrid_search_returns_per_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opensearch_url: str,
    index_name: str,
    make_store: StoreFactory,
) -> None:
    """test-plan §6 Benchmark runner: `hybrid` method (S5.2) - the anchored runner accepts
    `--method hybrid`, retrieves each case through the store's hybrid search under the case's
    ticker and filing-type filters, and writes `retrieval_method` and `mode` = `hybrid`."""
    monkeypatch.setenv("OPENSEARCH_URL", opensearch_url)
    chunks = build_gate_chunks()
    store = make_store(index_name)
    store.add_documents(chunks)
    fixture_path, source_dir = write_anchored_fixture(tmp_path)
    for module in (embeddings_module, opensearch_store):
        monkeypatch.setattr(module, "get_embeddings", FakeEmbedder)
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_shared_retrieval_benchmark",
            "--fixture", str(fixture_path),
            "--source-dir", str(source_dir),
            "--backend", "opensearch",
            "--collection", index_name,
            "--method", "hybrid",
            "--output", str(output),
        ],
    )  # fmt: skip

    main()

    result = json.loads(output.read_text(encoding="utf-8"))
    fixture_cases = {
        case["case_id"]: case
        for case in json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    }
    assert result["backend"] == "opensearch"
    assert result["retrieval_method"] == "hybrid"
    assert result["mode"] == "hybrid"
    assert result["provenance"]["collection"] == index_name
    assert {case["case_id"] for case in result["cases"]} == set(fixture_cases)
    for case in result["cases"]:
        source = fixture_cases[case["case_id"]]
        expected = store.search_hybrid(
            source["query"],
            filters=SearchFilters(
                ticker=source["ticker"], filing_type=source["filters"]["filing_type"]
            ),
            n_results=10,
        ).chunks
        assert expected, "precondition: the store's hybrid search finds chunks for every case"
        assert [row["chunk_id"] for row in case["retrieved"]] == [chunk.id for chunk in expected]
        assert [row["rank"] for row in case["retrieved"]] == list(range(1, len(expected) + 1))
