"""CM-4: copy the Gate A vectors from Qdrant into an OpenSearch index and wait until it is usable.

Traces to docs/test-plan.md §20 "Gate vector copy (CM-4, added 2026-11-08)" and
docs/retrieval-benchmark.md §9.1 ("Arms", "Before the run").

The source is an in-memory Qdrant collection in `QdrantVectorStore`'s payload layout, holding
vectors that the configured embedder would not produce for those texts. The configured embedding
provider points at a closed loopback port, so any embedding call fails the copy. Runs against the
local OpenSearch test container.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.vector_store import IndexDocument
from evaluation.validate_retrieval_fixture import AnchoredFixture
from tests.rag.gate_support import (
    SECTIONS,
    SOURCE_COLLECTION,
    build_gate_chunks,
    documents_in_index,
    load_fixture_from,
    other_vector,
    qdrant_source,
    source_vector,
    source_vectors,
)
from tests.rag.helpers import FakeEmbedder, raw_document

StoreFactory = Callable[..., Any]


# Imported late, as in test_embedding_arm_build.py, so each test fails on its own while the
# module under test does not exist yet.
def copy_vectors(source: QdrantClient, source_collection: str, index_name: str) -> int:
    from evaluation.copy_gate_vectors import copy_vectors as copy

    written: int = copy(source, source_collection, index_name)
    return written


def wait_until_ready(
    store: OpenSearchVectorStore,
    *,
    expected_count: int,
    fixture: AnchoredFixture,
    timeout_s: float,
    poll_s: float,
) -> None:
    from evaluation.copy_gate_vectors import wait_until_ready as wait

    wait(store, expected_count=expected_count, fixture=fixture, timeout_s=timeout_s, poll_s=poll_s)


@pytest.fixture
def chunks() -> list[IndexDocument]:
    return build_gate_chunks()


@pytest.fixture
def source(chunks: list[IndexDocument]) -> QdrantClient:
    return qdrant_source([(chunk, source_vector(chunk.text)) for chunk in chunks])


@pytest.fixture
def anchored(tmp_path: Path) -> AnchoredFixture:
    return load_fixture_from(tmp_path)


@pytest.fixture(autouse=True)
def store_is_found_through_the_environment(
    monkeypatch: pytest.MonkeyPatch, opensearch_url: str
) -> None:
    monkeypatch.setenv("OPENSEARCH_URL", opensearch_url)


# ---- copy_vectors ---------------------------------------------------------------------------


def test_copy_writes_every_point_with_chunk_id_text_and_metadata_unchanged(
    opensearch_url: str, index_name: str, chunks: list[IndexDocument], source: QdrantClient
) -> None:
    """test-plan §20 Gate vector copy: every point is written with its chunk ID, text and metadata."""
    assert len(chunks) > 50  # more than one Qdrant scroll page

    written = copy_vectors(source, SOURCE_COLLECTION, index_name)

    assert written == len(chunks)
    assert documents_in_index(opensearch_url, index_name) == len(chunks)
    copied: dict[str, tuple[str, dict[str, Any]]] = {}
    for chunk in chunks:
        status, document = raw_document(opensearch_url, index_name, chunk.id)
        assert status == 200, f"chunk {chunk.id} is not in the index under its chunk ID"
        stored = document["_source"]
        copied[chunk.id] = (stored["text"], {key: stored.get(key) for key in chunk.metadata})
    assert copied == {chunk.id: (chunk.text, chunk.metadata) for chunk in chunks}


def test_copy_carries_the_source_vectors_and_calls_no_embedding_provider(
    opensearch_url: str, index_name: str, chunks: list[IndexDocument], source: QdrantClient
) -> None:
    """test-plan §20 Gate vector copy: vectors unchanged, with no embedding call."""
    provider = FakeEmbedder()
    expected = source_vectors(source)

    copy_vectors(source, SOURCE_COLLECTION, index_name)

    for chunk in chunks:
        _, document = raw_document(opensearch_url, index_name, chunk.id)
        stored = document["_source"]["embedding"]
        assert stored == pytest.approx(expected[chunk.id], rel=1e-6)
        assert stored != pytest.approx(provider.embed_documents([chunk.text])[0], rel=1e-6)


@pytest.mark.parametrize("position", ["first", "last"])
def test_copy_fails_before_any_write_when_one_text_has_two_vectors(
    opensearch_url: str,
    index_name: str,
    chunks: list[IndexDocument],
    position: str,
) -> None:
    """test-plan §20 Gate vector copy: a text that carries two different vectors fails before any
    write (wherever in the source the conflicting point sits)."""
    twin = IndexDocument(
        id=f"{chunks[0].id}_twin", text=chunks[0].text, metadata=chunks[0].metadata
    )
    points = [(chunk, source_vector(chunk.text)) for chunk in chunks]
    conflicting = (twin, other_vector(twin.text))
    points = [conflicting, *points] if position == "first" else [*points, conflicting]
    source = qdrant_source(points)

    with pytest.raises(ValueError):
        copy_vectors(source, SOURCE_COLLECTION, index_name)

    assert documents_in_index(opensearch_url, index_name) == 0


def test_copy_keeps_chunks_that_repeat_a_text_with_the_same_vector(
    opensearch_url: str, index_name: str, chunks: list[IndexDocument]
) -> None:
    """test-plan §20 Gate vector copy: only a text with two *different* vectors fails; repeated
    boilerplate under distinct chunk IDs is copied under each ID."""
    repeat = IndexDocument(
        id=f"{chunks[0].id}_repeat", text=chunks[0].text, metadata=chunks[0].metadata
    )
    points = [(chunk, source_vector(chunk.text)) for chunk in [*chunks, repeat]]

    written = copy_vectors(qdrant_source(points), SOURCE_COLLECTION, index_name)

    assert written == len(chunks) + 1
    assert documents_in_index(opensearch_url, index_name) == len(chunks) + 1
    assert raw_document(opensearch_url, index_name, repeat.id)[0] == 200


# ---- wait_until_ready -----------------------------------------------------------------------

TIMEOUT_S = 0.6
POLL_S = 0.05


def _populated(
    make_store: StoreFactory, index: str, documents: list[IndexDocument]
) -> OpenSearchVectorStore:
    store: OpenSearchVectorStore = make_store(index)
    store.add_documents(documents)
    return store


def _chunk_holding_the_anchor_of(
    case_index: int, anchored: AnchoredFixture, chunks: list[IndexDocument]
) -> IndexDocument:
    gold = anchored.cases[case_index].gold_evidence[0]
    return next(
        chunk
        for chunk in chunks
        if chunk.metadata["accession_number"] == gold.accession_number
        and chunk.metadata["section_key"] == gold.section_key
        and gold.anchor_text in chunk.text
    )


def _with_whitespace_and_case_changed(document: IndexDocument) -> IndexDocument:
    return IndexDocument(
        id=document.id,
        text="  ".join(document.text.upper().split()) + "\n",
        metadata=document.metadata,
    )


@pytest.mark.parametrize(
    "rewrite",
    [lambda document: document, _with_whitespace_and_case_changed],
    ids=["verbatim", "case-and-whitespace-normalized"],
)
def test_ready_index_returns_without_error(
    make_store: StoreFactory,
    index_name: str,
    chunks: list[IndexDocument],
    anchored: AnchoredFixture,
    rewrite: Callable[[IndexDocument], IndexDocument],
) -> None:
    """test-plan §20 Gate vector copy: ready when `_count` equals the source count and every gold
    anchor matches a chunk (benchmark matcher: accession, section, normalized containment)."""
    store = _populated(make_store, index_name, [rewrite(chunk) for chunk in chunks])

    wait_until_ready(
        store, expected_count=len(chunks), fixture=anchored, timeout_s=TIMEOUT_S, poll_s=POLL_S
    )


def test_wait_blocks_until_the_index_catches_up_with_the_source(
    make_store: StoreFactory,
    index_name: str,
    chunks: list[IndexDocument],
    anchored: AnchoredFixture,
) -> None:
    """test-plan §20 Gate vector copy: the copy waits until the count and the anchors are right."""
    half = len(chunks) // 2
    store = _populated(make_store, index_name, chunks[:half])
    late_writer = make_store(index_name)
    failures: list[BaseException] = []

    def write_the_rest() -> None:
        time.sleep(0.5)
        try:
            late_writer.add_documents(chunks[half:])
        except BaseException as error:  # surfaced below, from the test thread
            failures.append(error)

    writer = threading.Thread(target=write_the_rest)
    writer.start()
    try:
        wait_until_ready(
            store, expected_count=len(chunks), fixture=anchored, timeout_s=20, poll_s=0.1
        )
        assert store.count == len(chunks)
    finally:
        writer.join()
    assert failures == []


def test_count_shortfall_fails_at_the_timeout_naming_both_counts(
    make_store: StoreFactory,
    index_name: str,
    chunks: list[IndexDocument],
    anchored: AnchoredFixture,
) -> None:
    """test-plan §20 Gate vector copy: fails (no benchmark run) if `_count` never reaches the
    source count before the timeout."""
    store = _populated(make_store, index_name, chunks)
    started = time.monotonic()

    with pytest.raises(RuntimeError) as failure:
        wait_until_ready(
            store,
            expected_count=len(chunks) + 1,
            fixture=anchored,
            timeout_s=TIMEOUT_S,
            poll_s=POLL_S,
        )

    assert time.monotonic() - started < TIMEOUT_S + 5
    assert str(len(chunks)) in str(failure.value)
    assert str(len(chunks) + 1) in str(failure.value)


@pytest.mark.parametrize("decoy", ["absent", "other-section", "other-accession"])
def test_unreachable_gold_anchor_fails_at_the_timeout_naming_the_case(
    make_store: StoreFactory,
    index_name: str,
    chunks: list[IndexDocument],
    anchored: AnchoredFixture,
    decoy: str,
) -> None:
    """test-plan §20 Gate vector copy: fails (no benchmark run) if a fixture gold anchor matches no
    chunk before the timeout, even when the count is right; the same text under another accession
    or section does not satisfy the anchor."""
    target = _chunk_holding_the_anchor_of(0, anchored, chunks)
    indexed = [chunk for chunk in chunks if chunk.id != target.id]
    if decoy != "absent":
        metadata = dict(target.metadata)
        if decoy == "other-section":
            metadata["section_key"] = next(
                s for s in SECTIONS if s != target.metadata["section_key"]
            )
        else:
            metadata["accession_number"] = "0000000000-24-000001"
        indexed.append(IndexDocument(id=f"{target.id}_decoy", text=target.text, metadata=metadata))
    store = _populated(make_store, index_name, indexed)
    started = time.monotonic()

    with pytest.raises(RuntimeError) as failure:
        wait_until_ready(
            store,
            expected_count=len(indexed),
            fixture=anchored,
            timeout_s=TIMEOUT_S,
            poll_s=POLL_S,
        )

    assert time.monotonic() - started < TIMEOUT_S + 5
    assert anchored.cases[0].case_id in str(failure.value)
