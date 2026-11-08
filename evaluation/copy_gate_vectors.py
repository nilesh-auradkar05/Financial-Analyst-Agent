"""Copy the Gate A vectors from a Qdrant collection into an OpenSearch index (CM-4).

The vectors are written as stored, never re-embedded, so the OpenSearch arm differs from the Qdrant
arm in backend only. The copy then waits until the index holds every point and every gold anchor of
the anchored fixture matches a chunk, so the benchmark never runs against a half-visible index.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.qdrant_store import DEFAULT_QDRANT_URL
from app.components.retrieval.vector_store import IndexDocument
from evaluation.run_shared_retrieval_benchmark import DEFAULT_SOURCE_DIR
from evaluation.validate_retrieval_fixture import (
    AnchoredFixture,
    load_anchored_fixture,
    normalize_text,
)

DEFAULT_SOURCE_COLLECTION = "sec_filings_bench_alpha_evidence_0_1_0"
DEFAULT_FIXTURE = Path("evaluation/fixtures/retrieval_anchored_benchmark_v1.json")
SCROLL_PAGE_SIZE = 256


class _StoredVectors(Embeddings):
    """Answers `embed_documents` from the source vectors; there is no query side to a copy."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("the vector copy must not embed")


def copy_vectors(source: QdrantClient, source_collection: str, index_name: str) -> int:
    """Write every point of `source_collection` into `index_name` and return the count written."""
    documents: list[IndexDocument] = []
    vectors: dict[str, list[float]] = {}
    offset = None
    while True:
        records, offset = source.scroll(
            source_collection,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for record in records:
            payload = dict(record.payload or {})
            chunk_id = str(payload.pop("_retrieval_id"))
            text = str(payload.pop("text"))
            vector = list(cast(list[float], record.vector))
            if vectors.setdefault(text, vector) != vector:
                raise ValueError(f"chunk {chunk_id!r} repeats a text that carries another vector")
            documents.append(IndexDocument(id=chunk_id, text=text, metadata=payload))
        if offset is None:
            break

    store = OpenSearchVectorStore(index_name=index_name, embeddings=_StoredVectors(vectors))
    return store.add_documents(documents)


def _unreachable_cases(store: OpenSearchVectorStore, fixture: AnchoredFixture) -> list[str]:
    """Case IDs with a gold anchor that no indexed chunk holds (same matcher as the benchmark)."""
    chunks = [
        (
            document.metadata.get("accession_number"),
            document.metadata.get("section_key"),
            normalize_text(document.text),
        )
        for document in store.iter_documents()
    ]
    return [
        case.case_id
        for case in fixture.cases
        if any(
            not any(
                accession == gold.accession_number
                and section_key == gold.section_key
                and normalize_text(gold.anchor_text) in text
                for accession, section_key, text in chunks
            )
            for gold in case.gold_evidence
        )
    ]


def wait_until_ready(
    store: OpenSearchVectorStore,
    *,
    expected_count: int,
    fixture: AnchoredFixture,
    timeout_s: float,
    poll_s: float = 2.0,
) -> None:
    """Return once the index holds `expected_count` documents and reaches every gold anchor."""
    deadline = time.monotonic() + timeout_s
    while True:
        count = store.count
        if count != expected_count:
            problem = (
                f"index {store.index_name!r} holds {count} documents, expected {expected_count}"
            )
        else:
            unreachable = _unreachable_cases(store, fixture)
            if not unreachable:
                return
            problem = f"gold anchors not found in the index for cases: {', '.join(unreachable)}"
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{problem} after {timeout_s:g}s")
        time.sleep(poll_s)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-collection", default=DEFAULT_SOURCE_COLLECTION)
    parser.add_argument("--target-index", required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--timeout-s", type=float, default=300)
    args = parser.parse_args(argv)

    source = QdrantClient(
        url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL), api_key=os.getenv("QDRANT_API_KEY")
    )
    try:
        fixture = load_anchored_fixture(args.fixture, source_dir=args.source_dir)
        expected = source.count(args.source_collection).count
        copy_vectors(source, args.source_collection, args.target_index)
        store = OpenSearchVectorStore(index_name=args.target_index)
        wait_until_ready(store, expected_count=expected, fixture=fixture, timeout_s=args.timeout_s)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"{args.target_index}: {store.count} documents ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
