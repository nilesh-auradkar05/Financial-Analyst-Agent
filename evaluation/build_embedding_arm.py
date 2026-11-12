"""Build an embedding arm's index from the baseline chunks (CM-5a).

Run under the arm's settings (EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS): the
baseline index is read chunk by chunk and re-embedded into the arm's own index, so every arm is
scored on the same chunks and the chunk IDs stay stable.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from evaluation.run_shared_retrieval_benchmark import DEFAULT_SOURCE_DIR
from evaluation.validate_retrieval_fixture import (
    AnchoredFixture,
    load_anchored_fixture,
    normalize_text,
)

BATCH_SIZE = 100
DEFAULT_FIXTURE = Path("evaluation/fixtures/retrieval_anchored_benchmark_v1.json")


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
            problem = (
                f"gold anchors not found in index {store.index_name!r} "
                f"for cases: {', '.join(unreachable)}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{problem} after {timeout_s:g}s")
        time.sleep(poll_s)


def build_embedding_arm(
    source: OpenSearchVectorStore,
    target: OpenSearchVectorStore,
    *,
    fixture: AnchoredFixture,
    timeout_s: float,
    poll_s: float = 2.0,
) -> dict[str, int]:
    """Copy every `source` chunk into `target` (re-embedded) and return the per-ticker counts.

    Writes become searchable late on OpenSearch Serverless, so the counts are checked only once
    `target` holds every source chunk and reaches every gold anchor of `fixture`.
    """
    tickers: Counter[str] = Counter()
    batch: list[IndexDocument] = []
    for document in source.iter_documents():
        tickers[str(document.metadata["ticker"])] += 1
        batch.append(document)
        if len(batch) == BATCH_SIZE:
            target.add_documents(batch)
            batch = []
    if batch:
        target.add_documents(batch)

    wait_until_ready(
        target,
        expected_count=tickers.total(),
        fixture=fixture,
        timeout_s=timeout_s,
        poll_s=poll_s,
    )
    counts = {ticker: target.count_documents(SearchFilters(ticker=ticker)) for ticker in tickers}
    mismatched = sorted(
        ticker for ticker, expected in tickers.items() if counts[ticker] != expected
    )
    if mismatched:
        raise RuntimeError(
            f"Arm index {target.index_name!r} differs from {source.index_name!r} for: {mismatched}"
        )
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", default="sec_filings")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--timeout-s", type=float, default=120)
    args = parser.parse_args(argv)

    source = OpenSearchVectorStore(index_name=args.source_index)
    target = OpenSearchVectorStore()
    if source.index_name == target.index_name:
        parser.error(
            f"source and target are both {source.index_name!r}; set the arm's EMBEDDING_* variables"
        )
    fixture = load_anchored_fixture(args.fixture, source_dir=DEFAULT_SOURCE_DIR)
    counts = build_embedding_arm(source, target, fixture=fixture, timeout_s=args.timeout_s)
    for ticker, count in counts.items():
        print(f"{ticker}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
