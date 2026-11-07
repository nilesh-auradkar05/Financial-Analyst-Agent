"""Build an embedding arm's index from the baseline chunks (CM-5a).

Run under the arm's settings (EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS): the
baseline index is read chunk by chunk and re-embedded into the arm's own index, so every arm is
scored on the same chunks and the chunk IDs stay stable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.vector_store import IndexDocument, SearchFilters

BATCH_SIZE = 100


def build_embedding_arm(
    source: OpenSearchVectorStore, target: OpenSearchVectorStore
) -> dict[str, int]:
    """Copy every `source` chunk into `target` (re-embedded) and return the per-ticker counts."""
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
    args = parser.parse_args(argv)

    source = OpenSearchVectorStore(index_name=args.source_index)
    target = OpenSearchVectorStore()
    if source.index_name == target.index_name:
        parser.error(
            f"source and target are both {source.index_name!r}; set the arm's EMBEDDING_* variables"
        )
    for ticker, count in build_embedding_arm(source, target).items():
        print(f"{ticker}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
