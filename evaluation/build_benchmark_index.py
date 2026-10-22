"""Index the sec_filing snapshots of an evidence release into a benchmark collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.components.retrieval.ingestion import build_index_documents
from app.components.retrieval.qdrant_store import QdrantVectorStore
from app.components.retrieval.vector_store import IndexDocument
from app.services.tools.sec_filings_tool import Filing, FilingMetaData, FilingSection
from dataops.replay import load_evidence_release_records

LIVE_COLLECTION = "sec_filings"


def build_benchmark_documents(
    release_id: str,
    *,
    registry_root: Path | str = Path("artifacts/dataops"),
) -> list[IndexDocument]:
    bundle = load_evidence_release_records(release_id, registry_root=registry_root)
    documents: list[IndexDocument] = []
    for record in bundle.records:
        if record.snapshot.source_type != "sec_filing":
            continue
        payload = record.payload
        filing = Filing(
            metadata=FilingMetaData(
                cik=payload["cik"],
                accession_number=payload["accession_number"],
                filing_type=payload["filing_type"],
                filing_date=payload["filing_date"],
                primary_document=payload["primary_document"],
                report_date=payload.get("report_date"),
                company_name=payload.get("company_name"),
                ticker=payload["ticker"],
            ),
            sections={
                name: FilingSection(name=name, content=section["content"])
                for name, section in payload["sections"].items()
            },
        )
        documents.extend(
            build_index_documents(filing, fetched_at=record.snapshot.fetched_at).documents
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a benchmark index from an evidence release."
    )
    parser.add_argument("--release", required=True, help="Evidence release id, <name:version>.")
    parser.add_argument("--collection", required=True, help="Qdrant collection to write.")
    args = parser.parse_args()

    collection = args.collection.strip()
    if not collection or collection == LIVE_COLLECTION:
        parser.error(f"--collection must be a non-empty name other than {LIVE_COLLECTION!r}")

    documents = build_benchmark_documents(args.release)
    store = QdrantVectorStore(collection_name=collection)
    written = store.add_documents(documents)
    print(f"Indexed {written} chunks into collection {collection!r}")


if __name__ == "__main__":
    main()
