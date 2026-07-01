from __future__ import annotations

import app.components.retrieval.ingestion as ingestion


def test_reingesting_same_filing_is_idempotent_through_store(
    monkeypatch,
    sample_filing,
    store,
) -> None:
    """test-plan §3: same filing yields stable chunk ids and no duplicates."""

    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)

    first = ingestion.ingest_filing(sample_filing)
    first_ids = sorted(store.documents)
    second = ingestion.ingest_filing(sample_filing)
    second_ids = sorted(store.documents)

    assert first.success is True
    assert second.success is True
    assert first_ids == second_ids
    assert first_ids
    assert len(second_ids) == len(set(second_ids))
    assert store.count_documents() == len(first_ids)


def test_replace_existing_rewrites_ticker_without_duplicate_chunks(
    monkeypatch,
    sample_filing,
    store,
) -> None:
    """test-plan §3: replacement path stays stable and duplicate-free."""

    monkeypatch.setattr(ingestion, "get_vector_store", lambda: store)

    ingestion.ingest_filing(sample_filing)
    first_ids = sorted(store.documents)
    result = ingestion.ingest_filing(sample_filing, replace_existing=True)

    assert result.success is True
    assert sorted(store.documents) == first_ids
    assert store.deleted_tickers == ["AAPL"]
    assert store.count_documents() == len(first_ids)
