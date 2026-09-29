"""Tests for app.services.run_store — persistent job state."""

import pytest

from app.services.run_store import FileBackedRunStore, RunRecord


@pytest.fixture
def tmp_store(tmp_path):
    return FileBackedRunStore(tmp_path / "runs.json")


class TestFileBackedRunStore:
    """Easy: basic CRUD."""

    def test_create_run(self, tmp_store):
        record = tmp_store.create_run("job-1", "AAPL")
        assert record.job_id == "job-1"
        assert record.ticker == "AAPL"
        assert record.status == "pending"

    def test_get_run(self, tmp_store):
        tmp_store.create_run("job-1", "AAPL")
        record = tmp_store.get_run("job-1")
        assert record is not None
        assert record.ticker == "AAPL"

    def test_get_nonexistent_returns_none(self, tmp_store):
        assert tmp_store.get_run("nope") is None

    def test_mark_running(self, tmp_store):
        tmp_store.create_run("job-1", "AAPL")
        record = tmp_store.mark_running("job-1")
        assert record.status == "running"

    def test_mark_completed(self, tmp_store):
        tmp_store.create_run("job-1", "AAPL")
        record = tmp_store.mark_completed("job-1", result={"memo": "test"})
        assert record.status == "completed"
        assert record.completed_at is not None

    def test_mark_failed(self, tmp_store):
        tmp_store.create_run("job-1", "AAPL")
        record = tmp_store.mark_failed("job-1", error="boom")
        assert record.status == "failed"
        assert record.error == "boom"


class TestPersistence:
    """Medium: survives re-instantiation."""

    def test_data_persists(self, tmp_path):
        path = tmp_path / "runs.json"
        store1 = FileBackedRunStore(path)
        store1.create_run("job-1", "AAPL")

        store2 = FileBackedRunStore(path)
        record = store2.get_run("job-1")
        assert record is not None
        assert record.ticker == "AAPL"


class TestStats:
    """Medium: stats reflect actual state."""

    def test_stats_counts(self, tmp_store):
        tmp_store.create_run("j1", "AAPL")
        tmp_store.create_run("j2", "MSFT")
        tmp_store.mark_running("j1")
        tmp_store.mark_completed("j2")

        stats = tmp_store.get_stats()
        assert stats["total_runs"] == 2
        assert stats["running"] == 1
        assert stats["completed"] == 1


class TestEdgeCases:
    """Hard/Edge cases."""

    def test_update_nonexistent_raises(self, tmp_store):
        with pytest.raises(KeyError):
            tmp_store.mark_running("nonexistent")

    def test_corrupted_file_recovers(self, tmp_path):
        path = tmp_path / "runs.json"
        path.write_text("not valid json", encoding="utf-8")
        store = FileBackedRunStore(path)
        assert store.list_runs() == []

    def test_ticker_uppercased(self, tmp_store):
        record = tmp_store.create_run("j1", "aapl")
        assert record.ticker == "AAPL"

    def test_company_name_stored(self, tmp_store):
        _ = tmp_store.create_run("j1", "AAPL", company_name="Apple Inc.")
        fetched = tmp_store.get_run("j1")
        assert fetched.company_name == "Apple Inc."

    def test_run_record_fields_not_misspelled(self):
        """Regression: the old code had 'Optinal' instead of 'Optional'.
        This test ensures RunRecord can be instantiated cleanly.
        """
        record = RunRecord(
            job_id="test",
            ticker="AAPL",
            status="pending",
            started_at="2024-01-01T00:00:00Z",
            completed_at=None,
        )
        assert record.completed_at is None


def test_idempotent_run_creation_is_atomic_and_detects_changed_body(tmp_store):
    first, created = tmp_store.create_idempotent_run(
        "job-1", "AAPL", principal="principal-hash", idempotency_key="key-hash",
        request_fingerprint="body-a",
    )
    same, created_again = tmp_store.create_idempotent_run(
        "job-2", "AAPL", principal="principal-hash", idempotency_key="key-hash",
        request_fingerprint="body-a",
    )

    assert created is True
    assert created_again is False
    assert same.job_id == first.job_id

    with pytest.raises(ValueError, match="different request"):
        tmp_store.create_idempotent_run(
            "job-3", "MSFT", principal="principal-hash", idempotency_key="key-hash",
            request_fingerprint="body-b",
        )


# Trace: docs/test-plan.md §1 atomic async idempotency.
def test_concurrent_idempotency_submissions_create_one_persisted_job(tmp_store):
    from concurrent.futures import ThreadPoolExecutor
    def create(index):
        return tmp_store.create_idempotent_run(str(index), "AAPL", principal="hash", idempotency_key="key", request_fingerprint="body")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create, range(16)))
    assert len({record.job_id for record, created in results}) == 1
    assert sum(created for record, created in results) == 1
    restored = FileBackedRunStore(tmp_store.path)
    assert len(restored.list_runs()) == 1
