from pathlib import Path

from api.run_store import FileBackedRunStore


def test_run_store_persists_record(tmp_path: Path):
    store_path = tmp_path / "run_store.json"
    store = FileBackedRunStore(store_path)

    store.create_run(job_id="job-1", ticker="AAPL", company_name="Apple")
    store.mark_running("job-1")
    store.mark_completed("job-1", result={"status": "ok"})

    reloaded = FileBackedRunStore(store_path)
    record = reloaded.get_run("job-1")

    assert record is not None
    assert record.job_id == "job-1"
    assert record.ticker == "AAPL"
    assert record.status == "completed"
    assert record.result == {"status": "ok"}

def test_run_store_marks_failure(tmp_path: Path):
    store = FileBackedRunStore(tmp_path / "run_store.json")

    store.create_run(job_id="job-2", ticker="MSFT")
    store.mark_failed("job-2", error="boom")

    record = store.get_run("job-2")
    assert record is not None
    assert record.status == "failed"
    assert record.error == "boom"
