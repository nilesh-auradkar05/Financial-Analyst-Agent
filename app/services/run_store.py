from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from loguru import logger


@dataclass
class RunRecord:
    job_id: str
    ticker: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    company_name: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

class FileBackedRunStore:
    """
    Small persistent run store.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        if not self.path.exists():
            self._write_all({})

    def create_run(self, job_id: str, ticker: str, company_name: Optional[str] = None) -> RunRecord:
        now = datetime.now(timezone.utc).isoformat()
        record = RunRecord(
            job_id=job_id,
            ticker=ticker.upper(),
            company_name=company_name,
            status="pending",
            started_at=now,
        )
        with self._lock:
            data = self._read_all()
            data[job_id] = asdict(record)
            self._write_all(data)
        return record

    def mark_running(self, job_id: str) -> RunRecord:
        return self._update(job_id, status="running")

    def mark_completed(self, job_id: str, result: Optional[dict[str, Any]] = None) -> RunRecord:
        return self._update(
            job_id,
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            result=result,
            error=None,
        )

    def mark_failed(self, job_id: str, error: str) -> RunRecord:
        return self._update(
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=error,
        )

    def get_run(self, job_id: str) -> Optional[RunRecord]:
        with self._lock:
            data = self._read_all()
            payload = data.get(job_id)
        return RunRecord(**payload) if payload else None

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            data = self._read_all()
            return [RunRecord(**payload) for payload in data.values()]

    def get_stats(self) -> dict[str, Any]:
        runs = self.list_runs()
        return {
            "total_runs": len(runs),
            "pending": sum(1 for r in runs if r.status == "pending"),
            "running": sum(1 for r in runs if r.status == "running"),
            "completed": sum(1 for r in runs if r.status == "completed"),
            "failed": sum(1 for r in runs if r.status == "failed"),
            "path": str(self.path),
        }

    def _update(self, job_id: str, **changes: Any) -> RunRecord:
        with self._lock:
            data = self._read_all()
            payload = data.get(job_id)
            if payload is None:
                raise KeyError(f"Run not found: {job_id}")
            payload.update(changes)
            data[job_id] = payload
            self._write_all(data)
        return RunRecord(**payload)

    def _read_all(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = self.path.read_text(encoding="utf-8").strip()
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            logger.warning("Run store file was corrupted; resetting to empty store")
            return {}

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.path)
