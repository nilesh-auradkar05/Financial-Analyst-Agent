"""Append-only eval registry: one validated JSON record per run (SPEC §11.2)."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.agents import graph
from app.services import llm

REGISTRY_ROOT = Path("evaluation/registry/runs")

NonEmptyStr = Annotated[str, Field(min_length=1)]


class RegistryError(Exception):
    """A record was refused: invalid, ceiling-violating, or its run_id already exists."""


class EvalRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")]
    git_sha: NonEmptyStr
    branch: NonEmptyStr
    model: NonEmptyStr
    temperature: float
    prompt_version: NonEmptyStr
    evidence_release: NonEmptyStr
    snapshot_hash: NonEmptyStr
    fixture_version: NonEmptyStr
    metrics: dict[str, float]
    classification: Literal["candidate", "approved"]
    result_path: str | None = None

    @model_validator(mode="after")
    def _live_evidence_is_never_approved(self) -> EvalRunRecord:
        if self.evidence_release == "live" and self.classification == "approved":
            raise ValueError("a run on live evidence cannot be approved")
        return self


def write_run(record: Mapping[str, Any], *, root: Path = REGISTRY_ROOT) -> Path:
    try:
        validated = EvalRunRecord.model_validate(dict(record))
    except ValidationError as exc:
        raise RegistryError(str(exc)) from exc
    path = root / f"{validated.run_id}.json"
    root.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(validated.model_dump(), indent=2))
    except FileExistsError as exc:
        raise RegistryError(f"run_id {validated.run_id} already exists") from exc
    return path


def new_run_id(git_sha: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{git_sha[:7]}-{secrets.token_hex(2)}"


def prompt_version() -> str:
    text = llm.ANALYST_SYSTEM_PROMPT + graph.DRAFT_PROMPT_TEMPLATE + graph.REVISE_PROMPT_TEMPLATE
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_snapshot_hash(evidence_release: str, snapshot_hashes: Iterable[str]) -> str:
    canonical = json.dumps(
        {"evidence_release": evidence_release, "snapshots": sorted(set(snapshot_hashes))},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
