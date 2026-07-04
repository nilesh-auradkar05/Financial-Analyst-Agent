from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataops.git_state import current_code_version

SourceType = Literal["sec_filing", "news_article", "market_quote", "sentiment_score"]
ReleaseType = Literal[
    "evidence_snapshot",
    "canonical_sections",
    "evidence_chunks",
    "vector_index",
    "retrieval_benchmark",
    "answer_eval",
    "quality_baseline",
]
ReleaseStatus = Literal["candidate", "approved", "rejected", "deprecated"]


def payload_hash_for(payload: bytes | str) -> str:
    payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(payload_bytes).hexdigest()


def snapshot_id_for(source_type: str, natural_key: str, payload_hash: str) -> str:
    identity = json.dumps(
        [source_type, natural_key, payload_hash],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{source_type}:{hashlib.sha256(identity).hexdigest()}"


class EvidenceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    source_type: SourceType
    ticker: str
    natural_key: str
    payload_hash: str
    fetched_at: datetime
    fetcher_name: str
    fetcher_version: str
    storage_uri: str
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, value: str) -> str:
        return value.upper()

    @field_validator("payload_hash")
    @classmethod
    def _valid_payload_hash(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("payload_hash must be a lowercase sha256 hex digest")
        return value

    @classmethod
    def from_payload(
        cls,
        *,
        source_type: SourceType,
        ticker: str,
        natural_key: str,
        payload: bytes | str,
        fetched_at: datetime,
        fetcher_name: str,
        fetcher_version: str,
        storage_uri: str,
        metadata: dict[str, str] | None = None,
    ) -> "EvidenceSnapshot":
        payload_hash = payload_hash_for(payload)
        return cls(
            snapshot_id=snapshot_id_for(source_type, natural_key, payload_hash),
            source_type=source_type,
            ticker=ticker,
            natural_key=natural_key,
            payload_hash=payload_hash,
            fetched_at=fetched_at,
            fetcher_name=fetcher_name,
            fetcher_version=fetcher_version,
            storage_uri=storage_uri,
            metadata=metadata or {},
        )


class DatasetReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_name: str
    dataset_version: str
    release_type: ReleaseType
    release_status: ReleaseStatus
    snapshot_ids: list[str]
    config_hash: str
    code_version: str
    created_at: datetime
    quality_report_uri: str
    artifact_uri: str
    parent_release_ids: list[str] = Field(default_factory=list)

    @property
    def release_id(self) -> str:
        return f"{self.dataset_name}:{self.dataset_version}"

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        dataset_version: str,
        release_type: ReleaseType,
        release_status: ReleaseStatus,
        snapshot_ids: list[str],
        config_hash: str,
        quality_report_uri: str,
        artifact_uri: str,
        parent_release_ids: list[str],
        code_version: str | None = None,
        created_at: datetime | None = None,
    ) -> "DatasetReleaseManifest":
        return cls(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            release_type=release_type,
            release_status=release_status,
            snapshot_ids=snapshot_ids,
            config_hash=config_hash,
            code_version=code_version or current_code_version(),
            created_at=created_at or datetime.now(timezone.utc),
            quality_report_uri=quality_report_uri,
            artifact_uri=artifact_uri,
            parent_release_ids=parent_release_ids,
        )
