"""Evidence snapshot hashing and freezing (SPEC section 7.x and 7.y).

The hash identifies the evidence a memo is drafted from plus the source manifest. Run
bookkeeping such as timestamps, steps, errors and memo text is deliberately excluded.
evidence_as_of is the oldest fetch time of that evidence, never the run start time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

EVIDENCE_FIELDS: tuple[str, ...] = (
    "news_articles",
    "stock_data",
    "filing_chunks",
    "sentiment_result",
)

_EMPTY_EVIDENCE: dict[str, Any] = {
    "news_articles": [],
    "stock_data": {},
    "filing_chunks": [],
    "sentiment_result": {},
}


def _evidence_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the four evidence fields, with absent or None fields as their empty value."""
    payload: dict[str, Any] = {}
    for field in EVIDENCE_FIELDS:
        value = state.get(field)
        payload[field] = _EMPTY_EVIDENCE[field] if value is None else value
    return payload


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    """Serialize a document to canonical UTF-8 JSON bytes."""
    text = json.dumps(
        document, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False
    )
    return text.encode("utf-8")


def compute_snapshot_hash(
    state: Mapping[str, Any],
    *,
    evidence_release: str | None = None,
    snapshot_ids: Sequence[str] = (),
) -> str:
    """Return the sha256 hex digest of the evidence and source manifest (SPEC 7.x, 7.z)."""
    document = {
        "evidence": _evidence_payload(state),
        "manifest": {
            "evidence_release": evidence_release,
            "snapshot_ids": sorted(snapshot_ids),
        },
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def oldest_timestamp(values: Iterable[str]) -> str | None:
    """Return the oldest ISO-8601 timestamp, with a UTC offset, or None when there are none."""
    instants: list[datetime] = []
    for value in values:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        instants.append(moment)
    return min(instants).isoformat() if instants else None


def freeze_evidence(
    state: Mapping[str, Any],
    *,
    evidence_release: str | None = None,
    snapshot_ids: Sequence[str] = (),
    as_of: str | None = None,
) -> dict[str, str]:
    """Return the snapshot hash, plus the oldest fetch time as evidence_as_of when known.

    evidence_as_of is the oldest fetch time of the evidence (SPEC 7.y), never the run start.
    It is omitted when no source contributed a fetch time.
    """
    evidence_as_of = (
        as_of
        if as_of is not None
        else oldest_timestamp((state.get("evidence_fetched_at") or {}).values())
    )
    frozen = {
        "snapshot_hash": compute_snapshot_hash(
            state, evidence_release=evidence_release, snapshot_ids=snapshot_ids
        ),
    }
    if evidence_as_of is not None:
        frozen["evidence_as_of"] = evidence_as_of
    return frozen
