"""Trace: docs/SPEC.md 7.x / 7.y / 7.z; docs/test-plan.md section 13 rows "Live lineage" and
"Replay lineage", and section 11 row "Snapshot lineage in response".

Live runs use faithful in-process fakes for every evidence entry point (no network).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tests.replay.support import HEX64, parse_iso

pytestmark = pytest.mark.replay


# --- Live lineage ------------------------------------------------------------------------


async def test_live_run_carries_hash_and_iso_evidence_as_of(live_world):
    """Live lineage: a live run ends with a 64-hex snapshot_hash and an ISO-8601 evidence_as_of
    (the run start time, SPEC 7.y)."""
    from app.agents.graph import run_agent

    live_world()
    before = datetime.now(timezone.utc)
    final = await run_agent("AAPL")
    after = datetime.now(timezone.utc)

    assert HEX64.match(final["snapshot_hash"])
    as_of = parse_iso(final["evidence_as_of"])
    assert before <= as_of <= after
    assert "evidence_release" not in final or not final["evidence_release"]


async def test_live_hash_follows_the_evidence_not_the_clock(live_world):
    """Live lineage: identical live evidence at two different times -> identical hash; changed
    evidence -> different hash."""
    from app.agents.graph import run_agent

    live_world()
    first = await run_agent("AAPL")
    second = await run_agent("AAPL")
    live_world(price_offset=5.0)
    moved = await run_agent("AAPL")

    assert first["started_at"] != second["started_at"]
    assert first["snapshot_hash"] == second["snapshot_hash"]
    assert moved["snapshot_hash"] != first["snapshot_hash"]


async def test_live_lineage_appears_in_analysis_response(live_world):
    """Live lineage: both fields appear in AnalysisResponse for a live run."""
    from app.agents.graph import run_agent
    from app.main import _format_response

    live_world()
    final = await run_agent("AAPL")
    response = _format_response(final)

    assert response.snapshot_hash == final["snapshot_hash"]
    assert response.evidence_as_of == final["evidence_as_of"]
    assert parse_iso(response.evidence_as_of) is not None


# --- AnalysisResponse --------------------------------------------------------------------


def test_format_response_copies_lineage_from_state():
    """Section 11 'Snapshot lineage in response': _format_response copies the two fields."""
    from app.main import _format_response

    digest = "ab" * 32
    response = _format_response(
        {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "investment_memo": "A memo.",
            "snapshot_hash": digest,
            "evidence_as_of": "2026-07-01T06:15:30+00:00",
        }
    )

    assert response.snapshot_hash == digest
    assert response.evidence_as_of == "2026-07-01T06:15:30+00:00"
    dumped = response.model_dump(mode="json")
    assert dumped["snapshot_hash"] == digest
    assert dumped["evidence_as_of"] == "2026-07-01T06:15:30+00:00"


def test_format_response_without_lineage_yields_none():
    """A state with no lineage (an error path, an older producer) yields None for both fields."""
    from app.main import _format_response

    response = _format_response(
        {"ticker": "AAPL", "company_name": "Apple Inc.", "investment_memo": "A memo."}
    )

    assert response.snapshot_hash is None
    assert response.evidence_as_of is None


def test_stored_response_without_lineage_still_loads():
    """Older persisted runs lack the two fields; they must still validate, with None defaults."""
    from app.main import _format_response
    from app.models import AnalysisResponse

    stored = _format_response(
        {"ticker": "AAPL", "company_name": "Apple Inc.", "investment_memo": "A memo."}
    ).model_dump(mode="json")
    stored.pop("snapshot_hash", None)
    stored.pop("evidence_as_of", None)
    assert "snapshot_hash" not in stored and "evidence_as_of" not in stored

    loaded = AnalysisResponse.model_validate(stored)

    assert loaded.ticker == "AAPL"
    assert loaded.snapshot_hash is None
    assert loaded.evidence_as_of is None


def test_response_lineage_round_trips_through_stored_json():
    """Lineage survives persistence: dump a run result and load it back unchanged."""
    from app.models import AnalysisResponse

    original = AnalysisResponse(
        ticker="AAPL",
        company_name="Apple Inc.",
        status="completed",
        snapshot_hash="cd" * 32,
        evidence_as_of="2026-07-01T06:15:30+00:00",
    )
    loaded = AnalysisResponse.model_validate_json(original.model_dump_json())

    assert loaded.snapshot_hash == "cd" * 32
    assert loaded.evidence_as_of == "2026-07-01T06:15:30+00:00"


# --- API surface -------------------------------------------------------------------------


def test_analysis_request_does_not_accept_an_evidence_release():
    """7.z: the release is an internal argument, not accepted from API clients. The request
    model forbids unknown fields, so the key is rejected rather than carried through."""
    from app.models import AnalysisRequest

    with pytest.raises(ValidationError) as raised:
        AnalysisRequest.model_validate(
            {"ticker": "AAPL", "evidence_release": "alpha-evidence:0.1.0"}
        )

    assert any(error["loc"] == ("evidence_release",) for error in raised.value.errors())


def test_analysis_request_schema_does_not_advertise_an_evidence_release():
    """7.z: nothing in the published request schema offers the field to clients."""
    from app.models import AnalysisRequest

    schema = AnalysisRequest.model_json_schema()

    assert "evidence_release" not in schema["properties"]
