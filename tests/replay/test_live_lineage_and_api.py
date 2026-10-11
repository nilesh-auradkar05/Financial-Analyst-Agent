"""Trace: docs/SPEC.md 7.x / 7.y / 7.z; docs/test-plan.md section 13 rows "Live lineage
(amended 2026-10-11)", "Live fetch times (2026-10-11)" and "Replay lineage", and section 11 row
"Snapshot lineage in response".

Live runs use faithful in-process fakes for every evidence entry point (no network).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tests.replay.support import HEX64, parse_iso

pytestmark = pytest.mark.replay


# --- Live lineage ------------------------------------------------------------------------


def days_ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


async def test_live_run_carries_hash_and_iso_evidence_as_of(live_world):
    """Live lineage (amended 2026-10-11): a live run ends with a 64-hex snapshot_hash and an
    ISO-8601 evidence_as_of with a UTC offset."""
    from app.agents.graph import run_agent

    live_world()
    final = await run_agent("AAPL")

    assert HEX64.match(final["snapshot_hash"])
    parse_iso(final["evidence_as_of"])
    assert "evidence_release" not in final or not final["evidence_release"]


async def test_live_evidence_as_of_is_the_filing_fetch_time_not_the_run_start(live_world):
    """Live fetch times: filings ingested before the run make evidence_as_of earlier than the
    run; it equals the oldest recorded fetched_at."""
    from app.agents.graph import run_agent

    older, newer = days_ago(30), days_ago(10)
    live_world(filing_times=[(newer.isoformat(), "2026-01-01"), (older.isoformat(), "2026-01-01")])
    final = await run_agent("AAPL")

    assert parse_iso(final["evidence_as_of"]) == older
    assert parse_iso(final["evidence_as_of"]) < parse_iso(final["started_at"])
    assert parse_iso(final["evidence_fetched_at"]["filings"]) == older


async def test_live_chunk_without_fetched_at_counts_from_its_filing_date(live_world):
    """Live fetch times: a chunk without fetched_at counts from its filing_date taken as
    midnight UTC, and it competes with chunks that do carry fetched_at."""
    from app.agents.graph import run_agent

    filing_date = days_ago(60).date()
    recorded = days_ago(5)
    live_world(filing_times=[(recorded.isoformat(), None), (None, filing_date.isoformat())])
    final = await run_agent("AAPL")

    midnight = datetime(filing_date.year, filing_date.month, filing_date.day, tzinfo=timezone.utc)
    assert parse_iso(final["evidence_fetched_at"]["filings"]) == midnight
    assert parse_iso(final["evidence_as_of"]) == midnight


async def test_live_chunk_with_no_time_at_all_contributes_nothing(live_world):
    """Live fetch times: a chunk with neither fetched_at nor filing_date adds no time; the
    other chunk still decides."""
    from app.agents.graph import run_agent

    recorded = days_ago(20)
    live_world(filing_times=[(None, None), (recorded.isoformat(), None)])
    final = await run_agent("AAPL")

    assert parse_iso(final["evidence_fetched_at"]["filings"]) == recorded
    assert parse_iso(final["evidence_as_of"]) == recorded


UNREADABLE = "not-a-time"


def filing_errors(final) -> list:
    return [e for e in final.get("errors", []) if e.get("step") == "retrieve_filings"]


async def test_live_unreadable_fetched_at_is_ignored_and_keeps_the_filing_evidence(live_world):
    """Live fetch times: a time that cannot be read is treated as absent and never costs the
    run its filing evidence; the readable chunk still decides."""
    from app.agents.graph import run_agent

    valid = days_ago(12)
    live_world(filing_times=[(UNREADABLE, None), (valid.isoformat(), None)])
    final = await run_agent("AAPL")

    assert final["filing_chunks"]
    assert filing_errors(final) == []
    assert parse_iso(final["evidence_fetched_at"]["filings"]) == valid
    assert parse_iso(final["evidence_as_of"]) == valid


async def test_live_unreadable_fetched_at_falls_back_to_the_filing_date(live_world):
    """Live fetch times: an unreadable fetched_at counts as absent, so the chunk counts from
    its filing_date (midnight UTC)."""
    from app.agents.graph import run_agent

    filing_date = days_ago(45).date()
    live_world(filing_times=[(UNREADABLE, filing_date.isoformat()), (None, None)])
    final = await run_agent("AAPL")

    midnight = datetime(filing_date.year, filing_date.month, filing_date.day, tzinfo=timezone.utc)
    assert final["filing_chunks"]
    assert filing_errors(final) == []
    assert parse_iso(final["evidence_fetched_at"]["filings"]) == midnight
    assert parse_iso(final["evidence_as_of"]) == midnight


async def test_live_all_times_unreadable_keeps_chunks_and_records_no_filing_time(live_world):
    """Live fetch times: with every time on every chunk unreadable the filing chunks remain, no
    'filings' time is recorded, and the run completes."""
    from app.agents.graph import run_agent

    live_world(filing_times=[(UNREADABLE, "13/45/2026"), (UNREADABLE, UNREADABLE)])
    final = await run_agent("AAPL")

    assert final["filing_chunks"]
    assert filing_errors(final) == []
    assert "filings" not in final.get("evidence_fetched_at", {})
    assert set(final["evidence_fetched_at"]) == {"news", "stock"}
    assert final["investment_memo"]
    assert HEX64.match(final["snapshot_hash"])


async def test_live_news_and_market_data_count_from_the_moment_they_were_fetched(live_world):
    """Live fetch times: without filings the result is the oldest of the news and stock fetch
    times, taken during the run, not the run start."""
    from app.agents.graph import run_agent

    live_world()
    before = datetime.now(timezone.utc)
    final = await run_agent("AAPL", include_filing_analysis=False)
    after = datetime.now(timezone.utc)

    assert set(final["evidence_fetched_at"]) == {"news", "stock"}
    fetched = [parse_iso(value) for value in final["evidence_fetched_at"].values()]
    assert all(before <= moment <= after for moment in fetched)
    assert parse_iso(final["evidence_as_of"]) == min(fetched)


async def test_live_source_that_returned_nothing_is_not_counted(live_world):
    """Live fetch times: an empty news result, a failed market-data fetch and an empty filing
    result add no key to evidence_fetched_at."""
    from app.agents.graph import run_agent

    recorded = days_ago(3)
    live_world(news_count=0, filing_times=[(recorded.isoformat(), "2026-01-01")])
    no_news = await run_agent("AAPL")
    live_world(stock_available=False, filing_times=[(recorded.isoformat(), "2026-01-01")])
    no_stock = await run_agent("AAPL")
    live_world(filing_times=[])
    no_filings = await run_agent("AAPL")

    assert set(no_news["evidence_fetched_at"]) == {"stock", "filings"}
    assert set(no_stock["evidence_fetched_at"]) == {"news", "filings"}
    assert set(no_filings["evidence_fetched_at"]) == {"news", "stock"}


async def test_live_run_with_no_evidence_has_no_evidence_as_of(live_world):
    """Live fetch times: when no source returned evidence evidence_as_of is absent, and the
    response carries None for it."""
    from app.agents.graph import run_agent
    from app.main import _format_response

    live_world(news_count=0, stock_available=False, filing_times=[])
    final = await run_agent("AAPL")

    assert "evidence_as_of" not in final
    assert not final.get("evidence_fetched_at")
    assert _format_response(final).evidence_as_of is None


async def test_live_hash_follows_the_evidence_not_the_clock(live_world):
    """Live lineage: identical live evidence fetched at two different times -> identical hash;
    changed evidence -> different hash."""
    from app.agents.graph import run_agent

    live_world(filing_times=[(days_ago(30).isoformat(), "2026-01-01")])
    first = await run_agent("AAPL")
    live_world(filing_times=[(days_ago(20).isoformat(), "2026-01-01")])
    second = await run_agent("AAPL")
    live_world(price_offset=5.0, filing_times=[(days_ago(30).isoformat(), "2026-01-01")])
    moved = await run_agent("AAPL")

    assert first["evidence_as_of"] != second["evidence_as_of"]
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
