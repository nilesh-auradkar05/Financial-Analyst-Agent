"""Trace: docs/SPEC.md 7.x (snapshot_hash) and 7.y (evidence_as_of);
docs/test-plan.md section 13 rows "Replay determinism", "Replay lineage", "Live lineage
(amended 2026-10-11)" and "Live fetch times (2026-10-11)", and section 11 row "Snapshot lineage
in response".

`app.dataops.snapshot` does not exist yet; it is imported inside each test so every test
fails for its own reason instead of failing the whole directory at collection.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

import pytest

from tests.replay.support import HEX64, parse_iso

pytestmark = pytest.mark.replay


def base_state() -> dict[str, Any]:
    return {
        "ticker": "AAPL",
        "news_articles": [
            {
                "title": "Apple services revenue rises",
                "url": "https://news.example.com/aapl/services",
                "source": "Example News",
                "snippet": "Services revenue rose.",
                "published_date": "2026-06-30",
            },
            {
                "title": "Apple expands AI features",
                "url": "https://news.example.com/aapl/ai",
                "source": "Wire",
                "snippet": "AI features expanded.",
                "published_date": "2026-07-02",
            },
        ],
        "stock_data": {"ticker": "AAPL", "current_price": 220.5, "market_cap": 3_400_000_000_000},
        "filing_chunks": [
            {"text": "Apple sells devices.", "section": "Business", "filing_type": "10-K"},
        ],
        "sentiment_result": {"overall_sentiment": "positive", "positive_count": 2},
        "started_at": "2026-07-04T10:00:00+00:00",
        "current_step": "research_news",
        "errors": [],
    }


def reversed_keys(value: Any) -> Any:
    """Same content, opposite dict insertion order, at every depth."""
    if isinstance(value, dict):
        return {key: reversed_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reversed_keys(item) for item in value]
    return value


def test_hash_is_64_lowercase_hex():
    """Replay lineage: snapshot_hash is a sha256 hex digest."""
    from app.dataops.snapshot import compute_snapshot_hash

    assert HEX64.match(compute_snapshot_hash(base_state()))


def test_hash_is_stable_for_the_same_input():
    """Replay determinism: same evidence in, same hash out."""
    from app.dataops.snapshot import compute_snapshot_hash

    assert compute_snapshot_hash(copy.deepcopy(base_state())) == compute_snapshot_hash(base_state())


def test_hash_ignores_dict_key_order():
    """Canonical JSON: key order is not evidence."""
    from app.dataops.snapshot import compute_snapshot_hash

    state = base_state()
    assert compute_snapshot_hash(reversed_keys(state)) == compute_snapshot_hash(state)


def test_hash_ignores_snapshot_id_order():
    """Source manifest ids are sorted, so the order callers pass them in does not matter."""
    from app.dataops.snapshot import compute_snapshot_hash

    ids = ["news_article:b", "market_quote:a", "sec_filing:c"]
    forward = compute_snapshot_hash(base_state(), evidence_release="r:1", snapshot_ids=ids)
    backward = compute_snapshot_hash(
        base_state(), evidence_release="r:1", snapshot_ids=list(reversed(ids))
    )
    assert forward == backward


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda s: s["news_articles"][0].update(snippet="Services revenue fell."),
            id="news_articles",
        ),
        pytest.param(lambda s: s["stock_data"].update(current_price=221.0), id="stock_data"),
        pytest.param(
            lambda s: s["filing_chunks"][0].update(text="Apple sells phones."), id="filing_chunks"
        ),
        pytest.param(
            lambda s: s["sentiment_result"].update(overall_sentiment="negative"),
            id="sentiment_result",
        ),
    ],
)
def test_hash_changes_when_any_evidence_field_changes(mutate):
    """7.x: the hash covers news, market data, filing chunks and sentiment."""
    from app.dataops.snapshot import compute_snapshot_hash

    original = base_state()
    changed = base_state()
    mutate(changed)
    assert compute_snapshot_hash(changed) != compute_snapshot_hash(original)


def test_hash_changes_when_evidence_is_removed():
    """Dropping a whole evidence class is a different snapshot, not the same one."""
    from app.dataops.snapshot import compute_snapshot_hash

    state = base_state()
    without_filings = base_state()
    without_filings["filing_chunks"] = []
    assert compute_snapshot_hash(without_filings) != compute_snapshot_hash(state)


def test_hash_depends_on_article_order_not_just_membership():
    """The evidence list order feeds the citation registry, so a reordered list is different."""
    from app.dataops.snapshot import compute_snapshot_hash

    state = base_state()
    swapped = base_state()
    swapped["news_articles"].reverse()
    assert compute_snapshot_hash(swapped) != compute_snapshot_hash(state)


def test_hash_changes_with_evidence_release():
    """7.z: the source manifest includes the release id."""
    from app.dataops.snapshot import compute_snapshot_hash

    a = compute_snapshot_hash(base_state(), evidence_release="alpha-evidence:0.1.0")
    b = compute_snapshot_hash(base_state(), evidence_release="alpha-evidence:0.2.0")
    live = compute_snapshot_hash(base_state())
    assert len({a, b, live}) == 3


def test_hash_changes_with_snapshot_ids():
    """7.z: the source manifest includes the snapshot ids."""
    from app.dataops.snapshot import compute_snapshot_hash

    a = compute_snapshot_hash(base_state(), evidence_release="r:1", snapshot_ids=["x:1", "x:2"])
    b = compute_snapshot_hash(base_state(), evidence_release="r:1", snapshot_ids=["x:1", "x:3"])
    none = compute_snapshot_hash(base_state(), evidence_release="r:1")
    assert len({a, b, none}) == 3


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda s: s.update(started_at="2030-01-01T00:00:00+00:00"), id="started_at"),
        pytest.param(lambda s: s.update(current_step="verify_memo"), id="current_step"),
        pytest.param(
            lambda s: s.update(errors=[{"step": "draft_memo", "message": "boom"}]), id="errors"
        ),
        pytest.param(lambda s: s.update(execution_time_ms=12345.0), id="execution_time_ms"),
        pytest.param(lambda s: s.update(investment_memo="A memo."), id="investment_memo"),
    ],
)
def test_hash_ignores_volatile_run_fields(mutate):
    """The hash identifies evidence, not the run that handled it."""
    from app.dataops.snapshot import compute_snapshot_hash

    state = base_state()
    mutated = base_state()
    mutate(mutated)
    assert compute_snapshot_hash(mutated) == compute_snapshot_hash(state)


FETCHED_AT = {
    "news": "2026-07-03T09:00:00+00:00",
    "stock": "2026-07-03T08:30:00+00:00",
    "filings": "2026-06-20T12:00:00+00:00",
}


def test_hash_ignores_evidence_fetched_at():
    """Live fetch times: fetch times are lineage, not evidence; they never change the hash."""
    from app.dataops.snapshot import compute_snapshot_hash

    state = base_state()
    stamped = base_state()
    stamped["evidence_fetched_at"] = dict(FETCHED_AT)
    restamped = base_state()
    restamped["evidence_fetched_at"] = {"news": "2030-01-01T00:00:00+00:00"}
    assert compute_snapshot_hash(stamped) == compute_snapshot_hash(state)
    assert compute_snapshot_hash(restamped) == compute_snapshot_hash(state)


def test_freeze_evidence_uses_the_oldest_fetch_time_as_evidence_as_of():
    """Live fetch times: evidence_as_of is the oldest evidence_fetched_at, never started_at."""
    from app.dataops.snapshot import compute_snapshot_hash, freeze_evidence, oldest_timestamp

    state = base_state()
    state["evidence_fetched_at"] = dict(FETCHED_AT)
    frozen = freeze_evidence(state, evidence_release="r:1", snapshot_ids=["x:2", "x:1"])

    assert set(frozen) == {"snapshot_hash", "evidence_as_of"}
    assert frozen["evidence_as_of"] == oldest_timestamp(FETCHED_AT.values())
    assert parse_iso(frozen["evidence_as_of"]) == min(parse_iso(v) for v in FETCHED_AT.values())
    assert frozen["evidence_as_of"] != state["started_at"]
    assert frozen["snapshot_hash"] == compute_snapshot_hash(
        state, evidence_release="r:1", snapshot_ids=["x:1", "x:2"]
    )


def test_freeze_evidence_never_uses_started_at():
    """Live lineage (amended 2026-10-11): a state with only started_at yields no evidence_as_of,
    and a missing started_at is not an error."""
    from app.dataops.snapshot import compute_snapshot_hash, freeze_evidence

    state = base_state()
    frozen = freeze_evidence(state)
    assert set(frozen) == {"snapshot_hash"}
    assert frozen["snapshot_hash"] == compute_snapshot_hash(state)

    no_start = base_state()
    del no_start["started_at"]
    assert set(freeze_evidence(no_start)) == {"snapshot_hash"}

    empty = base_state()
    empty["evidence_fetched_at"] = {}
    assert set(freeze_evidence(empty)) == {"snapshot_hash"}


def test_freeze_evidence_prefers_explicit_as_of():
    """7.y: an explicit as_of (the oldest fetched_at on replay) wins over evidence_fetched_at
    and started_at."""
    from app.dataops.snapshot import freeze_evidence

    state = base_state()
    state["evidence_fetched_at"] = dict(FETCHED_AT)
    frozen = freeze_evidence(state, as_of="2026-07-01T06:15:30+00:00")
    assert frozen["evidence_as_of"] == "2026-07-01T06:15:30+00:00"


# --- oldest_timestamp --------------------------------------------------------------------


def test_oldest_timestamp_compares_instants_across_offsets():
    """Live fetch times: the oldest is decided by absolute time. 10:00+05:30 is 04:30Z, older
    than 06:00Z although it sorts later as text."""
    from app.dataops.snapshot import oldest_timestamp

    values = ["2026-07-03T06:00:00+00:00", "2026-07-03T10:00:00+05:30", "2026-07-03T05:00:00Z"]
    assert sorted(values)[0] != "2026-07-03T10:00:00+05:30"

    result = oldest_timestamp(values)

    assert result is not None
    assert parse_iso(result) == min(parse_iso(v) for v in values)
    assert parse_iso(result) == parse_iso("2026-07-03T10:00:00+05:30")


def test_oldest_timestamp_treats_naive_values_as_utc():
    """Live fetch times: a value without an offset is read as UTC."""
    from app.dataops.snapshot import oldest_timestamp

    result = oldest_timestamp(["2026-07-03T06:00:00", "2026-07-03T08:00:00+00:00"])

    assert result is not None
    assert parse_iso(result) == datetime(2026, 7, 3, 6, 0, tzinfo=timezone.utc)


def test_oldest_timestamp_of_nothing_is_none():
    """Live fetch times: no values, no timestamp."""
    from app.dataops.snapshot import oldest_timestamp

    assert oldest_timestamp([]) is None
    assert oldest_timestamp(iter(())) is None


@pytest.mark.parametrize("value", ["2026-07-03T06:00:00", "2026-07-03T06:00:00+05:30"])
def test_oldest_timestamp_result_carries_a_utc_offset(value):
    """Live fetch times / 7.y: the result is ISO-8601 with an offset, even for naive input."""
    from app.dataops.snapshot import oldest_timestamp

    result = oldest_timestamp([value])

    assert result is not None
    parse_iso(result)
