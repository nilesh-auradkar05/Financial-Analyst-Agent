"""Trace: docs/SPEC.md 7.x (snapshot_hash) and 7.y (evidence_as_of);
docs/test-plan.md section 13 rows "Replay determinism" and "Replay lineage", and section 11
row "Snapshot lineage in response".

`app.dataops.snapshot` does not exist yet; it is imported inside each test so every test
fails for its own reason instead of failing the whole directory at collection.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.replay.support import HEX64

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


def test_freeze_evidence_returns_hash_and_run_start_as_evidence_as_of():
    """7.y: with no explicit as_of, evidence_as_of is the state's started_at."""
    from app.dataops.snapshot import compute_snapshot_hash, freeze_evidence

    state = base_state()
    frozen = freeze_evidence(state, evidence_release="r:1", snapshot_ids=["x:2", "x:1"])

    assert set(frozen) == {"snapshot_hash", "evidence_as_of"}
    assert frozen["evidence_as_of"] == state["started_at"]
    assert frozen["snapshot_hash"] == compute_snapshot_hash(
        state, evidence_release="r:1", snapshot_ids=["x:1", "x:2"]
    )


def test_freeze_evidence_prefers_explicit_as_of():
    """7.y: an explicit as_of (the oldest fetched_at on replay) wins over started_at."""
    from app.dataops.snapshot import freeze_evidence

    frozen = freeze_evidence(base_state(), as_of="2026-07-01T06:15:30+00:00")
    assert frozen["evidence_as_of"] == "2026-07-01T06:15:30+00:00"
