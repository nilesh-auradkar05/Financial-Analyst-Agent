"""Trace: docs/SPEC.md 7.x / 7.y / 7.z; docs/test-plan.md section 13 rows "Replay determinism",
"Runtime replay bypasses live tools", "Replay lineage", "Replay refuses bad input", and
section 11 row "Snapshot lineage in response".

`app.dataops` and the `evidence_release` argument of `run_agent` do not exist yet. Imports of
new modules happen inside the tests, so each test fails individually until it is implemented.
"""

from __future__ import annotations

import json

import pytest

from tests.replay.support import HEX64, evidence_view, parse_iso, register_foreign_release

pytestmark = pytest.mark.replay


async def replay(release, ticker="AAPL", **kwargs):
    from app.agents.graph import run_agent

    return await run_agent(
        ticker,
        evidence_release=release.release_id,
        evidence_registry_root=release.registry_root,
        **kwargs,
    )


# --- Replay determinism ------------------------------------------------------------------


async def test_two_replays_give_identical_hash_and_citation_registry(release, chat):
    """Replay determinism: two replays of one release -> identical snapshot_hash and registry."""
    first = await replay(release)
    second = await replay(release)

    assert HEX64.match(first["snapshot_hash"])
    assert first["snapshot_hash"] == second["snapshot_hash"]
    assert first["evidence_as_of"] == second["evidence_as_of"]
    assert first["citation_evidence"], "the replayed memo must have a citation registry"
    assert first["citation_evidence"] == second["citation_evidence"]
    assert first["investment_memo"] == second["investment_memo"]


async def test_lineage_in_response_is_stable_across_two_replays(release, chat):
    """Section 11 'Snapshot lineage in response': both fields present and stable across replays."""
    from app.main import _format_response

    first = _format_response(await replay(release))
    second = _format_response(await replay(release))

    assert first.snapshot_hash is not None and HEX64.match(first.snapshot_hash)
    assert first.evidence_as_of is not None
    assert (first.snapshot_hash, first.evidence_as_of) == (
        second.snapshot_hash,
        second.evidence_as_of,
    )


# --- Runtime replay bypasses live tools --------------------------------------------------


async def test_replay_runs_without_live_tools_and_uses_the_releases_evidence(
    release, chat, no_live_tools
):
    """Runtime replay bypasses live tools: every live path raises, the run still completes,
    and the evidence in the final state equals the ticker's snapshots in the release."""
    final = await replay(release)

    assert final["investment_memo"].strip()
    assert evidence_view(final) == {
        "news_articles": release.expected_news("AAPL"),
        "stock_data": release.expected_stock("AAPL"),
        "filing_chunks": release.expected_filing_chunks("AAPL"),
        "sentiment_result": release.expected_sentiment("AAPL"),
    }
    live_steps = {"research_news", "fetch_stock", "retrieve_filings", "analyze_sentiment"}
    assert [e for e in final["errors"] if e.get("step") in live_steps] == []


async def test_the_memo_is_drafted_from_the_releases_evidence_only(release, chat, no_live_tools):
    """The prompt the model sees carries this ticker's snapshot text and nothing from the other
    ticker (the fake model writes its memo from that prompt)."""
    final = await replay(release)

    assert len(chat.prompts_seen) == 1
    prompt = chat.prompts_seen[0]
    for article in release.expected_news("AAPL"):
        assert article["snippet"] in prompt
    for chunk in release.expected_filing_chunks("AAPL"):
        assert chunk["text"] in prompt
    for article in release.expected_news("MSFT"):
        assert article["snippet"] not in prompt
    registry_text = json.dumps(final["citation_evidence"])
    assert "Microsoft" not in registry_text


async def test_replay_reports_only_draft_and_verify_as_progress(release, chat, no_live_tools):
    """Section 13 / 7.z: only memo drafting and verification run. Node progress, when the run
    reports any, never names an evidence node."""
    from app.agents.graph import progress_sink

    reported: list[list[dict]] = []
    token = progress_sink.set(reported.append)
    try:
        await replay(release)
    finally:
        progress_sink.reset(token)

    names = {step["node"] for snapshot in reported for step in snapshot}
    assert names <= {"draft_memo", "verify_memo"}


async def test_flags_that_exclude_evidence_apply_on_replay(release, chat, no_live_tools):
    """Request controls still apply: filing analysis and news sentiment can be switched off."""
    final = await replay(release, include_filing_analysis=False, include_news_sentiment=False)

    assert final["filing_chunks"] == []
    assert final["sentiment_result"] == {}
    assert evidence_view(final)["news_articles"] == release.expected_news("AAPL")


async def test_max_news_articles_caps_replayed_news(release, chat, no_live_tools):
    """Request controls still apply: max_news_articles bounds the replayed articles."""
    final = await replay(release, max_news_articles=1)

    assert len(final["news_articles"]) == 1
    assert final["news_articles"][0]["url"] in {a["url"] for a in release.expected_news("AAPL")}


# --- Replay lineage ----------------------------------------------------------------------


async def test_replay_carries_hash_oldest_fetch_time_and_release_id(release, chat):
    """Replay lineage: 64-hex snapshot_hash; evidence_as_of = oldest fetched_at of the ticker's
    snapshots (compared as datetimes); the release id."""
    final = await replay(release)

    assert HEX64.match(final["snapshot_hash"])
    assert parse_iso(final["evidence_as_of"]) == release.oldest_fetched_at("AAPL")
    assert final["evidence_release"] == release.release_id


async def test_evidence_as_of_is_per_ticker_not_per_release(release, chat):
    """Replay lineage: MSFT holds the oldest fetch in the release; AAPL must not report it."""
    aapl = await replay(release, "AAPL")
    msft = await replay(release, "MSFT")

    assert release.oldest_fetched_at("MSFT") < release.oldest_fetched_at("AAPL")
    assert parse_iso(aapl["evidence_as_of"]) == release.oldest_fetched_at("AAPL")
    assert parse_iso(msft["evidence_as_of"]) == release.oldest_fetched_at("MSFT")


async def test_another_ticker_gives_a_different_hash(release, chat):
    """Replay lineage: same release, different ticker -> different snapshot_hash."""
    aapl = await replay(release, "AAPL")
    msft = await replay(release, "MSFT")

    assert aapl["snapshot_hash"] != msft["snapshot_hash"]
    assert msft["evidence_release"] == release.release_id


async def test_a_changed_payload_gives_a_different_hash(release, make_release, chat):
    """Replay lineage: a second release with one changed payload (AAPL price) -> different hash."""
    changed = make_release(version="0.2.0", aapl_price=999.0)

    original = await replay(release)
    altered = await replay(changed)

    assert changed.expected_stock("AAPL") != release.expected_stock("AAPL")
    assert altered["stock_data"]["current_price"] == 999.0
    assert altered["snapshot_hash"] != original["snapshot_hash"]


async def test_a_different_release_id_gives_a_different_hash(release, make_release, chat):
    """Replay lineage: identical payloads under another release id still differ (the source
    manifest names the release)."""
    twin = make_release(name="other-evidence", version="1.0.0")

    original = await replay(release)
    renamed = await replay(twin)

    assert evidence_view(renamed) == evidence_view(original)
    assert renamed["evidence_release"] == "other-evidence:1.0.0"
    assert renamed["snapshot_hash"] != original["snapshot_hash"]


# --- load_replay_state and state_from_evidence_records -----------------------------------


def test_load_replay_state_seeds_evidence_and_lineage(release):
    """Replay lineage via the loader: seeded evidence plus the three lineage keys."""
    from app.dataops.replay import load_replay_state

    state = load_replay_state(release.release_id, "AAPL", registry_root=release.registry_root)

    assert state["ticker"] == "AAPL"
    assert evidence_view(state) == {
        "news_articles": release.expected_news("AAPL"),
        "stock_data": release.expected_stock("AAPL"),
        "filing_chunks": release.expected_filing_chunks("AAPL"),
        "sentiment_result": release.expected_sentiment("AAPL"),
    }
    assert HEX64.match(state["snapshot_hash"])
    assert parse_iso(state["evidence_as_of"]) == release.oldest_fetched_at("AAPL")
    assert state["evidence_release"] == release.release_id


def test_load_replay_state_hash_reflects_the_evidence_flags(release):
    """7.z: the hash covers the evidence the memo is drafted from, so dropping filings changes it."""
    from app.dataops.replay import load_replay_state

    full = load_replay_state(release.release_id, "AAPL", registry_root=release.registry_root)
    no_filings = load_replay_state(
        release.release_id,
        "AAPL",
        registry_root=release.registry_root,
        include_filing_analysis=False,
    )

    assert no_filings["filing_chunks"] == []
    assert no_filings["snapshot_hash"] != full["snapshot_hash"]


def test_state_from_evidence_records_builds_state_from_snapshot_records(release):
    """The relocated helper keeps its behaviour: records in, seeded AgentState out."""
    from app.dataops.replay import state_from_evidence_records

    from dataops.evidence_release import load_snapshot_records

    records = [
        record
        for record in load_snapshot_records(release.snapshot_root)
        if record.snapshot.ticker == "AAPL"
    ]
    state = state_from_evidence_records(
        "AAPL",
        records,
        include_filing_analysis=True,
        include_news_sentiment=True,
        max_news_articles=10,
    )

    assert state["company_name"] == release.expected_stock("AAPL")["company_name"]
    assert evidence_view(state) == {
        "news_articles": release.expected_news("AAPL"),
        "stock_data": release.expected_stock("AAPL"),
        "filing_chunks": release.expected_filing_chunks("AAPL"),
        "sentiment_result": release.expected_sentiment("AAPL"),
    }


# --- Replay refuses bad input ------------------------------------------------------------

MALFORMED_IDS = [
    pytest.param("alpha-evidence", id="no-colon"),
    pytest.param(":0.1.0", id="empty-name"),
    pytest.param("alpha-evidence:", id="empty-version"),
    pytest.param("alpha evidence:0.1.0", id="space"),
    pytest.param("../registry:0.1.0", id="path-traversal-name"),
    pytest.param("alpha-evidence:../0.1.0", id="path-traversal-version"),
    pytest.param("alpha:0.1:0", id="extra-colon"),
]


@pytest.fixture
def bad_input_case(request, release, tmp_path):
    """Build the registry and arguments for one refusal case: (release_id, ticker, registry, hint)."""
    kind = request.param
    if kind == "unregistered":
        return (
            "alpha-evidence:9.9.9",
            "AAPL",
            release.registry_root,
            r"not found|not registered|unknown",
        )
    if kind == "other-type":
        registry = register_foreign_release(tmp_path, "baseline:1.0.0")
        return "baseline:1.0.0", "AAPL", registry, r"evidence_snapshot"
    if kind == "hash-drift":
        path = release.quote_file("AAPL")
        record = json.loads(path.read_text())
        record["payload"]["current_price"] = 1.0
        path.write_text(json.dumps(record), encoding="utf-8")
        return release.release_id, "AAPL", release.registry_root, r"payload_hash|hash"
    if kind == "no-ticker":
        return release.release_id, "ZZZZ", release.registry_root, r"ZZZZ"
    raise AssertionError(kind)


BAD_INPUT_KINDS = ["unregistered", "other-type", "hash-drift", "no-ticker"]


@pytest.mark.parametrize("release_id", MALFORMED_IDS)
async def test_run_agent_refuses_a_malformed_release_id_before_any_model_call(
    release_id, release, chat
):
    """Replay refuses bad input: a malformed release id raises ValueError, model never invoked."""
    from app.agents.graph import run_agent

    with pytest.raises(ValueError, match=r"name:version|<name:version>|release id"):
        await run_agent(
            "AAPL", evidence_release=release_id, evidence_registry_root=release.registry_root
        )
    assert chat.prompts_seen == []


@pytest.mark.parametrize("bad_input_case", BAD_INPUT_KINDS, indirect=True)
async def test_run_agent_refuses_bad_input_before_any_model_call(bad_input_case, chat):
    """Replay refuses bad input: unregistered release, release of another type, drifted payload
    hash and a ticker with no snapshots each raise ValueError with a clear message; the model is
    never invoked."""
    from app.agents.graph import run_agent

    release_id, ticker, registry_root, hint = bad_input_case
    with pytest.raises(ValueError, match=hint) as raised:
        await run_agent(ticker, evidence_release=release_id, evidence_registry_root=registry_root)

    assert str(raised.value).strip()
    assert chat.prompts_seen == []


@pytest.mark.parametrize("release_id", MALFORMED_IDS)
def test_load_replay_state_refuses_a_malformed_release_id(release_id, release):
    """Replay refuses bad input: the loader itself rejects malformed release ids."""
    from app.dataops.replay import load_replay_state

    with pytest.raises(ValueError, match=r"name:version|<name:version>|release id"):
        load_replay_state(release_id, "AAPL", registry_root=release.registry_root)


@pytest.mark.parametrize("bad_input_case", BAD_INPUT_KINDS, indirect=True)
def test_load_replay_state_refuses_bad_input_with_a_clear_message(bad_input_case):
    """Replay refuses bad input: the loader raises ValueError naming what is wrong."""
    from app.dataops.replay import load_replay_state

    release_id, ticker, registry_root, hint = bad_input_case
    with pytest.raises(ValueError, match=hint):
        load_replay_state(release_id, ticker, registry_root=registry_root)
