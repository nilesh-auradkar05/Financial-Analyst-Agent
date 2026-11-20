"""Trace: docs/SPEC.md 8.1, 8.5, 8.6; docs/adr/ADR-0017 section 2; docs/test-plan.md section 11 rows
"Verification failed, no repair", "Orphan citation", "Repair loop bounded (S6.4)", "Early stop
without improvement (S6.4)", "Not converged (S6.4)" and "Revision carries only failed claims
(S7.6)" (the part that reaches the revise prompt), and section 18 rows "Progress at completion"
and "Usage".

The verifier stays real. The only fake is the memo provider: `ScriptedChat` parses the numbered
sources out of the prompt it is given and writes one sentence per source. Each turn of its plan
says how many of those sentences to damage (a figure the source does not hold, or the citation
dropped); whether a draft passes, and its grounded rate, is decided by the real verifier. The
fake's own public records (`prompts_seen`, `memos`, `sources`, `reported`) are how a test sees
what the loop asked for.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.replay.support import (
    CLEAN,
    CLEAN_MARK,
    FIRST_DRAFT_FAILS,
    ORPHAN_FIGURE,
    TAGS,
    UNCITED_MARK,
    ScriptedChat,
    Turn,
)

pytestmark = pytest.mark.replay

TIMEOUT_SECONDS = 0.2
NODES = ["research_news", "fetch_stock", "retrieve_filings"]
FOLLOWING_NODES = ["analyze_sentiment", "draft_memo", "verify_memo"]


@pytest.fixture
def install_chat(monkeypatch: pytest.MonkeyPatch):
    from app.agents import graph

    def install(plan: list[Turn], **kwargs: Any) -> ScriptedChat:
        model = ScriptedChat(plan, **kwargs)
        monkeypatch.setattr(graph, "get_llm", lambda _settings: model)
        return model

    return install


@pytest.fixture
def short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agents.graph.settings.llm.request_timeout_seconds", TIMEOUT_SECONDS)


async def replay(release, **kwargs):
    from app.agents.graph import run_agent

    return await run_agent(
        "AAPL",
        evidence_release=release.release_id,
        evidence_registry_root=release.registry_root,
        **kwargs,
    )


def response_of(final):
    from app.main import _format_response

    return _format_response(final)


def status_of(final) -> str:
    status = response_of(final).status
    return str(getattr(status, "value", status))


def tag_of(memo: str) -> str:
    return next(tag for tag in TAGS if f"## Pass {tag}" in memo)


# --- 11 Repair loop bounded ---------------------------------------------------------------


async def test_a_draft_that_passes_costs_one_prompt_and_completes(release, install_chat):
    """11 'Repair loop bounded': the loop exits on the first passed=True; one draft is counted
    and the response carries it."""
    model = install_chat([CLEAN])
    final = await replay(release)

    assert len(model.prompts_seen) == 1
    assert final["draft_count"] == 1
    assert final["verification_result"]["passed"] is True
    assert status_of(final) == "completed"
    assert response_of(final).draft_count == 1


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param([FIRST_DRAFT_FAILS, CLEAN], id="passes-on-draft-2"),
        pytest.param([Turn(wrong=3), FIRST_DRAFT_FAILS, CLEAN], id="passes-on-draft-3"),
    ],
)
async def test_the_loop_exits_on_the_first_pass(release, install_chat, plan):
    """11 'Repair loop bounded': a later draft that passes ends the loop; the returned memo is
    that draft and the run is completed."""
    model = install_chat(plan)
    final = await replay(release)

    assert len(model.prompts_seen) == len(plan)
    assert final["draft_count"] == len(plan)
    assert tag_of(final["investment_memo"]) == TAGS[len(plan) - 1]
    assert final["verification_result"]["passed"] is True
    assert status_of(final) == "completed"
    assert response_of(final).draft_count == len(plan)


async def test_the_loop_stops_at_three_drafts_when_the_rate_keeps_rising(release, install_chat):
    """11 'Repair loop bounded' and 'Verification failed, no repair': at most 3 drafts
    (revision <= 2); a run that never passes is degraded, never completed."""
    plan = [Turn(wrong=4), Turn(wrong=3), Turn(wrong=2)]
    model = install_chat(plan)
    final = await replay(release)

    assert len(model.prompts_seen) == 3
    assert final["draft_count"] == 3
    assert final["verification_result"]["passed"] is False
    assert status_of(final) == "degraded"
    assert response_of(final).draft_count == 3


# --- 11 Early stop ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param([Turn(wrong=3), Turn(wrong=3)], id="rate-equal"),
        pytest.param([Turn(wrong=2), Turn(wrong=3)], id="rate-falls"),
    ],
)
async def test_no_third_draft_when_the_rate_does_not_rise(release, install_chat, plan):
    """11 'Early stop without improvement': grounded rate not rising from draft 1 to draft 2
    -> no third draft is requested; the run is degraded."""
    model = install_chat(plan)
    final = await replay(release)

    assert len(model.prompts_seen) == 2
    assert final["draft_count"] == 2
    assert final["verification_result"]["passed"] is False
    assert status_of(final) == "degraded"
    assert response_of(final).draft_count == 2


# --- 11 Not converged: the best draft -------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "best"),
    [
        pytest.param([Turn(wrong=2), Turn(wrong=3)], "Alpha", id="falling-keeps-draft-1"),
        pytest.param([Turn(wrong=3), Turn(wrong=3)], "Alpha", id="tie-keeps-the-earlier-draft"),
        pytest.param([Turn(wrong=4), Turn(wrong=3), Turn(wrong=2)], "Gamma", id="rising-draft-3"),
        pytest.param(
            [Turn(wrong=3), Turn(wrong=2), Turn(wrong=3)], "Beta", id="peak-in-the-middle"
        ),
    ],
)
async def test_a_run_that_does_not_converge_returns_the_best_draft(
    release, install_chat, plan, best
):
    """11 'Not converged': the best draft by grounded rate is returned; ties keep the earlier
    draft; the run is degraded."""
    model = install_chat(plan)
    final = await replay(release)

    assert len(model.prompts_seen) == len(plan)
    assert tag_of(final["investment_memo"]) == best
    assert final["verification_result"]["passed"] is False
    assert status_of(final) == "degraded"


async def test_a_tie_beyond_the_first_two_drafts_still_keeps_the_earlier_one(release, install_chat):
    """11 'Not converged': ties keep the earlier draft when the best rate is reached twice."""
    model = install_chat([Turn(wrong=3), Turn(wrong=2), Turn(wrong=2)])
    final = await replay(release)

    assert len(model.prompts_seen) == 3
    assert tag_of(final["investment_memo"]) == "Beta"


# --- 11 Not converged: uncited claims removed -----------------------------------------------


@pytest.mark.parametrize(
    ("plan", "best"),
    [
        pytest.param([Turn(uncited=1), Turn(uncited=1)], 0, id="removal-would-pass-but-did-not"),
        pytest.param(
            [Turn(wrong=3, uncited=1), Turn(wrong=2, uncited=1), Turn(wrong=1, uncited=1)],
            2,
            id="bound-reached",
        ),
    ],
)
async def test_the_returned_memo_has_no_uncited_claim_and_still_does_not_pass(
    release, install_chat, plan, best
):
    """11 'Not converged': uncited claims are removed from the returned draft; its
    verification describes that memo, and the run stays degraded even if the removal lifted
    the rates over the thresholds."""
    model = install_chat(plan)
    final = await replay(release)

    assert len(model.prompts_seen) == len(plan)
    chosen = model.memos[best]
    returned = final["investment_memo"]
    assert UNCITED_MARK in chosen, "premise: the draft the loop chose had an uncited claim"
    assert UNCITED_MARK not in returned
    for line in (raw.strip() for raw in chosen.splitlines()):
        if line and UNCITED_MARK not in line:
            assert line in returned
    assert tag_of(returned) == TAGS[best]

    summary = final["executive_summary"]
    assert summary.strip()
    assert UNCITED_MARK not in summary
    assert all(line.strip() in returned for line in summary.splitlines() if line.strip())

    verification = final["verification_result"]
    assert verification["claims"]
    assert all(
        not claim["missing_citation"] and claim["citations"] for claim in verification["claims"]
    )
    assert verification["passed"] is False
    assert status_of(final) == "degraded"


async def test_removing_uncited_claims_can_clear_the_thresholds_without_a_pass(
    release, install_chat
):
    """11 'Not converged': the recomputed verification of the returned memo meets both rate
    thresholds (coverage >= 0.80, grounded >= 0.75) yet passed stays False."""
    install_chat([Turn(uncited=1), Turn(uncited=1)])
    final = await replay(release)

    verification = final["verification_result"]
    assert verification["citation_coverage_rate"] >= 0.80
    assert verification["grounded_claim_rate"] >= 0.75
    assert verification["passed"] is False
    assert status_of(final) == "degraded"


# --- 11 Revision carries only failed claims ---------------------------------------------------


@pytest.mark.parametrize(
    ("first_draft", "failed_figures"),
    [
        pytest.param(Turn(wrong=2, marked=True), [8100, 8101], id="unsupported-claims"),
        pytest.param(Turn(orphan=True, marked=True), [int(ORPHAN_FIGURE)], id="orphan-citation"),
    ],
)
async def test_the_revise_prompt_names_the_failed_claims_and_carries_the_evidence(
    release, install_chat, first_draft, failed_figures
):
    """11 'Revision carries only failed claims' (S6.4 part): the revision prompt lists the failed
    claims (a figure of each appears in the previous draft and in the failed list), does not
    repeat the claims that passed, and carries the numbered evidence."""
    model = install_chat([first_draft, CLEAN])
    await replay(release)

    assert len(model.prompts_seen) == 2
    revision = model.prompts_seen[1]
    for figure in failed_figures:
        assert revision.count(str(figure)) >= 2, f"failed claim {figure} not named"
    passing = [
        f"{CLEAN_MARK}{'abcdefgh'[position]}"
        for position in range(first_draft.wrong, len(model.sources))
    ]
    assert passing
    for mark in passing:
        assert revision.count(mark) == 1, f"passing claim {mark} repeated"
    for number, core in model.sources:
        assert f"[{number}]" in revision
        assert core in revision
    if first_draft.orphan:
        assert revision.count("[99]") >= 2


async def test_an_orphan_citation_alone_triggers_a_revision(release, install_chat):
    """11 'Orphan citation': a draft that is fine except for a citation outside the registry
    fails, is revised once, and the revision without the orphan passes."""
    model = install_chat([Turn(orphan=True), CLEAN])
    final = await replay(release)

    assert len(model.prompts_seen) == 2
    assert final["draft_count"] == 2
    assert tag_of(final["investment_memo"]) == "Beta"
    assert final["verification_result"]["orphan_citations"] == []
    assert final["verification_result"]["passed"] is True
    assert status_of(final) == "completed"


async def test_an_orphan_citation_that_is_never_fixed_is_degraded(release, install_chat):
    """11 'Orphan citation': any [N] outside the registry on the returned memo -> passed False,
    orphan_citations=[N], degraded."""
    model = install_chat([Turn(orphan=True), Turn(orphan=True)])
    final = await replay(release)

    assert len(model.prompts_seen) == 2
    assert final["verification_result"]["orphan_citations"] == [99]
    assert final["verification_result"]["passed"] is False
    assert status_of(final) == "degraded"


# --- Failure paths ------------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["empty", "raise", "hang"])
async def test_nothing_to_revise_when_the_first_draft_never_arrives(
    release, install_chat, short_timeout, outcome
):
    """11 'Repair loop bounded' / Safe errors: an empty memo, a provider error or a timeout on
    draft 1 gives no memo to repair: one prompt, no revision, no invented memo, failed."""
    model = install_chat([Turn(outcome=outcome)])
    final = await replay(release)

    assert len(model.prompts_seen) == 1
    assert not (final.get("investment_memo") or "").strip()
    assert status_of(final) == "failed"


@pytest.mark.parametrize("outcome", ["raise", "hang"])
async def test_a_failed_revision_degrades_and_returns_the_first_draft_cleaned(
    release, install_chat, short_timeout, outcome
):
    """11 'Not converged': when the revision call fails the run ends degraded, not failed,
    returning draft 1 with its uncited claims removed; only one draft exists."""
    model = install_chat([Turn(wrong=1, uncited=1), Turn(outcome=outcome)])
    final = await replay(release)

    assert len(model.prompts_seen) == 2
    assert final["draft_count"] == 1
    assert len(final["draft_history"]) == 1
    returned = final["investment_memo"]
    assert tag_of(returned) == "Alpha"
    assert UNCITED_MARK not in returned
    assert all(
        line.strip() in returned
        for line in model.memos[0].splitlines()
        if line.strip() and UNCITED_MARK not in line
    )
    assert final["verification_result"]["passed"] is False
    assert status_of(final) == "degraded"
    assert response_of(final).draft_count == 1


# --- State: draft_history ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "grounded", "passed"),
    [
        pytest.param([CLEAN], [4], [True], id="one-draft"),
        pytest.param(
            [Turn(wrong=3), FIRST_DRAFT_FAILS, CLEAN],
            [1, 2, 4],
            [False, False, True],
            id="three-drafts",
        ),
        pytest.param([Turn(wrong=3), Turn(wrong=3)], [1, 1], [False, False], id="early-stop"),
    ],
)
async def test_draft_history_records_every_draft_in_order(
    release, install_chat, plan, grounded, passed
):
    """11 'Repair loop bounded': every draft's grounded rate and pass flag is recorded, in order;
    the rate is grounded claims over cited claims of that draft (four claims per draft here)."""
    model = install_chat(plan)
    final = await replay(release)

    history = final["draft_history"]
    assert len(model.prompts_seen) == len(plan)
    assert [entry["draft"] for entry in history] == list(range(1, len(plan) + 1))
    assert [entry["passed"] for entry in history] == passed
    assert [entry["grounded_claim_rate"] for entry in history] == pytest.approx(
        [count / len(model.sources) for count in grounded]
    )
    assert final["draft_count"] == len(history)


REVISE_MARK = "Zebracorn revision clause."


async def test_every_revision_prompt_is_formatted_from_the_revise_template(
    release, install_chat, monkeypatch
):
    """14 'prompt_version' (S6.4 amendment): `graph.REVISE_PROMPT_TEMPLATE` is the template every
    revision prompt is formatted from; text appended to it reaches each revision and no draft."""
    from app.agents import graph

    template = getattr(graph, "REVISE_PROMPT_TEMPLATE", "")
    monkeypatch.setattr(
        graph, "REVISE_PROMPT_TEMPLATE", template + "\n" + REVISE_MARK, raising=False
    )
    model = install_chat([Turn(wrong=3), Turn(wrong=2), CLEAN])
    await replay(release)

    assert len(model.prompts_seen) == 3
    assert REVISE_MARK not in model.prompts_seen[0]
    assert REVISE_MARK in model.prompts_seen[1]
    assert REVISE_MARK in model.prompts_seen[2]


def expected_reviewer_view(memo: str, registry: list[dict[str, Any]]):
    """The verifier's verdict on one draft, reduced to what `draft_history` must record."""
    import re

    from app.verification.grounding import evaluate_memo_grounding

    verdict = evaluate_memo_grounding(memo, registry)
    failed = [
        {"sentence": claim.sentence[:500], "reason": claim.reason}
        for claim in verdict.claims
        if not claim.supported
    ]
    valid = set(range(1, len(registry) + 1))
    orphans = sorted({int(n) for n in re.findall(r"\[(\d+)\]", memo)} - valid)
    return verdict, failed, orphans


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param([CLEAN], id="passing-draft-records-nothing-failed"),
        pytest.param([FIRST_DRAFT_FAILS, CLEAN], id="unsupported-claims"),
        pytest.param([Turn(wrong=1, uncited=1), CLEAN], id="unsupported-and-uncited"),
        pytest.param([Turn(orphan=True), CLEAN], id="orphan-citation"),
        pytest.param([Turn(wrong=3), Turn(wrong=2), CLEAN], id="three-drafts"),
        pytest.param([Turn(wrong=3), Turn(wrong=3)], id="early-stop"),
        pytest.param([Turn(uncited=1), Turn(uncited=1)], id="uncited-never-fixed"),
    ],
)
async def test_draft_history_records_each_drafts_failed_claims_and_orphans(
    release, install_chat, plan
):
    """11 'Reviewer reasons recorded' (S6.4, OBS-R3): each entry carries that draft's unsupported
    claims in memo order as {sentence, reason}, and its orphan citations; a passing draft
    records none. Expected values come from the real verifier applied to each draft's memo."""
    model = install_chat(plan)
    final = await replay(release)

    history = final["draft_history"]
    registry = final["citation_evidence"]
    assert len(history) == len(plan) == len(model.memos)
    views = [expected_reviewer_view(memo, registry) for memo in model.memos]
    for (_verdict, failed, orphans), turn in zip(views, plan):
        assert len(failed) == turn.wrong + turn.uncited + int(turn.orphan), "premise"
        assert (orphans == [99]) is turn.orphan, "premise"
    for entry, (verdict, failed, orphans) in zip(history, views):
        assert entry["grounded_claim_rate"] == pytest.approx(verdict.grounded_claim_rate)
        assert entry["failed_claims"] == failed
        assert entry["orphan_citations"] == orphans
        if entry["passed"]:
            assert entry["failed_claims"] == [] and entry["orphan_citations"] == []
    recorded_uncited = [
        claim for claim in history[0]["failed_claims"] if UNCITED_MARK in claim["sentence"]
    ]
    assert len(recorded_uncited) == plan[0].uncited
    assert all(claim["reason"] for claim in recorded_uncited)


async def test_a_failed_sentence_is_stored_cut_to_500_characters(release, install_chat):
    """11 'Reviewer reasons recorded' (S6.4): a failed claim's sentence is cut to at most 500
    characters; the reason is the verifier's, unchanged."""
    from app.verification.grounding import evaluate_memo_grounding

    model = install_chat([Turn(wrong=2, long=True), CLEAN])
    final = await replay(release)

    verdict = evaluate_memo_grounding(model.memos[0], final["citation_evidence"])
    uncut = [claim for claim in verdict.claims if not claim.supported]
    assert len(uncut) == 2
    assert all(len(claim.sentence) > 500 for claim in uncut), "premise: the sentences are long"
    recorded = final["draft_history"][0]["failed_claims"]
    assert [len(claim["sentence"]) for claim in recorded] == [500, 500]
    assert [claim["sentence"] for claim in recorded] == [c.sentence[:500] for c in uncut]
    assert [claim["reason"] for claim in recorded] == [c.reason for c in uncut]
    assert all(claim["reason"] for claim in recorded)


# --- 18 Usage -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param([FIRST_DRAFT_FAILS, CLEAN], id="two-drafts"),
        pytest.param([Turn(wrong=3), FIRST_DRAFT_FAILS, CLEAN], id="three-drafts"),
    ],
)
async def test_usage_is_the_total_across_every_draft_of_the_run(release, install_chat, plan):
    """18 'Usage' (S6.4 decision): token counts on the result total every provider call of the
    run; the model is the one the provider reported; the response counts drafts."""
    model = install_chat(plan, report_usage=True)
    final = await replay(release)

    assert len(model.reported) == len(plan)
    assert final["llm_usage"] == {
        "model": "echo-1",
        "input_tokens": sum(inp for inp, _ in model.reported),
        "output_tokens": sum(out for _, out in model.reported),
    }
    assert final["llm_usage"]["input_tokens"] > model.reported[-1][0]
    assert response_of(final).draft_count == len(plan)


async def test_a_provider_reporting_no_usage_yields_nulls_across_drafts(release, install_chat):
    """18 'Usage': a provider that reports none yields nulls, however many drafts ran."""
    model = install_chat([FIRST_DRAFT_FAILS, CLEAN])
    final = await replay(release)

    assert len(model.prompts_seen) == 2
    assert final["llm_usage"] == {"model": None, "input_tokens": None, "output_tokens": None}


# --- Response ---------------------------------------------------------------------------------


def test_a_state_without_a_draft_count_yields_none_in_the_response():
    """11: AnalysisResponse.draft_count defaults to None (older runs, error paths) and still
    loads from stored JSON that predates it."""
    from app.models import AnalysisResponse

    response = response_of(
        {"ticker": "AAPL", "company_name": "Apple Inc.", "investment_memo": "A memo."}
    )
    assert response.draft_count is None

    stored = response.model_dump(mode="json")
    stored.pop("draft_count", None)
    assert AnalysisResponse.model_validate(stored).draft_count is None


# --- Live graph ---------------------------------------------------------------------------------


async def test_live_run_that_passes_on_draft_one_reports_exactly_the_six_nodes(
    live_world, install_chat
):
    """18 'Progress at completion': a run that passes on draft 1 reports the six nodes in start
    order and no revise node."""
    from app.agents.graph import progress_sink, run_agent

    live_world()
    model = install_chat([CLEAN])
    snapshots: list[list[dict]] = []
    token = progress_sink.set(snapshots.append)
    try:
        final = await run_agent("AAPL")
    finally:
        progress_sink.reset(token)

    order = [step["node"] for step in snapshots[-1]]
    assert len(model.prompts_seen) == 1
    assert set(order[:3]) == set(NODES)
    assert order[3:] == FOLLOWING_NODES
    assert len(order) == 6
    assert final["draft_count"] == 1
    assert status_of(final) == "completed"


async def test_live_run_revises_a_failed_draft_once_and_completes(live_world, install_chat):
    """11 'Repair loop bounded': the live graph has the same loop as replay."""
    from app.agents.graph import run_agent

    live_world()
    model = install_chat([FIRST_DRAFT_FAILS, CLEAN])
    final = await run_agent("AAPL")

    assert len(model.prompts_seen) == 2
    assert final["draft_count"] == 2
    assert tag_of(final["investment_memo"]) == "Beta"
    assert final["verification_result"]["passed"] is True
    assert [entry["passed"] for entry in final["draft_history"]] == [False, True]
    assert status_of(final) == "completed"
    assert response_of(final).draft_count == 2
