"""DeepEval faithfulness answer-quality metric (docs/test-plan.md §14, S6.5; SPEC §4 judge ladder, §11.2).

``python -m evaluation.answer_quality`` has two commands: ``validate`` scores the claims of a labelled
S6.3 review file and compares the verdicts with the owner labels (weighted κ), ``score`` scores every
cited claim of a pinned approved baseline and records a candidate registry entry.

The judge is the one system boundary faked: ``FakeJudge`` stands in for the DeepEval model that
``evaluation.judge_models_interface.create_deepeval_judge_model`` returns. The metric itself is the
real ``deepeval.metrics.FaithfulnessMetric``, so the fake answers its four prompts (truths, claims,
verdicts, reason) by computing from the prompt text it receives: the truths are the evidence, a
claim is split into statements on ``;``, and a statement is ``yes`` when an evidence line says the
same thing with the same figures, ``no`` when an evidence line says it with other figures and
``borderline`` when nothing in the evidence says it. Synthetic baselines and review files are built
the way ``test_eval_replay_gate`` and ``test_judge_agreement`` build them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import socket
import statistics
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import pytest
from deepeval.metrics.base_metric import Verdict
from deepeval.metrics.faithfulness.schema import (
    Claims,
    FaithfulnessScoreReason,
    FaithfulnessVerdict,
    Truths,
    Verdicts,
)
from deepeval.models.base_model import DeepEvalBaseLLM

from evaluation.judge_models_interface import JudgeModelConfig
from tests.embedding_arms import isolate_environment
from tests.unit.test_eval_replay_gate import (
    BASELINE_SHAPES,
    RELEASE_ID,
    RUN_ID,
    registry_entries,
    write_baseline,
)
from tests.unit.test_judge_agreement import (
    MIXED,
    claim_id_of,
    edit_result,
    population,
    result_path,
    sample_argv,
)
from tests.unit.test_judge_ladder import AZURE_KEY, BEDROCK_ID, BEDROCK_KEY, DEPLOYMENT, SECRETS

KAPPA_GATE = 0.85
OUTLOOK = {
    "citation": 1,
    "title": "Outlook call",
    "text": "Management discussed the quarterly outlook.",
}
VALIDATE_FIELDS = {
    "baseline_run_id",
    "n",
    "judge_model",
    "temperature",
    "effort",
    "penalize_ambiguous_claims",
    "git_sha",
    "claims",
    "errors",
    "weighted_kappa",
    "unweighted_kappa",
    "table",
    "disagreements",
    "validated",
}
LINEAGE_FIELDS = (
    "run_id",
    "git_sha",
    "branch",
    "model",
    "temperature",
    "prompt_version",
    "evidence_release",
    "snapshot_hash",
    "fixture_version",
)


def aq() -> Any:
    import evaluation.answer_quality as module

    return module


class Result(NamedTuple):
    code: Any
    out: str
    err: str


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> Result:
    """Run the CLI; its exit code is the return value or the SystemExit code."""
    try:
        code: Any = aq().main(argv)
    except SystemExit as exit_:
        code = exit_.code
    captured = capsys.readouterr()
    return Result(code, captured.out, captured.err)


@pytest.fixture(autouse=True)
def socket_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """No connect leaves the process: the judge is faked, the baselines are files."""
    attempts: list[str] = []
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> NoReturn | None:
        if self.family == socket.AF_UNIX:
            return real_connect(self, address)
        attempts.append(repr(address))
        raise OSError(f"network access attempted: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield attempts
    assert attempts == [], f"connect attempted during test: {attempts}"


# --- Settings and the judge seam ---------------------------------------------------------------


@dataclass
class Env:
    """The environment a deployment supplies: which judge tiers are configured. Every value is a dummy."""

    monkeypatch: pytest.MonkeyPatch
    workdir: Path

    def configure(
        self, *, foundry: bool = True, bedrock: bool = False, blank: Sequence[str] = ()
    ) -> None:
        isolate_environment(self.monkeypatch, self.workdir)
        for name in ("RAG_EVAL_JUDGE_MODEL", "RAG_EVAL_JUDGE_PROVIDER", "DEEPEVAL_MODEL"):
            self.monkeypatch.delenv(name, raising=False)
        env = {
            "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
            # DeepEval's own settings refuse an empty endpoint, so it stays; the tier is the key and deployment.
            "AZURE_OPENAI_ENDPOINT": "http://127.0.0.1:9",
            "AZURE_API_KEY": AZURE_KEY if foundry else "",
            "AZURE_FOUNDRY_JUDGE_MODEL": DEPLOYMENT if foundry else "",
            "AWS_BEARER_TOKEN_BEDROCK": BEDROCK_KEY if bedrock else "",
            "AWS_REGION": "us-east-1",
        }
        for name, value in {**env, **dict.fromkeys(blank, "")}.items():
            self.monkeypatch.setenv(name, value)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Env:
    configured = Env(monkeypatch, tmp_path)
    configured.configure()
    return configured


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", text.lower()))


def _figures(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _statements(context: str) -> list[str]:
    return [
        part.strip()
        for line in context.split("\n")
        for part in re.split(r"(?<=[.!?])\s+", line)
        if part.strip()
    ]


def judge_statement(statement: str, context: str) -> FaithfulnessVerdict:
    """yes: an evidence line says the same with the same figures; no: same words, other figures."""
    same = [line for line in _statements(context) if _words(line) == _words(statement)]
    if any(_figures(line) == _figures(statement) for line in same):
        return FaithfulnessVerdict(verdict=Verdict.YES)
    if same:
        return FaithfulnessVerdict(verdict=Verdict.NO, reason=f"The evidence states: {same[0]}")
    return FaithfulnessVerdict(
        verdict=Verdict.BORDERLINE, reason="Nothing in the evidence states this."
    )


_TRUTHS = re.compile(r"\n\nText:\n(.*)\n\nJSON:\s*$", re.DOTALL)
_CLAIMS = re.compile(r"\n\nAI Output:\n(.*)\n\nJSON:\s*$", re.DOTALL)
_VERDICTS = re.compile(r"\n\nRetrieval Contexts:\n(.*)\n\nClaims:\n(.*)\n\nJSON:\s*$", re.DOTALL)
_REASON = re.compile(
    r"\n\nFaithfulness Score:\n(\S+)\n\nContradictions:\n(.*)\n\nJSON:\s*$", re.DOTALL
)


@dataclass
class Seam:
    """What the faked judge boundary saw: the configs it was built from and every prompt it received."""

    configs: list[JudgeModelConfig] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)  # each case's `actual_output`
    verdict_cases: list[tuple[str, list[str]]] = field(default_factory=list)  # (context, claims)
    fail_marker: str | None = None
    garble_marker: str | None = None

    def contexts_of(self, claim: str) -> set[str]:
        """The retrieval context text of every case whose only statement is `claim`."""
        return {context for context, claims in self.verdict_cases if claims == [claim]}


class FakeJudge(DeepEvalBaseLLM):  # type: ignore[misc]
    """A DeepEval model that computes each reply from the prompt it gets (no baked answers)."""

    def __init__(self, name: str, seam: Seam) -> None:
        self.name = name
        self.seam = seam

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return str(self.name)

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)

    def generate(self, prompt: str, schema: Any = None) -> Any:
        seam = self.seam
        seam.prompts.append(prompt)
        if seam.fail_marker and seam.fail_marker in prompt:
            raise RuntimeError("judge unavailable")
        step = schema.__name__
        if seam.garble_marker and seam.garble_marker in prompt and step == "Claims":
            return "I am unable to answer that."
        if step == "Truths":
            match = _TRUTHS.search(prompt)
            assert match, prompt[-300:]
            return Truths(truths=[match.group(1)])
        if step == "Claims":
            match = _CLAIMS.search(prompt)
            assert match, prompt[-300:]
            seam.outputs.append(match.group(1))
            return Claims(claims=[p.strip() for p in match.group(1).split(";") if p.strip()])
        if step == "Verdicts":
            match = _VERDICTS.search(prompt)
            assert match, prompt[-300:]
            context, claims = match.group(1), ast.literal_eval(match.group(2))
            seam.verdict_cases.append((context, claims))
            return Verdicts(verdicts=[judge_statement(claim, context) for claim in claims])
        assert step == "FaithfulnessScoreReason", step
        match = _REASON.search(prompt)
        assert match, prompt[-300:]
        contradictions = ast.literal_eval(match.group(2))
        because = "; ".join(contradictions) or "every statement is in the evidence"
        return FaithfulnessScoreReason(reason=f"The score is {match.group(1)} because {because}.")


@pytest.fixture
def seam(monkeypatch: pytest.MonkeyPatch) -> Seam:
    """The judge the metric talks to: whatever `create_deepeval_judge_model` is asked for."""
    recorded = Seam()

    def create(config: JudgeModelConfig) -> FakeJudge:
        recorded.configs.append(config)
        return FakeJudge(str(config.model), recorded)

    monkeypatch.setattr("evaluation.judge_models_interface.create_deepeval_judge_model", create)
    return recorded


# --- Synthetic baselines and review files --------------------------------------------------------


def registry_root(workdir: Path) -> Path:
    return workdir / "registry" / "runs"


def records(workdir: Path) -> list[str]:
    return sorted(path.name for path in registry_root(workdir).glob("*.json"))


def baseline(workdir: Path, shapes: Sequence[tuple[int, int]] = BASELINE_SHAPES) -> Path:
    """An approved token-replay baseline whose runs carry the (ticker, repeat) of a repeated run."""
    root = write_baseline(workdir, recorded_shapes=shapes)

    def add_repeats(data: dict[str, Any]) -> None:
        for number, raw in enumerate(data["raw_runs"]):
            raw["repeat"] = (2, 0, 1)[number % 3]

    edit_result(workdir, add_repeats)
    return root


def score_argv(workdir: Path, out_dir: Path, *, baseline_id: str | None = RUN_ID) -> list[str]:
    return [
        "score",
        *(["--baseline", baseline_id] if baseline_id else []),
        "--out-dir",
        str(out_dir),
        "--registry-root",
        str(registry_root(workdir)),
    ]


def validate_argv(workdir: Path, review: Path, out: Path) -> list[str]:
    return [
        "validate",
        "--review",
        str(review),
        "--out",
        str(out),
        "--registry-root",
        str(registry_root(workdir)),
    ]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def result_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("*.json")) if out_dir.exists() else []


@dataclass
class Review:
    """A labelled review file over the MIXED baseline (N = 90 supported, 30 unsupported, 25 each drawn)."""

    path: Path
    heuristic: dict[str, bool]
    owner: dict[str, bool]
    judged: dict[str, bool]  # what the faked judge decides from each claim's evidence
    weights: dict[str, float]
    supported_ids: list[str]
    unsupported_ids: list[str]

    def edit(self, change: Callable[[dict[str, Any]], None]) -> None:
        data = load(self.path)
        change(data)
        self.path.write_text(json.dumps(data))


def labelled_review(
    workdir: Path,
    *,
    judge_flip: tuple[Sequence[int], Sequence[int]] = ((), ()),
    owner_flip: tuple[Sequence[int], Sequence[int]] = ((), ()),
) -> Review:
    """The S6.3 `sample` file of the MIXED baseline (25 claims per heuristic stratum), labelled.

    The owner's label is the heuristic verdict, flipped for the given positions in the sorted claim
    ids of the supported and unsupported strata. The faked judge decides from the evidence alone, so
    the evidence of a claim it should flip is replaced: a supported claim gets evidence that never
    mentions it (the judge finds nothing), an unsupported claim gets evidence quoting it.
    """
    from evaluation import judge_agreement

    write_baseline(workdir, recorded_shapes=MIXED)
    path = workdir / "review.json"
    assert judge_agreement.main(sample_argv(workdir, path, k=25, seed=7)) == 0
    data = load(path)
    by_id = {c.claim_id: c for c in population(workdir)}
    ids = [item["claim_id"] for item in data["claims"]]
    supported = sorted(i for i in ids if by_id[i].supported)
    unsupported = sorted(i for i in ids if not by_id[i].supported)
    assert (len(supported), len(unsupported)) == (25, 25)
    assert data["strata"] == {"supported": {"N": 90, "n": 25}, "unsupported": {"N": 30, "n": 25}}
    owner_flipped = {supported[p] for p in owner_flip[0]} | {unsupported[p] for p in owner_flip[1]}
    judge_flipped = {supported[p] for p in judge_flip[0]} | {unsupported[p] for p in judge_flip[1]}
    heuristic = {i: by_id[i].supported for i in ids}
    for item in data["claims"]:
        claim_id = item["claim_id"]
        item["owner_label"] = (
            "supported" if heuristic[claim_id] != (claim_id in owner_flipped) else "unsupported"
        )
        if claim_id in judge_flipped:
            item["evidence"] = (
                [OUTLOOK]
                if heuristic[claim_id]
                else [{"citation": 1, "title": "Restated figures", "text": item["claim"]}]
            )
    path.write_text(json.dumps(data))
    weights = {i: (3.6 if heuristic[i] else 1.2) for i in ids}
    return Review(
        path,
        heuristic,
        {i: heuristic[i] != (i in owner_flipped) for i in ids},
        {i: heuristic[i] != (i in judge_flipped) for i in ids},
        weights,
        supported,
        unsupported,
    )


# --- Faithfulness claim scoring -----------------------------------------------------------------


def custom_baseline(
    workdir: Path, memo: str, edit_registry: Callable[[list[dict[str, Any]]], None] | None = None
) -> Path:
    """A two-run baseline whose recorded drafts are both `memo` (optionally over an edited registry)."""
    root = baseline(workdir, ((0, 0), (0, 0)))

    def replace(data: dict[str, Any]) -> None:
        for raw in data["raw_runs"]:
            raw["recorded_memo"] = memo
            if edit_registry:
                edit_registry(raw["recorded_registry"])

    edit_result(workdir, replace)
    return root


SCORED_MEMO = "\n".join(
    [
        "Segment 1 revenue was $101 billion in fiscal 2024 [2][1].",
        "Segment 3 revenue was $903 billion in fiscal 2024 [4].",
        "Segment 5 revenue was $105 billion in fiscal 2024; Segment 5 margin was 12 percent [6].",
        "Segment 7 revenue was $107 billion in fiscal 2024 [25].",
        "Segment 9 headcount grew sharply [10].",
        "Segment 3 revenue was $103 billion in fiscal 2024 [25][4].",
        "Segment 11 revenue was $111 billion in fiscal 2024.",
    ]
)
# Sentence of each cited claim as the runtime verifier reports it (the citation marker is dropped),
# with the score the faked judge's statements give it:
#   1 yes -> 1.0 · 1 no -> 0.0 · yes + borderline -> 0.5 · evidence only a missing entry -> 0.0
#   a statement no evidence line makes -> 0.0 · a missing entry plus one that states it -> 1.0
EXPECTED_SCORES = {
    "Segment 1 revenue was $101 billion in fiscal 2024 .": 1.0,
    "Segment 3 revenue was $903 billion in fiscal 2024 .": 0.0,
    "Segment 5 revenue was $105 billion in fiscal 2024; Segment 5 margin was 12 percent .": 0.5,
    "Segment 7 revenue was $107 billion in fiscal 2024 .": 0.0,
    "Segment 9 headcount grew sharply .": 0.0,
    "Segment 3 revenue was $103 billion in fiscal 2024 .": 1.0,
}


def test_a_claim_is_supported_only_at_score_1_and_an_ambiguous_statement_counts_against_it(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Faithfulness claim scoring (S6.5): each cited claim is one metric case; supported iff the
    score is exactly 1.0, so one statement the judge cannot find makes the claim unsupported (score
    0.5), and a statement the evidence does not state (borderline) counts against the claim
    (penalize_ambiguous_claims; the DeepEval default would score it 1.0); every claim keeps the
    metric's own reason; the uncited claim is not scored."""
    custom_baseline(env.workdir, SCORED_MEMO)
    out_dir = env.workdir / "scored"

    result = run(score_argv(env.workdir, out_dir), capsys)

    assert result.code == 0, result
    [result_file] = result_files(out_dir)
    runs = load(result_file)["runs"]
    assert len(runs) == 2
    for number, scored in enumerate(runs, start=1):
        claims = {c["claim_id"]: c for c in scored["claims"]}
        expected = {claim_id_of(number, s): score for s, score in EXPECTED_SCORES.items()}
        assert {i: c["score"] for i, c in claims.items()} == pytest.approx(expected)
        assert {i: c["supported"] for i, c in claims.items()} == {
            i: score == 1.0 for i, score in expected.items()
        }
        for claim in claims.values():
            assert isinstance(claim["reason"], str) and claim["reason"].strip()
            assert f"{claim['score']:.2f}" in claim["reason"]
        assert scored["errors"] == []
        assert scored["cited_claims"] == len(EXPECTED_SCORES) == 6
        assert scored["supported_claims"] == 2
        assert scored["faithful_claim_rate"] == pytest.approx(2 / 6)


def test_each_claim_is_scored_on_its_cited_evidence_title_and_full_text_in_citation_order(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Faithfulness claim scoring (S6.5): `retrieval_context` holds one entry per cited evidence
    item in citation order (the order the memo cites them: [2][1] is entry 2 then entry 1), each its
    title, a newline and its full text, uncapped and unescaped; an item marked `missing` adds no entry
    (a claim citing only a missing entry has an empty context)."""
    long_text = "Segment 3 revenue was $103 billion in fiscal 2024. " + (
        "Operating notes continue & expand <see below>. " * 80
    )
    assert len(long_text) > 3000

    def edit(entries: list[dict[str, Any]]) -> None:
        entries[1]["title"] = "Segment 1 report & outlook"
        entries[1]["text"] = "Segment 1 revenue was $101 billion in fiscal 2024. R&D <est> & more."
        entries[3]["text"] = long_text

    custom_baseline(env.workdir, SCORED_MEMO, edit)
    edited = registry_entries()
    edit(edited)
    entries = {entry["index"]: entry for entry in edited}

    def entry_text(index: int) -> str:
        return f"{entries[index]['title']}\n{entries[index]['text']}"

    result = run(score_argv(env.workdir, env.workdir / "scored"), capsys)

    assert result.code == 0, result
    assert seam.contexts_of("Segment 1 revenue was $101 billion in fiscal 2024 .") == {
        f"{entry_text(2)}\n\n{entry_text(1)}"
    }
    assert seam.contexts_of("Segment 3 revenue was $903 billion in fiscal 2024 .") == {
        entry_text(4)
    }
    assert seam.contexts_of("Segment 3 revenue was $103 billion in fiscal 2024 .") == {
        entry_text(4)
    }
    assert seam.contexts_of("Segment 7 revenue was $107 billion in fiscal 2024 .") == {""}
    assert len(entry_text(4)) > 3000


def test_the_actual_output_is_the_claim_sentence_and_only_cited_claims_are_cases(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Faithfulness claim scoring (S6.5): `actual_output` is the claim sentence; a claim with no
    citation is not a case."""
    custom_baseline(env.workdir, SCORED_MEMO)

    result = run(score_argv(env.workdir, env.workdir / "scored"), capsys)

    assert result.code == 0, result
    assert sorted(set(seam.outputs)) == sorted(EXPECTED_SCORES)
    assert len(seam.outputs) == 2 * len(EXPECTED_SCORES)


@pytest.mark.parametrize("failure", ["judge-raises", "judge-reply-unparseable"])
def test_a_claim_whose_metric_fails_is_an_error_with_its_claim_id_never_a_verdict(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    """§14 Faithfulness claim scoring (S6.5) and Faithfulness baseline (S6.5): a claim whose metric
    raises (or whose judge reply has no score) is an error with its claim_id and the error text,
    never a verdict and never dropped; the rates are null, the result is still written, no registry
    record is added and the run exits 1."""
    root = baseline(env.workdir)
    edit_result(
        env.workdir,
        lambda data: data["raw_runs"][1].update(
            recorded_memo=data["raw_runs"][1]["recorded_memo"].replace(
                "Segment 0 revenue", "Segment 0 POISON revenue"
            )
        ),
    )
    [poisoned] = [c for c in population(env.workdir) if "POISON" in c.sentence and c.cited]
    assert poisoned.claim_id.startswith("2-")
    if failure == "judge-raises":
        seam.fail_marker = "POISON"
    else:
        seam.garble_marker = "POISON"
    before = records(env.workdir)
    out_dir = env.workdir / "scored"

    result = run(score_argv(env.workdir, out_dir), capsys)

    assert result.code == 1, result
    [result_file] = result_files(out_dir)
    data = load(result_file)
    assert data["faithful_claim_rate_mean"] is None
    assert data["faithful_claim_rate_stdev"] is None
    scored = data["runs"][1]
    assert scored["faithful_claim_rate"] is None
    [error] = scored["errors"]
    assert error["claim_id"] == poisoned.claim_id
    assert error["error"].strip()
    if failure == "judge-raises":
        assert "judge unavailable" in error["error"]
    assert poisoned.claim_id not in [c["claim_id"] for c in scored["claims"]]
    assert data["runs"][0]["errors"] == [] and data["runs"][2]["errors"] == []
    assert records(env.workdir) == before == [f"{RUN_ID}.json"]
    assert root.exists()


# --- Faithfulness baseline ------------------------------------------------------------------------


def test_score_writes_per_run_rates_the_lineage_result_and_one_candidate_registry_record(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Faithfulness baseline (S6.5): every cited claim of every recorded run is scored; the result
    carries the lineage keys (model the judge model, temperature null for the Foundry tier,
    prompt_version, evidence_release and snapshot_hash from the baseline's registry record,
    fixture_version the first 16 hex of the sha256 of the baseline's result file), the baseline run
    id, effort, and per run (ticker, repeat) the cited-claim count, supported count, faithful_claim_rate
    (supported / cited), per-claim {claim_id, score, supported, reason} and errors; the mean and
    sample stdev of faithful_claim_rate over runs; exactly one registry record (candidate, metrics the
    mean and stdev, result_path naming the result)."""
    baseline(env.workdir)
    raw_runs = load(result_path(env.workdir))["raw_runs"]
    fixture_version = hashlib.sha256(result_path(env.workdir).read_bytes()).hexdigest()[:16]
    # (unsupported, uncited) per run of 20 claims: cited 20 / 18 / 16, supported 20 / 16 / 12.
    assert BASELINE_SHAPES == ((0, 0), (2, 2), (4, 4))
    expected_rates = [20 / 20, 16 / 18, 12 / 16]
    claims_by_run: dict[str, list[Any]] = {}
    for claim in population(env.workdir):
        claims_by_run.setdefault(claim.claim_id.split("-")[0], []).append(claim)
    out_dir = env.workdir / "scored"

    result = run(score_argv(env.workdir, out_dir), capsys)

    assert result.code == 0, result
    assert result.err.strip() == ""
    [result_file] = result_files(out_dir)
    data = load(result_file)
    for key, value in {
        "model": DEPLOYMENT,
        "temperature": None,
        "prompt_version": "p" * 16,
        "evidence_release": RELEASE_ID,
        "snapshot_hash": "a" * 64,
        "fixture_version": fixture_version,
        "baseline_run_id": RUN_ID,
        "effort": "high",
    }.items():
        assert data[key] == value, key
    for key in ("run_id", "git_sha", "branch"):
        assert isinstance(data[key], str) and data[key].strip(), key
    assert len(data["runs"]) == len(raw_runs) == 3
    for number, (scored, raw, rate) in enumerate(
        zip(data["runs"], raw_runs, expected_rates, strict=True), start=1
    ):
        cited = [c for c in claims_by_run[str(number)] if c.cited]
        assert (scored["ticker"], scored["repeat"]) == (raw["ticker"], raw["repeat"])
        assert scored["cited_claims"] == len(cited)
        assert scored["supported_claims"] == sum(c.supported for c in cited)
        assert scored["faithful_claim_rate"] == pytest.approx(rate)
        assert {c["claim_id"]: c["supported"] for c in scored["claims"]} == {
            c.claim_id: c.supported for c in cited
        }
        assert {c["claim_id"]: c["score"] for c in scored["claims"]} == pytest.approx(
            {c.claim_id: 1.0 if c.supported else 0.0 for c in cited}
        )
        assert scored["errors"] == []
    assert data["faithful_claim_rate_mean"] == pytest.approx(statistics.mean(expected_rates))
    assert data["faithful_claim_rate_stdev"] == pytest.approx(statistics.stdev(expected_rates))

    [new_record] = [name for name in records(env.workdir) if name != f"{RUN_ID}.json"]
    record = load(registry_root(env.workdir) / new_record)
    assert record["run_id"] == data["run_id"] == new_record.removesuffix(".json")
    assert record["classification"] == "candidate"
    assert record["metrics"] == {
        "faithful_claim_rate_mean": pytest.approx(statistics.mean(expected_rates)),
        "faithful_claim_rate_stdev": pytest.approx(statistics.stdev(expected_rates)),
    }
    assert Path(record["result_path"]).resolve() == result_file.resolve()
    for key in LINEAGE_FIELDS:
        assert record[key] == data[key], key
    assert seam.configs and all(config.provider == "opus" for config in seam.configs)


@pytest.mark.parametrize(
    ("argument", "baseline_args", "mutate", "words"),
    [
        (None, {}, None, ("baseline",)),
        ("no-such-run", {}, None, ("no-such-run",)),
        (RUN_ID, {"classification": "candidate"}, None, ("approved", "candidate")),
        (RUN_ID, {"evidence_release": "live"}, None, ("live",)),
        (RUN_ID, {"checker": "semantic"}, None, ("token", "semantic", "checker")),
        (
            RUN_ID,
            {},
            lambda data: [run_.pop("recorded_memo") for run_ in data["raw_runs"]],
            ("draft", "memo"),
        ),
    ],
    ids=[
        "missing-baseline-flag",
        "unknown-run-id",
        "candidate-record",
        "live-record-placed-by-hand",
        "result-scored-with-semantic-checker",
        "draftless-result",
    ],
)
def test_score_refuses_a_baseline_that_is_not_a_pinned_approved_token_replay_with_drafts(
    env: Env,
    seam: Seam,
    capsys: pytest.CaptureFixture[str],
    argument: str | None,
    baseline_args: dict[str, Any],
    mutate: Callable[[dict[str, Any]], Any] | None,
    words: tuple[str, ...],
) -> None:
    """§14 Faithfulness baseline (S6.5): refuses, like the regression gate, a missing, unknown,
    candidate, live, non-token or draftless baseline: exit 1, nothing written, before any judge call."""
    write_baseline(env.workdir, recorded_shapes=BASELINE_SHAPES, **baseline_args)
    if mutate:
        edit_result(env.workdir, mutate)
    before = records(env.workdir)
    out_dir = env.workdir / "scored"

    result = run(score_argv(env.workdir, out_dir, baseline_id=argument), capsys)

    assert result.code == 1, result
    assert result_files(out_dir) == []
    assert records(env.workdir) == before
    assert seam.prompts == []
    assert any(word in result.err.lower() for word in words), result.err


# --- Single judge tier ----------------------------------------------------------------------------

BOTH = {"foundry": True, "bedrock": True}
NEITHER = {"foundry": False, "bedrock": False}
INCOMPLETE_FOUNDRY_ONLY = {
    "foundry": True,
    "bedrock": False,
    "blank": ("AZURE_FOUNDRY_JUDGE_MODEL",),
}


@dataclass
class Prepared:
    argv: list[str]
    produced: Path  # the validate file, or the score output directory


def prepare(env: Env, command: str) -> Prepared:
    """A valid baseline and arguments for `command`: only the judge settings can make it refuse."""
    if command == "validate":
        review = labelled_review(env.workdir, owner_flip=((0,), ()))
        out = env.workdir / "validation.json"
        return Prepared(validate_argv(env.workdir, review.path, out), out)
    baseline(env.workdir)
    out_dir = env.workdir / "scored"
    return Prepared(score_argv(env.workdir, out_dir), out_dir)


def produced_files(prepared: Prepared) -> list[Path]:
    if prepared.produced.suffix == ".json":
        return [prepared.produced] if prepared.produced.exists() else []
    return result_files(prepared.produced)


@pytest.mark.parametrize("command", ["validate", "score"])
@pytest.mark.parametrize(
    "tiers",
    [BOTH, NEITHER, INCOMPLETE_FOUNDRY_ONLY],
    ids=["both-tiers", "no-tier", "foundry-without-its-deployment"],
)
def test_the_commands_refuse_unless_exactly_one_judge_tier_is_configured(
    env: Env,
    seam: Seam,
    capsys: pytest.CaptureFixture[str],
    command: str,
    tiers: dict[str, Any],
) -> None:
    """§14 Single judge tier (S6.5): `validate` and `score` exit 1 and write nothing, before any judge
    call, unless exactly one judge tier (Foundry or Bedrock) is configured; the message names the
    settings and no secret value."""
    prepared = prepare(env, command)
    before = records(env.workdir)
    env.configure(**tiers)

    result = run(prepared.argv, capsys)

    assert result.code == 1, result
    assert produced_files(prepared) == []
    assert records(env.workdir) == before
    assert seam.prompts == []
    message = f"{result.out}\n{result.err}"
    assert "AZURE_FOUNDRY_JUDGE_MODEL" in message
    assert "AWS_BEARER_TOKEN_BEDROCK" in message
    assert not any(secret in message for secret in SECRETS)


@pytest.mark.parametrize("command", ["validate", "score"])
@pytest.mark.parametrize(
    ("tiers", "model", "temperature"),
    [
        ({"foundry": True, "bedrock": False}, DEPLOYMENT, None),
        ({"foundry": False, "bedrock": True}, BEDROCK_ID, 0.0),
        (
            {"foundry": True, "bedrock": False, "blank": ("AWS_BEARER_TOKEN_BEDROCK",)},
            DEPLOYMENT,
            None,
        ),
        (
            {"foundry": True, "bedrock": True, "blank": ("AZURE_API_KEY",)},
            BEDROCK_ID,
            0.0,
        ),
    ],
    ids=["foundry-only", "bedrock-only", "foundry-with-blank-bedrock-token", "foundry-without-key"],
)
def test_the_commands_run_on_exactly_one_configured_tier_and_record_its_model_and_temperature(
    env: Env,
    seam: Seam,
    capsys: pytest.CaptureFixture[str],
    command: str,
    tiers: dict[str, Any],
    model: str,
    temperature: float | None,
) -> None:
    """§14 Single judge tier (S6.5) with Faithfulness validation / baseline (S6.5): one configured
    tier is enough; the judge model is the one settings give (Foundry deployment, else the Bedrock
    id) and the temperature is null for the Foundry tier, the configured 0 for Bedrock."""
    prepared = prepare(env, command)
    env.configure(**tiers)

    result = run(prepared.argv, capsys)

    assert result.code == 0, result
    [produced] = produced_files(prepared)
    data = load(produced)
    assert data["temperature"] == temperature
    assert data["judge_model" if command == "validate" else "model"] == model
    assert {config.model for config in seam.configs} == {model}
    assert seam.prompts


# --- Faithfulness validation ----------------------------------------------------------------------


def test_validate_scores_every_claim_and_computes_the_stratum_weighted_kappa_against_the_owner(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Faithfulness validation (S6.5): every claim of the review file is scored; the file holds
    the baseline run id, n, judge model, temperature (null, Foundry), effort, penalize_ambiguous_claims,
    git sha, per claim {claim_id, score, supported, reason}, errors, and the stratum-weighted κ (claim
    weight N/n of its heuristic stratum: 3.6 supported, 1.2 unsupported) and the unweighted κ against
    the owner labels, the weighted 2x2 table, every disagreement by claim_id and `validated`.

    The faked judge decides from the evidence: 4 supported claims get evidence that does not state
    them (the judge says borderline: unsupported under penalize_ambiguous_claims) and 3 unsupported
    ones get evidence quoting them. The owner flips 4 supported and 3 unsupported labels, so the
    judge's claims s0-s3 / u0-u2 and the owner's s2-s5 / u2-u4 overlap on s2, s3 and u2.
    Weighted table (judge rows, owner columns) [[69.6, 9.6], [9.6, 31.2]] of 120: p_o = .84,
    p_e = .66² + .34² = .5512, κ_w = .2888 / .4488; unweighted [[20, 4], [4, 22]] of 50: p_o = .84,
    p_e = .48² + .52² = .5008, κ_u = .3392 / .4992. Below the 0.85 gate: not validated.
    """
    review = labelled_review(
        env.workdir, judge_flip=((0, 1, 2, 3), (0, 1, 2)), owner_flip=((2, 3, 4, 5), (2, 3, 4))
    )
    sup, uns = review.supported_ids, review.unsupported_ids
    out = env.workdir / "validation.json"

    result = run(validate_argv(env.workdir, review.path, out), capsys)

    assert result.code == 0, result
    data = load(out)
    assert set(data) == VALIDATE_FIELDS
    assert data["baseline_run_id"] == RUN_ID
    assert data["n"] == 50
    assert data["judge_model"] == DEPLOYMENT
    assert data["temperature"] is None
    assert data["effort"] == "high"
    assert data["penalize_ambiguous_claims"] is True
    assert isinstance(data["git_sha"], str) and data["git_sha"].strip()
    assert data["errors"] == []
    assert {c["claim_id"]: c["supported"] for c in data["claims"]} == review.judged
    assert {c["claim_id"]: c["score"] for c in data["claims"]} == pytest.approx(
        {i: 1.0 if supported else 0.0 for i, supported in review.judged.items()}
    )
    assert all(isinstance(c["reason"], str) and c["reason"].strip() for c in data["claims"])
    assert [cell for row in data["table"] for cell in row] == pytest.approx([69.6, 9.6, 9.6, 31.2])
    assert data["weighted_kappa"] == pytest.approx(0.2888 / 0.4488, abs=1e-6)
    assert data["unweighted_kappa"] == pytest.approx(0.3392 / 0.4992, abs=1e-6)
    order = sorted(review.judged)
    from evaluation import judge_agreement

    pairs = [(review.judged[i], review.owner[i]) for i in order]
    assert data["weighted_kappa"] == pytest.approx(
        judge_agreement.cohen_kappa(pairs, [review.weights[i] for i in order]), abs=1e-9
    )
    assert sorted(data["disagreements"]) == sorted(
        [sup[0], sup[1], sup[4], sup[5], uns[0], uns[1], uns[3], uns[4]]
    )
    assert data["validated"] is False


def test_validate_passes_a_judge_whose_weighted_kappa_clears_the_gate_at_n_50(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str]
) -> None:
    """§14 Faithfulness validation (S6.5): validated = n >= 50 and weighted κ >= 0.85.

    The judge agrees with the heuristic and the owner flips one supported label: weighted table
    [[86.4, 3.6], [0, 30]], p_o = .97, p_e = .75 * .72 + .25 * .28 = .61, κ_w = .36 / .39 = 12/13 (the
    unweighted κ is .96); the one disagreement is that claim."""
    review = labelled_review(env.workdir, owner_flip=((0,), ()))
    out = env.workdir / "validation.json"

    result = run(validate_argv(env.workdir, review.path, out), capsys)

    assert result.code == 0, result
    data = load(out)
    assert data["n"] == 50
    assert data["weighted_kappa"] == pytest.approx(12 / 13, abs=1e-6)
    assert data["unweighted_kappa"] == pytest.approx(0.96, abs=1e-6)
    assert data["weighted_kappa"] >= KAPPA_GATE
    assert data["disagreements"] == [review.supported_ids[0]]
    assert data["validated"] is True


@pytest.mark.parametrize("failure", ["judge-raises", "judge-reply-unparseable"])
def test_validate_with_an_errored_claim_writes_the_file_without_kappa_and_exits_1(
    env: Env, seam: Seam, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    """§14 Faithfulness validation (S6.5): an errored claim is recorded under `errors` with its
    claim_id and never as a verdict; the file is written with κ and the table null and `validated`
    false; the exit code is 1."""
    review = labelled_review(env.workdir, owner_flip=((0,), ()))
    victim = review.unsupported_ids[0]

    def poison(data: dict[str, Any]) -> None:
        for item in data["claims"]:
            if item["claim_id"] == victim:
                item["claim"] = f"{item['claim']} POISON"

    review.edit(poison)
    if failure == "judge-raises":
        seam.fail_marker = "POISON"
    else:
        seam.garble_marker = "POISON"
    out = env.workdir / "validation.json"

    result = run(validate_argv(env.workdir, review.path, out), capsys)

    assert result.code == 1, result
    data = load(out)
    assert set(data) == VALIDATE_FIELDS
    [error] = data["errors"]
    assert error["claim_id"] == victim and error["error"].strip()
    if failure == "judge-raises":
        assert "judge unavailable" in error["error"]
    assert victim not in [c["claim_id"] for c in data["claims"]]
    assert len(data["claims"]) == 49
    assert data["weighted_kappa"] is None
    assert data["unweighted_kappa"] is None
    assert data["table"] is None
    assert data["validated"] is False


def drop_a_claim(data: dict[str, Any]) -> None:
    data["claims"].pop()


def blank_a_label(data: dict[str, Any]) -> None:
    data["claims"][3]["owner_label"] = None


def misspell_a_label(data: dict[str, Any]) -> None:
    data["claims"][3]["owner_label"] = "partially supported"


@pytest.mark.parametrize(
    "mutate",
    [drop_a_claim, blank_a_label, misspell_a_label],
    ids=["49-claims", "missing-owner-label", "invalid-owner-label"],
)
def test_validate_refuses_a_review_that_is_too_small_or_not_fully_labelled_before_any_judge_call(
    env: Env,
    seam: Seam,
    capsys: pytest.CaptureFixture[str],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """§14 Faithfulness validation (S6.5): a review file with fewer than 50 claims, or a missing or
    invalid `owner_label`, exits 1 and writes nothing, before any judge call."""
    review = labelled_review(env.workdir)
    review.edit(mutate)
    out = env.workdir / "validation.json"

    result = run(validate_argv(env.workdir, review.path, out), capsys)

    assert result.code == 1, result
    assert not out.exists()
    assert seam.prompts == []
    assert result.err.strip()
