"""DeepEval faithfulness answer-quality metric (SPEC §4 judge ladder, §11.2; docs/test-plan.md §14, S6.5).

``python -m evaluation.answer_quality validate`` scores the claims of a labelled S6.3 review file and
compares the verdicts with the owner labels (stratum-weighted Cohen's κ); ``score`` scores every cited
claim of a pinned approved baseline and records a candidate registry entry. A claim is supported only
when its faithfulness score is exactly 1.0 (ambiguous statements count against it).
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.config import LLMSettings
from dataops.git_state import git_state
from evaluation import judge_models_interface
from evaluation.eval_replay import RefusalError, load_baseline, read_json
from evaluation.judge_agreement import (
    KAPPA_GATE,
    LABELS,
    MIN_CLAIMS,
    UndefinedKappaError,
    _cited_claims,
    _table,
    _write_json,
    cohen_kappa,
)
from evaluation.registry import REGISTRY_ROOT, new_run_id, write_run
from evaluation.run_rag_quality_eval import _run_with_retries

EFFORT = "high"
OUT_DIR = Path("evaluation/answer_quality_res")
TIER_SETTINGS = (
    "exactly one judge tier must be configured: Foundry (AZURE_OPENAI_ENDPOINT, AZURE_API_KEY, "
    "AZURE_FOUNDRY_JUDGE_MODEL) or Bedrock (AWS_BEARER_TOKEN_BEDROCK, AWS_REGION)"
)


def _judge() -> tuple[Any, judge_models_interface.JudgeModelConfig, str | None]:
    """The DeepEval judge, its config and its effort, once exactly one judge tier is configured."""
    llm = LLMSettings(_env_file=None)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    foundry = llm.azure_deployment(llm.azure_foundry_judge_model) is not None
    token = llm.aws_bearer_token_bedrock
    bedrock = bool(token and token.get_secret_value().strip() and llm.aws_region)
    if foundry == bedrock:
        raise RefusalError(TIER_SETTINGS)
    config = judge_models_interface.judge_config_from_env(
        provider="opus", model=None, temperature=0.0, top_p=None, max_tokens=None
    )
    return judge_models_interface.create_deepeval_judge_model(config), config, EFFORT if foundry else None


def _score_claim(
    judge: Any, item: dict[str, Any], args: argparse.Namespace
) -> tuple[float, str | None]:
    metric = FaithfulnessMetric(
        penalize_ambiguous_claims=True, include_reason=True, model=judge, async_mode=False
    )
    case = LLMTestCase(
        input=f"Write an investment memo for {item['ticker']}.",
        actual_output=item["claim"],
        retrieval_context=[
            f"{entry['title']}\n{entry['text']}"
            for entry in item["evidence"]
            if not entry.get("missing")
        ],
    )
    _run_with_retries(
        lambda: metric.measure(case, _show_indicator=False),
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    if metric.score is None:
        raise ValueError("the metric returned no score")
    return metric.score, metric.reason


def _score_claims(
    judge: Any, items: Sequence[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in items:
        try:
            score, reason = _score_claim(judge, item, args)
        except Exception as exc:  # noqa: BLE001 - a failed claim is recorded as an error, never dropped
            errors.append({"claim_id": item["claim_id"], "error": f"{type(exc).__name__}: {exc}"})
        else:
            scored.append(
                {
                    "claim_id": item["claim_id"],
                    "score": score,
                    "supported": score == 1.0,
                    "reason": reason,
                }
            )
    return scored, errors


def _validate(args: argparse.Namespace) -> int:
    review = read_json(args.review, "review file")
    try:
        items = review["claims"]
        order = [item["claim_id"] for item in items]
        owner = {item["claim_id"]: item.get("owner_label") for item in items}
        strata = review["strata"]
        run_id = review["baseline_run_id"]
    except (KeyError, TypeError) as exc:
        raise RefusalError(f"review file is malformed: missing {exc}") from exc
    if len(order) < MIN_CLAIMS:
        raise RefusalError(f"{len(order)} claims reviewed, fewer than the required {MIN_CLAIMS}")
    if bad := [i for i in order if owner[i] not in LABELS]:
        raise RefusalError(f"missing or invalid owner_label for {len(bad)} claims, first {bad[0]}")
    claims = _cited_claims(load_baseline(run_id, args.registry_root))
    if unknown := [i for i in order if i not in claims]:
        raise RefusalError(f"claim {unknown[0]} is not a cited claim of baseline {run_id}")
    judge, config, effort = _judge()

    scored, errors = _score_claims(judge, items, args)
    report: dict[str, Any] = {
        "baseline_run_id": run_id,
        "n": len(order),
        "judge_model": config.model,
        "temperature": config.temperature,
        "effort": effort,
        "penalize_ambiguous_claims": True,
        "git_sha": git_state()["commit"],
        "claims": scored,
        "errors": errors,
        "weighted_kappa": None,
        "unweighted_kappa": None,
        "table": None,
        "disagreements": [],
        "validated": False,
    }
    if not errors:
        judged = {c["claim_id"]: c["supported"] for c in scored}
        pairs = [(judged[i], owner[i] == "supported") for i in order]
        weights = [
            strata[LABELS[not claims[i][1]]]["N"] / strata[LABELS[not claims[i][1]]]["n"]
            for i in order
        ]
        try:
            report["weighted_kappa"] = cohen_kappa(pairs, weights)
            report["unweighted_kappa"] = cohen_kappa(pairs)
        except UndefinedKappaError as exc:
            raise RefusalError(f"κ is undefined: {exc}") from exc
        report["table"] = _table(pairs, weights)
        report["disagreements"] = [i for i, (j, o) in zip(order, pairs, strict=True) if j != o]
        report["validated"] = report["weighted_kappa"] >= KAPPA_GATE
    _write_json(args.out, report)
    if errors:
        print(f"{len(errors)} of {len(order)} claims errored: see {args.out}", file=sys.stderr)
    return 1 if errors else 0


def _score(args: argparse.Namespace) -> int:
    if not args.baseline:
        raise RefusalError("--baseline <run_id> is required")
    runs = load_baseline(args.baseline, args.registry_root)
    record = read_json(args.registry_root / f"{args.baseline}.json", "registry record")
    result_file = Path(record["result_path"])
    judge, config, effort = _judge()

    cited = _cited_claims(runs)
    git = git_state()
    scored_runs: list[dict[str, Any]] = []
    for number, run in enumerate(runs, start=1):
        items = [item for claim_id, (item, _) in cited.items() if claim_id.startswith(f"{number}-")]
        scored, errors = _score_claims(judge, items, args)
        supported = sum(claim["supported"] for claim in scored)
        scored_runs.append(
            {
                "ticker": run["ticker"],
                "repeat": run["repeat"],
                "cited_claims": len(items),
                "supported_claims": supported,
                "faithful_claim_rate": supported / len(items) if items and not errors else None,
                "claims": scored,
                "errors": errors,
            }
        )
    rates = [run["faithful_claim_rate"] for run in scored_runs]
    complete = None not in rates
    lineage = {
        "run_id": new_run_id(git["commit"]),
        "git_sha": git["commit"],
        "branch": git["branch"],
        "model": config.model,
        "temperature": config.temperature,
        "prompt_version": record["prompt_version"],
        "evidence_release": record["evidence_release"],
        "snapshot_hash": record["snapshot_hash"],
        "fixture_version": hashlib.sha256(result_file.read_bytes()).hexdigest()[:16],
    }
    metrics = {
        "faithful_claim_rate_mean": statistics.mean(rates) if complete else None,
        "faithful_claim_rate_stdev": statistics.stdev(rates) if complete else None,
    }
    out_path = args.out_dir / f"{lineage['run_id']}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_path,
        {
            **lineage,
            "baseline_run_id": args.baseline,
            "effort": effort,
            "penalize_ambiguous_claims": True,
            "runs": scored_runs,
            **metrics,
        },
    )
    if not complete:
        print(f"runs without a faithful_claim_rate (errored or no cited claims): see {out_path}", file=sys.stderr)
        return 1
    write_run(
        {**lineage, "metrics": metrics, "classification": "candidate", "result_path": str(out_path)},
        root=args.registry_root,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepEval faithfulness answer quality (S6.5)")
    commands = parser.add_subparsers(dest="command", required=True)
    retries = argparse.ArgumentParser(add_help=False)
    retries.add_argument("--registry-root", type=Path, default=REGISTRY_ROOT)
    retries.add_argument("--max-retries", type=int, default=3)
    retries.add_argument("--retry-backoff-seconds", type=float, default=30.0)
    validate = commands.add_parser(
        "validate", parents=[retries], help="score a labelled review file and compare with the owner"
    )
    validate.add_argument("--review", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True)
    score = commands.add_parser(
        "score", parents=[retries], help="score every cited claim of a pinned baseline"
    )
    score.add_argument("--baseline", help="run_id of an approved registry record")
    score.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    try:
        return {"validate": _validate, "score": _score}[args.command](args)
    except RefusalError as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
