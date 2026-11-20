"""eval-replay regression gate (SPEC §11.2, docs/test-plan.md §14).

``python -m evaluation.eval_replay --baseline <run_id>`` re-scores the drafts recorded in a pinned
approved baseline with the current runtime verifier, offline, and fails when a candidate mean rate
falls more than 2σ below the baseline's recorded per-run rates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from app.verification.grounding import evaluate_memo_grounding
from evaluation.registry import REGISTRY_ROOT

METRICS = ("grounded_claim_rate", "citation_coverage_rate")
SIGMA_LIMIT = 2


class RefusalError(Exception):
    """The baseline cannot pin the gate; the message says why."""


def read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RefusalError(f"cannot read {what} {path}: {type(exc).__name__}") from exc
    if not isinstance(data, dict):
        raise RefusalError(f"{what} {path} is not a JSON object")
    return data


def load_baseline(run_id: str, root: Path) -> list[dict[str, Any]]:
    record_path = root / f"{run_id}.json"
    if record_path.parent != root or not record_path.is_file():
        raise RefusalError(f"baseline {run_id!r} names no registry record under {root}")
    record = read_json(record_path, "registry record")
    if record.get("classification") != "approved":
        raise RefusalError(
            f"baseline {run_id} is {record.get('classification')!r}; "
            "only an approved run can pin the gate, not a candidate"
        )
    if record.get("evidence_release") == "live":
        raise RefusalError(
            f"baseline {run_id} was recorded on live evidence, which cannot be pinned"
        )
    if not record.get("result_path"):
        raise RefusalError(f"baseline {run_id} has no result file to re-score")
    result = read_json(Path(record["result_path"]), "result file")
    if result.get("grounding_checker") != "token":
        raise RefusalError(
            f"baseline {run_id} was scored with the {result.get('grounding_checker')!r} checker; "
            "the gate re-scores with the token checker"
        )
    try:
        runs: list[dict[str, Any]] = [
            {key: run[key] for key in (*METRICS, "ticker", "recorded_memo", "recorded_registry")}
            for run in result["raw_runs"]
        ]
    except (KeyError, TypeError) as exc:
        raise RefusalError(f"result file of baseline {run_id} holds no recorded drafts") from exc
    if len(runs) < 2:
        raise RefusalError(f"baseline {run_id} has fewer than two runs, so it has no deviation")
    return runs


def _gate(run_id: str, root: Path) -> bool:
    runs = load_baseline(run_id, root)
    verdicts = [evaluate_memo_grounding(r["recorded_memo"], r["recorded_registry"]) for r in runs]
    print(f"baseline {run_id}: {len(runs)} runs re-scored offline")
    failed = []
    for metric in METRICS:
        recorded = [float(r[metric]) for r in runs]
        rescored = [float(getattr(v, metric)) for v in verdicts]
        mean, sigma = statistics.mean(recorded), statistics.stdev(recorded)
        threshold = mean - SIGMA_LIMIT * sigma
        candidate = statistics.mean(rescored)
        passed = candidate >= threshold
        if not passed:
            failed.append(metric)
        print(
            f"{metric}: baseline mean {mean:.4f} sigma {sigma:.4f} threshold {threshold:.4f} "
            f"candidate mean {candidate:.4f} {'PASS' if passed else 'FAIL'}"
        )
        for number, (was, now) in enumerate(zip(recorded, rescored, strict=True), start=1):
            print(f"  run {number}: recorded {was:.4f} candidate {now:.4f} delta {now - was:+.4f}")
    if failed:
        print(f"FAIL: {', '.join(failed)} more than {SIGMA_LIMIT} sigma below the baseline")
    return not failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-score a pinned baseline's recorded drafts")
    parser.add_argument("--baseline", help="run_id of an approved registry record")
    parser.add_argument("--registry-root", type=Path, default=REGISTRY_ROOT)
    args = parser.parse_args(argv)
    if not args.baseline:
        print("--baseline <run_id> is required", file=sys.stderr)
        return 1
    try:
        return 0 if _gate(args.baseline, args.registry_root) else 1
    except RefusalError as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
