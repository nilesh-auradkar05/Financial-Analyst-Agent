"""Run the shared retrieval benchmark across configured retrieval methods."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation.validate_retrieval_fixture import load_fixture

BackendName = Literal["chroma", "qdrant"]
RetrievalMode = Literal["query", "sections", "section_aware"]

DEFAULT_FIXTURE = Path("evaluation/fixtures/retrieval_shared_benchmark_v1.json")
DEFAULT_OUTPUT_DIR = Path("evaluation/results/shared_benchmark")


class MethodSpec(BaseModel):
    """A retrieval method run configuration."""

    model_config = ConfigDict(frozen=True)

    name: str
    backend: BackendName
    mode: RetrievalMode
    env: dict[str, str] = Field(default_factory=dict)


DEFAULT_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(name="qdrant_query", backend="qdrant", mode="query"),
    MethodSpec(name="qdrant_section_aware", backend="qdrant", mode="section_aware"),
    MethodSpec(name="chroma_query", backend="chroma", mode="query"),
)


def _selected_methods(names: list[str] | None) -> list[MethodSpec]:
    methods_by_name = {method.name: method for method in DEFAULT_METHODS}
    if not names:
        return list(DEFAULT_METHODS)

    unknown = sorted(name for name in names if name not in methods_by_name)
    if unknown:
        valid = ", ".join(sorted(methods_by_name))
        raise SystemExit(f"Unknown method(s): {unknown}. Valid methods: {valid}")

    return [methods_by_name[name] for name in names]


def _load_case_payloads(fixture_path: Path) -> list[dict[str, object]]:
    # Validate first. Then use the original JSON shape so evaluation.retrieval_main
    # stays the single source of execution behavior.
    load_fixture(fixture_path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("validated fixture unexpectedly did not load as a list")
    return payload


def _write_mode_fixture(
    cases: list[dict[str, object]],
    *,
    mode: RetrievalMode,
    directory: Path,
) -> Path:
    mode_cases = [{**case, "mode": mode} for case in cases]
    output_path = directory / f"fixture_{mode}.json"
    output_path.write_text(json.dumps(mode_cases, indent=2) + "\n", encoding="utf-8")
    return output_path


def _run_method(
    method: MethodSpec,
    *,
    fixture_path: Path,
    output_path: Path,
    continue_on_error: bool,
) -> int:
    env = os.environ.copy()
    env["VECTOR_BACKEND"] = method.backend
    env.update(method.env)

    command = [
        sys.executable,
        "-m",
        "evaluation.retrieval_main",
        str(fixture_path),
        "--output",
        str(output_path),
    ]
    print(f"\n==> Running {method.name}")
    print(f"    backend={method.backend} mode={method.mode} output={output_path}")

    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0 and not continue_on_error:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared retrieval benchmark matrix.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method name to run. May be passed multiple times. Defaults to all configured methods.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = _selected_methods(args.methods)
    cases = _load_case_payloads(args.fixture)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "fixture": str(args.fixture),
        "case_count": len(cases),
        "methods": [method.model_dump() for method in methods],
        "outputs": {},
    }

    with tempfile.TemporaryDirectory(prefix="retrieval_shared_fixture_") as tmp:
        tmp_dir = Path(tmp)
        for method in methods:
            method_fixture = _write_mode_fixture(cases, mode=method.mode, directory=tmp_dir)
            output_path = run_dir / f"{method.name}.json"
            manifest["outputs"][method.name] = str(output_path)  # type: ignore[index]

            if args.dry_run:
                print(f"DRY RUN: would run {method.name} with fixture {method_fixture}")
                continue

            return_code = _run_method(
                method,
                fixture_path=method_fixture,
                output_path=output_path,
                continue_on_error=args.continue_on_error,
            )
            if return_code != 0:
                manifest.setdefault("failed_methods", {})  # type: ignore[call-overload]
                manifest["failed_methods"][method.name] = return_code  # type: ignore[index]

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark manifest: {manifest_path}")


if __name__ == "__main__":
    main()
