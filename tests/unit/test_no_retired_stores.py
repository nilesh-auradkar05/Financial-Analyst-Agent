"""Trace: docs/test-plan.md §5 "No retired stores (CM-7)" (ADR-0016 decision 1).

OpenSearch is the only vector store. The lockfile holds no Qdrant or Chroma package, importing the
application, retrieval and evaluation modules loads neither client, and the Compose stack defines
no `qdrant` service and no `QDRANT_*` or `CHROMA_*` variable. The `api` service reaches the
`opensearch` service of the same file through its OPENSEARCH_URL. Files are read as data; the
imports run in a fresh interpreter so nothing this test session loaded can mask a leak.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

from evaluation.run_shared_retrieval_benchmark import main

ROOT = Path(__file__).resolve().parents[2]
RETIRED_PACKAGES = {"qdrant-client", "chromadb"}
RETIRED_MODULES = {"qdrant_client", "chromadb"}
RETIRED_VARIABLE_PREFIXES = ("QDRANT_", "CHROMA_")
IMPORTED_MODULES = (
    "app.main",
    "app.components.retrieval.vector_store",
    "app.components.retrieval.opensearch_store",
    "app.components.retrieval.hybrid_retrieve",
    "app.components.retrieval.reranked_hybrid_retrieve",
    "evaluation.run_shared_retrieval_benchmark",
    "evaluation.build_embedding_arm",
    "evaluation.retrieval_main",
)


def _compose() -> dict[str, Any]:
    loaded = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _environment(service: dict[str, Any]) -> dict[str, str | None]:
    """A service's `environment` as a mapping, whether Compose wrote it as a map or a list."""
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return {str(name): None if value is None else str(value) for name, value in environment.items()}
    pairs = (entry.partition("=") for entry in environment)
    return {name: value if separator else None for name, separator, value in pairs}


def _default_of(value: str) -> str:
    """The value Compose uses when the variable is unset: `${NAME:-default}` -> `default`."""
    return re.sub(r"\$\{[^}:]+:?-([^}]*)\}", r"\1", value)


def test_lockfile_holds_no_retired_store_package() -> None:
    """test-plan §5 No retired stores: `uv.lock` holds no `qdrant-client` or `chromadb` package."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {package["name"] for package in lock["package"]}

    assert names & RETIRED_PACKAGES == set()


def test_importing_app_retrieval_and_evaluation_loads_no_retired_client() -> None:
    """test-plan §5 No retired stores: importing the app, retrieval and evaluation modules loads
    neither `qdrant_client` nor `chromadb`."""
    code = (
        "import importlib, json, sys\n"
        f"for name in {list(IMPORTED_MODULES)!r}:\n"
        "    importlib.import_module(name)\n"
        f"banned = {sorted(RETIRED_MODULES)!r}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in banned)))\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def test_compose_defines_no_retired_store_service_or_variable() -> None:
    """test-plan §5 No retired stores: no `qdrant` service, no `QDRANT_*` or `CHROMA_*` variable,
    and the default backend is not a retired one."""
    services = _compose()["services"]

    retired_variables = sorted(
        f"{name}.{variable}"
        for name, service in services.items()
        for variable in _environment(service)
        if variable.startswith(RETIRED_VARIABLE_PREFIXES)
    )
    backends = {
        name: _default_of(value)
        for name, service in services.items()
        if (value := _environment(service).get("VECTOR_BACKEND"))
    }

    assert "qdrant" not in services
    assert retired_variables == []
    assert set(backends.values()) <= {"opensearch"}


def test_compose_api_service_reaches_an_opensearch_service_in_the_same_file() -> None:
    """test-plan §5 No retired stores: the `api` service reaches an `opensearch` service in the
    same file. Rule: the file defines an `opensearch` service and the host of the api service's
    OPENSEARCH_URL (its default when the variable is unset) is `opensearch`."""
    services = _compose()["services"]

    api_url = _environment(services["api"]).get("OPENSEARCH_URL")

    assert "opensearch" in services
    assert api_url is not None
    assert urlparse(_default_of(api_url)).hostname == "opensearch"


def test_no_compose_file_defines_a_qdrant_service() -> None:
    """test-plan §5 No retired stores: no other Compose file in the repo root defines a `qdrant`
    service."""
    with_qdrant = sorted(
        path.name
        for path in ROOT.glob("docker-compose*.yml")
        if "qdrant" in ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("services") or {})
    )

    assert with_qdrant == []

@pytest.mark.parametrize("backend", ["qdrant", "chroma"])
def test_benchmark_runner_rejects_a_retired_backend(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    backend: str,
) -> None:
    """test-plan §5 No retired stores (CM-7 sprint row): `--backend` accepts only `opensearch`;
    a retired name is a usage error before the runner reads a fixture or opens a store."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_shared_retrieval_benchmark",
            "--backend", backend,
            "--output-dir", str(tmp_path),
            "--dry-run",
        ],
    )  # fmt: skip

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 2
    assert "opensearch" in capsys.readouterr().err
