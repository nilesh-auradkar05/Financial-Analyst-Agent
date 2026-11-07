from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
RETRIEVAL = APP / "components" / "retrieval"
EMBEDDING_CLIENT_MODULES = {"langchain_ollama", "ollama"}
LANGCHAIN_EMBEDDING_NAMES = {
    "langchain_aws": {"BedrockEmbeddings"},
    "langchain_openai": {"OpenAIEmbeddings", "AzureOpenAIEmbeddings"},
}


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_app_never_imports_evaluation() -> None:
    """test-plan §11 Layering: `app/` has no import from `evaluation/`."""
    offenders = [
        f"{path.relative_to(APP.parent)}: {name}"
        for path in sorted(APP.rglob("*.py"))
        for name in _imported_modules(path)
        if name == "evaluation" or name.startswith("evaluation.")
    ]
    assert offenders == []


def _embedding_client_uses(path: Path) -> list[str]:
    """Imports and references that build an embedding provider client, lazy imports included."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] in EMBEDDING_CLIENT_MODULES
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in EMBEDDING_CLIENT_MODULES:
                found.append(node.module)
            found.extend(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name in LANGCHAIN_EMBEDDING_NAMES.get(node.module, set())
            )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.attr in LANGCHAIN_EMBEDDING_NAMES.get(node.value.id, set())
        ):
            found.append(f"{node.value.id}.{node.attr}")
        elif isinstance(node, ast.Constant) and node.value == "bedrock-runtime":
            found.append("bedrock-runtime client")
    return found


def test_no_embedding_provider_client_is_built_outside_the_retrieval_boundary() -> None:
    """test-plan §21 Provider client stays inside the boundary: no module outside
    `app/components/retrieval/` (in `app/`, `evaluation/` or `dataops/`) imports an Ollama,
    Bedrock or OpenAI embedding client or opens a bedrock-runtime client."""
    offenders = [
        f"{path.relative_to(ROOT)}: {use}"
        for package in ("app", "evaluation", "dataops")
        for path in sorted((ROOT / package).rglob("*.py"))
        if RETRIEVAL not in path.parents
        for use in _embedding_client_uses(path)
    ]
    assert offenders == []
