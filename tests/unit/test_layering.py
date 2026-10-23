from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"


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
