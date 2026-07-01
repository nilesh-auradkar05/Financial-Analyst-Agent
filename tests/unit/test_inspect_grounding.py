from __future__ import annotations

import importlib


def test_inspect_grounding_imports_without_running_agent() -> None:
    module = importlib.import_module("evaluation.inspect_grounding")

    assert callable(module.main)
