"""Compose provider settings: docs/test-plan.md §16 (dummy credentials only)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("values", [
    {"AWS_BEARER_TOKEN_BEDROCK": "dummy-bedrock", "AWS_REGION": "us-east-1",
     "AWS_BEDROCK_MODEL": "bedrock-model", "CLAUDE_LLM_MODEL": "claude-a", "OPENAI_LLM_MODEL": "gpt-b",
     "ANTHROPIC_API_KEY": "dummy-anthropic", "OPENAI_API_KEY": "dummy-openai",
     "AZURE_API_KEY": "dummy-azure", "AZURE_FOUNDRY_MODEL": "dummy-deployment",
     "AZURE_PROJECT_ENDPOINT": "https://example.invalid/project",
     "AZURE_OPENAI_ENDPOINT": "https://example.invalid/openai"},
    {"AWS_BEARER_TOKEN": "dummy-alias"},
    {},
])
def test_compose_preserves_provider_settings(tmp_path, values):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose CLI required (no daemon needed)")
    env_file = tmp_path / "dummy.env"
    env_file.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    process_env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("AWS_", "AZURE_", "ANTHROPIC_", "OPENAI_", "CLAUDE_", "LLM_", "OLLAMA_", "COMPOSE_"))}
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["docker", "compose", "--project-directory", str(root), "--env-file", str(env_file),
         "-f", str(root / "docker-compose.yml"), "config", "--format", "json"],
        env=process_env, capture_output=True, text=True, check=True,
    )
    rendered = json.loads(result.stdout)["services"]["api"]["environment"]
    for key, value in values.items():
        assert rendered[key] == value
    for key in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_BEDROCK_MODEL", "CLAUDE_LLM_MODEL", "OPENAI_LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if key not in values:
            assert rendered.get(key) is None
