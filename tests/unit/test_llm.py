"""Tests for app.services.llm.

Public surface: ``get_llm`` and ``ANALYST_SYSTEM_PROMPT``. The health probe for a local model
server no longer exists (CM-6e); `/health` is covered by tests/unit/test_health_embedding.py.
"""

from langchain_core.runnables import Runnable

from app.config import Settings
from app.services.llm import ANALYST_SYSTEM_PROMPT, get_llm


class TestGetLLM:
    """test-plan §8 Config: "Typed settings load" / provider selection via typed settings."""

    def test_cloud_chain_needs_no_provider_selector(self) -> None:
        assert isinstance(get_llm(Settings(_env_file=None)), Runnable)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]


class TestAnalystPrompt:
    """test-plan §7 RAG/memo: "Citation mapping" and "Missing evidence" (prompt-level contract)."""

    def test_requires_numbered_citations(self) -> None:
        assert "[N]" in ANALYST_SYSTEM_PROMPT or "[1]" in ANALYST_SYSTEM_PROMPT

    def test_forbids_unsupported_facts_and_invented_citations(self) -> None:
        text = ANALYST_SYSTEM_PROMPT.lower()
        assert "never invent citation numbers" in text
        assert "unsupported" in text


class TestNoLocalModelServer:
    """test-plan §1 REST hardening (health depends on no local model server) and §25."""

    def test_the_ollama_health_probe_is_gone(self) -> None:
        import app.services.llm as llm

        assert not hasattr(llm, "check_ollama_health")


def test_system_prompt_requires_citations():
    assert "[N]" in ANALYST_SYSTEM_PROMPT
