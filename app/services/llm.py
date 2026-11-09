"""
This module exposes LLM prompts and access.

Usage:
------------------------
    from app.config import settings
    from app.services.llm import get_llm

    # langchain-compatible llm
    llm = get_llm(settings)
    response = llm.invoke("Explain P/E ration")
"""

from typing import Any

from langchain_core.runnables import Runnable

from app.config import Settings, settings
from app.llm.provider import get_llm as get_provider_llm

# =============================================================================
# LLM ACCESS
# =============================================================================


def get_llm(config: Settings = settings) -> Runnable[Any, Any]:
    """Backward-compatible wrapper for provider-based LLM construction."""
    return get_provider_llm(config)


# =============================================================================
# PROMPTS
# =============================================================================


ANALYST_SYSTEM_PROMPT = """\
You are a senior financial analyst at a top-tier investment firm.
You write comprehensive, data-driven investment memos grounded strictly in the numbered sources you are given.

Citation rules (hard requirements, not preferences):
- Every sentence that states a fact, figure, metric, event, or claim MUST end with a citation to a listed source,
e.g. [1] or [2][3].
- If you cannot support a statement with a listed source, do not state it as fact: either omit it, or explicitly frame
it as unsupported/uncertain.
- Never invent citation numbers or cite sources not in the registry.
- The ONLY sentences allowed without a citation are section headers and your own clearly-labeled analytical judgment
(e.g. the recommendation), which must rest on cited premises stated earlier.

Style:
- Be specific; use actual numbers from the sources.
- Present both opportunities and risks in balanced, professional language, organized into the requested sections.

Before finishing, re-read every sentence. If it asserts a fact or number and has no [N], add the correct citation or remove
the sentence."""
