"""
This module exposes LLM prompts and health checks.

Usage:
------------------------
    from app.config import settings
    from app.services.llm import get_llm

    # langchain-compatible llm
    llm = get_llm(settings)
    response = llm.invoke("Explain P/E ration")
"""

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from app.config import Settings, settings
from app.llm.provider import get_llm as get_provider_llm

# =============================================================================
# LLM ACCESS
# =============================================================================


def get_llm(config: Settings = settings) -> BaseChatModel:
    """Backward-compatible wrapper for provider-based LLM construction."""
    return get_provider_llm(config)


# =============================================================================
# HEALTH CHECK
# =============================================================================


async def check_ollama_health(*, log_failure: bool = False) -> bool:
    """Check if Ollama server is running and model is available."""
    tags_url = f"{settings.ollama.base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                tags_url,
                timeout=5.0,
            )

            if response.status_code != 200:
                if log_failure:
                    logger.warning(
                        f"Ollama health check failed at {tags_url}: HTTP {response.status_code}"
                    )
                return False

            # Check if our model is available
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            model_available = any(settings.ollama.llm_model in m for m in models)
            if not model_available and log_failure:
                logger.warning(
                    f"Ollama model {settings.ollama.llm_model!r} not found. "
                    f"Available models: {models}"
                )
            return model_available

    except Exception as exc:
        if log_failure:
            logger.warning(f"Ollama health check failed at {tags_url}: {exc}")
        return False


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
