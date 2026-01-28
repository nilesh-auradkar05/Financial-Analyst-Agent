"""
This module provides integration with ollama for local LLM inference.

I am using Qwen3-VL model which can:
    - Process text prompts
    - Analyze images (in this context it is useful to process charts, tables, diagrams from SEC filings).

Usage:
------------------------
    from models.llm import get_llm, analyze_image, generate_text

    # langchain-compatible llm
    llm = get_llm()
    response = llm.invoke("Explain P/E ration")

    # direct text generation
    response = await generate_text("Summarize these risk factors: .......")

    # Vision analysis (for chart/tables)
    analysis = await analyze_image(
        image_path = "chart.png",
        prompt="Describe the revenue trend shown in this chart."
    )

"""

import asyncio
import sys
from pathlib import Path

import httpx
from langchain_ollama import ChatOllama

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings

# =============================================================================
# LLM ACCESS
# =============================================================================


def get_llm(temperature: float = 0.0) -> ChatOllama:
    """
    Get configured ChatOllama instance.

    Args:
        temperature: Override default temperature

    Returns:
        ChatOllama ready for use

    Example:
        llm = get_llm()
        response = await llm.ainvoke([
            {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": "Analyze AAPL..."},
        ])
        print(response.content)
    """
    return ChatOllama(
        model=settings.ollama.llm_model,
        base_url=settings.ollama.base_url,
        temperature=temperature or settings.ollama.temperature,
        # num_ctx=8192,  # Context window size
        # num_predict=2048,  # Max tokens to generate
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================


async def check_ollama_health() -> bool:
    """Check if Ollama server is running and model is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ollama.base_url}/api/tags",
                timeout=5.0,
            )

            if response.status_code != 200:
                return False

            # Check if our model is available
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            return any(settings.ollama.llm_model in m for m in models)

    except Exception:
        return False


# =============================================================================
# PROMPTS
# =============================================================================


ANALYST_SYSTEM_PROMPT = """You are a senior financial analyst at a top-tier investment firm.
Your role is to provide comprehensive, data-driven analysis of companies.

Guidelines:
- Be specific and use actual numbers from the provided data
- Cite sources using [N] notation
- Present balanced analysis including both opportunities and risks
- Use professional, clear language
- Structure your analysis with clear sections"""


MEMO_TEMPLATE = """Based on the following research data, write a comprehensive investment memo for {company_name} ({ticker}).

{context}

Structure your memo with these sections:
1. Executive Summary (2-3 key takeaways)
2. Company Overview
3. Recent Developments
4. Financial Highlights
5. Risk Factors
6. Investment Thesis
7. Conclusion

Use [N] citations when referencing specific data points."""


# =============================================================================
# CLI TEST
# =============================================================================


async def _main():
    """Test LLM connection."""
    print("Checking Ollama health...")

    healthy = await check_ollama_health()

    if healthy:
        print(f"Ollama is running with model: {settings.ollama.llm_model}")

        print("\nTesting generation...")
        llm = get_llm()
        response = await llm.ainvoke("Say 'Hello, Alpha Analyst!' in exactly those words.")
        print(f"Response: {response.content}")
    else:
        print(" Ollama not available")
        print(f"   Expected model: {settings.ollama.llm_model}")
        print(f"   URL: {settings.ollama.base_url}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
