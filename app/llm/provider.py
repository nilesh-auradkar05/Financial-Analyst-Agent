"""Runtime chat-model provider factory.

Converse-API families (provider == "bedrock"):
  - Anthropic Claude  -> additionalModelRequestFields={"thinking": {...}}  (enabled/adaptive)
  - DeepSeek V3.2     -> additionalModelRequestFields={"thinking": {"type": "enabled"}}  (hybrid thinking)
  - OpenAI gpt-oss    -> reasoning effort low|medium|high
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from app.config import Settings

# Confirm exact IDs/regions with `aws bedrock list-foundation-models` / list-inference-profiles.
MODEL_PRESETS: dict[str, str] = {
    "claude-sonnet": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "deepseek-v3": "deepseek.v3.2",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    # frontier (provider must be "bedrock_openai"):
    "gpt-5.5": "openai.gpt-5.5",
    "gpt-5.4": "openai.gpt-5.4",
}


def _model_family(model_id: str) -> str:
    mid = model_id.lower()
    if "anthropic.claude" in mid:
        return "claude"
    if "openai.gpt-oss" in mid:
        return "openai_oss"
    if "deepseek" in mid:
        return "deepseek"
    return "other"


def get_llm(settings: Settings) -> BaseChatModel:
    """Return a configured chat model using the active provider settings."""
    provider = settings.llm.provider.lower()
    model_id = MODEL_PRESETS.get(settings.llm.model, settings.llm.model)
    mode = settings.llm.thinking_mode.lower()

    # --- OpenAI frontier (gpt-5.x) via Responses API on the mantle endpoint -----------------
    if provider == "bedrock_openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("bedrock_openai provider requires `langchain-openai`.") from exc

        token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if not token:
            raise ValueError(
                "OpenAI frontier models need a Bedrock API key. "
                "Generate one and set AWS_BEARER_TOKEN_BEDROCK."
            )
        region = settings.llm.aws_region or "us-east-2"
        base_url = f"https://bedrock-mantle.{region}.api.aws/openai/v1"

        kwargs: dict = {
            "model": model_id,
            "base_url": base_url,
            "api_key": token,
            "use_responses_api": True,
            "max_tokens": settings.llm.max_tokens,
        }
        if mode != "off":
            kwargs["reasoning_effort"] = settings.llm.thinking_effort
        logger.info(
            "Bedrock-OpenAI (Responses) model='{}' region={} effort={} | VERIFY interop",
            model_id,
            region,
            settings.llm.thinking_effort if mode != "off" else "off",
        )
        return ChatOpenAI(**kwargs)

    # --- Converse-API families: Claude / DeepSeek / gpt-oss ---------------------------------
    if provider == "bedrock":
        if not settings.llm.aws_region:
            raise ValueError(
                "LLM provider is bedrock but AWS region is missing. Set AWS_REGION."
            )
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Bedrock provider requires `langchain-aws` and `awscrt`."
            ) from exc

        family = _model_family(model_id)
        kwargs = {
            "model": model_id,
            "max_tokens": settings.llm.max_tokens,
            "region_name": settings.llm.aws_region,
        }

        if family == "claude" and mode in {"enabled", "adaptive"}:
            if mode == "enabled":
                if settings.llm.thinking_budget_tokens < 1024:
                    raise ValueError("thinking_budget_tokens must be >= 1024")
                if settings.llm.thinking_budget_tokens >= settings.llm.max_tokens:
                    raise ValueError(
                        "thinking_budget_tokens must be < max_tokens "
                        f"({settings.llm.thinking_budget_tokens} >= {settings.llm.max_tokens})"
                    )
                kwargs["additional_model_request_fields"] = {
                    "thinking": {"type": "enabled", "budget_tokens": settings.llm.thinking_budget_tokens}
                }
            else:  # adaptive
                kwargs["additional_model_request_fields"] = {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": settings.llm.thinking_effort},
                }
            reasoning_desc = f"claude/thinking={mode}"  # temperature omitted (model default)

        elif family == "deepseek" and mode != "off":
            # DeepSeek V3.2 hybrid thinking. Native enable field is {"thinking": {"type": "enabled"}}.
            # Thinking mode ignores temperature; reasoning returns as reasoning_content.
            # VERIFY: confirm the response carries a reasoning block (see smoke check).
            kwargs["additional_model_request_fields"] = {"thinking": {"type": "enabled"}}
            reasoning_desc = "deepseek/thinking=enabled (verify field)"

        elif family == "openai_oss" and mode != "off":
            # gpt-oss adjustable reasoning effort (low|medium|high). VERIFY exact Converse field.
            kwargs["additional_model_request_fields"] = {"reasoning_effort": settings.llm.thinking_effort}
            kwargs["temperature"] = settings.llm.temperature
            reasoning_desc = f"gpt-oss/effort={settings.llm.thinking_effort} (verify field)"

        else:
            if mode != "off" and family == "other":
                logger.warning(
                    "thinking_mode={} requested but model family '{}' has no known reasoning "
                    "config here; ignoring it (no reasoning override applied).",
                    mode,
                    family,
                )
            kwargs["temperature"] = settings.llm.temperature
            reasoning_desc = f"{family}/thinking=off"

        logger.info(
            "Bedrock model='{}' family={} | {} | max_tokens={}",
            model_id,
            family,
            reasoning_desc,
            settings.llm.max_tokens,
        )
        return ChatBedrockConverse(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info("Using Ollama chat model '{}' via '{}'", settings.ollama.llm_model, settings.ollama.base_url)
        return ChatOllama(
            model=settings.ollama.llm_model,
            base_url=settings.ollama.base_url,
            temperature=settings.llm.temperature,
        )

    raise ValueError(
        f"Unsupported LLM provider: {settings.llm.provider!r}. "
        "Supported: 'bedrock' (Claude/DeepSeek/gpt-oss), 'bedrock_openai' (gpt-5.x), 'ollama'."
    )


__all__ = ["get_llm", "MODEL_PRESETS"]
