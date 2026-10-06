"""Runtime chat-model factory: Bedrock primary, then personal Claude and OpenAI fallbacks.

Bedrock Converse-API families:
  - Anthropic Claude  -> additionalModelRequestFields={"thinking": {...}}  (enabled/adaptive)
  - DeepSeek V3.2     -> additionalModelRequestFields={"thinking": {"type": "enabled"}}  (hybrid thinking)
  - OpenAI gpt-oss    -> reasoning effort low|medium|high
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from functools import partial
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from loguru import logger
from pydantic import SecretStr

from app.config import Settings

# Confirm exact IDs/regions with `aws bedrock list-foundation-models` / list-inference-profiles.
MODEL_PRESETS: dict[str, str] = {
    "claude-sonnet": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "deepseek-v3": "deepseek.v3.2",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
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


def model_metadata(settings: Settings) -> dict:
    """Effective factory settings; native model spans remain authoritative."""
    model = MODEL_PRESETS.get(settings.llm.model, settings.llm.model)
    family = _model_family(model)
    omit_temperature = (
        family == "claude" and (
            settings.llm.thinking_mode in {"enabled", "adaptive"}
            or "claude-sonnet-5" in model.lower()
        )
        or family == "deepseek" and settings.llm.thinking_mode != "off"
    )
    return {"provider": "bedrock", "model": model,
            "temperature": None if omit_temperature else settings.llm.temperature}


def _bedrock_model(settings: Settings) -> BaseChatModel:
    """Return the configured Bedrock Converse chat model."""
    model_id = MODEL_PRESETS.get(settings.llm.model, settings.llm.model)
    mode = settings.llm.thinking_mode.lower()

    if not settings.llm.aws_region:
        raise ValueError("Bedrock AWS region is missing. Set AWS_REGION.")
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Bedrock provider requires `langchain-aws` and `awscrt`."
        ) from exc

    family = _model_family(model_id)
    kwargs: dict[str, Any] = {
        "model": model_id,
        "max_tokens": settings.llm.max_tokens,
        "region_name": settings.llm.aws_region,
        "timeout": math.ceil(settings.llm.request_timeout_seconds),
        "max_retries": 0,
    }
    token = settings.llm.aws_bearer_token_bedrock
    if token and token.get_secret_value().strip():
        kwargs["bedrock_api_key"] = token

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
        if "claude-sonnet-5" not in model_id.lower():
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


def fallback_count(settings: Settings) -> int:
    """Number of bounded cloud attempts, including Bedrock."""
    return 1 + len(settings.llm.personal_fallbacks())


def _personal_model(settings: Settings, provider: str, model: str, key: SecretStr) -> BaseChatModel:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": key.get_secret_value(),
        "max_tokens": settings.llm.max_tokens,
        "timeout": settings.llm.request_timeout_seconds,
        "max_retries": 0,
    }
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kwargs)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)


# One fallback chain per Settings object, so each provider's client is built once per process.
# ponytail: keyed by object identity and never evicted; the app has one Settings. Key by
# value if settings ever become per-request.
_chains: dict[int, tuple[Settings, Runnable[Any, Any]]] = {}


def get_llm(settings: Settings) -> Runnable[Any, Any]:
    """Use Bedrock first, then personal Claude, then personal OpenAI (each only with a key)."""
    if cached := _chains.get(id(settings)):
        return cached[1]
    providers: list[tuple[str, Callable[[], BaseChatModel]]] = [("bedrock", lambda: _bedrock_model(settings))]
    for name, model, key in settings.llm.personal_fallbacks():
        providers.append((name, partial(_personal_model, settings, name, model, key)))

    timeout = settings.llm.request_timeout_seconds

    def lazy_model(name: str, build_model: Callable[[], BaseChatModel]) -> Runnable[Any, Any]:
        built: list[BaseChatModel] = []

        def build() -> BaseChatModel:
            # A failed build raises before the append, so it is retried on the next request.
            if not built:
                built.append(build_model())
            return built[0]

        def invoke(input: Any, config: RunnableConfig) -> Any:
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                return pool.submit(lambda: build().invoke(input, config=config)).result(timeout=timeout)
            except FutureTimeoutError as exc:
                raise TimeoutError(f"{name} attempt timed out") from exc
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

        async def ainvoke(input: Any, config: RunnableConfig) -> Any:
            async def attempt() -> Any:
                model = await asyncio.to_thread(build)
                return await model.ainvoke(input, config=config)

            try:
                return await asyncio.wait_for(attempt(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"{name} attempt timed out") from exc

        return RunnableLambda(invoke, afunc=ainvoke).with_config(run_name=f"{name}_attempt")

    attempts = [lazy_model(name, build) for name, build in providers]
    chain = attempts[0].with_fallbacks(attempts[1:])
    _chains[id(settings)] = (settings, chain)
    return chain


__all__ = ["get_llm", "MODEL_PRESETS"]
