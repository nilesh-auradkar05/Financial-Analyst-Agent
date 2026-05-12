"""
Provider-neutral judge model adapters for RAG quality evaluation.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

load_dotenv()

class JudgeProvider(StrEnum):
    """Supported judge model providers."""

    DEEPEVAL_DEFAULT = "deepeval_default"
    LANGCHAIN_ANTHROPIC = "langchain_anthropic"
    CHAT_NVIDIA = "chat_nvidia"

class JudgeModelConfig(BaseModel):
    """Runtime config for LLM-as-Judge providers"""

    provider: JudgeProvider = Field(default=JudgeProvider.DEEPEVAL_DEFAULT)
    model: str | None = None
    api_key: SecretStr | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def clean_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

def judge_config_from_env(
    *,
    provider: str | None,
    model: str | None,
    temperature: float,
    top_p: float | None,
    max_tokens: int | None,
    extra_body_json: str | None,
) -> JudgeModelConfig:
    """Build judge config from CLI values and env variables."""

    selected_provider = (
        provider
        or os.getenv("RAG_EVAL_JUDGE_PROVIDER")
        or JudgeProvider.DEEPEVAL_DEFAULT.value
    )
    selected_model = model or os.getenv("RAG_EVAL_JUDGE_MODEL")

    extra_body: dict[str, Any] = {}
    raw_extra_body = extra_body_json or os.getenv("RAG_EVAL_JUDGE_EXTRA_BODY")
    if raw_extra_body:
        parsed = json.loads(raw_extra_body)
        if not isinstance(parsed, dict):
            raise ValueError("Judge extra body must be a JSON object.")
        extra_body = parsed

    api_key: SecretStr | None = None
    if selected_provider == JudgeProvider.CHAT_NVIDIA:
        raw_key = os.getenv("NVIDIA_API_KEY")
        api_key = SecretStr(raw_key) if raw_key else None
        selected_model = selected_model or "deepseek-ai/deepseek-v4-pro"
        max_tokens = max_tokens or 16384
        top_p = top_p if top_p is not None else 0.95
        if not extra_body:
            extra_body = {
                "chat_template_kwargs": {
                    "thinking_mode": False
                }
            }
    elif selected_provider == JudgeProvider.LANGCHAIN_ANTHROPIC:
        raw_key = os.getenv("ANTHROPIC_API_KEY")
        api_key = SecretStr(raw_key) if raw_key else None
        selected_model = selected_model or "claude-haiku-4-5-20251001"
        max_tokens = max_tokens or 4096

    return JudgeModelConfig(
        provider=JudgeProvider(selected_provider),
        model=selected_model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )

def _secret_value(secret: SecretStr | None) -> str | None:
    if secret is None:
        return None
    return secret.get_secret_value()

def create_langchain_judge_llm(config: JudgeModelConfig) -> Any | None:
    """Create a LangChain chat model for RAGAS and custom DeepEval judges."""

    if config.provider == JudgeProvider.DEEPEVAL_DEFAULT:
        return None

    if not config.model:
        raise ValueError(f"model is required for provider={config.provider}")

    if config.provider == JudgeProvider.CHAT_NVIDIA:
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as exc:
            raise ImportError(
                "Install langchain-nvidia-ai-endpoints"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
        }
        api_key = _secret_value(config.api_key)
        if api_key:
            kwargs["api_key"] = api_key
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        if config.extra_body:
            kwargs["extra_body"] = config.extra_body

        return ChatNVIDIA(**kwargs)

    if config.provider == JudgeProvider.LANGCHAIN_ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Install langchain-anthropic"
            ) from exc

        kwargs = {
            "model": config.model,
            "temperature": config.temperature,
        }
        api_key = _secret_value(config.api_key)
        if api_key:
            kwargs["api_key"] = api_key
        if config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p

        return ChatAnthropic(**kwargs)

    raise ValueError(f"Unsupported judge provider: {config.provider}")

def _content_from_message(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))

        return "\n".join(parts)

    return str(content)

def _coerce_structured_output(raw: Any, schema: type[BaseModel]) -> BaseModel:
    if isinstance(raw, schema):
        return raw
    if isinstance(raw, BaseModel):
        return schema.model_validate(raw.model_dump())
    if isinstance(raw, dict):
        return schema.model_validate(raw)

    text = _content_from_message(raw)
    try:
        return schema.model_validate_json(text)
    except Exception:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return schema.model_validate_json(cleaned)

def _schema_prompt(prompt: str, schema: type[BaseModel]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    return (
        f"{prompt}\n\n"
        "Return only valid JSON that conforms to this JSON schema. "
        "Do not wrap the JSON in markdown.\n"
        f"{schema_json}"
    )

def create_ragas_judge_embeddings(config: JudgeModelConfig) -> Any | None:
    """Create RAGAS compatible embeddings for metrics."""

    provider = os.getenv("RAGAS_EMBEDDINGS_PROVIDER")
    if provider is None:
        provider = "openai" if config.provider == JudgeProvider.DEEPEVAL_DEFAULT else "huggingface"
        provider = provider.strip().lower()

    if provider in {"", "none", "disabled", "off"}:
        return None

    if provider == "openai":
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Install openai ragas"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Required for RAGAS OpenAI Embeddings"
            )

        model = os.getenv("RAGAS_EMBEDDINGS_MODEL") or "text-embedding-3-small"
        return OpenAIEmbeddings(client=AsyncOpenAI(api_key=api_key), model=model)

    if provider == "huggingface":
        try:
            from ragas.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers"
            ) from exc

        model = os.getenv("RAGAS_EMBEDDINGS_model") or "sentence-transformers/all-MiniLM-L12-v2"
        return HuggingFaceEmbeddings(model=model)

    raise ValueError(
        "Unsupported RAGAS_EMBEDDINGS_PROVIDER"
    )

def create_deepeval_judge_model(config: JudgeModelConfig) -> Any | str | None:
    """
    Return a DeepEval-compatible model object
    """

    if config.provider == JudgeProvider.DEEPEVAL_DEFAULT:
        return config.model

    chat_model = create_langchain_judge_llm(config)
    if chat_model is None:
        return config.model

    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError as exc:
        raise ImportError("Install deepeval") from exc

    class LangChainDeepEvalJudge(DeepEvalBaseLLM):
        """DeepEval adapter over a Langchain chat model."""

        def __init__(self, model: Any, model_name: str) -> None:
            self.model = model
            self.model_name = model_name

        def load_model(self) -> Any:
            return self.model

        def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
            model = self.load_model()
            if schema is not None:
                if hasattr(model, "with_structured_output"):
                    structured_model = model.with_structured_output(schema)
                    return _coerce_structured_output(structured_model.invoke(prompt), schema)
                response = model.invoke(_schema_prompt(prompt, schema))
                return _coerce_structured_output(response, schema)

            return _content_from_message(model.invoke(prompt))

        async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
            model = self.load_model()
            if schema is not None:
                if hasattr(model, "with_structured_output"):
                    structured_model = model.with_structured_output(schema)
                    response = await structured_model.ainvoke(prompt)
                    return _coerce_structured_output(response, schema)

                response = await model.ainvoke(_schema_prompt(prompt, schema))
                return _coerce_structured_output(response, schema)

            response = await model.ainvoke(prompt)
            return _content_from_message(response)

        def get_model_name(self) -> str:
            return self.model_name

    return LangChainDeepEvalJudge(model=chat_model, model_name=f"{config.provider}:{config.model}")

def _ragas_model_kwargs(config: JudgeModelConfig) -> dict[str, Any]:
    """Return kwargs accepted by RAGAS llm_factory model adapters."""

    kwargs: dict[str, Any] = {"temperature": config.temperature}
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.extra_body:
        kwargs.update(config.extra_body)
    return kwargs

def _create_ragas_default_llm(config: JudgeModelConfig) -> Any | None:
    """Create a default OpenAI-backed RAGAS judge when explicitly configured."""

    if not config.model:
        # Keep backward compatibility: DeepEval can pick a native default, but
        # RAGAS v0.4 collections cannot safely do that without a client.
        return None

    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise ImportError(
            "Install RAGAS/OpenAI judge support with: "
            "uv add --group eval ragas openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for RAGAS with deepeval_default + "
            "--judge-model. Use --judge-provider langchain_anthropic or "
            "chat_nvidia instead."
        )

    kwargs = _ragas_model_kwargs(config)
    client = OpenAI(api_key=api_key)
    return llm_factory(config.model, provider="openai", client=client, **kwargs)


def _create_ragas_anthropic_llm(config: JudgeModelConfig) -> Any:
    """Create an Anthropic-backed RAGAS v0.4 judge."""

    if not config.model:
        raise ValueError("model is required for RAGAS provider=langchain_anthropic")

    try:
        from anthropic import AsyncAnthropic
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise ImportError(
            "Install Anthropic RAGAS judge support with: "
            "uv add --group eval anthropic ragas"
        ) from exc

    api_key = _secret_value(config.api_key) or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for RAGAS Anthropic judging")

    kwargs = _ragas_model_kwargs(config)
    client = AsyncAnthropic(api_key=api_key)
    llm = llm_factory(
        config.model,
        provider="anthropic",
        client=client,
        adapter="instructor",
        **kwargs,
    )

    # Claude 4.x rejects requests that include both `temperature` and `top_p`.
    # RAGAS' InstructorAdapter always seeds InstructorModelArgs(), which carries
    # a default `top_p=0.1`, and `_ragas_model_kwargs` always seeds
    # `temperature`. Enforce mutual exclusion on the live model_args dict
    # (Anthropic uses pass-through, so this is what gets sent at request time):
    # prefer the caller's explicit `top_p` when provided, otherwise keep
    # `temperature` and drop the auto-injected `top_p`.
    model_args = getattr(llm, "model_args", None)
    if isinstance(model_args, dict):
        if config.top_p is not None:
            model_args.pop("temperature", None)
        else:
            model_args.pop("top_p", None)

    return llm


def _create_ragas_nvidia_llm(config: JudgeModelConfig) -> Any:
    """Create a NVIDIA NIM/build.nvidia.com-backed RAGAS v0.4 judge."""

    if not config.model:
        raise ValueError("model is required for RAGAS provider=chat_nvidia")

    try:
        from litellm import OpenAI as LiteLLMOpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise ImportError(
            "Install ragas litellm"
        ) from exc

    api_key = (
        _secret_value(config.api_key)
        or os.getenv("NVIDIA_NIM_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
    )
    if not api_key:
        raise ValueError("NVIDIA_API_KEY or NVIDIA_NIM_API_KEY is required for RAGAS NVIDIA judging")

    # LiteLLM expects NVIDIA_NIM_API_KEY for its nvidia_nim provider.
    os.environ.setdefault("NVIDIA_NIM_API_KEY", api_key)

    kwargs = _ragas_model_kwargs(config)

    # LiteLLM routes NVIDIA NIM calls via the nvidia_nim/ prefix. Do not mutate
    # config.model so DeepEval/ChatNVIDIA can still use the original name.
    model = config.model
    if not model.startswith("nvidia_nim/"):
        model = f"nvidia_nim/{model}"

    client = LiteLLMOpenAI(api_key=api_key)
    return llm_factory(
        model,
        provider="litellm",
        client=client,
        adapter="litellm",
        **kwargs,
    )

def create_ragas_judge_llm(config: JudgeModelConfig) -> Any | None:
    """Create a RAGAS v0.4+ compatible judge LLM."""

    if config.provider == JudgeProvider.DEEPEVAL_DEFAULT:
        return _create_ragas_default_llm(config)
    if config.provider == JudgeProvider.LANGCHAIN_ANTHROPIC:
        return _create_ragas_anthropic_llm(config)
    if config.provider == JudgeProvider.CHAT_NVIDIA:
        return _create_ragas_nvidia_llm(config)
    raise ValueError(f"Unsupported RAGAS judge provider: {config.provider}")
