"""
Provider-neutral judge model adapters for RAG quality evaluation.
"""

from __future__ import annotations

import asyncio
import json
import os
from enum import StrEnum
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

load_dotenv()

def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"

def _is_async_sync_mismatch(exc: BaseException) -> bool:
    """Detect async-call failures caused by sync-only model/client objects"""

    text = _error_text(exc).lower()
    fragments = (
        "agenerate",
        "async",
        "sync object",
        "does not support async",
        "object cannot be used in 'await'",
        "can't be used in 'await'",
        "has no attribute 'ainvoke'",
        "has no attribute 'agenerate'",
    )

    return any(fragment in text for fragment in fragments)

def _invoke_model(model: Any, prompt: str) -> Any:
    """Invoke a LangChain-like model synchronously."""

    if hasattr(model, "invoke"):
        return model.invoke(prompt)

    raise TypeError(f"Model does not support sync invoke(): {type(model).__name__}")

async def _ainvoke_or_thread(model: Any, prompt: str) -> Any:
    """
    Invoke a LangChain-like model asynchronously.

    Falls back to running sync invoke() in a thread when the model/client
    does not actually support async execution.
    """

    if hasattr(model, "ainvoke"):
        try:
            return await model.ainvoke(prompt)
        except Exception as exc:
            if not _is_async_sync_mismatch(exc):
                raise

    if hasattr(model, "invoke"):
        return await asyncio.to_thread(model.invoke, prompt)

    raise TypeError(f"Model does not support invoke() or ainvoke(): {type(model).__name__}")

class JudgeProvider(StrEnum):
    """Supported judge model providers."""

    DEEPEVAL_DEFAULT = "deepeval_default"
    LANGCHAIN_ANTHROPIC = "langchain_anthropic"

class JudgeModelConfig(BaseModel):
    """Runtime config for LLM-as-Judge providers"""

    provider: JudgeProvider = Field(default=JudgeProvider.DEEPEVAL_DEFAULT)
    model: str | None = None
    api_key: SecretStr | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)

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
) -> JudgeModelConfig:
    """Build judge config from CLI values and env variables."""

    selected_provider = (
        provider
        or os.getenv("RAG_EVAL_JUDGE_PROVIDER")
        or JudgeProvider.DEEPEVAL_DEFAULT.value
    )
    selected_model = model or os.getenv("RAG_EVAL_JUDGE_MODEL")

    api_key: SecretStr | None = None
    if selected_provider == JudgeProvider.LANGCHAIN_ANTHROPIC:
        raw_key = os.getenv("ANTHROPIC_API_KEY")
        api_key = SecretStr(raw_key) if raw_key else None
        selected_model = selected_model or "claude-sonnet-4-6"
        max_tokens = max_tokens or 4096

    return JudgeModelConfig(
        provider=JudgeProvider(selected_provider),
        model=selected_model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

def _secret_value(secret: SecretStr | None) -> str | None:
    if secret is None:
        return None
    return secret.get_secret_value()


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def create_langchain_judge_llm(config: JudgeModelConfig) -> Any | None:
    """Create a LangChain chat model for RAGAS and custom DeepEval judges."""

    if config.provider == JudgeProvider.DEEPEVAL_DEFAULT:
        return None

    if not config.model:
        raise ValueError(f"model is required for provider={config.provider}")

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

        model = os.getenv("RAGAS_EMBEDDINGS_MODEL") or "sentence-transformers/all-MiniLM-L12-v2"
        device = os.getenv("RAGAS_EMBEDDINGS_DEVICE") or "cpu"
        local_files_only = _env_bool("RAGAS_EMBEDDINGS_LOCAL_FILES_ONLY", default=True)
        return HuggingFaceEmbeddings(
            model=model,
            device=device,
            local_files_only=local_files_only,
        )

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

            def _generate_once() -> Any:
                if schema is not None:
                    if hasattr(model, "with_structured_output"):
                        structured_model = model.with_structured_output(schema)
                        return _coerce_structured_output(
                            _invoke_model(structured_model, prompt),
                            schema,
                        )
                    response = _invoke_model(model, _schema_prompt(prompt, schema))
                    return _coerce_structured_output(response, schema)

                return _content_from_message(_invoke_model(model, prompt))

            return _generate_once()

        async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
            model = self.load_model()

            async def _generate_once() -> Any:
                if schema is not None:
                    if hasattr(model, "with_structured_output"):
                        structured_model = model.with_structured_output(schema)
                        response = await _ainvoke_or_thread(structured_model, prompt)
                        return _coerce_structured_output(response, schema)

                    response = await _ainvoke_or_thread(model, _schema_prompt(prompt, schema))
                    return _coerce_structured_output(response, schema)

                response = await _ainvoke_or_thread(model, prompt)
                return _content_from_message(response)

            return await _generate_once()

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
    return kwargs

def _create_ragas_default_llm(config: JudgeModelConfig) -> Any | None:
    """Create a default OpenAI-backed RAGAS judge when explicitly configured."""

    if not config.model:
        # Keep backward compatibility: DeepEval can pick a native default, but
        # RAGAS v0.4 collections cannot safely do that without a client.
        return None

    try:
        from openai import AsyncOpenAI
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
            "configure OPENAI_API_KEY."
        )

    kwargs = _ragas_model_kwargs(config)
    client = AsyncOpenAI(api_key=api_key)
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

def create_ragas_judge_llm(config: JudgeModelConfig) -> Any | None:
    """Create a RAGAS v0.4+ compatible judge LLM."""

    if config.provider == JudgeProvider.DEEPEVAL_DEFAULT:
        return _create_ragas_default_llm(config)
    if config.provider == JudgeProvider.LANGCHAIN_ANTHROPIC:
        return _create_ragas_anthropic_llm(config)
    raise ValueError(f"Unsupported RAGAS judge provider: {config.provider}")
