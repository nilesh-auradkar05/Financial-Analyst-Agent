"""Run RAGAS and DeepEval over generated RAG quality samples.=
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal, TypeVar, cast

from pydantic import BaseModel

from evaluation.build_rag_quality_samples import RAGQualitySample
from evaluation.judge_models_interface import (
    JudgeModelConfig,
    JudgeProvider,
    create_deepeval_judge_model,
    create_ragas_judge_embeddings,
    create_ragas_judge_llm,
    judge_config_from_env,
)

JudgeName = Literal["ragas", "deepeval"]
T = TypeVar("T")

class MetricScore(BaseModel):
    """Normalized metric result across RAGAS and DeepEval."""

    case_id: str
    judge: str
    metric: str
    score: float | None = None
    passed: bool | None = None
    reason: str | None = None
    error: str | None = None


class RagQualityReport(BaseModel):
    """Persisted RAG quality report."""

    generated_at: str
    sample_count: int
    judges: list[str]
    summary: dict[str, Any]
    scores: list[MetricScore]


def load_samples(path: str | Path) -> list[RAGQualitySample]:
    """Load generated RAG quality samples from JSON."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("RAG quality samples must be a JSON list")
    return [RAGQualitySample.model_validate(item) for item in payload]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate_context(contexts: Iterable[str], *, max_chars: int) -> list[str]:
    """Trim contexts to reduce judge cost and hosted-model rate-limit pain."""

    if max_chars <= 0:
        return [context for context in contexts if context.strip()]
    return [context[:max_chars] for context in contexts if context.strip()]


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_retryable_exception(exc: BaseException) -> bool:
    text = _error_text(exc).lower()
    retryable_fragments = (
        "429",
        "too many requests",
        "rate limit",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "service unavailable",
        "overloaded",
    )
    return any(fragment in text for fragment in retryable_fragments)


def _sleep_for_retry(attempt: int, *, base_seconds: float) -> None:
    if base_seconds <= 0:
        return
    time.sleep(base_seconds * (2 ** max(attempt - 1, 0)))


async def _async_sleep_for_retry(attempt: int, *, base_seconds: float) -> None:
    if base_seconds <= 0:
        return
    await asyncio.sleep(base_seconds * (2 ** max(attempt - 1, 0)))


def _run_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> T:
    """Run a sync judge call with conservative retry handling."""

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _is_retryable_exception(exc):
                raise
            _sleep_for_retry(attempt, base_seconds=retry_backoff_seconds)


async def _run_with_retries_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    retry_backoff_seconds: float,
) -> T:
    """Run an async judge call with conservative retry handling."""

    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - external judge failures are normalized later
            attempt += 1
            if attempt > max_retries or not _is_retryable_exception(exc):
                raise
            await _async_sleep_for_retry(attempt, base_seconds=retry_backoff_seconds)


def _constructor_kwargs(metric_cls: type[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs for DeepEval metric constructors across minor versions."""

    try:
        signature = inspect.signature(metric_cls)
    except (TypeError, ValueError):
        return kwargs

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def run_deepeval(
    samples: list[RAGQualitySample],
    *,
    judge_config: JudgeModelConfig,
    threshold: float,
    max_context_chars: int,
    async_mode: bool,
    request_delay_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[MetricScore]:
    """Run DeepEval RAG metrics over generated samples."""

    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        return [
            MetricScore(
                case_id=sample.case_id,
                judge="deepeval",
                metric="import_error",
                error=f"DeepEval is not importable: {exc}",
            )
            for sample in samples
        ]

    try:
        judge_model = create_deepeval_judge_model(judge_config)
    except Exception as exc:
        return [
            MetricScore(
                case_id=sample.case_id,
                judge="deepeval",
                metric="configuration_error",
                error=_error_text(exc),
            )
            for sample in samples
        ]

    base_kwargs: dict[str, Any] = {
        "threshold": threshold,
        "include_reason": True,
        "async_mode": async_mode,
    }
    if judge_model is not None:
        base_kwargs["model"] = judge_model

    metric_classes = [
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    ]

    metrics = [
        metric_cls(**_constructor_kwargs(metric_cls, base_kwargs))
        for metric_cls in metric_classes
    ]

    scores: list[MetricScore] = []
    for sample in samples:
        test_case = LLMTestCase(
            input=sample.input,
            actual_output=sample.actual_output,
            expected_output=sample.expected_output,
            context=sample.context or [sample.expected_output],
            retrieval_context=_truncate_context(
                sample.retrieval_context,
                max_chars=max_context_chars,
            ),
        )

        for metric in metrics:
            metric_name = metric.__class__.__name__
            try:
                _run_with_retries(
                    lambda: metric.measure(test_case),
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                scores.append(
                    MetricScore(
                        case_id=sample.case_id,
                        judge="deepeval",
                        metric=metric_name,
                        score=_safe_float(getattr(metric, "score", None)),
                        passed=bool(getattr(metric, "success", False)),
                        reason=getattr(metric, "reason", None),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - external judge failures should not kill report
                scores.append(
                    MetricScore(
                        case_id=sample.case_id,
                        judge="deepeval",
                        metric=metric_name,
                        error=_error_text(exc),
                    )
                )

            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)

    return scores


def _load_ragas_collection_metric_classes() -> list[type[Any]]:
    """Load RAGAS v0.4+ collections metrics.
    """

    try:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "RAGAS collections metrics are not importable. Install/upgrade with: "
            "uv add --group eval -U ragas"
        ) from exc

    return [
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    ]


def _instantiate_ragas_metric(metric_cls: type[Any], *, llm: Any | None, embeddings: Any | None,) -> Any:
    """Instantiate a RAGAS metric with the configured judge LLM when supported."""

    kwargs: dict[str, Any] = {}
    parameters: Mapping[str, inspect.Parameter] = {}
    try:
        signature = inspect.signature(metric_cls)
        parameters = signature.parameters
    except (TypeError, ValueError):
        parameters = {}

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if llm is not None and (accepts_kwargs or "llm" in parameters):
        kwargs["llm"] = llm
    if embeddings is not None and (accepts_kwargs or "embeddings" in parameters):
        kwargs["embeddings"] = embeddings

    return metric_cls(**kwargs)


def _ragas_kwargs_for_metric(
    metric: Any,
    sample: RAGQualitySample,
    *,
    max_context_chars: int,
) -> dict[str, Any]:
    """Return candidate kwargs for RAGAS metric scoring."""

    contexts = _truncate_context(sample.retrieval_context, max_chars=max_context_chars)
    metric_name = metric.__class__.__name__

    if metric_name == "Faithfulness":
        return {
            "user_input": sample.input,
            "response": sample.actual_output,
            "retrieved_contexts": contexts,
        }

    if metric_name in {"AnswerRelevancy", "ResponseRelevancy"}:
        return {
            "user_input": sample.input,
            "response": sample.actual_output,
        }

    if metric_name in {"ContextPrecision", "ContextPrecisionWithReference"}:
        return {
            "user_input": sample.input,
            "reference": sample.expected_output,
            "retrieved_contexts": contexts,
        }

    if metric_name == "ContextRecall":
        return {
            "user_input": sample.input,
            "reference": sample.expected_output,
            "retrieved_contexts": contexts,
        }

    available_values: dict[str, Any] = {
        "user_input": sample.input,
        "response": sample.actual_output,
        "reference": sample.expected_output,
        "retrieved_contexts": contexts,
    }
    try:
        signature = inspect.signature(metric.ascore)
    except (TypeError, ValueError):
        raise ValueError(
            f"Unsupported RAGAS metric {metric_name}: cannot inspect ascore signature"
        )

    kwargs: dict[str, Any] = {}
    missing_required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name in {"self", "callbacks"}:
            continue
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if parameter_name in available_values:
            kwargs[parameter_name] = available_values[parameter_name]
        elif parameter.default is inspect.Parameter.empty:
            missing_required.append(parameter_name)

    if missing_required:
        raise ValueError(f"Unsupported RAGAS metric {metric_name}: missing required fields: {missing_required} for ascore signature {signature}")

    return kwargs


async def _score_ragas_metric(
    metric: Any,
    sample: RAGQualitySample,
    *,
    max_context_chars: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> Any:
    """Score one RAGAS collections metric."""

    kwargs = _ragas_kwargs_for_metric(
        metric,
        sample,
        max_context_chars=max_context_chars,
    )

    return await _run_with_retries_async(
        lambda: metric.ascore(**kwargs),
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )

def _extract_ragas_score(result: Any) -> tuple[float | None, str | None]:
    """Extract numeric value and optional reason from RAGAS MetricResult."""

    if isinstance(result, dict):
        return _safe_float(result.get("value") or result.get("score")), result.get("reason")
    return _safe_float(getattr(result, "value", result)), getattr(result, "reason", None)


async def run_ragas(
    samples: list[RAGQualitySample],
    *,
    judge_config: JudgeModelConfig,
    threshold: float,
    max_context_chars: int,
    request_delay_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[MetricScore]:
    """Run RAGAS v0.4+ collections metrics over generated RAG samples."""

    try:
        metric_classes = _load_ragas_collection_metric_classes()
        ragas_llm = create_ragas_judge_llm(judge_config)
        ragas_embeddings = create_ragas_judge_embeddings(judge_config)
        metrics = [
            _instantiate_ragas_metric(metric_class, llm=ragas_llm, embeddings=ragas_embeddings)
            for metric_class in metric_classes
        ]
    except Exception as exc:
        return [
            MetricScore(
                case_id=sample.case_id,
                judge="ragas",
                metric="configuration_error",
                error=_error_text(exc),
            )
            for sample in samples
        ]

    scores: list[MetricScore] = []
    for sample in samples:
        for metric in metrics:
            metric_name = metric.__class__.__name__
            try:
                result = await _score_ragas_metric(
                    metric,
                    sample,
                    max_context_chars=max_context_chars,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                score_value, reason = _extract_ragas_score(result)
                scores.append(
                    MetricScore(
                        case_id=sample.case_id,
                        judge="ragas",
                        metric=metric_name,
                        score=score_value,
                        passed=None if score_value is None else score_value >= threshold,
                        reason=reason,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - external judge failures should not kill report
                scores.append(
                    MetricScore(
                        case_id=sample.case_id,
                        judge="ragas",
                        metric=metric_name,
                        error=_error_text(exc),
                    )
                )

            if request_delay_seconds > 0:
                await asyncio.sleep(request_delay_seconds)

    return scores


def summarize_scores(
    scores: list[MetricScore],
    *,
    judge_config: JudgeModelConfig,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    errors = 0
    for score in scores:
        if score.error:
            errors += 1
            continue
        if score.score is None:
            continue
        key = f"{score.judge}.{score.metric}"
        grouped.setdefault(key, []).append(score.score)

    return {
        "metric_averages": {
            key: sum(values) / len(values)
            for key, values in sorted(grouped.items())
            if values
        },
        "metric_counts": {key: len(values) for key, values in sorted(grouped.items())},
        "error_count": errors,
        "judge_provider": str(judge_config.provider),
        "judge_model": judge_config.model,
    }


def save_report(report: RagQualityReport, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(), indent=2))
    return output


def log_summary_to_langsmith(report: RagQualityReport, *, experiment_name: str) -> None:
    """LangSmith logging."""

    tracing_enabled = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "false"))
    if tracing_enabled.lower() not in {"1", "true", "yes", "on"}:
        return
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return

    try:
        from langsmith import traceable  # type: ignore
    except ImportError:
        return

    @traceable(name=experiment_name, run_type="chain", tags=["evaluation", "rag-quality"])
    def _emit(summary: dict[str, Any]) -> dict[str, Any]:
        return summary

    try:
        _emit(report.summary)
    except Exception:
        # Do not let observability break evaluation. We learned this one already.
        return


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("evaluation/reports") / f"rag_quality_eval_{timestamp}.json"


def _parse_judges(value: str) -> list[JudgeName]:
    judges = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"ragas", "deepeval"}
    unknown = sorted(set(judges) - allowed)
    if unknown:
        raise ValueError(f"Unknown judges: {unknown}. Allowed judges: {sorted(allowed)}")
    return [cast(JudgeName, judge) for judge in judges if judge in allowed]


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _build_judge_config(args: argparse.Namespace) -> JudgeModelConfig:
    model = args.judge_model or args.deepeval_model
    return judge_config_from_env(
        provider=args.judge_provider,
        model=model,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        max_tokens=args.judge_max_tokens,
        extra_body_json=args.judge_extra_body_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS/DeepEval RAG quality evals")
    parser.add_argument("samples", help="Path to generated RAG quality samples JSON")
    parser.add_argument("--judges", default="ragas,deepeval", help="Comma-separated: ragas,deepeval")
    parser.add_argument("--output", default=str(_default_output_path()))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max-context-chars", type=int, default=2500)
    parser.add_argument("--langsmith-experiment", default="rag-quality-eval")

    parser.add_argument(
        "--judge-provider",
        choices=[provider.value for provider in JudgeProvider],
        default=os.getenv("RAG_EVAL_JUDGE_PROVIDER", JudgeProvider.DEEPEVAL_DEFAULT.value),
        help="Judge model provider: deepeval_default, langchain_anthropic, chat_nvidia",
    )
    parser.add_argument("--judge-model", default=os.getenv("RAG_EVAL_JUDGE_MODEL"), help="Judge model name", required=True)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--judge-extra-body-json", default=None)
    parser.add_argument(
        "--ragas-embeddings-provider",
        choices=["openai", "huggingface", "none"],
        default=os.getenv("RAGAS_EMBEDDINGS_PROVIDER"),
        help="RAGAS embeddings provider for metrics such as AnswerRelevancy",
    )
    parser.add_argument(
        "--ragas-embeddings-model",
        default=os.getenv("RAGAS_EMBEDDINGS_MODEL"),
        help="Embedding model for the selected RAGAS embeddings provider.",
    )
    parser.add_argument(
        "--deepeval-model",
        default=os.getenv("DEEPEVAL_MODEL"),
        help="Legacy alias for --judge-model; kept so old commands do not explode.",
    )
    parser.add_argument(
        "--deepeval-async-mode",
        type=_parse_bool,
        default=False,
        help="Whether DeepEval metrics may run internal judge calls asynchronously.",
    )
    parser.add_argument(
        "--judge-request-delay-seconds",
        type=float,
        default=0.0,
        help="Delay between metric calls; useful for hosted judge rate limits.",
    )
    parser.add_argument("--judge-max-retries", type=int, default=2)
    parser.add_argument("--judge-retry-backoff-seconds", type=float, default=5.0)

    args = parser.parse_args()

    if args.ragas_embeddings_provider:
        os.environ["RAGAS_EMBEDDINGS_PROVIDER"] = args.ragas_embeddings_provider
    if args.ragas_embeddings_model:
        os.environ["RAGAS_EMBEDDINGS_MODEL"] = args.ragas_embeddings_model

    samples = load_samples(args.samples)
    judges = _parse_judges(args.judges)
    judge_config = _build_judge_config(args)

    scores: list[MetricScore] = []
    if "ragas" in judges:
        scores.extend(
            asyncio.run(
                run_ragas(
                    samples,
                    judge_config=judge_config,
                    threshold=args.threshold,
                    max_context_chars=args.max_context_chars,
                    request_delay_seconds=args.judge_request_delay_seconds,
                    max_retries=args.judge_max_retries,
                    retry_backoff_seconds=args.judge_retry_backoff_seconds,
                )
            )
        )

    if "deepeval" in judges:
        scores.extend(
            run_deepeval(
                samples,
                judge_config=judge_config,
                threshold=args.threshold,
                max_context_chars=args.max_context_chars,
                async_mode=args.deepeval_async_mode,
                request_delay_seconds=args.judge_request_delay_seconds,
                max_retries=args.judge_max_retries,
                retry_backoff_seconds=args.judge_retry_backoff_seconds,
            )
        )

    report = RagQualityReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        sample_count=len(samples),
        judges=list(judges),
        summary=summarize_scores(scores, judge_config=judge_config),
        scores=scores,
    )
    output_path = save_report(report, args.output)
    log_summary_to_langsmith(report, experiment_name=args.langsmith_experiment)

    print(json.dumps(report.summary, indent=2))
    print(f"Saved RAG quality report: {output_path}")


if __name__ == "__main__":
    main()
