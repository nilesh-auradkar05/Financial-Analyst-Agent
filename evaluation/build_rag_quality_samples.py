"""Generate RAG quality samples for RAGAS/DeepEval evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RAGQualityEvalCase(BaseModel):
    """A memo-level RAG quality case."""

    id: str = Field(min_length=3)
    ticker: str = Field(min_length=1)
    input: str = Field(min_length=10)
    expected_output: str = Field(min_length=10)
    tags: list[str] = Field(default_factory=list)
    include_filing_analysis: bool = True
    include_news_sentiment: bool = False
    max_news_articles: int = Field(default=0, ge=0, le=20)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

class RAGQualitySample(BaseModel):
    """Generated output plus retrieved context for judge-based RAG metrics."""

    case_id: str
    ticker: str
    input: str
    actual_output: str
    expected_output: str
    retrieval_context: list[str]
    context: list[str]
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

def load_cases(path: str | Path) -> list[RAGQualityEvalCase]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("RAG Quality fixture must be a JSON list.")
    return [RAGQualityEvalCase.model_validate(item) for item in payload]

def _context_from_state(state: dict[str, Any]) -> list[str]:
    evidence = state.get("citation_evidence") or []
    contexts: list[str] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text:
            contexts.append(text)

    return contexts

async def _run_case(case: RAGQualityEvalCase) -> RAGQualitySample:
    from app.agents.graph import run_agent

    state = await run_agent(
        ticker=case.ticker,
        include_filing_analysis=case.include_filing_analysis,
        include_news_sentiment=case.include_news_sentiment,
        max_news_articles=case.max_news_articles,
    )
    state_dict = dict(state)
    actual_output = str(state_dict.get("investment_memo") or "")
    retrieval_context = _context_from_state(state_dict)
    return RAGQualitySample(
        case_id=case.id,
        ticker=case.ticker,
        input=case.input,
        actual_output=actual_output,
        expected_output=case.expected_output,
        retrieval_context=retrieval_context,
        context=[case.expected_output],
        tags=case.tags,
        metadata={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "include_filing_analysis": case.include_filing_analysis,
            "include_news_sentiment": case.include_news_sentiment,
            "max_news_articles": case.max_news_articles,
            "execution_time_ms": state_dict.get("execution_time_ms"),
            "verification_result": state_dict.get("verification_result"),
            "errors": state_dict.get("errors", []),
        },
    )

async def build_samples(
    cases: list[RAGQualityEvalCase],
    *,
    case_id: str | None,
    limit: int | None = None,
) -> list[RAGQualitySample]:
    selected = [case for case in cases if case_id is None or case.id == case_id]
    if not selected:
        raise ValueError(f"No matching RAG quality cases found for case_id={case_id!r}")
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"--limit must be a positive integer, got {limit}")
        selected = selected[:limit]
    samples: list[RAGQualitySample] = []
    for case in selected:
        samples.append(await _run_case(case))

    return samples

def save_samples(samples: list[RAGQualitySample], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([sample.model_dump() for sample in samples], indent=2))

    return output

def _default_output_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return Path("evaluation/results") / f"rag_quality_samples_{ts}.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Build generated samples for RAG quality evals.")
    parser.add_argument(
        "--fixture",
        default="evaluation/fixtures/rag_quality_eval_cases_v1.json",
        help="Path to RAG quality eval case fixture.",
    )
    parser.add_argument("--case", help="Optional case id to run")
    parser.add_argument("--output", default=str(_default_output_path()))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cases to generate samples for (inclusive). Applied after --case filtering.",
    )
    args = parser.parse_args()

    cases = load_cases(args.fixture)
    samples = asyncio.run(build_samples(cases, case_id=args.case, limit=args.limit))
    output_path = save_samples(samples, args.output)
    print(f"Saved {len(samples)} generated RAG quality samples to {output_path}")

if __name__ == "__main__":
    main()
