"""
FINANCIAL ANALYST AGENT SYSTEM - EVALUATION

USAGE:
    poetry run python -m evaluation AAPL
    poetry run python -m evaluation --all
    poetry run python -m evaulation AAPL --full # Include Ragas/DeepEval
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

# ======================================================================
# RESULT DATACLASS
# ======================================================================

@dataclass
class EvalResult:
    """Evaluation result"""
    ticker: str
    passed: bool = False
    score: float = 0.0

    # Completeness checks
    has_memo: bool = False
    has_stock_data: bool = False
    has_news: bool = False
    has_sentiment: bool = False
    has_citations: bool = False

    # Counts
    memo_words: int = 0
    citation_count: int = 0
    news_count: int = 0
    chunk_count: int = 0

    # Quality
    quality_score: float = 0.0
    issues: str = ""

    # Optional Ragas/DeepEval
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    hallucination: Optional[float] = None

    # Meta
    time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

# ======================================================================
# HEURISTIC CHECKS
# ======================================================================

def check_memo(memo: str, ticker: str) -> tuple[float, list[str]]:
    """Check memo quality. Returns (0-1 score, list of issues)"""
    issues = []
    score = 0.0

    words = len(memo.split())

    # Length
    if words < 200:
        issues.append(f"short ({words}) words")
        score -= 0.3

    # Ticker mentioned
    if ticker.upper() not in memo.upper():
        issues.append("ticker missing")
        score -= 0.2

    # Key sections present
    sections = ["summary", "overview", "risk", "financial", "conclusion"]
    found = sum(1 for s in sections if s.lower() in memo.lower())
    if found < 3:
        issues.append(f"sections ({found}/5 )")
        score -= 0.2

    # Citations
    if "[1]" not in memo and "Source:" not in memo:
        issues.append("no citations")
        score -= 0.1

    # Numeric data
    numbers = re.findall(r'\$[\d,]+|\d+%|\d+%\.\d+', memo)
    if len(numbers) < 3:
        issues.append("few numbers")
        score -= 0.1

    return max(0.0, score), issues

# ======================================================================
# OPTIONAL: RAGAS
# ======================================================================

def run_ragas(question: str, answer: str, contexts: list[str]) -> dict:
    """Run Ragas. Return {} if not available."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        ds = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])

        return {
            "faithfulness": result.get("faithfulness"),
            "relevance": result.get("answer_relevancy"),
        }

    except ImportError:
        return {}
    except Exception as e:
        logger.debug(f"Ragas: {e}")
        return {}

# ======================================================================
# OPTIONAL: DEEPEVAL
# ======================================================================

def run_deepeval(input_text: str, output: str, context: list[str]) -> dict:
    """Run DeepEval. Returns {} if not availables."""
    try:
        from deepeval.metrics import HallucinationMetric
        from deepeval.test_case import LLMTestCase

        tc = LLMTestCase(input=input_text, actual_output=output, context=context)
        metric = HallucinationMetric(threshold=0.3)
        metric.measure(tc)

        return {"hallucination": metric.score}

    except ImportError:
        return {}
    except Exception as e:
        logger.debug(f"DeepEval: {e}")
        return {}

# ======================================================================
# MAIN EVALUATION
# ======================================================================

async def evaluate_ticker(ticker: str, run_full: bool = False) -> EvalResult:
    """Evaluate a single ticker."""
    r = EvalResult(ticker=ticker)
    start = time.time()

    try:
        from agents.graph import run_agent

        logger.info(f"Evaluating {ticker}...")
        out = await run_agent(ticker)

        # Completeness checks
        r.has_memo = bool(out.get("investment_memo"))
        r.has_stock_data = bool(out.get("stock_data"))
        r.has_news = bool(out.get("news_articles", [])) > 0
        r.has_sentiment = bool(out.get("sentiment_result"))
        r.has_citations = len(out.get("citations", [])) > 0

        # Counts
        memo = out.get("investment_memo", "")
        r.memo_words = len(memo.split())
        r.citation_count = len(out.get("citations", []))
        r.news_count = len(out.get("news_articles", []))
        r.chunk_count = len(out.get("filing_chunks", []))

        if not r.has_memo:
            r.error = "No memo generated"
            r.time_ms = (time.time() - start) * 1000
            return r

        # Quality check
        r.quality_score, issues = check_memo(memo, ticker)
        r.issues = ", ".join(issues) if issues else ""

        # Optional: Ragas/DeepEval
        if run_full:
            contexts = [c.get("text", "") for c in out.get("filing_chunks", [])]

            ragas = run_ragas(f"Analyze {ticker}", memo, contexts)
            r.faithfulness_score = ragas.get("faithfulness")
            r.relevance_score = ragas.get("relevance")

            de = run_deepeval(f"Analyze {ticker}", memo, contexts)
            r.hallucination = de.get("hallucination")

        # Overall score
        scores = [r.quality_score]
        completeness = sum([r.has_memo, r.has_stock_data, r.has_news, r.has_sentiment, r.has_citations]) / 5
        scores.append(completeness)

        if r.faithfulness_score is not None:
            scores.append(r.faithfulness_score)
        if r.relevance_score is not None:
            scores.append(r.relevance_score)
        if r.hallucination is not None:
            scores.append(r.hallucination)

        r.score = sum(scores) / len(scores)
        r.passed = r.score >= 0.6 and r.has_memo

    except ImportError as e:
        r.error = str(e)
    except Exception as e:
        r.error = str(e)
        logger.error(f"Failed: {e}")

    r.time_ms = (time.time() - start) * 1000
    return r

async def evaluate_all(tickers: list[str] = None, run_full: bool = False) -> list[EvalResult]:
    """Evaluate multiple tickers"""
    tickers = tickers or ["AAPL", "MSFT", "GOOG", "NVDA", "INTC"]
    results = []

    for ticker in tickers:
        result = await evaluate_ticker(ticker, run_full)
        results.append(result)

        status = "PASSED" if result.passed else "FAILED"
        logger.info(f"{ticker}: {status} score={result.score:.2f}")

    return results

# ======================================================================
# Output
# ======================================================================

def print_results(results: list[EvalResult]):
    """Print summary"""
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)

    for r in results:
        s = "✅" if r.passed else "❌"
        print(f"{s} {r.ticker:5} | score={r.score:.2f} | words={r.memo_words:4} | {r.time_ms/1000:.1f}s")
        if r.error:
            print(f"        └─ {r.error}")
        elif r.issues:
            print(f"        └─ Issues: {r.issues}")

    print("-" * 55)
    passed = sum(1 for r in results if r.passed)
    avg = sum(r.score for r in results) / len(results) if results else 0
    print(f"Passed: {passed}/{len(results)} | Avg Score: {avg:.2f}")

def save_json(results: list[EvalResult], path: str):
    """Save to JSON"""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved: {path}")

# ======================================================================
# CLI
# ======================================================================

async def main():
    import argparse

    p = argparse.ArgumentParser(description="Alpha Analyst Evaluation")
    p.add_argument("ticker", nargs="?", help="Ticker (or use --all)")
    p.add_argument("--all", action="store_true", help="Run all test tickers")
    p.add_argument("--full", action="store_true", help="Include Ragas/DeepEval")
    p.add_argument("-o", "--output", help="Save JSON to file")

    args = p.parse_args()

    print("\n🎯 Alpha Analyst Evaluation\n")

    # Check what's available
    try:
        from ragas import evaluate
        print("  Ragas: ✅")
    except ImportError:
        print("  Ragas: ❌ (pip install ragas)")

    try:
        from deepeval import evaluate
        print("  DeepEval: ✅")
    except ImportError:
        print("  DeepEval: ❌ (pip install deepeval)")

    print()

    if args.all:
        results = await evaluate_all(run_full=args.full)
    elif args.ticker:
        results = [await evaluate_ticker(args.ticker.upper(), args.full)]
    else:
        p.print_help()
        return

    print_results(results)

    if args.output:
        save_json(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
