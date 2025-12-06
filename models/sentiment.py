"""
This module provides financial sentiment analysis using FinBERT model.

FinBERT Overview:
-----------------------
    - FinBERT is a pre-trained sentiment analysis model for financial text. It classifies text into:
        - Positive: Bullish sentiment, good news, growth indicators
        - Negative: Bearish sentiment, bad news, risk indicators
        - Neutral: Factual statements, no clear sentiment
    - Why i used FinBert over Other sentiment models?
        - "Revenue declined 5%" -> Negative
        - "The company cut costs" -> Could be positive or negative depending on the prior context. Company investing in new Tech for which it may need to cut costs. or it is a sign of a company in trouble.
        - "Volatility increased" -> Neutral/Negative in finance, but general models might mis-classify this.

Model: ProsusAI/finbert (HuggingFace)

Usage:
----------------------
    from models.sentiment import analyze_sentiment, SentimentAnalyzer

    # Single text analysis
    result = analyze_sentiment("Apple reported record earning, beating expectations by 10%")
    print(f"Sentiment: {result.label} ({result.confidence:.2%})")

    # Batch analysis
    texts = ["Stock surged 10%", "Company faces lawsuit", "Quarterly report released"]
    results = analyze_sentiment_batch(texts)

    # Analyze news articles
    from tools.web_search_tool import NewsArticle
    articles = [...] # from web search tool
    analyzed = analyze_articles(articles)
"""

import torch
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from loguru import logger

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import settings

# DATA Models

@dataclass
class SentimentResult:
    """
    Result of sentiment analysis on a single text.

    Attributes:
        text: The analyzed text
        label: Sentiment label (positive, negative, neutral)
        confidence: Confidence score
        scores: Dict of all label scorees
        positive_score: Score for positive sentiment
        negative_score: Score for negative sentiment
        neutral_score: Score for neutral sentiment
    """

    text: str
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def positive_score(self) -> float:
        """Score for positive sentiment"""
        return self.scores.get("positive", 0.0)

    @property
    def negative_score(self) -> float:
        """Score for negative sentiment"""
        return self.scores.get("negative", 0.0)

    @property
    def neutral_score(self) -> float:
        """Score for neutral sentiment"""
        return self.scores.get("neutral", 0.0)

    @property
    def is_positive(self) -> bool:
        """Return if sentiment is positive"""
        return self.label == "positive"

    @property
    def is_negative(self) -> bool:
        """Return if sentiment is negative"""
        return self.label == "negative"

    @property
    def is_neutral(self) -> bool:
        """Return if sentiment is neutral"""
        return self.label == "neutral"

    @property
    def text_preview(self) -> str:
        """Truncated text for display (100 chars)"""
        if len(self.text) <= 100:
            return self.text

        return self.text[:97] + "..."

    def to_dict(self) -> dict:
        """convert to dictionary for JSON serialization"""
        return {
            "text": self.text_preview,
            "label": self.label,
            "confidence": self.confidence,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }

    def __str__(self) -> str:
        """Human-readable representation"""
        return f"sentiment: {self.label.upper()} ({self.confidence:.1%}): {self.text_preview}"

@dataclass
class ArticleSentiment:
    """
    Sentiment analysis result for a news article.

    Combines the article metadata with sentiment analysis.
    Used for the agent's sentiment analysis step.

    Attributes:
        title: Article title
        url: Article URL (REQUIRED FOR CITATION for human intervention)
        source: News source
        published_date: Publication date
        sentiment: SentimentResult for the article
        snippet: Original article snippet
    """

    title: str
    url: str
    source: str
    sentiment: SentimentResult
    published_date: Optional[str] = None
    snippet: Optional[str] = None
    
    def to_citation(self, index: int) -> str:
        """Format as citation with Sentiment"""
        emoji = {"positive": "up-arrow", "negative": "down-arrow", "neutral": "side-arrow"}.get(self.sentiment.label, "question-mark")
        date_str = f", {self.published_date}" if self.published_date else ""
        return (
            f"[{index}] {emoji} {self.title} - {self.source}{date_str}\n"
            f"        Sentiment: {self.sentiment.label} ({self.sentiment.confidence:.1%})\n"
            f"        {self.url}"
        )

@dataclass
class SentimentSummary:
    """
    Aggregated sentiment summary across multiple texts/articles.

    Provides overall sentiment metrics for a collectioon of analyzed items.
    Used in the investment memo to summarize news sentiment.

    Attributes:
        total_count: Total number of items analyzed
        positive_count: Number of positive items
        negative_count: Number of negative items
        neutral_count: Number of neutral items
        average_positive_score: Mean positive socre across all items
        average_negative_score: Mean negative score across all items
        results: Individual sentiment results
        overall_sentiment: Derived overall sentiment
        timestamp: When analysis waas performed
    """

    total_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_positive_score: float
    average_negative_score: float
    results: list[SentimentResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def overall_sentiment(self) -> str:
        """
        Derive overall sentiment from the results.

            - if positive > negative by 2+ items -> positive
            - if negative > positive by 2+ items -> negative
            - Otherwise -> neutral/mixed
        """
        diff = self.positive_count - self.negative_count

        if diff >= 2:
            return "positive"
        elif diff <= -2:
            return "negative"
        elif self.positive_count > self.negative_count:
            return 'slightly positive'
        elif self.negative_count > self.positive_count:
            return "slightly negative"
        else:
            return "neutral"

    @property
    def sentiment_ratio(self) -> str:
        """Ration of positive to negative sentiment"""
        if self.negative_count == 0:
            if self.negative_count > 0:
                return "All positive"
            return "All Neutral"

        return f"{self.positive_count}:{self.negative_count} (Positive to Negative Ratio)"

    def to_summary(self) -> str:
        """
        Format as human-readable summary for the investment memo.

        Returns:
            Multi-line summary string
        """
        emoji = {
            "positive": "up-arrow",
            "negative": "down-arrow",
            "neutral": "side-arrow",
            "slightly positive": "up-arrow",
            "slightly negative": "down-arrow",
        }.get(self.overall_sentiment, "question-mark")

        return f"""{emoji} Overall Sentiment: {self.overall_sentiment.upper()}
        
        Distribution:
            - Positive: {self.positive_count} ({self.positive_count/self.total_count:.0%})
            - Negative: {self.negative_count} ({self.negative_count/self.total_count:.0%})
            - Neutral: {self.neutral_count} ({self.neutral_count/self.total_count:.0%})

        Average Scores:
            - Positive Score: {self.average_positive_score:.2%}
            - Negative Score: {self.average_negative_score:.2%}

        Analyzed {self.total_count} items.
        """

    def to_context(self) -> str:
        """Format for LLM context"""
        return f"""News Sentiment Analysis:
        
        Overall: {self.overall_sentiment}
        Distribution: {self.positive_count} positive, {self.negative_count} negative, {self.neutral_count} neutral
        Ratio: {self.sentiment_ratio}
        Average Positive Score: {self.average_positive_score:.2%}
        Average Negative Score: {self.average_negative_score:.2%}
        """

# Sentiment Analyzer

class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT model.

    This class handles:
        - Model loading and device management
        - Text tokenization and single/batch processing
        - Result formatting
    
    The model is loaded lazily on first use to avoid slow imports

    Example:
    -------------
        analyzer = SentimentAnalyzer()

        # Single analysis
        result = analyzer.analyze("Stock prices surged on earning beat")

        # Batch analysis
        results = analyzer.analyze_batch([
            "Revenue grew 20% year-over-year",
            "Company announces layoffs",
            "Quarterly report released today",
        ])

        # Get summary statistics
        summary = analyzer.summarize(reesults)
    """

    LABELS = ["positive", "negative", "neutral"]

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name. Defaults to settings.finbert.model_name.
            device: Device to use for inference. Auto-detected if None.
        """
        self.model_name = model_name or settings.finbert.model_name

        # set device
        if device:
            self.device = device
        elif settings.finbert.device:
            self.device = settings.finbert.device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Lazy loaded model and tokenizer
        self._model = None
        self._tokenizer = None

        logger.info(
            f"SentimentAnalyzer initialized "
            f"(model={self.model_name}, device={self.device})"
        )

    @property
    def model(self):
        """Lazy-load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer on first access."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """
        Load the FinBERT model and tokenizer.

        This is called lazily on first access to avoid slow imports.
        """

        logger.info(f"Loading FinBERT model: {self.model_name}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Move to device
            self._model = self._model.to(self.device)
        
            self._model.eval()

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load FinBERT model: {e}")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            result = analyzer.analyze("Apple Stock surges on strong iPhone sales")
            print(f"{result.label}: {result.confidence:.2%}")
        """

        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently.

        Batching better uses GPU parallelism.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once.

        Returns:
            List of SentimentResult objects

        Example:
            texts = ["Good news", "Bad news", "No news"]
            results = analyzer.analyze_batch(texts)
            for text, result in zip(texts, results):
                print(f"{text}: {result.label}")
        """
        if not texts:
            return []

        logger.info(f"Analyzing {len(texts)} texts in batches of {batch_size}")

        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self._analyze_batch_internal(batch_texts)
            all_results.extend(batch_results)

        return all_results

    def _analyze_batch_internal(self, texts: list[str]) -> list[SentimentResult]:
        """
        Internal method to analyze a single batch.

        Args:
            texts: Batch of texts to analyze

        Returns:
            List of SentimentResult objects
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,     # BERT's max sequence length
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self._model(**inputs)

            # Getting probabilities via softmax
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert to results
        results = []
        probs_cpu = probs.cpu().numpy()

        for idx, text in enumerate(texts):
            scores_arr = probs_cpu[idx]

            # Create scores dict
            scores = {
                label: float(scores_arr[i])
                for i, label in enumerate(self.LABELS)
            }

            # Get predicted label
            predicted_idx = scores_arr.argmax()
            label = self.LABELS[predicted_idx]
            confidence = float(scores_arr[predicted_idx])

            results.append(SentimentResult(
                text=text,
                label=label,
                confidence=confidence,
                scores=scores,
            ))

        return results

    def summarize(self, results: list[SentimentResult]) -> SentimentSummary:
        """
        Create summary statistics from multiple sentiment results.

        Args:
            results: List of SentimentResult objects

        Returns:
            SentimentSummary with aggregate statistics
        """
        if not results:
            return SentimentSummary(
                total_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                average_positive_score=0.0,
                average_negative_score=0.0,
                results=[],
            )

        positive_count = sum(1 for result in results if result.is_positive)
        negative_count = sum(1 for result in results if result.is_negative)
        neutral_count = sum(1 for result in results if result.is_neutral)

        avg_positive = sum(result.positive_score for result in results) / len(results)
        avg_negative = sum(result.negative_score for result in results) / len(results)

        return SentimentSummary(
            total_count=len(results),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            average_positive_score=avg_positive,
            average_negative_score=avg_negative,
            results=results,
        )

# Wrapper Convenience Functions

_default_analyzer: Optional[SentimentAnalyzer] = None

def _get_default_analyzer() -> SentimentAnalyzer:
    """Get default analyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SentimentAnalyzer()
    return _default_analyzer

def analyze_sentiment(text: str) -> SentimentResult:
    """
    Analyze sentiment of a single text. (wrapper convenience function)

    Args:
        text: Text to analyze

    Returns:
        SentimentResult with label and scores

    Example:
        result = analyze_sentiment("Stock prices fell sharply")
        print(f"Sentiment: {result.label}")
    """
    return _get_default_analyzer().analyze(text)

def analyze_sentiment_batch(texts: list[str]) -> list[SentimentResult]:
    """
    Analyze sentiment of list of texts. (Wrapper convenience function)

    Args:
        texts: List of texts to analyze.

    Returns:
        List of SentimentResult objects.
    """
    return _get_default_analyzer().analyze_batch(texts)

def analyze_articles(articles: list) -> list[ArticleSentiment]:
    """
    Analyze sentiment of news articles.

    This class integrates with NewsArticle objects from the web search tool.

    Args:
        articles: List of NewsArticle objects

    Returns:
        List of ArticleSentiment objects with sentiment analysis.

    Example:
        from tools.web_search_tool import NewsArticle

        search_results = search_news("Tesla earnings")
        analyzed = analyze_articles(search_results.articles)

        for article in analyzed:
            print(article.to_citation(1))
    """
    if not articles:
        return []

    analyzer = _get_default_analyzer()

    # Extract snippets for analysis
    texts = [article.snippet for article in articles]

    # Batch analyze
    results = analyzer.analyze_batch(texts)

    # Combine with article metadata
    analyzed = []
    for article, sentiment in zip(articles, results):
        analyzed.append(ArticleSentiment(
            title=article.title,
            url=article.url,
            source=article.source,
            published_date=article.published_date,
            snippet=article.snippet,
            sentiment=sentiment,
        ))

    return analyzed

def get_sentiment_summary(texts: list[str]) -> SentimentSummary:
    """
    Get aggregated sentiment summary for multiple texts.

    Args:
        texts: List of texts to analyze.

    Returns:
        SentimentSummary with statistics.
    """
    analyzer = _get_default_analyzer()
    results = analyzer.analyze_batch(texts)
    return analyzer.summarize(results)

# Testing
def _main():
    """Test the sentiment analysis module."""

    import sys

    print(f"\n{'='*60}")
    print("Sentiment Analysis Module - Testing")
    print(f"{'='*60}\n")

    # Initialize analyzer
    print("1. Loading FinBERT model ....")
    analyzer = SentimentAnalyzer()

    # Force load the model
    _ = analyzer.model
    print(f"      Model loaded on {analyzer.device}")

    # Test single analysis
    print("2. Analyzing Text analysis....")
    test_texts = [
        "Apple reported record quarterly earnings, beating analyst eexpectations.",
        "The company announced massive layoffs affecting 10,000 employees.",
        "The quarterly report will be released next Tuesday.",
        "Stock prices plummeted after the CEO resigned unexpectedly.",
        "Revenue grew 25% year-over-year, driven by strong iPhone sales.",
    ]

    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"    {result}")

    print(f"\n{'='*60}")

    # Test batch analysis
    print("3. Batch Analysis & summary....")
    results = analyzer.analyze_batch(test_texts)
    summary = analyzer.summarize(results)

    print(f"\n{summary.to_summary()}")

    # Test with custom input
    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        print(f"\n4. Custom Input: {custom_text}")
        print(f"    Input: {custom_text}")
        result = analyzer.analyze(custom_text)
        print(f"    {result}")

    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    _main()