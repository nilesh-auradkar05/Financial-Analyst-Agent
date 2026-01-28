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

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from langsmith import traceable
from loguru import logger
from transformers import PreTrainedTokenizerBase

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
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
    ):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name. Defaults to settings.finbert.model_name.
            device: Device to use for inference. Auto-detected if None.
        """
        self._model = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.model_name = model_name or settings.finbert.model_name

        # set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def _load_model(self):
        """Load model and tokenizer on first use."""

        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading {self.model_name} on {self.device}...")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        logger.info(
            f"SentimentAnalyzer initialized "
            f"(model={self.model_name}, device={self.device})"
        )

    @traceable(name="sentiment_analyze_batch", run_type="chain", tags=["sentiment"])
    def analyze_batch(self, texts: list[str], batch_size: int = 16) -> list[SentimentResult]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing (affects memory usage)

        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []

        import torch
        self._load_model()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Sentiment model failed to load.")

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            # Build results
            for j, text in enumerate(batch_texts):
                scores = probs[j].cpu().numpy()
                label_idx = int(scores.argmax())

                results.append(SentimentResult(
                    text=text,
                    label=self.LABELS[label_idx],
                    confidence=float(scores[label_idx]),
                    scores={
                        "positive": float(scores[0]),
                        "negative": float(scores[1]),
                        "neutral": float(scores[2]),
                    },
                ))

        logger.info(f"Analyzed {len(texts)} texts")
        return results

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        results = self.analyze_batch([text])
        return results[0] if results else SentimentResult(
            text=text, label="neutral", confidence=0.0, scores={}
        )

# Wrapper Convenience Functions

_default_analyzer: Optional[SentimentAnalyzer] = None

def _get_default_analyzer() -> SentimentAnalyzer:
    """Get default analyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SentimentAnalyzer()
    return _default_analyzer

@traceable(name="analyze_sentiment_batch", run_type="chain", tags=["sentiment"])
def analyze_sentiment_batch(texts: list[str]) -> list[SentimentResult]:
    """
    Analyze sentiment of list of texts. (Wrapper convenience function)

    Args:
        texts: List of texts to analyze.

    Returns:
        List of SentimentResult objects.
    """
    return _get_default_analyzer().analyze_batch(texts)


# Testing
def _main():
    """Test the sentiment analysis module."""

    import sys

    texts = sys.argv[1:] if len(sys.argv) > 1 else [
        "Apple reported record-breaking quarterly revenue, beating analyst expectations.",
        "The company faces significant regulatory challenges and potential fines.",
        "Stock price remained stable amid market volatility.",
    ]

    print(f"\nAnalyzing {len(texts)} texts...\n")

    results = analyze_sentiment_batch(texts)

    for result in results:
        emoji = "📈" if result.is_positive else "📉" if result.is_negative else "➡️"
        print(f"{emoji} {result.label.upper()} ({result.confidence:.0%})")
        print(f"   {result.text[:80]}...")
        print()

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    _main()
