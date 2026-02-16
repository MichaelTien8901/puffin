"""News-driven trading signal generator."""

import pandas as pd

from puffin.ai.llm_provider import LLMProvider
from puffin.ai.sentiment import batch_sentiment


class NewsSignalGenerator:
    """Generate trading signals from news sentiment."""

    def __init__(
        self,
        provider: LLMProvider,
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
        min_confidence: float = 0.5,
        min_articles: int = 2,
    ):
        self.provider = provider
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.min_confidence = min_confidence
        self.min_articles = min_articles

    def generate(self, articles: list[dict]) -> pd.DataFrame:
        """Generate trading signals from a batch of articles.

        Args:
            articles: List of dicts with 'text' and optional 'timestamp'.

        Returns:
            DataFrame with columns: ticker, signal, score, confidence, n_articles.
        """
        sentiment = batch_sentiment(self.provider, articles)

        records = []
        for ticker, data in sentiment.items():
            if data["n_articles"] < self.min_articles:
                signal = 0.0
            elif data["confidence"] < self.min_confidence:
                signal = 0.0
            elif data["score"] >= self.bullish_threshold:
                signal = min(data["score"], 1.0)
            elif data["score"] <= self.bearish_threshold:
                signal = max(data["score"], -1.0)
            else:
                signal = 0.0

            records.append({
                "ticker": ticker,
                "signal": signal,
                "score": data["score"],
                "confidence": data["confidence"],
                "n_articles": data["n_articles"],
            })

        return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()
