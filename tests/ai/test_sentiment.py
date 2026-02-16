"""Tests for AI sentiment analysis (with mocked LLM responses)."""

import json
import pytest

from puffin.ai.llm_provider import LLMProvider
from puffin.ai.sentiment import analyze_sentiment, batch_sentiment
from puffin.ai.signals import NewsSignalGenerator


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, response: str = ""):
        super().__init__(cache_dir="/tmp/puffin_test_cache")
        self.response = response
        self.call_count = 0

    def analyze(self, text: str, prompt: str) -> dict:
        self.call_count += 1
        return {"response": self.response, "model": "mock", "usage": {}}

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.call_count += 1
        return self.response


class TestSentimentAnalysis:
    def test_bullish_sentiment(self):
        response = json.dumps({
            "sentiment": "bullish",
            "score": 0.8,
            "confidence": 0.9,
            "reasoning": "Strong earnings beat",
            "tickers_mentioned": ["AAPL"],
        })
        provider = MockProvider(response=response)
        result = analyze_sentiment(provider, "Apple reports record earnings")
        assert result["sentiment"] == "bullish"
        assert result["score"] == 0.8
        assert "AAPL" in result["tickers_mentioned"]

    def test_bearish_sentiment(self):
        response = json.dumps({
            "sentiment": "bearish",
            "score": -0.6,
            "confidence": 0.7,
            "reasoning": "Revenue decline",
            "tickers_mentioned": ["TSLA"],
        })
        provider = MockProvider(response=response)
        result = analyze_sentiment(provider, "Tesla misses revenue estimates")
        assert result["sentiment"] == "bearish"
        assert result["score"] < 0

    def test_batch_sentiment(self):
        response = json.dumps({
            "sentiment": "bullish",
            "score": 0.5,
            "confidence": 0.8,
            "reasoning": "Positive outlook",
            "tickers_mentioned": ["AAPL"],
        })
        provider = MockProvider(response=response)
        articles = [
            {"text": "Apple launches new product"},
            {"text": "Apple revenue grows 20%"},
        ]
        result = batch_sentiment(provider, articles)
        assert "AAPL" in result
        assert result["AAPL"]["n_articles"] == 2


class TestNewsSignalGenerator:
    def test_generates_signals(self):
        response = json.dumps({
            "sentiment": "bullish",
            "score": 0.7,
            "confidence": 0.8,
            "reasoning": "Strong growth",
            "tickers_mentioned": ["MSFT"],
        })
        provider = MockProvider(response=response)
        generator = NewsSignalGenerator(provider, min_articles=1)
        articles = [
            {"text": "Microsoft cloud revenue surges"},
            {"text": "Microsoft beats expectations"},
        ]
        signals = generator.generate(articles)
        assert len(signals) > 0
        assert "signal" in signals.columns

    def test_min_articles_filter(self):
        response = json.dumps({
            "sentiment": "bullish",
            "score": 0.7,
            "confidence": 0.8,
            "reasoning": "Good news",
            "tickers_mentioned": ["NVDA"],
        })
        provider = MockProvider(response=response)
        generator = NewsSignalGenerator(provider, min_articles=5)
        articles = [{"text": "NVIDIA stock rises"}]
        signals = generator.generate(articles)
        # Should have zero signal due to min_articles filter
        if len(signals) > 0:
            assert signals.loc["NVDA", "signal"] == 0.0
