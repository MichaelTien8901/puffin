"""LLM-powered sentiment analysis for financial text."""

import json
from datetime import datetime

import numpy as np
import pandas as pd

from puffin.ai.llm_provider import LLMProvider


SENTIMENT_PROMPT = """Analyze the following financial news article and provide a sentiment assessment.
Respond in JSON format with these fields:
- sentiment: "bullish", "bearish", or "neutral"
- score: float from -1.0 (very bearish) to 1.0 (very bullish)
- confidence: float from 0.0 to 1.0
- reasoning: brief explanation (1-2 sentences)
- tickers_mentioned: list of stock ticker symbols mentioned

Respond ONLY with valid JSON, no other text."""


def analyze_sentiment(provider: LLMProvider, text: str) -> dict:
    """Analyze sentiment of a single text.

    Args:
        provider: LLM provider instance.
        text: Financial news text to analyze.

    Returns:
        Dict with sentiment, score, confidence, reasoning, tickers_mentioned.
    """
    result = provider.analyze(text, SENTIMENT_PROMPT)
    try:
        parsed = json.loads(result["response"])
    except json.JSONDecodeError:
        # Try to extract JSON from response
        response = result["response"]
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
        else:
            parsed = {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "reasoning": "Failed to parse LLM response",
                "tickers_mentioned": [],
            }

    parsed["model"] = result.get("model", "unknown")
    return parsed


def batch_sentiment(
    provider: LLMProvider,
    articles: list[dict],
    time_decay: float = 0.95,
) -> dict:
    """Analyze sentiment for a batch of articles with time-weighted aggregation.

    Args:
        provider: LLM provider instance.
        articles: List of dicts with 'text' and optional 'timestamp' keys.
        time_decay: Decay factor for older articles (0-1).

    Returns:
        Dict with aggregated sentiment per ticker.
    """
    results = []
    for article in articles:
        sentiment = analyze_sentiment(provider, article["text"])
        sentiment["timestamp"] = article.get("timestamp", datetime.now().isoformat())
        results.append(sentiment)

    # Aggregate by ticker
    ticker_sentiments: dict[str, list] = {}
    for i, result in enumerate(results):
        weight = time_decay ** (len(results) - 1 - i)
        for ticker in result.get("tickers_mentioned", []):
            if ticker not in ticker_sentiments:
                ticker_sentiments[ticker] = []
            ticker_sentiments[ticker].append({
                "score": result["score"] * weight,
                "confidence": result["confidence"] * weight,
                "weight": weight,
            })

    aggregated = {}
    for ticker, sentiments in ticker_sentiments.items():
        total_weight = sum(s["weight"] for s in sentiments)
        if total_weight > 0:
            avg_score = sum(s["score"] for s in sentiments) / total_weight
            avg_confidence = sum(s["confidence"] for s in sentiments) / total_weight
        else:
            avg_score = 0.0
            avg_confidence = 0.0

        aggregated[ticker] = {
            "score": avg_score,
            "confidence": avg_confidence,
            "n_articles": len(sentiments),
            "signal": "bullish" if avg_score > 0.3 else "bearish" if avg_score < -0.3 else "neutral",
        }

    return aggregated
