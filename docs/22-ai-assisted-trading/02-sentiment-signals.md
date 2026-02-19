---
layout: default
title: "Sentiment & Signals"
parent: "Part 22: AI-Assisted Trading"
nav_order: 2
---

# Sentiment & Signals

## Overview

Raw news text is not directly useful for a trading system. This section covers the pipeline that transforms unstructured financial text into quantitative trading signals: first extracting sentiment with `analyze_sentiment`, then aggregating across articles with `batch_sentiment` and time-decay weighting, and finally converting scores into filtered signals via `NewsSignalGenerator`.

The key modules are:

- `puffin.ai.sentiment` -- single-text and batch sentiment analysis
- `puffin.ai.signals` -- the `NewsSignalGenerator` that produces a signals DataFrame
- `puffin.ai.news` -- RSS feed fetcher for sourcing articles

## Single-Article Sentiment Analysis

The `analyze_sentiment` function sends a financial text to your LLM provider with a structured prompt that requests JSON output:

```python
from puffin.ai.providers import ClaudeProvider
from puffin.ai.sentiment import analyze_sentiment

provider = ClaudeProvider()

article = """
Tesla shares surged 12% after the company announced record quarterly
deliveries, beating analyst expectations. The strong demand signals a
rebound in the EV market.
"""

sentiment = analyze_sentiment(provider, article)

print(sentiment)
# {
#     "sentiment": "bullish",
#     "score": 0.75,         # -1.0 (very bearish) to 1.0 (very bullish)
#     "confidence": 0.85,    # 0.0 to 1.0
#     "reasoning": "Record deliveries and strong demand signal positive momentum",
#     "tickers_mentioned": ["TSLA"],
#     "model": "claude-sonnet-4-5-20250929"
# }
```

The function uses a carefully crafted `SENTIMENT_PROMPT` defined in `puffin.ai.sentiment` that instructs the LLM to respond with valid JSON containing five fields: `sentiment`, `score`, `confidence`, `reasoning`, and `tickers_mentioned`.

{: .note }
If the LLM returns malformed JSON, `analyze_sentiment` attempts to extract JSON from the response by locating the first `{` and last `}`. If that also fails, it returns a neutral sentiment with zero confidence.

### How the Prompt Works

The prompt from the source code:

```python
from puffin.ai.sentiment import SENTIMENT_PROMPT

print(SENTIMENT_PROMPT)
# Analyze the following financial news article and provide a sentiment assessment.
# Respond in JSON format with these fields:
# - sentiment: "bullish", "bearish", or "neutral"
# - score: float from -1.0 (very bearish) to 1.0 (very bullish)
# - confidence: float from 0.0 to 1.0
# - reasoning: brief explanation (1-2 sentences)
# - tickers_mentioned: list of stock ticker symbols mentioned
#
# Respond ONLY with valid JSON, no other text.
```

The `SENTIMENT_PROMPT` is passed to `provider.analyze()` as the `prompt` argument, while the article text goes into the `text` argument. The provider combines them into a single LLM message.

## Batch Sentiment with Time Decay

When analyzing multiple articles about the same tickers, newer articles should carry more weight. The `batch_sentiment` function handles this with exponential time-decay weighting:

```python
from puffin.ai.sentiment import batch_sentiment

articles = [
    {
        "text": "Tesla announces new Gigafactory...",
        "timestamp": "2024-01-15T10:00:00"
    },
    {
        "text": "Tesla faces regulatory scrutiny...",
        "timestamp": "2024-01-14T15:30:00"
    },
    {
        "text": "EV market shows strong growth...",
        "timestamp": "2024-01-13T09:00:00"
    },
]

# Newer articles weighted more heavily (time_decay=0.95)
aggregated = batch_sentiment(provider, articles, time_decay=0.95)

print(aggregated)
# {
#     "TSLA": {
#         "score": 0.35,        # Weighted average across articles
#         "confidence": 0.72,
#         "n_articles": 3,
#         "signal": "bullish"   # "bullish" | "bearish" | "neutral"
#     },
#     ...
# }
```

The `time_decay` parameter (0 to 1) exponentially reduces the weight of older articles. A value of 0.95 means each older article has 95% the weight of the next newer one. The function:

1. Calls `analyze_sentiment` on each article individually
2. Groups results by ticker (from `tickers_mentioned`)
3. Applies exponential decay weights based on article order
4. Computes weighted-average score and confidence per ticker
5. Assigns a signal label: `"bullish"` if score > 0.3, `"bearish"` if score < -0.3, otherwise `"neutral"`

{: .tip }
Use `time_decay=1.0` for equal weighting across all articles, or lower values like `0.8` when you want to strongly favor the most recent news.

## Fetching News from RSS Feeds

The `puffin.ai.news` module provides `fetch_rss_news` to pull articles from financial RSS feeds:

```python
from puffin.ai.news import fetch_rss_news

# Fetch from default feeds (Yahoo Finance, CNBC)
articles = fetch_rss_news(max_articles=20)

# Or specify custom feeds
articles = fetch_rss_news(
    feeds=[
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    ],
    max_articles=10,
)

# Each article has: text, title, source, timestamp, link
for article in articles[:3]:
    print(f"[{article['source']}] {article['title']}")
```

The fetcher combines each article's title and summary into the `text` field, which is what `analyze_sentiment` expects.

## News-Driven Trading Signals

The `NewsSignalGenerator` converts batch sentiment into concrete trading signals with configurable thresholds:

```python
from puffin.ai.signals import NewsSignalGenerator

generator = NewsSignalGenerator(
    provider=provider,
    bullish_threshold=0.3,   # Score > 0.3 triggers a buy signal
    bearish_threshold=-0.3,  # Score < -0.3 triggers a sell signal
    min_confidence=0.5,      # Require at least 50% confidence
    min_articles=2,          # Require at least 2 articles per ticker
)

# Generate signals from a batch of articles
signals = generator.generate(articles)

print(signals)
#           signal  score  confidence  n_articles
# ticker
# TSLA        0.75   0.75        0.85           3
# AAPL        0.00   0.15        0.60           1  # Below min_articles
# NVDA       -0.40  -0.40        0.70           2
```

The `signal` column ranges from -1.0 (strong sell) to +1.0 (strong buy), or 0.0 for neutral/insufficient data. The generator applies four filters in order:

1. **Minimum articles** -- tickers with fewer than `min_articles` get signal 0.0
2. **Minimum confidence** -- tickers below `min_confidence` get signal 0.0
3. **Bullish threshold** -- scores above the threshold pass through (clamped to 1.0)
4. **Bearish threshold** -- scores below the threshold pass through (clamped to -1.0)

{: .warning }
Signals based on a single article are unreliable. The `min_articles` filter exists for good reason -- always require at least 2-3 corroborating articles before acting on a signal.

## Integrating with Your Strategy

Plug the signal generator into a Puffin `Strategy` subclass to create a news-driven trading system:

```python
from puffin.strategies import Strategy
from puffin.ai.signals import NewsSignalGenerator

class NewsSentimentStrategy(Strategy):
    """Trading strategy driven by news sentiment signals."""

    def __init__(self, signal_generator: NewsSignalGenerator):
        self.signal_generator = signal_generator
        self.news_cache = []

    def on_news(self, article: dict):
        """Called when a new article arrives from the news feed."""
        self.news_cache.append(article)

    def generate_signals(self, data, timestamp):
        """Generate trading signals from accumulated news articles."""
        if not self.news_cache:
            return {}

        # Generate sentiment-based signals
        signals_df = self.signal_generator.generate(self.news_cache)

        # Clear cache after processing
        self.news_cache = []

        # Convert DataFrame to dict for the backtester
        return signals_df["signal"].to_dict()

    def get_parameters(self):
        return {
            "bullish_threshold": self.signal_generator.bullish_threshold,
            "bearish_threshold": self.signal_generator.bearish_threshold,
            "min_confidence": self.signal_generator.min_confidence,
            "min_articles": self.signal_generator.min_articles,
        }
```

### Combining Sentiment with Technical Signals

News sentiment works best as a confirmation overlay on top of technical or factor-based signals:

```python
def combined_signal(technical_signal, sentiment_signal, sentiment_weight=0.3):
    """Blend a technical signal with a sentiment signal.

    Args:
        technical_signal: Signal from technical analysis (-1.0 to 1.0).
        sentiment_signal: Signal from news sentiment (-1.0 to 1.0).
        sentiment_weight: Weight given to sentiment (0.0 to 1.0).

    Returns:
        Blended signal value.
    """
    tech_weight = 1.0 - sentiment_weight
    blended = tech_weight * technical_signal + sentiment_weight * sentiment_signal
    return max(-1.0, min(1.0, blended))
```

{: .tip }
A sentiment weight of 0.2 to 0.3 is a good starting point. Sentiment is a useful confirming signal but should not dominate your allocation decisions.

## Tuning Signal Parameters

The thresholds in `NewsSignalGenerator` directly control your signal sensitivity:

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `bullish_threshold` | More buy signals (noisier) | Fewer buy signals (higher conviction) |
| `bearish_threshold` | More sell signals (noisier) | Fewer sell signals (higher conviction) |
| `min_confidence` | Accept uncertain signals | Only high-confidence signals |
| `min_articles` | React to single articles | Require corroboration |

A conservative configuration for live trading:

```python
generator = NewsSignalGenerator(
    provider=provider,
    bullish_threshold=0.5,
    bearish_threshold=-0.5,
    min_confidence=0.7,
    min_articles=3,
)
```

## Source Code

- Sentiment analysis: [`puffin/ai/sentiment.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/sentiment.py)
- Signal generator: [`puffin/ai/signals.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/signals.py)
- News fetcher: [`puffin/ai/news.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/news.py)

## Next Steps

Now that you can generate signals from news, proceed to [AI Agent Portfolio Manager](03-ai-agent) to let an AI agent recommend portfolio allocation changes based on these signals, positions, and market data.
