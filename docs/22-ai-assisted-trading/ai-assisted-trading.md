---
layout: default
title: "Part 22: AI-Assisted Trading"
nav_order: 23
permalink: /22-ai-assisted-trading/
---

# AI-Assisted Trading

## Overview

Large Language Models (LLMs) like Claude and GPT-4 bring powerful new capabilities to algorithmic trading: analyzing sentiment in news articles, generating portfolio recommendations, and creating comprehensive market reports. In this chapter, you'll learn how to integrate LLMs into your trading system using Puffin's AI module.

The `puffin.ai` module provides:
- **LLM provider abstraction**: Switch between Claude, OpenAI, or custom providers
- **Sentiment analysis**: Extract bullish/bearish signals from news text
- **News-driven signals**: Generate trading signals from news sentiment
- **AI agent portfolio management**: Let an AI agent recommend portfolio allocation
- **Market report generation**: Automated analysis reports

## LLM Provider Abstraction

Puffin uses an abstract `LLMProvider` interface so you can swap between Claude, OpenAI, or any custom provider without changing your trading code:

```python
from puffin.ai import LLMProvider, ClaudeProvider, OpenAIProvider

# Abstract interface
class LLMProvider(ABC):
    @abstractmethod
    def analyze(self, text: str, prompt: str) -> dict:
        """Analyze text with a specific prompt.

        Returns:
            Dict with 'response', 'model', 'usage' keys.
        """

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a prompt."""
```

### Using ClaudeProvider

Claude excels at nuanced financial analysis and structured JSON output:

```python
from puffin.ai import ClaudeProvider

# Initialize with API key (or use ANTHROPIC_API_KEY env var)
provider = ClaudeProvider(
    api_key="your-api-key",
    model="claude-sonnet-4-5-20250929",  # Default
    cache_dir=".cache/llm",  # Response caching
    cache_ttl=3600,  # 1 hour cache TTL
)

# Analyze text
result = provider.analyze(
    text="Apple reported Q4 earnings beating estimates by 15%...",
    prompt="Extract sentiment and key financial metrics from this earnings report."
)

print(result["response"])
print(f"Model: {result['model']}")
print(f"Tokens: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")
```

### Using OpenAIProvider

OpenAI's GPT models are also supported:

```python
from puffin.ai import OpenAIProvider

provider = OpenAIProvider(
    api_key="your-api-key",
    model="gpt-4o",  # Default
)

result = provider.generate(
    prompt="Summarize the key technical levels for SPY right now.",
    system="You are a technical analysis expert."
)
print(result)
```

{: .note }
All providers cache responses to minimize API costs. The cache key is based on the provider, model, text, and prompt.

## Sentiment Analysis Pipeline

The `analyze_sentiment` function uses an LLM to extract sentiment from financial news:

```python
from puffin.ai import ClaudeProvider, analyze_sentiment

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
#     "score": 0.75,  # -1.0 (bearish) to 1.0 (bullish)
#     "confidence": 0.85,  # 0.0 to 1.0
#     "reasoning": "Record deliveries and strong demand signal positive momentum",
#     "tickers_mentioned": ["TSLA"],
#     "model": "claude-sonnet-4-5-20250929"
# }
```

### Batch Sentiment with Time Decay

When analyzing multiple articles, use `batch_sentiment` to aggregate sentiment by ticker with time-weighted decay:

```python
from puffin.ai import batch_sentiment

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
#         "score": 0.35,  # Weighted average
#         "confidence": 0.72,
#         "n_articles": 3,
#         "signal": "bullish"  # "bullish" | "bearish" | "neutral"
#     },
#     ...
# }
```

The `time_decay` parameter (0-1) exponentially reduces the weight of older articles. A value of 0.95 means each older article has 95% the weight of the next newer one.

## News-Driven Trading Signals

The `NewsSignalGenerator` converts news sentiment into concrete trading signals:

```python
from puffin.ai import NewsSignalGenerator

generator = NewsSignalGenerator(
    provider=provider,
    bullish_threshold=0.3,  # Score > 0.3 → buy signal
    bearish_threshold=-0.3,  # Score < -0.3 → sell signal
    min_confidence=0.5,  # Require 50% confidence
    min_articles=2,  # Require at least 2 articles per ticker
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

The signal column ranges from -1.0 (strong sell) to +1.0 (strong buy), or 0.0 for neutral/insufficient data.

### Integrating with Your Strategy

```python
from puffin.strategies import Strategy

class NewsSentimentStrategy(Strategy):
    def __init__(self, signal_generator):
        self.signal_generator = signal_generator
        self.news_cache = []

    def on_news(self, article):
        """Called when new article arrives."""
        self.news_cache.append(article)

    def generate_signals(self, data, timestamp):
        """Generate trading signals."""
        if not self.news_cache:
            return {}

        # Generate sentiment signals
        signals_df = self.signal_generator.generate(self.news_cache)

        # Clear cache after processing
        self.news_cache = []

        # Convert to dict for backtester
        return signals_df["signal"].to_dict()
```

## AI Agent Portfolio Management

The `PortfolioAgent` is an AI agent that recommends portfolio allocation changes based on positions, market data, sentiment, and strategy signals:

```python
from puffin.ai import PortfolioAgent

agent = PortfolioAgent(provider)

# Current portfolio state
positions = {
    "AAPL": {"qty": 100, "avg_price": 175.0, "current_price": 182.5},
    "TSLA": {"qty": 50, "avg_price": 245.0, "current_price": 238.0},
}

# Recent market data
market_data = {
    "AAPL": {"price": 182.5, "change_pct": 2.3, "volume": 58_000_000},
    "TSLA": {"price": 238.0, "change_pct": -1.8, "volume": 125_000_000},
    "NVDA": {"price": 495.0, "change_pct": 5.2, "volume": 42_000_000},
}

# Optional: sentiment from news
sentiment = {
    "AAPL": {"score": 0.4, "confidence": 0.7},
    "TSLA": {"score": -0.3, "confidence": 0.6},
    "NVDA": {"score": 0.8, "confidence": 0.9},
}

# Optional: signals from your trading strategy
strategy_signals = {
    "AAPL": 0.2,
    "TSLA": -0.5,
    "NVDA": 0.9,
}

# Get AI recommendations
recommendations = agent.recommend(
    positions=positions,
    market_data=market_data,
    sentiment=sentiment,
    strategy_signals=strategy_signals,
)

print(recommendations)
# {
#     "recommendations": [
#         {
#             "ticker": "NVDA",
#             "action": "buy",
#             "target_allocation_pct": 25.0,
#             "reasoning": "Strong momentum and positive sentiment"
#         },
#         {
#             "ticker": "TSLA",
#             "action": "reduce",
#             "target_allocation_pct": 15.0,
#             "reasoning": "Negative sentiment and sell signal"
#         },
#         {
#             "ticker": "AAPL",
#             "action": "hold",
#             "target_allocation_pct": 30.0,
#             "reasoning": "Solid performance, maintain position"
#         }
#     ],
#     "market_outlook": "bullish",
#     "risk_assessment": "medium",
#     "summary": "Rotate from TSLA to NVDA given strong momentum"
# }
```

### Audit Trail

The agent maintains a decision log for compliance and debugging:

```python
# Get full decision history
log = agent.get_decision_log()

for entry in log:
    print(f"Time: {entry['timestamp']}")
    print(f"Inputs: {entry['inputs']}")
    print(f"Output: {entry['output']['summary']}")
    print()
```

### Using Agent Recommendations

```python
def execute_agent_recommendations(recommendations, broker, current_portfolio):
    """Convert agent recommendations to actual trades."""
    total_value = sum(p["qty"] * p["current_price"] for p in current_portfolio.values())

    for rec in recommendations["recommendations"]:
        ticker = rec["ticker"]
        target_pct = rec["target_allocation_pct"] / 100.0
        target_value = total_value * target_pct

        current_value = current_portfolio.get(ticker, {}).get("qty", 0) * \
                       current_portfolio.get(ticker, {}).get("current_price", 0)

        diff_value = target_value - current_value

        if abs(diff_value) > 100:  # Minimum trade size
            if diff_value > 0:
                # Buy
                qty = int(diff_value / market_data[ticker]["price"])
                broker.submit_order(ticker, "buy", qty)
            else:
                # Sell
                qty = int(abs(diff_value) / market_data[ticker]["price"])
                broker.submit_order(ticker, "sell", qty)
```

{: .warning }
Always review AI agent recommendations before executing trades. The agent is a tool to augment your decision-making, not replace it.

## Market Report Generation

Generate comprehensive market analysis reports with `generate_market_report`:

```python
from puffin.ai import generate_market_report

watchlist_data = {
    "AAPL": {
        "price": 182.5,
        "change_pct": 2.3,
        "high_52w": 199.6,
        "low_52w": 164.1,
        "rsi": 58.2,
        "macd_signal": 1,  # Bullish
    },
    "TSLA": {
        "price": 238.0,
        "change_pct": -1.8,
        "high_52w": 299.3,
        "low_52w": 152.4,
        "rsi": 42.1,
        "macd_signal": -1,  # Bearish
    },
}

sentiment = {
    "AAPL": {"score": 0.4, "confidence": 0.7, "signal": "bullish"},
    "TSLA": {"score": -0.3, "confidence": 0.6, "signal": "bearish"},
}

signals = {
    "AAPL": 0.2,
    "TSLA": -0.5,
}

report = generate_market_report(
    provider=provider,
    watchlist_data=watchlist_data,
    sentiment=sentiment,
    signals=signals,
)

print(report)
```

Example output:

```markdown
# Market Analysis Report

## AAPL - Apple Inc.
**Price Action**: Trading at $182.50, up 2.3% on the day. Currently in the
upper half of its 52-week range ($164.10 - $199.60), showing sustained strength.

**Technical Setup**: RSI at 58.2 indicates healthy momentum without being
overbought. MACD showing bullish crossover, confirming upward momentum.

**Sentiment & Signals**: Positive news sentiment (score: 0.40) with high
confidence. Strategy generating mild buy signal (0.2). Overall outlook: bullish.

**Key Levels**: Support at $175, resistance at $190.


## TSLA - Tesla Inc.
**Price Action**: Trading at $238.00, down 1.8%. In the middle of 52-week
range ($152.40 - $299.30), showing consolidation after recent volatility.

**Technical Setup**: RSI at 42.1 suggests some weakness but not oversold.
MACD bearish, indicating downward pressure.

**Sentiment & Signals**: Negative news sentiment (score: -0.30) with moderate
confidence. Strategy generating sell signal (-0.5). Overall outlook: bearish.

**Key Levels**: Support at $230, resistance at $250.


## Overall Market Assessment

Mixed signals with AAPL showing strength while TSLA faces headwinds. Favor
quality names with positive momentum over speculative positions. Monitor TSLA
for potential reversal if it holds $230 support.
```

## Complete Example: News-Driven Trading Bot

Here's a complete example that ties everything together:

```python
from datetime import datetime, timedelta
from puffin.ai import (
    ClaudeProvider,
    NewsSignalGenerator,
    PortfolioAgent,
    generate_market_report,
)
from puffin.data import NewsProvider

# Initialize components
llm_provider = ClaudeProvider()
news_provider = NewsProvider(api_key="your-news-api-key")
signal_generator = NewsSignalGenerator(llm_provider)
portfolio_agent = PortfolioAgent(llm_provider)

# Fetch recent news
articles = news_provider.fetch_news(
    symbols=["AAPL", "TSLA", "NVDA"],
    start_date=datetime.now() - timedelta(days=1),
)

# Generate sentiment-based signals
signals = signal_generator.generate(articles)
print("Sentiment Signals:")
print(signals)

# Get current portfolio state
positions = get_current_positions()  # Your broker API
market_data = get_market_data(["AAPL", "TSLA", "NVDA"])

# Get AI agent recommendations
recommendations = portfolio_agent.recommend(
    positions=positions,
    market_data=market_data,
    sentiment=signals[["score", "confidence"]].to_dict("index"),
    strategy_signals=signals["signal"].to_dict(),
)

print("\nAgent Recommendations:")
for rec in recommendations["recommendations"]:
    print(f"{rec['ticker']}: {rec['action']} → {rec['target_allocation_pct']}%")
    print(f"  Reason: {rec['reasoning']}")

# Generate daily report
report = generate_market_report(
    provider=llm_provider,
    watchlist_data=market_data,
    sentiment=signals.to_dict("index"),
)

# Save report
with open(f"reports/market_report_{datetime.now():%Y%m%d}.md", "w") as f:
    f.write(report)

print("\nReport saved!")
```

## Best Practices

### 1. Cost Management

LLM API calls can get expensive. Use caching aggressively:

```python
# Cache responses for 1 hour (default)
provider = ClaudeProvider(cache_ttl=3600)

# Longer cache for historical analysis
provider = ClaudeProvider(cache_ttl=86400)  # 24 hours
```

### 2. Rate Limiting

Avoid hitting API rate limits:

```python
import time

def analyze_articles_batch(provider, articles, batch_size=10, delay=1.0):
    """Process articles in batches with delays."""
    results = []
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        for article in batch:
            results.append(analyze_sentiment(provider, article["text"]))
        time.sleep(delay)  # Delay between batches
    return results
```

### 3. Fallback Providers

Switch to a backup provider if the primary fails:

```python
class RobustProvider:
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def analyze(self, text, prompt):
        try:
            return self.primary.analyze(text, prompt)
        except Exception as e:
            print(f"Primary failed: {e}. Using fallback.")
            return self.fallback.analyze(text, prompt)

provider = RobustProvider(
    primary=ClaudeProvider(),
    fallback=OpenAIProvider(),
)
```

### 4. Validation

Always validate LLM outputs before using them for trading:

```python
def validate_sentiment(sentiment):
    """Ensure sentiment output is valid."""
    required_keys = {"sentiment", "score", "confidence", "reasoning"}
    if not required_keys.issubset(sentiment.keys()):
        raise ValueError("Missing required sentiment fields")

    if not -1.0 <= sentiment["score"] <= 1.0:
        raise ValueError("Score out of range")

    if not 0.0 <= sentiment["confidence"] <= 1.0:
        raise ValueError("Confidence out of range")

    return sentiment
```

### 5. Human Oversight

Never fully automate trading based solely on LLM outputs:

```python
def execute_with_approval(recommendations, require_manual_approval=True):
    """Execute trades with optional manual approval."""
    if require_manual_approval:
        print("\nRecommendations:")
        for rec in recommendations["recommendations"]:
            print(f"{rec['ticker']}: {rec['action']}")

        approval = input("\nApprove? (yes/no): ")
        if approval.lower() != "yes":
            print("Aborted.")
            return

    # Execute trades...
```

## Performance Considerations

### Token Usage

Monitor your token consumption:

```python
total_tokens = 0

def track_usage(result):
    global total_tokens
    usage = result.get("usage", {})
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    total_tokens += tokens
    print(f"Tokens used: {tokens} (total: {total_tokens})")

result = provider.analyze(text, prompt)
track_usage(result)
```

### Response Time

LLM calls can be slow (1-5 seconds). Use async for better performance:

```python
import asyncio
from anthropic import AsyncAnthropic

class AsyncClaudeProvider(ClaudeProvider):
    async def analyze_async(self, text, prompt):
        """Async version of analyze."""
        client = AsyncAnthropic(api_key=self.api_key)
        message = await client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": f"{prompt}\n\nText:\n{text}"}],
        )
        return {
            "response": message.content[0].text,
            "model": self.model,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }

# Process multiple articles in parallel
async def analyze_parallel(provider, articles):
    tasks = [provider.analyze_async(a["text"], SENTIMENT_PROMPT) for a in articles]
    return await asyncio.gather(*tasks)

# Run
provider = AsyncClaudeProvider()
results = asyncio.run(analyze_parallel(provider, articles))
```

## Next Steps

Now that you understand AI-assisted trading, you're ready to move to live trading:

- [Part 23: Live Trading](../../23-live-trading/) - Connect to brokers and execute real trades

## Further Reading

- [Anthropic Claude API Documentation](https://docs.anthropic.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain for LLM Applications](https://python.langchain.com/)
