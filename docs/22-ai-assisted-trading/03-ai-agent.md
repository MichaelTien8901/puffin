---
layout: default
title: "AI Agent Portfolio Manager"
parent: "Part 22: AI-Assisted Trading"
nav_order: 3
---

# AI Agent Portfolio Manager

## Overview

The `PortfolioAgent` is an AI agent that recommends portfolio allocation changes based on your current positions, market data, sentiment scores, and strategy signals. Rather than making trades directly, it produces structured JSON recommendations that you review and execute. It also maintains a full decision audit log for compliance and debugging.

This section also covers `generate_market_report`, which produces human-readable markdown analysis reports for your watchlist.

## Portfolio Recommendations

The agent takes four inputs and returns structured allocation recommendations:

```python
from puffin.ai.providers import ClaudeProvider
from puffin.ai.agent import PortfolioAgent

provider = ClaudeProvider()
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

# Optional: sentiment scores from the pipeline
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

{: .note }
The agent includes tickers from `market_data` that are not in your current `positions`, allowing it to recommend new positions alongside adjustments to existing ones.

## How the Agent Builds Its Prompt

Internally, `PortfolioAgent._build_prompt()` constructs a structured markdown prompt with four sections:

1. **Current Positions** -- each ticker with shares, average price, and current price
2. **Market Data** -- price, daily change percentage, and volume for all tickers
3. **Sentiment Analysis** (optional) -- sentiment score and confidence per ticker
4. **Strategy Signals** (optional) -- signal direction and strength per ticker

The agent passes this prompt to `provider.generate()` with the `AGENT_SYSTEM_PROMPT`, which instructs the LLM to respond in a specific JSON schema:

```python
from puffin.ai.agent import AGENT_SYSTEM_PROMPT

# The system prompt constrains the LLM to produce valid JSON
# with recommendations, market_outlook, risk_assessment, and summary
```

The system prompt specifies four valid actions: `"buy"`, `"sell"`, `"hold"`, and `"reduce"`, along with a `target_allocation_pct` for each ticker.

## Decision Audit Trail

Every call to `agent.recommend()` is logged with inputs, outputs, and metadata. This audit trail is essential for compliance and for understanding why the agent made specific recommendations:

```python
# Make several recommendation calls...
recommendations_1 = agent.recommend(positions, market_data)
recommendations_2 = agent.recommend(positions, market_data, sentiment=sentiment)

# Retrieve full decision history
log = agent.get_decision_log()

for entry in log:
    print(f"Time: {entry['timestamp']}")
    print(f"  Positions analyzed: {entry['inputs']['n_positions']}")
    print(f"  Tickers in market data: {entry['inputs']['n_tickers']}")
    print(f"  Had sentiment: {entry['inputs']['has_sentiment']}")
    print(f"  Had signals: {entry['inputs']['has_signals']}")
    print(f"  Prompt length: {entry['prompt_length']} chars")
    print(f"  Summary: {entry['output']['summary']}")
    print()
```

Each log entry records:

| Field | Description |
|-------|-------------|
| `timestamp` | ISO-format datetime of the recommendation |
| `inputs.n_positions` | Number of current positions |
| `inputs.n_tickers` | Number of tickers in market data |
| `inputs.has_sentiment` | Whether sentiment data was provided |
| `inputs.has_signals` | Whether strategy signals were provided |
| `prompt_length` | Character length of the constructed prompt |
| `output` | The full recommendation JSON |

{: .tip }
Persist the decision log to disk or a database for long-running systems. The in-memory list resets when the agent is garbage-collected.

## Executing Recommendations

Convert the agent's recommendations into actual broker orders. Always apply a minimum trade size to avoid tiny or uneconomical trades:

```python
def execute_agent_recommendations(recommendations, broker, current_portfolio, market_data):
    """Convert agent recommendations to actual trades.

    Args:
        recommendations: Output from agent.recommend().
        broker: Broker instance with submit_order() method.
        current_portfolio: Dict of {ticker: {qty, current_price}}.
        market_data: Dict of {ticker: {price, ...}}.
    """
    total_value = sum(
        p["qty"] * p["current_price"]
        for p in current_portfolio.values()
    )

    for rec in recommendations["recommendations"]:
        ticker = rec["ticker"]
        target_pct = rec["target_allocation_pct"] / 100.0
        target_value = total_value * target_pct

        current_value = (
            current_portfolio.get(ticker, {}).get("qty", 0)
            * current_portfolio.get(ticker, {}).get("current_price", 0)
        )

        diff_value = target_value - current_value

        if abs(diff_value) > 100:  # Minimum trade size in dollars
            price = market_data[ticker]["price"]
            if diff_value > 0:
                qty = int(diff_value / price)
                broker.submit_order(ticker, "buy", qty)
            else:
                qty = int(abs(diff_value) / price)
                broker.submit_order(ticker, "sell", qty)
```

{: .warning }
Always review AI agent recommendations before executing trades. The agent is a tool to augment your decision-making, not replace it. Never fully automate trading based solely on LLM outputs.

## Human Oversight

For production systems, require manual approval before executing any recommended trades:

```python
def execute_with_approval(recommendations, require_manual_approval=True):
    """Execute trades with optional manual approval gate."""
    if require_manual_approval:
        print("\nAgent Recommendations:")
        print(f"Outlook: {recommendations['market_outlook']}")
        print(f"Risk: {recommendations['risk_assessment']}")
        print(f"Summary: {recommendations['summary']}\n")

        for rec in recommendations["recommendations"]:
            print(f"  {rec['ticker']}: {rec['action']} -> {rec['target_allocation_pct']}%")
            print(f"    Reason: {rec['reasoning']}")

        approval = input("\nApprove these trades? (yes/no): ")
        if approval.lower() != "yes":
            print("Trades aborted.")
            return

    # Execute trades...
    print("Executing approved trades.")
```

## Market Report Generation

The `generate_market_report` function produces a comprehensive markdown analysis of your watchlist by sending market data, sentiment, and signals to the LLM:

```python
from puffin.ai.reports import generate_market_report

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

The report includes per-ticker sections with price action, technical setup, sentiment, key levels, and an overall market assessment.

{: .tip }
Save daily reports to disk for a historical archive. They are useful for reviewing what the AI "thought" at each point in time.

## Complete Example: News-Driven Trading Bot

Here is a complete example that ties the full pipeline together -- fetching news, generating sentiment signals, getting AI agent recommendations, and producing a daily report:

```python
from datetime import datetime
from puffin.ai.providers import ClaudeProvider
from puffin.ai.signals import NewsSignalGenerator
from puffin.ai.agent import PortfolioAgent
from puffin.ai.reports import generate_market_report
from puffin.ai.news import fetch_rss_news

# Initialize components
llm_provider = ClaudeProvider()
signal_generator = NewsSignalGenerator(
    provider=llm_provider,
    bullish_threshold=0.3,
    bearish_threshold=-0.3,
    min_confidence=0.5,
    min_articles=2,
)
portfolio_agent = PortfolioAgent(llm_provider)

# Step 1: Fetch recent news
articles = fetch_rss_news(max_articles=20)
print(f"Fetched {len(articles)} articles")

# Step 2: Generate sentiment-based signals
signals = signal_generator.generate(articles)
print("\nSentiment Signals:")
print(signals)

# Step 3: Get current portfolio state (from your broker)
positions = {
    "AAPL": {"qty": 100, "avg_price": 175.0, "current_price": 182.5},
    "TSLA": {"qty": 50, "avg_price": 245.0, "current_price": 238.0},
}
market_data = {
    "AAPL": {"price": 182.5, "change_pct": 2.3, "volume": 58_000_000},
    "TSLA": {"price": 238.0, "change_pct": -1.8, "volume": 125_000_000},
    "NVDA": {"price": 495.0, "change_pct": 5.2, "volume": 42_000_000},
}

# Step 4: Get AI agent recommendations
recommendations = portfolio_agent.recommend(
    positions=positions,
    market_data=market_data,
    sentiment=signals[["score", "confidence"]].to_dict("index") if len(signals) > 0 else None,
    strategy_signals=signals["signal"].to_dict() if len(signals) > 0 else None,
)

print("\nAgent Recommendations:")
for rec in recommendations["recommendations"]:
    print(f"  {rec['ticker']}: {rec['action']} -> {rec['target_allocation_pct']}%")
    print(f"    Reason: {rec['reasoning']}")

# Step 5: Generate daily report
report = generate_market_report(
    provider=llm_provider,
    watchlist_data=market_data,
    sentiment=signals[["score", "confidence"]].to_dict("index") if len(signals) > 0 else None,
)

# Save report
report_path = f"reports/market_report_{datetime.now():%Y%m%d}.md"
with open(report_path, "w") as f:
    f.write(report)
print(f"\nReport saved to {report_path}")

# Step 6: Review decision log
print("\nDecision Log:")
for entry in portfolio_agent.get_decision_log():
    print(f"  {entry['timestamp']}: {entry['output']['summary']}")
```

## Best Practices Summary

| Practice | Why It Matters |
|----------|---------------|
| Cache aggressively | Reduces API costs by 80%+ on repeated analyses |
| Validate all outputs | LLMs can return malformed JSON or out-of-range values |
| Require human approval | AI recommendations should augment, not replace, judgment |
| Log every decision | Essential for compliance, debugging, and performance review |
| Use fallback providers | Avoid downtime if your primary LLM provider has an outage |
| Set minimum article counts | Single-article signals are noisy and unreliable |

## Source Code

- Portfolio agent: [`puffin/ai/agent.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/agent.py)
- Market reports: [`puffin/ai/reports.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/reports.py)
- News fetcher: [`puffin/ai/news.py`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai/news.py)
- Full module: [`puffin/ai/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ai)

## Further Reading

- [Anthropic Claude API Documentation](https://docs.anthropic.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain for LLM Applications](https://python.langchain.com/)

## Next Steps

With the AI-assisted trading pipeline complete, you are ready to move to live trading:

- [Part 23: Live Trading](../23-live-trading/) -- Connect to brokers and execute real trades
