"""AI agent for portfolio management."""

import json
from datetime import datetime

from puffin.ai.llm_provider import LLMProvider


AGENT_SYSTEM_PROMPT = """You are an AI portfolio management assistant. You analyze market data,
sentiment signals, and strategy outputs to recommend portfolio allocation changes.

Always respond in JSON format with:
{
    "recommendations": [
        {
            "ticker": "SYMBOL",
            "action": "buy" | "sell" | "hold" | "reduce",
            "target_allocation_pct": float,
            "reasoning": "brief explanation"
        }
    ],
    "market_outlook": "bullish" | "bearish" | "neutral",
    "risk_assessment": "low" | "medium" | "high",
    "summary": "1-2 sentence overall assessment"
}"""


class PortfolioAgent:
    """AI agent that makes portfolio allocation recommendations."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.decision_log: list[dict] = []

    def recommend(
        self,
        positions: dict[str, dict],
        market_data: dict[str, dict],
        sentiment: dict[str, dict] | None = None,
        strategy_signals: dict[str, float] | None = None,
    ) -> dict:
        """Generate portfolio recommendations.

        Args:
            positions: Current positions {ticker: {qty, avg_price, current_price}}.
            market_data: Recent market info {ticker: {price, change_pct, volume}}.
            sentiment: Sentiment scores {ticker: {score, confidence}}.
            strategy_signals: Strategy signals {ticker: signal_value}.

        Returns:
            Dict with recommendations, market_outlook, risk_assessment, summary.
        """
        prompt = self._build_prompt(positions, market_data, sentiment, strategy_signals)
        response = self.provider.generate(prompt, system=AGENT_SYSTEM_PROMPT)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
            else:
                result = {
                    "recommendations": [],
                    "market_outlook": "neutral",
                    "risk_assessment": "medium",
                    "summary": "Unable to parse AI response.",
                }

        # Log the decision
        self.decision_log.append({
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "n_positions": len(positions),
                "n_tickers": len(market_data),
                "has_sentiment": sentiment is not None,
                "has_signals": strategy_signals is not None,
            },
            "output": result,
            "prompt_length": len(prompt),
        })

        return result

    def _build_prompt(self, positions, market_data, sentiment, signals) -> str:
        parts = ["Review the following portfolio and market data, then recommend allocation changes.\n"]

        parts.append("## Current Positions")
        if positions:
            for ticker, pos in positions.items():
                parts.append(f"- {ticker}: {pos.get('qty', 0)} shares @ ${pos.get('avg_price', 0):.2f} "
                           f"(current: ${pos.get('current_price', 0):.2f})")
        else:
            parts.append("- No current positions (100% cash)")

        parts.append("\n## Market Data")
        for ticker, data in market_data.items():
            parts.append(f"- {ticker}: ${data.get('price', 0):.2f} "
                       f"({data.get('change_pct', 0):+.2f}%) "
                       f"vol: {data.get('volume', 0):,}")

        if sentiment:
            parts.append("\n## Sentiment Analysis")
            for ticker, sent in sentiment.items():
                parts.append(f"- {ticker}: score={sent.get('score', 0):.2f} "
                           f"confidence={sent.get('confidence', 0):.2f}")

        if signals:
            parts.append("\n## Strategy Signals")
            for ticker, signal in signals.items():
                direction = "BUY" if signal > 0 else "SELL" if signal < 0 else "NEUTRAL"
                parts.append(f"- {ticker}: {direction} (strength={abs(signal):.2f})")

        return "\n".join(parts)

    def get_decision_log(self) -> list[dict]:
        """Return the full decision log for audit."""
        return self.decision_log
