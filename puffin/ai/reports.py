"""AI-generated market analysis reports."""

from puffin.ai.llm_provider import LLMProvider


REPORT_PROMPT = """Generate a concise market analysis report for the following watchlist.
Include for each ticker: price action summary, key technical levels, sentiment,
and any notable signals. End with an overall market assessment.
Format as clean markdown."""


def generate_market_report(
    provider: LLMProvider,
    watchlist_data: dict[str, dict],
    sentiment: dict[str, dict] | None = None,
    signals: dict[str, float] | None = None,
) -> str:
    """Generate a market analysis report.

    Args:
        provider: LLM provider.
        watchlist_data: {ticker: {price, change_pct, high_52w, low_52w, rsi, macd_signal}}.
        sentiment: Optional sentiment data.
        signals: Optional strategy signals.

    Returns:
        Markdown-formatted report.
    """
    parts = ["Analyze the following market data and generate a report:\n"]

    for ticker, data in watchlist_data.items():
        parts.append(f"### {ticker}")
        parts.append(f"- Price: ${data.get('price', 0):.2f} ({data.get('change_pct', 0):+.2f}%)")
        if "high_52w" in data:
            parts.append(f"- 52W Range: ${data['low_52w']:.2f} – ${data['high_52w']:.2f}")
        if "rsi" in data:
            parts.append(f"- RSI: {data['rsi']:.1f}")
        if "macd_signal" in data:
            parts.append(f"- MACD Signal: {'Bullish' if data['macd_signal'] > 0 else 'Bearish'}")

        if sentiment and ticker in sentiment:
            s = sentiment[ticker]
            parts.append(f"- Sentiment: {s.get('signal', 'neutral')} (score={s.get('score', 0):.2f})")

        if signals and ticker in signals:
            sig = signals[ticker]
            parts.append(f"- Strategy Signal: {'BUY' if sig > 0 else 'SELL' if sig < 0 else 'NEUTRAL'}")

    prompt = "\n".join(parts)
    return provider.generate(prompt, system=REPORT_PROMPT)
