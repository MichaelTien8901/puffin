"""
AI-assisted trading workflow with sentiment and signals.

This example demonstrates:
1. Sentiment analysis integration
2. AI agent decision making
3. Signal combination
4. Risk-adjusted execution
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from puffin.data import YFinanceProvider
from puffin.ai import analyze_sentiment, PortfolioAgent, NewsSignalGenerator
from puffin.risk import kelly_criterion, PortfolioRiskManager
from puffin.strategies import Strategy


class AIAssistedStrategy(Strategy):
    """AI-assisted trading strategy combining technical and sentiment signals."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate AI-assisted signals."""
        df = data.copy()

        # Technical signals
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        df['signal'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
        df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1

        return df

    def get_parameters(self) -> dict:
        return {'sma_fast': 20, 'sma_slow': 50}


def main():
    """Run AI-assisted workflow."""
    print("=" * 60)
    print("AI-Assisted Trading Workflow")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading market data...")
    provider = YFinanceProvider()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    ticker = 'AAPL'
    data = provider.fetch_historical(
        symbol=ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
    )

    # 2. Generate technical signals
    print("\n2. Generating technical signals...")
    strategy = AIAssistedStrategy()
    data_with_signals = strategy.generate_signals(data)

    latest = data_with_signals.iloc[-1]
    print(f"   Ticker: {ticker}")
    print(f"   Current price: ${latest['close']:.2f}")
    print(f"   Technical signal: {latest['signal']}")

    # 3. Sentiment analysis (mock - would use real LLM provider in production)
    print("\n3. Sentiment analysis...")
    print("   Note: Requires LLM API key for real sentiment analysis.")
    print("   Using simulated sentiment for demo.")

    simulated_sentiment = np.random.uniform(-0.5, 0.5)
    print(f"   Sentiment score: {simulated_sentiment:.2f}")

    # 4. Position sizing with Kelly
    print("\n4. Position sizing...")
    win_rate = 0.55
    win_loss_ratio = 1.5
    kelly_pct = kelly_criterion(
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        fraction=0.5,
    )
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Win/loss ratio: {win_loss_ratio:.2f}")
    print(f"   Half-Kelly position size: {kelly_pct:.1%} of equity")

    # 5. Risk assessment
    print("\n5. Risk assessment...")
    portfolio_rm = PortfolioRiskManager()

    equity_curve = pd.Series(np.cumsum(np.random.randn(100) * 1000) + 100000)
    ok, dd = portfolio_rm.check_drawdown(equity_curve, max_dd=0.15)

    print(f"   Current drawdown: {dd:.2%}")
    print(f"   Within limits: {ok}")
    print(f"   Risk status: {'GREEN - Safe to trade' if ok else 'RED - Reduce exposure'}")

    print("\n" + "=" * 60)
    print("AI workflow complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
