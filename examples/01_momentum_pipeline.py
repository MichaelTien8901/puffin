"""
Complete momentum strategy pipeline with risk management and monitoring.

This example demonstrates a full trading workflow:
1. Data loading and preparation
2. Technical indicator calculation
3. Momentum strategy with backtesting
4. Risk management
5. Performance monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from puffin.data import YFinanceProvider
from puffin.strategies import MomentumStrategy
from puffin.backtest import Backtester
from puffin.risk import (
    volatility_based,
    PortfolioRiskManager,
)
from puffin.monitor import PnLTracker, TradeLog, TradeRecord, BenchmarkComparison


def main():
    """Run complete momentum pipeline."""
    print("=" * 60)
    print("Momentum Strategy Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading data...")
    provider = YFinanceProvider()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)

    ticker = 'AAPL'
    print(f"   Loading {ticker}...")
    data = provider.fetch_historical(
        symbol=ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
    )

    print(f"   Loaded {len(data)} bars")

    # 2. Initialize strategy
    print("\n2. Initializing momentum strategy...")
    strategy = MomentumStrategy(
        fast_window=10,
        slow_window=50,
    )

    # 3. Run backtest
    print("\n3. Running backtest...")
    backtester = Backtester(
        initial_capital=100000,
        commission_pct=0.001,
    )

    result = backtester.run(data, strategy)
    metrics = result.metrics()

    print(f"\n   Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"   Trade Count: {metrics.get('trade_count', 0)}")

    # 4. Risk metrics
    print("\n4. Portfolio risk metrics...")
    portfolio_rm = PortfolioRiskManager()

    equity_curve = pd.Series(result.equity_curve)
    ok, dd = portfolio_rm.check_drawdown(equity_curve, max_dd=0.15)
    print(f"   Current drawdown: {dd:.2%}")
    print(f"   Within limits: {ok}")

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0:
        var = portfolio_rm.compute_var(returns, confidence=0.95)
        es = portfolio_rm.compute_expected_shortfall(returns, confidence=0.95)
        print(f"   95% VaR: {var:.4f}")
        print(f"   Expected Shortfall: {es:.4f}")

    # 5. Position sizing
    print("\n5. Position sizing...")
    atr_value = data['close'].diff().abs().rolling(14).mean().iloc[-1]
    pos_size = volatility_based(equity=100000, atr=atr_value, risk_pct=0.02)
    print(f"   ATR-based position size: {pos_size:.0f} shares")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
