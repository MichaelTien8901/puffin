---
layout: default
title: "Putting It All Together"
parent: "Part 6: Trading Strategies"
nav_order: 2
---

# Putting It All Together

This chapter demonstrates how to combine all the Puffin modules into complete, production-ready trading systems. We'll walk through integrating data pipelines, strategies, risk management, backtesting, and monitoring.

## Complete Trading System Architecture

A production trading system consists of several interconnected components:

```
Data Pipeline → Feature Engineering → Strategy → Risk Management
                                         ↓
                                    Execution
                                         ↓
                              Monitoring & Analysis
```

## Example 1: End-to-End Momentum Pipeline

Let's build a complete momentum trading system that incorporates all best practices.

### Step 1: Data and Features

```python
from puffin.data import YFinanceProvider
from puffin.factors.technical import TechnicalIndicators
import pandas as pd

# Load data
provider = YFinanceProvider()
data = provider.get_historical(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01',
    interval='1d'
)

# Calculate indicators
ti = TechnicalIndicators()
data['sma_50'] = ti.sma(data['close'], 50)
data['sma_200'] = ti.sma(data['close'], 200)
data['rsi'] = ti.rsi(data['close'], 14)
data['atr'] = ti.atr(data['high'], data['low'], data['close'], 14)
```

### Step 2: Strategy with Risk Management

```python
from puffin.strategies import Strategy
from puffin.risk import (
    volatility_based,
    StopLossManager,
    FixedStop,
    TrailingStop,
    PortfolioRiskManager
)
from puffin.risk.stop_loss import Position as RiskPosition
from datetime import datetime


class ProductionMomentumStrategy(Strategy):
    """Production-ready momentum strategy."""

    def __init__(self, risk_pct=0.02, atr_multiplier=2.0):
        super().__init__()
        self.risk_pct = risk_pct
        self.atr_multiplier = atr_multiplier
        self.stop_manager = StopLossManager()
        self.portfolio_rm = PortfolioRiskManager()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate entry/exit signals."""
        df = data.copy()
        df['signal'] = 0

        # Entry: Price crosses above SMA50, RSI > 50
        entry = (
            (df['close'] > df['sma_50']) &
            (df['close'].shift(1) <= df['sma_50'].shift(1)) &
            (df['rsi'] > 50) &
            (df['sma_50'] > df['sma_200'])  # Trend filter
        )

        # Exit: Price crosses below SMA50 or RSI < 30
        exit = (
            (df['close'] < df['sma_50']) |
            (df['rsi'] < 30)
        )

        df.loc[entry, 'signal'] = 1
        df.loc[exit, 'signal'] = -1

        return df

    def on_bar(self, bar: pd.Series, portfolio) -> Signal:
        """Process each bar with risk management."""
        position = portfolio.positions.get(bar.name, 0)
        ticker = bar.get('ticker', 'AAPL')

        # Check portfolio risk first
        equity_curve = pd.Series(portfolio.equity_curve)
        if len(equity_curve) > 0:
            # Circuit breaker check
            if self.portfolio_rm.circuit_breaker(equity_curve, threshold=0.15):
                return Signal(0, 0)  # Halt trading

            # Drawdown check
            ok, dd = self.portfolio_rm.check_drawdown(equity_curve, max_dd=0.10)
            if not ok:
                return Signal(-1, 0) if position != 0 else Signal(0, 0)

        # Check stops
        if position != 0 and self.stop_manager.check_stops(ticker, bar['close']):
            self.stop_manager.remove_position(ticker)
            return Signal(-1, 0)

        # Entry signal
        if bar.get('signal', 0) == 1 and position == 0:
            # Calculate position size
            position_size = volatility_based(
                equity=portfolio.equity,
                atr=bar['atr'],
                risk_pct=self.risk_pct,
                multiplier=self.atr_multiplier
            )

            # Set up stops
            risk_position = RiskPosition(
                ticker=ticker,
                entry_price=bar['close'],
                entry_time=bar.name if isinstance(bar.name, datetime) else datetime.now(),
                quantity=position_size,
                side='long',
                metadata={'atr': bar['atr']}
            )

            self.stop_manager.add_position(risk_position)
            self.stop_manager.add_stop(ticker, FixedStop(stop_distance=bar['atr'] * 2))
            self.stop_manager.add_stop(ticker, TrailingStop(trail_distance=bar['atr'] * 1.5))

            return Signal(1, position_size)

        # Exit signal
        elif bar.get('signal', 0) == -1 and position != 0:
            self.stop_manager.remove_position(ticker)
            return Signal(-1, 0)

        return Signal(0, 0)
```

### Step 3: Backtesting and Analysis

```python
from puffin.backtest import Backtester
from puffin.monitor import PnLTracker

# Prepare data with signals
data_with_signals = strategy.generate_signals(data)

# Run backtest
backtest = Backtester(
    data=data_with_signals,
    strategy=strategy,
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

result = backtest.run()

# Analyze performance
metrics = PnLTracker(result['equity_curve'])

print(f"Total Return: {result['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio():.2f}")
print(f"Max Drawdown: {metrics.max_drawdown():.2%}")
print(f"Win Rate: {metrics.win_rate():.2%}")
print(f"Total Trades: {len(result['trades'])}")
```

### Step 4: Monitoring and Logging

```python
from puffin.monitor import PnLTracker, TradeLog, TradeRecord, BenchmarkComparison

# Initialize monitoring
pnl_tracker = PnLTracker(initial_cash=100000)
trade_log = TradeLog()

# Record trades
for trade in result['trades']:
    trade_record = TradeRecord(
        timestamp=trade['entry_time'],
        ticker='AAPL',
        side='buy' if trade['type'] == 'long' else 'sell',
        qty=trade['size'],
        price=trade['entry_price'],
        commission=trade.get('commission', 0),
        slippage=trade.get('slippage', 0),
        strategy='momentum'
    )
    trade_log.record(trade_record)

# Export for analysis
trade_log.export_csv('trades.csv')

# Compare to benchmark
bc = BenchmarkComparison()
strategy_returns = pd.Series(result['equity_curve']).pct_change()
benchmark_returns = provider.get_historical('SPY', '2023-01-01', '2024-01-01')['close'].pct_change()

metrics = bc.compare(strategy_returns.dropna(), benchmark_returns.dropna())
print(f"Alpha: {metrics['alpha']:.4f}")
print(f"Beta: {metrics['beta']:.2f}")
print(f"Information Ratio: {metrics['ir']:.2f}")
```

## Example 2: Multi-Strategy Portfolio

Combine multiple strategies for diversification:

```python
from puffin.strategies import Strategy

class MultiStrategyPortfolio:
    """Portfolio combining multiple strategies."""

    def __init__(self, strategies: dict, weights: dict):
        """
        Initialize multi-strategy portfolio.

        Parameters
        ----------
        strategies : dict
            Dictionary of strategy_name -> Strategy
        weights : dict
            Dictionary of strategy_name -> weight (sum to 1.0)
        """
        self.strategies = strategies
        self.weights = weights
        self.portfolio_rm = PortfolioRiskManager()

    def generate_signals(self, data_dict: dict) -> dict:
        """
        Generate signals for all strategies.

        Parameters
        ----------
        data_dict : dict
            Dictionary of ticker -> DataFrame

        Returns
        -------
        dict
            Dictionary of ticker -> combined_signal
        """
        signals = {}

        for ticker, data in data_dict.items():
            combined_signal = 0

            for strat_name, strategy in self.strategies.items():
                strat_data = strategy.generate_signals(data)

                if len(strat_data) > 0:
                    latest_signal = strat_data['signal'].iloc[-1]
                    combined_signal += latest_signal * self.weights[strat_name]

            signals[ticker] = combined_signal

        return signals


# Example usage
strategies = {
    'momentum': MomentumStrategy(),
    'mean_reversion': MeanReversionStrategy(),
    'breakout': BreakoutStrategy()
}

weights = {
    'momentum': 0.4,
    'mean_reversion': 0.3,
    'breakout': 0.3
}

portfolio = MultiStrategyPortfolio(strategies, weights)
```

## Example 3: Live Trading Integration

Integrate with live trading:

```python
from puffin.broker import AlpacaBroker
from puffin.monitor import SystemHealth

class ProductionTradingSystem:
    """Complete production trading system."""

    def __init__(self, strategy, broker, data_provider):
        self.strategy = strategy
        self.broker = broker
        self.data_provider = data_provider

        # Monitoring
        self.pnl_tracker = PnLTracker(initial_cash=broker.get_account()['cash'])
        self.trade_log = TradeLog()
        self.health = SystemHealth(alert_callback=self.send_alert)

        # Risk management
        self.portfolio_rm = PortfolioRiskManager()

    def send_alert(self, message, level):
        """Send alert via email/Slack."""
        print(f"[{level.upper()}] {message}")
        # Implement actual alerting here

    def run(self):
        """Run live trading loop."""
        while True:
            try:
                # Health checks
                self.health.check_data_feed(self.data_provider)
                self.health.check_broker_connection(self.broker)

                # Get latest data
                data = self.data_provider.get_latest()

                # Generate signals
                signals = self.strategy.generate_signals(data)

                # Execute trades
                for ticker, signal in signals.items():
                    if signal != 0:
                        order = self.broker.place_order(
                            ticker=ticker,
                            side='buy' if signal > 0 else 'sell',
                            quantity=abs(signal)
                        )

                        # Log trade
                        trade = TradeRecord(
                            timestamp=datetime.now(),
                            ticker=ticker,
                            side='buy' if signal > 0 else 'sell',
                            qty=abs(signal),
                            price=order['fill_price'],
                            commission=order['commission'],
                            slippage=order['slippage'],
                            strategy=self.strategy.__class__.__name__
                        )
                        self.trade_log.record(trade)

                # Update monitoring
                positions = self.broker.get_positions()
                prices = {p['ticker']: p['current_price'] for p in positions}
                self.pnl_tracker.update(positions, prices)

                # Risk checks
                equity_curve = pd.Series([h['equity'] for h in self.pnl_tracker.history])
                if self.portfolio_rm.circuit_breaker(equity_curve, threshold=0.20):
                    self.health.alert("Circuit breaker triggered!", level='critical')
                    break

                # Sleep
                time.sleep(60)  # 1 minute

            except Exception as e:
                self.health.alert(f"Error in trading loop: {e}", level='error')
                time.sleep(60)


# Run live trading
system = ProductionTradingSystem(
    strategy=strategy,
    broker=broker,
    data_provider=provider
)
system.run()
```

## Best Practices Checklist

### Data Pipeline
- [ ] Multiple data sources for redundancy
- [ ] Data validation and cleaning
- [ ] Proper handling of corporate actions
- [ ] Missing data handling strategy

### Strategy Development
- [ ] Clear entry/exit rules
- [ ] Parameter optimization with walk-forward
- [ ] Out-of-sample testing
- [ ] Multiple timeframe analysis

### Risk Management
- [ ] Position sizing rules
- [ ] Multiple stop loss types
- [ ] Portfolio-level risk limits
- [ ] Circuit breakers

### Backtesting
- [ ] Realistic commissions and slippage
- [ ] Proper handling of survivorship bias
- [ ] Forward testing on recent data
- [ ] Monte Carlo simulations

### Monitoring
- [ ] Real-time P&L tracking
- [ ] Trade logging
- [ ] System health monitoring
- [ ] Automated alerts

### Production
- [ ] Error handling and recovery
- [ ] Logging and debugging
- [ ] Backup and disaster recovery
- [ ] Performance monitoring

## Common Pitfalls to Avoid

1. **Overfitting**
   - Use cross-validation
   - Test on multiple time periods
   - Keep strategy simple

2. **Look-ahead Bias**
   - Never use future data
   - Be careful with indicators that "peek"
   - Properly align time series

3. **Ignoring Transaction Costs**
   - Include realistic commissions
   - Model slippage
   - Consider market impact

4. **Poor Risk Management**
   - Always use stops
   - Don't risk too much per trade
   - Monitor portfolio correlation

5. **Lack of Monitoring**
   - Log everything
   - Set up alerts
   - Review performance regularly

## Source Code

Browse the implementation: [`puffin/strategies/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/strategies)

## Next Steps

Now that you understand how to build complete trading systems:

1. Start with paper trading
2. Test with small capital
3. Monitor and adjust
4. Scale gradually

See the `examples/` directory for complete, runnable implementations:
- `01_momentum_pipeline.py` - Complete momentum strategy
- `02_ml_pipeline.py` - Machine learning workflow
- `03_ai_workflow.py` - AI-assisted trading
- `04_boosting_long_short.py` - Long-short portfolio
- `05_deep_rl_agent.py` - Reinforcement learning agent

## Further Reading

- [Risk Management]({{ site.baseurl }}/24-risk-management/)
- [Monitoring & Analytics]({{ site.baseurl }}/25-monitoring-analytics/)
- [Live Trading]({{ site.baseurl }}/23-live-trading/01-live-trading)
- [Backtesting]({{ site.baseurl }}/07-backtesting/)
