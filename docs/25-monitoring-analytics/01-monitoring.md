---
layout: default
title: "Chapter 1: Monitoring & Analytics"
parent: "Part 25: Monitoring & Analytics"
nav_order: 1
---

# Chapter 1: Monitoring & Analytics

Comprehensive monitoring and analytics are essential for understanding strategy performance and making informed trading decisions. The Puffin framework provides tools for trade logging, P&L tracking, benchmark comparison, and system health monitoring.

## Trade Logging

The `TradeLog` class records all trade executions for analysis and audit purposes.

### Recording Trades

```python
from puffin.monitor import TradeLog, TradeRecord
from datetime import datetime

# Create trade log
log = TradeLog()

# Record a trade
trade = TradeRecord(
    timestamp=datetime.now(),
    ticker='AAPL',
    side='buy',
    qty=100,
    price=150.25,
    commission=1.00,
    slippage=0.05,
    strategy='momentum',
    metadata={'signal_strength': 0.85}
)

log.record(trade)
```

### Exporting Trade Data

```python
# Export to CSV
log.export_csv('trades.csv')

# Export to JSON
log.export_json('trades.json')

# Load from file
log2 = TradeLog()
log2.load_json('trades.json')
```

### Filtering Trades

```python
# Filter by ticker
aapl_trades = log.filter(ticker='AAPL')

# Filter by strategy
momentum_trades = log.filter(strategy='momentum')

# Filter by date range
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

recent_trades = log.filter(date_range=(start_date, end_date))

# Combine filters
aapl_momentum = log.filter(
    ticker='AAPL',
    strategy='momentum',
    date_range=(start_date, end_date)
)
```

### Trade Summary

```python
summary = log.summary()

print(f"Total trades: {summary['total_trades']}")
print(f"Buy trades: {summary['total_buy']}")
print(f"Sell trades: {summary['total_sell']}")
print(f"Total commission: ${summary['total_commission']:.2f}")
print(f"Total slippage: ${summary['total_slippage']:.2f}")
print(f"Avg trade size: ${summary['avg_trade_size']:.2f}")
print(f"Strategies: {summary['strategies']}")
print(f"Tickers: {summary['tickers']}")
```

## P&L Tracking

The `PnLTracker` class monitors profit and loss with attribution.

### Basic Usage

```python
from puffin.monitor import PnLTracker

# Initialize with starting capital
tracker = PnLTracker(initial_cash=100000.0)

# Record trades
tracker.record_trade(
    ticker='AAPL',
    quantity=100,
    price=150.0,
    side='buy',
    commission=1.0
)

# Update prices
tracker.positions['AAPL'].current_price = 155.0

# Check P&L
print(f"Cash: ${tracker.cash:,.2f}")
print(f"Equity: ${tracker.equity():,.2f}")
print(f"Unrealized P&L: ${tracker.unrealized_pnl():,.2f}")
print(f"Total P&L: ${tracker.total_pnl():,.2f}")
```

### Recording Trades

```python
# Buy trade
tracker.record_trade(
    ticker='AAPL',
    quantity=100,
    price=150.0,
    side='buy',
    commission=1.0
)

# Sell trade (realizes P&L)
tracker.record_trade(
    ticker='AAPL',
    quantity=50,
    price=155.0,
    side='sell',
    commission=1.0
)

print(f"Realized P&L: ${tracker.realized_pnl:,.2f}")
```

### Updating Positions

```python
from puffin.monitor.pnl import Position

# Create positions with current prices
positions = {
    'AAPL': Position(
        ticker='AAPL',
        quantity=100,
        avg_price=150.0,
        current_price=155.0,
        strategy='momentum'
    ),
    'GOOGL': Position(
        ticker='GOOGL',
        quantity=50,
        avg_price=2800.0,
        current_price=2850.0,
        strategy='mean_reversion'
    )
}

# Update tracker with latest prices
prices = {'AAPL': 155.0, 'GOOGL': 2850.0}
tracker.update(positions, prices)
```

### P&L Time Series

```python
# Daily P&L
daily_pnl = tracker.daily_pnl()
print(daily_pnl.tail())

# Cumulative P&L
cumulative_pnl = tracker.cumulative_pnl()
print(cumulative_pnl.tail())

# Plot
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Daily P&L
daily_pnl.plot(kind='bar', ax=ax1, color='green')
ax1.set_title('Daily P&L')
ax1.set_ylabel('P&L ($)')

# Cumulative P&L
cumulative_pnl.plot(ax=ax2, linewidth=2)
ax2.set_title('Cumulative P&L')
ax2.set_ylabel('P&L ($)')

plt.tight_layout()
plt.show()
```

### P&L Attribution

```python
# By strategy
strategy_attr = tracker.attribution_by_strategy()
print("\nP&L by Strategy:")
print(strategy_attr)

# By asset
asset_attr = tracker.attribution_by_asset()
print("\nP&L by Asset:")
print(asset_attr)

# Performance summary
summary = tracker.performance_summary()
print(f"\nInitial Cash: ${summary['initial_cash']:,.2f}")
print(f"Current Equity: ${summary['current_equity']:,.2f}")
print(f"Total Return: {summary['total_return']:.2f}%")
print(f"Realized P&L: ${summary['realized_pnl']:,.2f}")
print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
print(f"Open Positions: {summary['num_positions']}")
```

## Benchmark Comparison

Compare strategy performance against benchmarks like SPY or market indices.

### Basic Comparison

```python
from puffin.monitor import BenchmarkComparison
import pandas as pd

bc = BenchmarkComparison()

# Your strategy returns
strategy_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.015])

# Benchmark returns (e.g., SPY)
benchmark_returns = pd.Series([0.005, 0.015, -0.005, 0.02, 0.01])

# Compare
metrics = bc.compare(strategy_returns, benchmark_returns)

print(f"Alpha: {metrics['alpha']:.4f}")
print(f"Beta: {metrics['beta']:.4f}")
print(f"Information Ratio: {metrics['ir']:.4f}")
print(f"Tracking Error: {metrics['tracking_error']:.4f}")
print(f"Correlation: {metrics['correlation']:.4f}")
print(f"Outperformance: {metrics['outperformance']:.2f}%")
```

**Metrics Explained:**
- **Alpha**: Excess return above benchmark (adjusted for beta)
- **Beta**: Systematic risk / market exposure
- **Information Ratio**: Risk-adjusted outperformance
- **Tracking Error**: Volatility of excess returns
- **Correlation**: How closely strategy follows benchmark

### Visualization

```python
# Plot comparison
fig = bc.plot_comparison(strategy_returns, benchmark_returns)
plt.show()
```

The comparison plot includes:
1. Cumulative returns
2. Excess returns
3. Rolling correlation
4. Performance metrics

### Rolling Metrics

```python
# Calculate rolling performance metrics
rolling = bc.rolling_metrics(
    strategy_returns,
    benchmark_returns,
    window=60  # 60-day rolling window
)

print(rolling.tail())

# Plot rolling alpha
rolling['alpha'].plot(figsize=(12, 6))
plt.title('Rolling 60-Day Alpha')
plt.ylabel('Alpha')
plt.grid(True)
plt.show()
```

## System Health Monitoring

Monitor system health and send alerts when issues arise.

### Data Feed Health

```python
from puffin.monitor import SystemHealth

# Create health monitor with alert callback
def alert_callback(message, level):
    print(f"[{level.upper()}] {message}")
    # Could send email, Slack message, etc.

health = SystemHealth(alert_callback=alert_callback)

# Check data feed
class DataProvider:
    def get_latest_timestamp(self):
        from datetime import datetime, timedelta
        return datetime.now() - timedelta(seconds=30)

provider = DataProvider()
status = health.check_data_feed(provider)

print(f"Data feed status: {status['status']}")
print(f"Latency: {status['latency']:.1f}s")
```

### Broker Connection Health

```python
# Check broker connection
class Broker:
    def is_connected(self):
        return True

    def last_heartbeat(self):
        from datetime import datetime
        return datetime.now()

broker = Broker()
status = health.check_broker_connection(broker)

print(f"Broker status: {status['status']}")
print(f"Connected: {status['connected']}")
```

### Manual Alerts

```python
# Send custom alerts
health.alert("Low liquidity detected", level='warning')
health.alert("Order execution failed", level='error')
health.alert("Daily loss limit reached", level='critical')
```

### Overall Health

```python
# Get overall system health
overall = health.get_overall_health()

print(f"Overall status: {overall['status']}")
print(f"Checks: {overall['checks']}")
```

## Streamlit Dashboard

The Puffin framework includes a Streamlit dashboard for real-time monitoring.

### Basic Setup

```python
from puffin.monitor.dashboard import create_dashboard

# Create dashboard with your data
create_dashboard(
    pnl_tracker=tracker,
    trade_log=log
)
```

### Running the Dashboard

```bash
# Save dashboard code to file
cat > dashboard_app.py << 'EOF'
from puffin.monitor import PnLTracker, TradeLog
from puffin.monitor.dashboard import create_dashboard

# Load your data
tracker = PnLTracker(initial_cash=100000)
log = TradeLog()

# ... populate with real data ...

# Create dashboard
create_dashboard(pnl_tracker=tracker, trade_log=log)
EOF

# Run Streamlit
streamlit run dashboard_app.py
```

### Dashboard Features

The dashboard includes multiple pages:

1. **Portfolio Overview**
   - Total equity
   - P&L summary
   - Open positions
   - Strategy attribution

2. **Daily P&L**
   - Daily P&L bar chart
   - Win rate statistics
   - Best/worst days

3. **Equity Curve**
   - Equity over time
   - Drawdown chart
   - Recovery metrics

4. **Open Positions**
   - Position details
   - Unrealized P&L
   - Position allocation pie chart

5. **Trade Log**
   - Filterable trade history
   - Trade summary statistics
   - Commission and slippage totals

## Complete Example

Here's a complete monitoring workflow:

```python
from puffin.monitor import PnLTracker, TradeLog, BenchmarkComparison, SystemHealth
from datetime import datetime
import pandas as pd

# Initialize monitoring components
tracker = PnLTracker(initial_cash=100000)
log = TradeLog()
bc = BenchmarkComparison()
health = SystemHealth()

# Trading loop
for bar in data:
    # Generate signals
    signal = strategy.generate_signal(bar)

    # Execute trade
    if signal:
        # Record trade
        trade = TradeRecord(
            timestamp=bar.timestamp,
            ticker=bar.ticker,
            side=signal.side,
            qty=signal.quantity,
            price=bar.close,
            commission=1.0,
            slippage=0.05,
            strategy='my_strategy'
        )
        log.record(trade)

        # Update P&L tracker
        tracker.record_trade(
            ticker=bar.ticker,
            quantity=signal.quantity,
            price=bar.close,
            side=signal.side,
            commission=1.0
        )

    # Update positions with current prices
    for ticker, position in tracker.positions.items():
        position.current_price = get_current_price(ticker)

    # Check system health
    health.check_data_feed(data_provider)
    health.check_broker_connection(broker)

    # Monitor risk
    equity_curve = pd.Series([h['equity'] for h in tracker.history])
    if len(equity_curve) > 0:
        ok, dd = portfolio_rm.check_drawdown(equity_curve, max_dd=0.15)
        if not ok:
            health.alert(f"Drawdown {dd:.1%} exceeds limit", level='warning')

# End of day analysis
print("\n=== Daily Summary ===")
summary = log.summary()
print(f"Trades: {summary['total_trades']}")
print(f"Commission: ${summary['total_commission']:.2f}")

pnl_summary = tracker.performance_summary()
print(f"Equity: ${pnl_summary['current_equity']:,.2f}")
print(f"Total Return: {pnl_summary['total_return']:.2f}%")

# Compare to benchmark
strategy_returns = tracker.daily_pnl() / tracker.initial_cash
benchmark_returns = get_spy_returns()
metrics = bc.compare(strategy_returns, benchmark_returns)
print(f"Alpha: {metrics['alpha']:.4f}")
print(f"Sharpe vs Benchmark: {metrics['ir']:.2f}")
```

## Best Practices

1. **Trade Logging**
   - Log every trade execution
   - Include metadata for analysis
   - Export logs regularly for backup
   - Use structured logging format

2. **P&L Tracking**
   - Update positions frequently
   - Separate realized and unrealized P&L
   - Track attribution by strategy and asset
   - Monitor intraday P&L swings

3. **Benchmarking**
   - Choose appropriate benchmark
   - Track rolling metrics
   - Monitor correlation changes
   - Calculate risk-adjusted metrics

4. **System Health**
   - Check health frequently
   - Set up alert notifications
   - Monitor data feed latency
   - Track broker connection status

5. **Dashboard**
   - Review daily
   - Look for unusual patterns
   - Monitor drawdowns
   - Check position concentration

## Next Steps

- See [Risk Management](/docs/24-risk-management/01-risk-management.md) for risk controls
- See [Live Trading](/docs/23-live-trading/01-live-trading.md) for production deployment
- Check `examples/` for complete implementations
