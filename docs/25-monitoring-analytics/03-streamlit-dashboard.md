---
layout: default
title: "Streamlit Dashboard"
parent: "Part 25: Monitoring & Analytics"
nav_order: 3
---

# Streamlit Dashboard

## Overview

Numbers in a terminal are fine for debugging, but for day-to-day monitoring you need a visual dashboard that surfaces portfolio health at a glance. Puffin ships with a ready-to-use Streamlit dashboard that combines the `PnLTracker`, `TradeLog`, and `BenchmarkComparison` modules into a multi-page web application.

The dashboard provides five views:

1. **Portfolio Overview** -- equity, P&L, cash allocation, and strategy attribution
2. **Daily P&L** -- bar chart with win/loss colouring and summary statistics
3. **Equity Curve** -- equity over time with drawdown analysis
4. **Open Positions** -- position details, unrealized P&L, and allocation pie chart
5. **Trade Log** -- filterable, sortable trade history with aggregate metrics

{: .note }
Streamlit is not included in Puffin's core dependencies. Install it with `pip install streamlit` before running the dashboard.

## Quick Start

The fastest way to launch the dashboard is with the `create_dashboard` function.

```python
from puffin.monitor.dashboard import create_dashboard
from puffin.monitor.pnl import PnLTracker
from puffin.monitor.trade_log import TradeLog

# Initialize your monitoring objects
tracker = PnLTracker(initial_cash=100_000.0)
log = TradeLog()

# ... populate with trades and position updates ...

# Create and launch dashboard
create_dashboard(pnl_tracker=tracker, trade_log=log)
```

### Running the Dashboard

Save your dashboard code to a Python file and launch it with the Streamlit CLI.

```bash
# Save dashboard code to file
cat > dashboard_app.py << 'EOF'
from puffin.monitor.pnl import PnLTracker
from puffin.monitor.trade_log import TradeLog
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

{: .tip }
Add `--server.port 8502` to run on a custom port, and `--server.headless true` for deployment on a remote server without a browser.

## Dashboard Pages in Detail

### Portfolio Overview

The overview page uses Streamlit's `st.metric` widget to display four key numbers at the top: total equity, total P&L, cash balance, and open position count. Below the metrics, a strategy attribution table breaks down unrealized P&L and market value by strategy.

```python
from puffin.monitor.dashboard import create_dashboard
from puffin.monitor.pnl import PnLTracker, Position

tracker = PnLTracker(initial_cash=100_000.0)

# Simulate some positions for the overview
tracker.record_trade(
    ticker='AAPL', quantity=100, price=150.0, side='buy', commission=1.0
)
tracker.record_trade(
    ticker='GOOGL', quantity=20, price=2800.0, side='buy', commission=1.0
)

# Mark to market
tracker.positions['AAPL'].current_price = 158.0
tracker.positions['AAPL'].strategy = 'momentum'
tracker.positions['GOOGL'].current_price = 2850.0
tracker.positions['GOOGL'].strategy = 'mean_reversion'

# The overview page will show:
# - Total Equity: cash + positions value
# - Total P&L: realized + unrealized
# - Strategy Attribution table
summary = tracker.performance_summary()
print(f"Equity: ${summary['current_equity']:,.2f}")
print(f"Total Return: {summary['total_return']:.2f}%")
```

### Daily P&L Page

The daily P&L page renders a colour-coded bar chart (green for gains, red for losses) and computes four summary statistics: average daily P&L, best day, worst day, and win rate.

```python
from puffin.monitor.pnl import PnLTracker
import pandas as pd

tracker = PnLTracker(initial_cash=100_000.0)

# After a series of trades and update() calls, the history is populated
# daily_pnl() returns the per-day change in total P&L
daily = tracker.daily_pnl()

if len(daily) > 0:
    win_rate = (daily > 0).sum() / len(daily) * 100
    print(f"Average Daily P&L: ${daily.mean():,.2f}")
    print(f"Best Day: ${daily.max():,.2f}")
    print(f"Worst Day: ${daily.min():,.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
```

{: .warning }
The win rate metric can be misleading for strategies with asymmetric payoffs. A strategy that wins 40% of the time but has a 3:1 reward-to-risk ratio outperforms one that wins 60% with a 0.5:1 ratio. Always look at win rate alongside average win size and average loss size.

### Equity Curve and Drawdown

The equity curve page plots total equity over time with a dashed line at the initial capital level. Below it, a drawdown chart shows the percentage decline from the running peak, filled in red for visual impact.

Three drawdown statistics are computed:

- **Current Drawdown** -- where you stand right now relative to the peak
- **Max Drawdown** -- the worst peak-to-trough decline
- **Recovery Time** -- how many days it took to recover from the max drawdown (or "Not recovered" if still underwater)

```python
from puffin.monitor.pnl import PnLTracker
import pandas as pd
import numpy as np

tracker = PnLTracker(initial_cash=100_000.0)
# ... populate history via update() calls ...

if len(tracker.history) > 0:
    df = pd.DataFrame(tracker.history)
    equity_series = pd.Series(df['equity'].values, index=df['timestamp'])
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100

    print(f"Current Drawdown: {drawdown.iloc[-1]:.2f}%")
    print(f"Max Drawdown: {drawdown.min():.2f}%")
```

{: .note }
Drawdowns are expressed as negative percentages. A max drawdown of -15% means the portfolio lost 15% from its peak before recovering.

### Open Positions Page

This page displays a formatted table of all open positions with columns for quantity, average price, current price, cost basis, market value, unrealized P&L, and return percentage. A conditional colour gradient highlights winning (green) and losing (red) positions.

Below the table, a pie chart shows position allocation by market value.

```python
from puffin.monitor.pnl import PnLTracker

tracker = PnLTracker(initial_cash=100_000.0)

# Record several positions
tracker.record_trade(
    ticker='AAPL', quantity=100, price=150.0, side='buy', commission=1.0
)
tracker.record_trade(
    ticker='MSFT', quantity=50, price=380.0, side='buy', commission=1.0
)
tracker.record_trade(
    ticker='TSLA', quantity=30, price=250.0, side='buy', commission=1.0
)

# Mark to market
tracker.positions['AAPL'].current_price = 158.0
tracker.positions['MSFT'].current_price = 375.0
tracker.positions['TSLA'].current_price = 265.0

# Attribution by asset
asset_attr = tracker.attribution_by_asset()
print(asset_attr.to_string())
```

### Trade Log Page

The trade log page provides three filter dropdowns (ticker, strategy, side) and renders the filtered results in a scrollable table. Below the table, summary statistics show total trades, total commission, total slippage, and average trade size.

```python
from puffin.monitor.trade_log import TradeLog, TradeRecord
from datetime import datetime

log = TradeLog()

# Record multiple trades
for ticker, price in [('AAPL', 150.0), ('MSFT', 380.0), ('TSLA', 250.0)]:
    log.record(TradeRecord(
        timestamp=datetime.now(),
        ticker=ticker,
        side='buy',
        qty=100,
        price=price,
        commission=1.0,
        slippage=0.05,
        strategy='momentum'
    ))

summary = log.summary()
print(f"Total Trades: {summary['total_trades']}")
print(f"Total Commission: ${summary['total_commission']:.2f}")
print(f"Total Slippage: ${summary['total_slippage']:.2f}")
print(f"Avg Trade Size: ${summary['avg_trade_size']:,.2f}")
```

## Complete Monitoring Workflow

Here is a complete example that wires together all monitoring components into a trading loop and then launches the dashboard.

```python
from puffin.monitor.pnl import PnLTracker
from puffin.monitor.trade_log import TradeLog, TradeRecord
from puffin.monitor.benchmark import BenchmarkComparison
from puffin.monitor.health import SystemHealth
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

{: .tip }
For production deployment, save the `tracker` and `log` to disk at the end of every session and reload them on startup. This ensures you have a continuous performance record across restarts.

## Best Practices

1. **Dashboard Review**
   - Review the dashboard daily, ideally at market close
   - Look for unusual patterns in the daily P&L distribution
   - Monitor drawdowns and set mental or automated circuit breakers
   - Check position concentration to avoid overexposure

2. **Deployment**
   - Run the dashboard on a dedicated server or cloud instance
   - Use `streamlit run --server.headless true` for remote access
   - Set up auto-refresh with `st.experimental_rerun()` for live data
   - Password-protect the dashboard in production with Streamlit's native auth

3. **Data Persistence**
   - Export trade logs to CSV/JSON at the end of every session
   - Store equity history snapshots in a database for long-term analysis
   - Back up all monitoring data alongside your strategy configuration

## Source Code

The dashboard implementation lives in the following file:

- [`puffin/monitor/dashboard.py`](https://github.com/MichaelTien8901/puffin/blob/main/puffin/monitor/dashboard.py) -- `create_dashboard()` with five Streamlit pages: Portfolio Overview, Daily P&L, Equity Curve, Open Positions, and Trade Log

Browse the full monitoring module: [`puffin/monitor/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/monitor)

## Next Steps

- See [Risk Management]({{ site.baseurl }}/24-risk-management/) for risk controls that integrate with the monitoring pipeline
- See [Live Trading]({{ site.baseurl }}/23-live-trading/01-live-trading) for production deployment patterns
- Check `examples/` for complete end-to-end implementations
