---
layout: default
title: "Trade Logging & P&L"
parent: "Part 25: Monitoring & Analytics"
nav_order: 1
---

# Trade Logging & P&L

## Overview

Every algorithmic trading system needs two foundational capabilities: a complete audit trail of executed trades and accurate, real-time profit-and-loss tracking. Without these, you cannot debug strategy behaviour, satisfy regulatory requirements, or make informed decisions about position sizing and risk.

Puffin provides two tightly integrated modules for this purpose:

- **`TradeLog`** and **`TradeRecord`** -- capture every execution with metadata, then filter, export, and summarise the history.
- **`PnLTracker`** and **`Position`** -- maintain a live position book, compute realized and unrealized P&L, and produce time-series and attribution reports.

{: .warning }
Always persist your trade log to disk (CSV or JSON) at the end of every trading session. In-memory logs are lost on restart.

## Trade Recording

The `TradeRecord` dataclass captures a single execution. It serializes to and from dictionaries and supports arbitrary metadata for strategy-specific annotations.

```python
from puffin.monitor.trade_log import TradeLog, TradeRecord
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

{: .tip }
Attach a `metadata` dictionary to every `TradeRecord` so you can later analyse which signal parameters led to the best fills.

### Exporting Trade Data

The `TradeLog` supports CSV and JSON for portability. CSV is convenient for spreadsheet analysis; JSON preserves nested metadata structures.

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

Filter the log by ticker, strategy, date range, or any combination. Filters return a list of `TradeRecord` objects suitable for further analysis.

```python
from datetime import datetime, timedelta

# Filter by ticker
aapl_trades = log.filter(ticker='AAPL')

# Filter by strategy
momentum_trades = log.filter(strategy='momentum')

# Filter by date range
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

The `summary()` method returns aggregate statistics across all recorded trades.

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

{: .note }
The `avg_trade_size` is the average notional value (`qty * price`) across all trades, not the average quantity.

## P&L Tracking

The `PnLTracker` maintains a live position book and computes P&L in real time. It separates realized P&L (from closed trades) and unrealized P&L (from open positions marked to market).

### Basic Usage

```python
from puffin.monitor.pnl import PnLTracker

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

### Closing Positions

When you sell, `PnLTracker` automatically computes the realized gain or loss using the average cost basis.

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

{: .warning }
Selling more shares than you hold raises a `ValueError`. Check `tracker.positions[ticker].quantity` before submitting a sell order.

### Updating Positions with Market Data

Use `update()` to push new prices into the tracker and record a history snapshot for time-series analysis.

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

After accumulating history snapshots, generate daily and cumulative P&L series for charting and further analysis.

```python
import matplotlib.pyplot as plt

# Daily P&L
daily_pnl = tracker.daily_pnl()
print(daily_pnl.tail())

# Cumulative P&L
cumulative_pnl = tracker.cumulative_pnl()
print(cumulative_pnl.tail())

# Plot
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

Attribution lets you decompose total P&L by strategy or by individual asset. This is critical for understanding which strategies are earning and which are dragging performance.

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

{: .tip }
When running multiple strategies in parallel, assign a unique `strategy` string to every `Position` so the attribution report cleanly separates contributions.

## Best Practices

1. **Trade Logging**
   - Log every trade execution without exception
   - Include metadata for post-trade analysis
   - Export logs regularly for backup and compliance
   - Use structured logging format (JSON for programmatic access, CSV for ad-hoc analysis)

2. **P&L Tracking**
   - Update positions frequently (at least on every bar)
   - Separate realized and unrealized P&L in all reporting
   - Track attribution by strategy and asset
   - Monitor intraday P&L swings to detect anomalies

## Source Code

The trade logging and P&L tracking implementations live in the following files:

- [`puffin/monitor/trade_log.py`](https://github.com/MichaelTien8901/puffin/blob/main/puffin/monitor/trade_log.py) -- `TradeLog` and `TradeRecord`
- [`puffin/monitor/pnl.py`](https://github.com/MichaelTien8901/puffin/blob/main/puffin/monitor/pnl.py) -- `PnLTracker` and `Position`
- [`puffin/monitor/__init__.py`](https://github.com/MichaelTien8901/puffin/blob/main/puffin/monitor/__init__.py) -- Public API exports
