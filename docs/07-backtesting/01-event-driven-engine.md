---
layout: default
title: "Event-Driven Engine"
parent: "Part 7: Backtesting"
nav_order: 1
---

# Event-Driven Engine

## Overview

Before deploying a strategy with real money, you need to simulate it against historical data. Backtesting answers the question: "How would this strategy have performed in the past?" The quality of the answer depends entirely on the realism of the simulation.

## Why Event-Driven?

There are two approaches to backtesting:

| Approach | Pros | Cons |
|----------|------|------|
| **Vectorized** | Fast, simple | Unrealistic, lookahead bias risk |
| **Event-driven** | Realistic, no lookahead | Slower, more complex |

**Vectorized backtesting** computes signals across the entire dataset at once using array operations. This is fast, but it makes it easy to accidentally use future information when calculating signals (lookahead bias). It also cannot model realistic order execution, since all signals are computed simultaneously.

**Event-driven backtesting** processes data one bar at a time, just like live trading. At each bar, the engine:

1. Receives the latest price data
2. Executes any pending orders from the previous bar
3. Passes the visible history to the strategy
4. Collects new signals and converts them to orders

Puffin uses event-driven backtesting because it mirrors how live trading actually works: you receive a bar, make a decision, submit an order, and it fills on the next bar.

## Lookahead Bias Prevention

{: .important }
Orders generated on bar T execute on bar T+1. The strategy only sees data up to bar T. This prevents lookahead bias.

The `Backtester.run()` method enforces this by slicing the data at each step. The strategy's `generate_signals()` method receives only `data.loc[:current_date]`, never the full dataset. Pending orders are filled against the next bar's open price, matching how a market order placed after the close would execute in reality.

```python
# Inside the engine loop (simplified):
for i, date in enumerate(dates):
    # Strategy sees only data up to current bar
    history = data.loc[:date]

    # Execute orders from PREVIOUS bar against current bar's open
    for order in pending_orders:
        fill = self._try_fill(order, current_bar, ...)

    # Generate new signals -- these become orders for the NEXT bar
    signals = strategy.generate_signals(history)
```

## Basic Usage

```python
from puffin.backtest import Backtester, SlippageModel, CommissionModel
from puffin.strategies import MomentumStrategy

strategy = MomentumStrategy(short_window=20, long_window=50)

bt = Backtester(
    initial_capital=100_000,
    slippage=SlippageModel(fixed=0.01),
    commission=CommissionModel(flat=1.0),
)

result = bt.run(strategy, data)
print(result.metrics())
```

The `Backtester` accepts three parameters:

- **`initial_capital`** -- Starting cash balance (default: 100,000).
- **`slippage`** -- A `SlippageModel` instance controlling price impact (see [Execution Models](02-execution-models)).
- **`commission`** -- A `CommissionModel` instance controlling trading costs (see [Execution Models](02-execution-models)).

The `run()` method takes a `Strategy` instance and a pandas DataFrame of OHLCV data. For multi-asset backtests, pass a DataFrame with a `(Date, Symbol)` MultiIndex. The engine auto-detects the format and iterates over all symbols.

## Order Types

The engine supports three order types:

- **Market orders** -- Fill at the next bar's open price (plus slippage).
- **Limit orders** -- Fill only if the price reaches the limit. Buy limits fill when the low is at or below the limit price; sell limits fill when the high is at or above.
- **Stop orders** -- Trigger when the price crosses the stop level. Used for stop-loss exits.

```python
from puffin.backtest.engine import Order

# Market order (default)
Order(symbol="AAPL", side="buy", qty=100)

# Limit order
Order(symbol="AAPL", side="buy", qty=100, order_type="limit", limit_price=150.0)

# Stop order
Order(symbol="AAPL", side="sell", qty=100, order_type="stop", stop_price=140.0)
```

## Exercises

1. Backtest the `MomentumStrategy` on SPY from 2018--2024 with `short_window=20` and `long_window=50`. What is the Sharpe ratio?
2. Change the windows to `short_window=10` and `long_window=30`. How do the results compare?
3. Try running the backtest without any slippage or commission models. How much do the metrics change when you add realistic costs?
