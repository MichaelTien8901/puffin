---
layout: default
title: "Stop Losses"
parent: "Part 24: Risk Management"
nav_order: 2
---

# Stop Losses

Stop losses protect against excessive losses by defining exit conditions for open positions. The `puffin.risk.stop_loss` module provides multiple stop loss strategies, all inheriting from the `StopLoss` abstract base class and implementing a `.check()` method.

All stop types work with the `Position` dataclass, which tracks entry price, entry time, quantity, side (long or short), and running high/low prices:

```python
from puffin.risk.stop_loss import Position
from datetime import datetime

position = Position(
    ticker='AAPL',
    entry_price=100.0,
    entry_time=datetime.now(),
    quantity=100,
    side='long'
)
```

---

## Fixed Stop

A fixed stop loss triggers at a specified distance from the entry price. It does not move once set.

```python
from puffin.risk import FixedStop
from puffin.risk.stop_loss import Position
from datetime import datetime

# Create stop loss
stop = FixedStop(stop_distance=5.0, price_based=True)

# Create position
position = Position(
    ticker='AAPL',
    entry_price=100.0,
    entry_time=datetime.now(),
    quantity=100,
    side='long'
)

# Check if triggered
if stop.check(current_price=94.0, entry_price=100.0, position=position):
    print("Stop loss triggered!")
```

{: .note }
> When `price_based=True`, `stop_distance` is in absolute price units ($5 below entry). When `price_based=False`, it is a percentage (e.g., 0.05 for 5% below entry).

---

## Trailing Stop

A trailing stop follows favorable price movement, locking in gains as the price advances. It never moves backward.

```python
from puffin.risk import TrailingStop

# 5-point trailing stop
stop = TrailingStop(trail_distance=5.0, price_based=True)

# As price rises, stop trails behind
# Entry: $100, Price moves to $110
# Stop now at $105 (5 points below high)
```

{: .tip }
> Trailing stops are powerful on trending instruments. They let winners run while still providing a safety net if the trend reverses.

---

## ATR Stop

An ATR-based stop adapts to the instrument's volatility. The stop distance is computed as a multiple of the Average True Range (ATR), which must be provided in the position's `metadata` dictionary.

```python
from puffin.risk import ATRStop
from puffin.risk.stop_loss import Position
from datetime import datetime

# 2x ATR stop
stop = ATRStop(atr_multiplier=2.0, trailing=True)

position = Position(
    ticker='AAPL',
    entry_price=100.0,
    entry_time=datetime.now(),
    quantity=100,
    side='long',
    metadata={'atr': 3.0}  # ATR must be in metadata
)

# Stop distance = 2.0 * 3.0 = 6.0
```

{: .important }
> You must include the `atr` key in the position's `metadata` dictionary. If it is missing, the ATR stop cannot compute a distance and will raise an error.

---

## Time Stop

A time-based stop exits a position after a specified duration, regardless of price. This is useful for mean-reversion strategies or event-driven trades where the thesis has a limited time horizon.

```python
from puffin.risk import TimeStop

# Exit after 10 bars
stop = TimeStop(max_bars=10)

# Or exit after 1 hour
stop = TimeStop(max_seconds=3600)
```

---

## Stop Loss Manager

The `StopLossManager` orchestrates multiple stops per position. It triggers an exit when **any** of the registered stops fires.

```python
from puffin.risk import StopLossManager, FixedStop, TimeStop

manager = StopLossManager()

# Add position
manager.add_position(position)

# Add multiple stops
manager.add_stop('AAPL', FixedStop(stop_distance=5.0))
manager.add_stop('AAPL', TimeStop(max_bars=20))

# Check all stops
if manager.check_stops('AAPL', current_price=94.0):
    print("One or more stops triggered!")

# Get current stop prices
stop_prices = manager.get_stop_prices('AAPL', current_price=100.0)
print(f"Stop prices: {stop_prices}")
```

{: .tip }
> A common production pattern is to combine a trailing stop (to lock in profits) with a time stop (to avoid holding stale positions) and a fixed stop (as a worst-case backstop).

---

## Exercises

1. **Multi-stop strategy.** Set up a `StopLossManager` with a fixed stop at $3 below entry, a trailing stop at $2.5 below the high, and a time stop of 15 bars. Simulate a price path that first rises from $100 to $108 over 5 bars, then declines to $104 over 5 bars. Which stop fires first?

2. **ATR stop vs. fixed stop.** For a stock with ATR = 4.0, compare a 2x ATR stop to a fixed $5 stop. Under what volatility conditions does the ATR stop give more room? When does it give less?

3. **Trailing stop backtest.** Using historical daily data for a stock of your choice (via `yfinance`), implement a simple trend-following strategy that enters on a 50-day high breakout and exits using a `TrailingStop` with a trail distance of 2x ATR. Compare the results against a fixed 10% stop.
