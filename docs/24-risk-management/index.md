---
layout: default
title: "Part 24: Risk Management"
nav_order: 25
---

# Risk Management

Risk management is critical for long-term trading success. The Puffin framework provides comprehensive tools for position sizing, stop loss management, and portfolio-level risk controls.

## Position Sizing

Position sizing determines how much capital to allocate to each trade. The `puffin.risk.position_sizing` module provides three main approaches:

### Fixed Fractional

The fixed fractional method risks a fixed percentage of equity on each trade:

```python
from puffin.risk import fixed_fractional

# Risk 2% of $100,000 equity with a $5 stop
position_size = fixed_fractional(
    equity=100000,
    risk_pct=0.02,  # 2%
    stop_distance=5.0
)
print(f"Position size: {position_size} shares")  # 400 shares
```

**Formula:** Position Size = (Equity × Risk%) / Stop Distance

### Kelly Criterion

The Kelly Criterion calculates optimal position size based on win rate and win/loss ratio:

```python
from puffin.risk import kelly_criterion

# 55% win rate, 1.5 win/loss ratio
kelly_pct = kelly_criterion(
    win_rate=0.55,
    win_loss_ratio=1.5,
    fraction=0.5  # Half Kelly for conservatism
)
print(f"Optimal position size: {kelly_pct*100:.1f}% of equity")
```

**Formula:** Kelly% = (Win Rate × Win/Loss Ratio - (1 - Win Rate)) / Win/Loss Ratio

**Warning:** Full Kelly can be aggressive. Use half Kelly (fraction=0.5) or quarter Kelly (fraction=0.25) for more conservative sizing.

### Volatility-Based

Volatility-based sizing adjusts position size based on ATR (Average True Range):

```python
from puffin.risk import volatility_based

# Adjust for volatility
position_size = volatility_based(
    equity=100000,
    atr=3.0,
    risk_pct=0.02,
    multiplier=2.0  # 2x ATR stop
)
print(f"Position size: {position_size} shares")
```

**Benefits:**
- High volatility → smaller positions
- Low volatility → larger positions
- Automatically adapts to market conditions

## Stop Loss Management

Stop losses protect against excessive losses. The `puffin.risk.stop_loss` module provides multiple stop loss strategies.

### Fixed Stop

A fixed stop loss at a specified distance from entry:

```python
from puffin.risk import FixedStop, Position
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

### Trailing Stop

A trailing stop follows favorable price movement:

```python
from puffin.risk import TrailingStop

# 5-point trailing stop
stop = TrailingStop(trail_distance=5.0, price_based=True)

# As price rises, stop trails behind
# Entry: $100, Price moves to $110
# Stop now at $105 (5 points below high)
```

### ATR Stop

An ATR-based stop adapts to volatility:

```python
from puffin.risk import ATRStop

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

### Time Stop

A time-based stop exits after a duration:

```python
from puffin.risk import TimeStop

# Exit after 10 bars
stop = TimeStop(max_bars=10)

# Or exit after 1 hour
stop = TimeStop(max_seconds=3600)
```

### Stop Loss Manager

Manage multiple stops per position:

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

## Portfolio Risk Management

Portfolio-level risk controls prevent catastrophic losses.

### Drawdown Monitoring

{: .tip }
> **Plain English:** Drawdown is the "max pain" — the deepest valley your account falls into from a peak. If your account grew to $110,000 then dropped to $90,000, that's a $20,000 (18%) drawdown. It answers the question: "What's the worst it got before things improved?"

Monitor and limit drawdown:

```python
from puffin.risk import PortfolioRiskManager
import pandas as pd

rm = PortfolioRiskManager()

equity_curve = pd.Series([100000, 105000, 110000, 95000, 98000])

# Check drawdown
ok, current_dd = rm.check_drawdown(equity_curve, max_dd=0.15)

if not ok:
    print(f"Warning: Drawdown {current_dd:.1%} exceeds limit!")
```

### Exposure Limits

Control total portfolio exposure:

```python
from puffin.risk.portfolio_risk import Position

positions = [
    Position('AAPL', 100, 150, 15000, 0.5),
    Position('GOOGL', 50, 200, 10000, 0.33)
]

ok, exposure = rm.check_exposure(positions, max_exposure=1.0)

if not ok:
    print(f"Warning: Exposure {exposure:.1%} exceeds limit!")
```

### Circuit Breaker

Halt trading when drawdown exceeds threshold:

```python
# Circuit breaker at 20% drawdown
triggered = rm.circuit_breaker(equity_curve, threshold=0.20)

if triggered:
    print("Circuit breaker triggered! Trading halted.")
    # Stop all trading until manual reset

# Reset when ready to resume
rm.reset_circuit_breaker()
```

{: .tip }
> **Plain English — Sharpe Ratio:** The Sharpe Ratio is a score for your trading efficiency. Think of it as: "How much stress do I have to endure for every dollar I make?" A high Sharpe means smooth, consistent returns. A low Sharpe means a roller coaster ride — is the destination really worth it?

### Value at Risk (VaR)

Calculate potential loss at confidence level:

```python
returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])

# Historical VaR at 95% confidence
var = rm.compute_var(returns, confidence=0.95, method='historical')
print(f"95% VaR: {var:.2%}")

# Expected Shortfall (CVaR)
es = rm.compute_expected_shortfall(returns, confidence=0.95)
print(f"Expected Shortfall: {es:.2%}")
```

### Concentration Metrics

Monitor portfolio concentration:

```python
metrics = rm.concentration_metrics(positions)

print(f"HHI: {metrics['hhi']:.3f}")
print(f"Max position: {metrics['max_weight']:.1%}")
print(f"Top 5 positions: {metrics['top5_weight']:.1%}")
```

**Interpretation:**
- HHI close to 1.0 → highly concentrated
- HHI close to 0 → well diversified
- Max weight > 20% → risky concentration

## Complete Example

Here's a complete risk management workflow:

```python
from puffin.risk import (
    PortfolioRiskManager,
    StopLossManager,
    FixedStop,
    TrailingStop,
    volatility_based,
)
import pandas as pd

# Initialize risk managers
portfolio_rm = PortfolioRiskManager()
stop_manager = StopLossManager()

# Position sizing
equity = 100000
atr = 3.0
position_size = volatility_based(
    equity=equity,
    atr=atr,
    risk_pct=0.02,
    multiplier=2.0
)

# Create position
from puffin.risk.stop_loss import Position
from datetime import datetime

position = Position(
    ticker='AAPL',
    entry_price=150.0,
    entry_time=datetime.now(),
    quantity=position_size,
    side='long',
    metadata={'atr': atr}
)

# Set up stops
stop_manager.add_position(position)
stop_manager.add_stop('AAPL', FixedStop(stop_distance=6.0))
stop_manager.add_stop('AAPL', TrailingStop(trail_distance=4.0))

# Monitor in trading loop
current_price = 155.0

# Check stops
if stop_manager.check_stops('AAPL', current_price):
    print("Stop triggered - exit position")

# Check portfolio risk
equity_curve = pd.Series([100000, 102000, 103000, 101000])
ok, dd = portfolio_rm.check_drawdown(equity_curve, max_dd=0.10)

if not ok:
    print(f"Drawdown alert: {dd:.1%}")

# Circuit breaker
if portfolio_rm.circuit_breaker(equity_curve, threshold=0.15):
    print("HALT: Circuit breaker triggered")
```

## Best Practices

1. **Position Sizing**
   - Never risk more than 1-2% per trade
   - Use volatility-adjusted sizing
   - Scale position size with confidence

2. **Stop Losses**
   - Always use stops
   - Place stops beyond normal volatility
   - Consider using multiple stop types
   - Trail stops on profitable trades

3. **Portfolio Risk**
   - Monitor drawdown continuously
   - Limit total exposure
   - Implement circuit breakers
   - Track concentration metrics
   - Calculate VaR regularly

4. **Risk/Reward**
   - Target at least 2:1 reward/risk ratio
   - Use wider stops in volatile markets
   - Tighten stops in trending markets

## Next Steps

- See [Monitoring & Analytics](/docs/25-monitoring-analytics/) for performance tracking
- See [Live Trading](/docs/23-live-trading/01-live-trading) for integrating risk management
- Check `examples/` for complete implementations
