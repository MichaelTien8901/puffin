---
layout: default
title: "Position Sizing"
parent: "Part 24: Risk Management"
nav_order: 1
---

# Position Sizing

Position sizing determines how much capital to allocate to each trade. Getting this right is arguably more important than the entry signal itself -- even a mediocre strategy can survive if position sizes are managed well, and a great strategy can blow up if they are not.

The `puffin.risk.position_sizing` module provides three main approaches: fixed fractional, Kelly criterion, and volatility-based sizing.

---

## Fixed Fractional

The fixed fractional method risks a fixed percentage of equity on each trade. It is the simplest and most widely used approach.

**Formula:**

```
Position Size = (Equity x Risk%) / Stop Distance
```

For example, risking 2% of a $100,000 account with a $5 stop distance gives a position of 400 shares.

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

{: .note }
> The function raises `ValueError` if equity is non-positive, `risk_pct` is outside (0, 1], or `stop_distance` is non-positive.

---

## Kelly Criterion

The Kelly Criterion calculates the theoretically optimal position size based on your win rate and the ratio of average win to average loss. It maximizes the long-run geometric growth rate of your equity.

**Formula:**

```
Kelly% = (Win Rate x Win/Loss Ratio - (1 - Win Rate)) / Win/Loss Ratio
```

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

{: .warning }
> **Full Kelly can be aggressive.** Full Kelly sizing maximizes long-run growth but can produce severe drawdowns along the way. In practice, most traders use half Kelly (`fraction=0.5`) or quarter Kelly (`fraction=0.25`) for more conservative sizing. The `fraction` parameter defaults to `0.5` for this reason.

---

## Volatility-Based Sizing

Volatility-based sizing adjusts position size using ATR (Average True Range) so that each trade risks roughly the same dollar amount regardless of how volatile the instrument is.

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
- High volatility leads to smaller positions
- Low volatility leads to larger positions
- Automatically adapts to changing market conditions

{: .tip }
> Volatility-based sizing is especially useful when trading a diversified portfolio of instruments with different volatility profiles. It normalizes risk across positions so that a single high-volatility name does not dominate the portfolio's P&L.

---

## Exercises

1. **Compare sizing methods.** Using a $50,000 account, calculate the position size for a stock at $120 with a $4 stop distance using both fixed fractional (2% risk) and volatility-based sizing (ATR = 2.5, multiplier = 2.0). Which gives a larger position? Why?

2. **Kelly sensitivity.** Compute the full Kelly percentage for win rates of 0.45, 0.50, 0.55, and 0.60, all with a win/loss ratio of 1.5. At what win rate does Kelly first recommend a positive allocation? What happens at 0.45?

3. **Fractional Kelly.** A strategy has a 60% win rate and a 2.0 win/loss ratio. Calculate the full Kelly, half Kelly, and quarter Kelly allocations. Plot a simulated equity curve for each over 200 trades (you can generate random win/loss outcomes using `numpy.random`).
