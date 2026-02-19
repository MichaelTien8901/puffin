---
layout: default
title: "Portfolio Risk Controls"
parent: "Part 24: Risk Management"
nav_order: 3
---

# Portfolio Risk Controls

Individual position sizing and stop losses are necessary but not sufficient. Portfolio-level risk controls prevent catastrophic losses by monitoring aggregate exposure, drawdown, and concentration. The `puffin.risk.portfolio_risk` module provides the `PortfolioRiskManager` class for this purpose.

```python
from puffin.risk import PortfolioRiskManager
import pandas as pd

rm = PortfolioRiskManager()
```

---

## Drawdown Monitoring

{: .tip }
> **Plain English:** Drawdown is the "max pain" -- the deepest valley your account falls into from a peak. If your account grew to $110,000 then dropped to $90,000, that's a $20,000 (18%) drawdown. It answers the question: "What's the worst it got before things improved?"

The `check_drawdown` method compares the current drawdown against a maximum threshold and returns a tuple of `(ok, current_dd)`:

```python
equity_curve = pd.Series([100000, 105000, 110000, 95000, 98000])

# Check drawdown
ok, current_dd = rm.check_drawdown(equity_curve, max_dd=0.15)

if not ok:
    print(f"Warning: Drawdown {current_dd:.1%} exceeds limit!")
```

{: .note }
> The method calculates a running maximum over the equity curve and measures the current value's distance from that peak. A drawdown of 0.1364 (13.6%) in the example above is within the 15% limit, so `ok` would be `True`.

---

## Exposure Limits

Total portfolio exposure should be bounded to prevent over-leveraging. The `check_exposure` method sums position weights and compares against a maximum:

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

{: .note }
> The `Position` dataclass in `puffin.risk.portfolio_risk` has five fields: `ticker`, `quantity`, `current_price`, `market_value`, and `weight` (portfolio weight). This is different from the `Position` in `puffin.risk.stop_loss`, which tracks entry price and time for stop calculations.

---

## Circuit Breaker

A circuit breaker halts all trading when drawdown exceeds a critical threshold. Once triggered, trading remains halted until a manual reset:

```python
# Circuit breaker at 20% drawdown
triggered = rm.circuit_breaker(equity_curve, threshold=0.20)

if triggered:
    print("Circuit breaker triggered! Trading halted.")
    # Stop all trading until manual reset

# Reset when ready to resume
rm.reset_circuit_breaker()
```

{: .warning }
> A circuit breaker is your last line of defense. Set the threshold high enough that normal volatility does not trip it, but low enough that it catches genuine tail events before they become fatal.

---

## Value at Risk (VaR) and Expected Shortfall

Value at Risk (VaR) estimates the maximum loss at a given confidence level over a specified time horizon. Expected Shortfall (also called CVaR or Conditional VaR) goes further by averaging the losses in the tail beyond VaR.

```python
returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02])

# Historical VaR at 95% confidence
var = rm.compute_var(returns, confidence=0.95, method='historical')
print(f"95% VaR: {var:.2%}")

# Expected Shortfall (CVaR)
es = rm.compute_expected_shortfall(returns, confidence=0.95)
print(f"Expected Shortfall: {es:.2%}")
```

{: .tip }
> VaR answers "What is the worst loss I can expect on 19 out of 20 days?" Expected Shortfall answers "Given that today is one of those bad 1-in-20 days, how bad could it actually get?" Expected Shortfall is generally considered a superior risk measure because it captures tail risk.

---

## Concentration Metrics

Concentrated portfolios amplify idiosyncratic risk. The `concentration_metrics` method computes the Herfindahl-Hirschman Index (HHI), the maximum single-position weight, and the combined weight of the top five positions:

```python
metrics = rm.concentration_metrics(positions)

print(f"HHI: {metrics['hhi']:.3f}")
print(f"Max position: {metrics['max_weight']:.1%}")
print(f"Top 5 positions: {metrics['top5_weight']:.1%}")
```

**Interpretation:**
- HHI close to 1.0 indicates a highly concentrated portfolio
- HHI close to 0 indicates a well-diversified portfolio
- Max weight above 20% suggests risky concentration in a single name

---

## Complete Example

Here is a complete risk management workflow that ties together position sizing, stop losses, and portfolio-level controls:

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

---

{: .tip }
> **Plain English -- Sharpe Ratio:** The Sharpe Ratio is a score for your trading efficiency. Think of it as: "How much stress do I have to endure for every dollar I make?" A high Sharpe means smooth, consistent returns. A low Sharpe means a roller coaster ride -- is the destination really worth it?

---

## Exercises

1. **Drawdown analysis.** Generate a synthetic equity curve using a random walk (`numpy.random.normal`) over 252 trading days. Compute the maximum drawdown. Repeat 1,000 times and plot the distribution of maximum drawdowns. What is the median worst drawdown for a strategy with a 10% annualized return and 15% annualized volatility?

2. **VaR comparison.** Using daily returns for SPY over the past year (via `yfinance`), compute the 95% and 99% historical VaR. Then compute the parametric (Gaussian) VaR assuming normally distributed returns. How do the two methods compare? Which is more conservative?

3. **Circuit breaker calibration.** A strategy has a Sharpe ratio of 1.5 and annualized volatility of 12%. What drawdown threshold should you set for the circuit breaker so that it triggers less than once per year on average? (Hint: simulate equity paths and measure circuit breaker frequency at different thresholds.)

4. **Concentration monitoring.** Build a portfolio of 10 positions with random weights drawn from a Dirichlet distribution. Compute the HHI and max-weight metrics using `rm.concentration_metrics()`. How does the HHI change as you vary the Dirichlet concentration parameter?
