---
layout: default
title: "Execution Models"
parent: "Part 7: Backtesting"
nav_order: 2
---

# Execution Models

## Overview

A backtest is only as good as its execution model. If you assume zero trading costs and perfect fills, your results will be wildly optimistic. This chapter covers slippage and commission models -- the two components that bridge the gap between theoretical signals and realistic performance.

## Slippage Models

Slippage is the difference between the expected fill price and the actual fill price. It arises from the bid-ask spread, market impact, and latency. The `SlippageModel` dataclass supports three modes:

### Fixed Slippage

A constant dollar amount per share, independent of price. Suitable for liquid large-cap stocks with tight spreads.

```python
from puffin.backtest import SlippageModel

# Fixed slippage: $0.01 per share
SlippageModel(fixed=0.01)
```

### Percentage Slippage

A fraction of the fill price. Better for modeling slippage across stocks at different price levels or for less liquid names.

```python
# Percentage slippage: 0.1% of price
SlippageModel(pct=0.001)
```

### Combined Slippage

Use both components together for a more realistic model that captures a base spread plus proportional market impact.

```python
# Combined: $0.005 fixed + 0.05% of price
SlippageModel(fixed=0.005, pct=0.0005)
```

Under the hood, `SlippageModel.calculate(price)` returns `fixed + price * pct`. For buy orders, the slippage is added to the fill price; for sell orders, it is subtracted.

## Commission Models

Commissions are the fees charged by your broker for executing trades. The `CommissionModel` dataclass supports three fee structures:

### Flat Fee

A fixed dollar amount per order, regardless of size. Common with discount brokers.

```python
from puffin.backtest import CommissionModel

# Flat fee per order
CommissionModel(flat=5.0)
```

### Per-Share Fee

A fee proportional to the number of shares traded. Used by some institutional brokers (e.g., Interactive Brokers' tiered pricing).

```python
# Per-share fee: $0.005 per share
CommissionModel(per_share=0.005)
```

### Percentage Fee

A fee proportional to the total trade value. Common in non-US equity markets and for some crypto exchanges.

```python
# Percentage of trade value: 0.1%
CommissionModel(pct=0.001)
```

The calculation is `flat + per_share * qty + price * qty * pct`. You can combine all three components for a complex fee schedule.

{: .tip }
Start with `SlippageModel(fixed=0.01)` and `CommissionModel(flat=1.0)` as reasonable defaults for US equities. Many retail brokers now offer commission-free trading, but slippage still exists.

## Performance Metrics

After a backtest completes, call `result.metrics()` to get a dictionary of performance statistics:

```python
metrics = result.metrics()
# {
#     'total_return': 0.15,
#     'annualized_return': 0.12,
#     'annualized_volatility': 0.18,
#     'sharpe_ratio': 1.2,
#     'max_drawdown': -0.08,
#     'win_rate': 0.55,
#     'profit_factor': 1.8,
#     'total_trades': 42,
# }
```

### Key Metrics Explained

| Metric | Description | What to look for |
|--------|-------------|-----------------|
| **Total return** | Cumulative percentage gain/loss | Positive, obviously |
| **Annualized return** | Compounded annual growth rate | Compare to benchmark (S&P ~10%) |
| **Annualized volatility** | Standard deviation of returns, annualized | Lower is better at equal return |
| **Sharpe ratio** | Risk-adjusted return (excess return / volatility) | > 1.0 is good, > 2.0 is excellent |
| **Max drawdown** | Largest peak-to-trough decline | How much pain you must endure |
| **Win rate** | Fraction of winning trades | Context-dependent (trend followers are often < 50%) |
| **Profit factor** | Gross profits / gross losses | > 1.0 means profitable, > 1.5 is solid |
| **Total trades** | Number of executed fills | Too few = not statistically significant |

{: .note }
No single metric tells the full story. A high Sharpe ratio with only 5 trades is not meaningful. Always consider metrics together with the equity curve and drawdown chart.

## Visualization

The `BacktestResult` object provides a built-in plotting method that shows the equity curve and drawdown together:

```python
result.plot()  # Equity curve + drawdown chart
```

This generates a two-panel chart: the top panel shows portfolio value over time, and the bottom panel shows the drawdown (percentage decline from peak equity) in red. Look for long, deep drawdowns -- they indicate periods where the strategy is losing money and may signal regime changes.

## Exercises

1. Run a backtest with zero slippage and zero commission, then re-run with `SlippageModel(fixed=0.02)` and `CommissionModel(per_share=0.005)`. Compare the Sharpe ratios.
2. Experiment with percentage-based slippage. At what slippage level does a profitable strategy become unprofitable?
3. After running a backtest, examine the equity curve using `result.plot()`. Identify the largest drawdown period -- what market event caused it?
