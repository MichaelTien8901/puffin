---
layout: default
title: "Financial Metrics"
parent: "Part 1: Market Foundations"
nav_order: 4
---

# Financial Metrics

## Overview

To evaluate trading strategies, you need a common language of performance metrics. This chapter covers the essential metrics with Python implementations you'll use throughout the rest of the guide.

## Returns

### Simple Returns

The percentage change in price from one period to the next.

```python
import numpy as np
import pandas as pd

# Simple return
def simple_return(prices: pd.Series) -> pd.Series:
    return prices.pct_change()

# Example
prices = pd.Series([100, 102, 99, 105, 103])
returns = simple_return(prices)
# [NaN, 0.02, -0.0294, 0.0606, -0.0190]
```

### Log Returns

Log returns are additive across time, making them useful for multi-period analysis.

```python
def log_return(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))

# Log returns ≈ simple returns for small values
# but sum correctly: total_log_return = sum(daily_log_returns)
```

{: .tip }
Use **simple returns** for portfolio calculations and reporting. Use **log returns** for statistical analysis and modeling.

### Cumulative Returns

```python
def cumulative_return(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1

# Starting with $10,000
initial_capital = 10_000
equity_curve = initial_capital * (1 + returns).cumprod()
```

## Volatility

Volatility measures the dispersion of returns — how much prices fluctuate.

```python
def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)

# Example: if daily std is 1%, annualized vol ≈ 15.87%
daily_std = 0.01
annual_vol = daily_std * np.sqrt(252)  # 0.1587
```

{: .note }
252 is the typical number of trading days per year for US equities. Use 365 for crypto (24/7 markets).

## Sharpe Ratio

The **Sharpe ratio** measures risk-adjusted return — how much return you earn per unit of risk.

```python
def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
```

### Interpreting Sharpe Ratios

| Sharpe Ratio | Interpretation |
|-------------|----------------|
| < 0 | Losing money (underperforming risk-free rate) |
| 0 – 1.0 | Below average |
| 1.0 – 2.0 | Good |
| 2.0 – 3.0 | Very good |
| > 3.0 | Excellent (verify — may indicate overfitting) |

{: .warning }
A Sharpe ratio above 3.0 in backtesting often indicates **overfitting** to historical data. Be skeptical of strategies with extremely high Sharpe ratios.

## Drawdown

**Drawdown** measures the decline from a portfolio's peak value — the worst-case loss an investor would experience.

```python
def drawdown(equity_curve: pd.Series) -> pd.DataFrame:
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return pd.DataFrame({
        'equity': equity_curve,
        'peak': peak,
        'drawdown': dd
    })

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()

# Example: max_drawdown of -0.15 means the portfolio lost 15% from its peak
```

### Drawdown Duration

How long the portfolio stayed below its previous peak.

```python
def drawdown_duration(equity_curve: pd.Series) -> int:
    peak = equity_curve.cummax()
    underwater = equity_curve < peak
    # Count consecutive True values
    groups = (~underwater).cumsum()
    durations = underwater.groupby(groups).sum()
    return int(durations.max()) if len(durations) > 0 else 0
```

## Alpha and Beta

**Beta** measures a strategy's sensitivity to market movements. **Alpha** is the excess return after accounting for market exposure.

```python
def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252
) -> tuple[float, float]:
    # Beta = Cov(strategy, benchmark) / Var(benchmark)
    covariance = np.cov(strategy_returns.dropna(), benchmark_returns.dropna())
    beta = covariance[0, 1] / covariance[1, 1]

    # Alpha (annualized) = strategy_return - risk_free - beta * (benchmark_return - risk_free)
    rf_daily = risk_free_rate / periods_per_year
    alpha = (
        (strategy_returns.mean() - rf_daily)
        - beta * (benchmark_returns.mean() - rf_daily)
    ) * periods_per_year

    return alpha, beta
```

### Interpreting Alpha and Beta

| Metric | Value | Meaning |
|--------|-------|---------|
| Beta = 1.0 | — | Moves with the market |
| Beta > 1.0 | — | More volatile than the market |
| Beta < 1.0 | — | Less volatile than the market |
| Beta ≈ 0 | — | Market-neutral strategy |
| Alpha > 0 | — | Outperforming after adjusting for risk |
| Alpha < 0 | — | Underperforming after adjusting for risk |

## Win Rate and Profit Factor

Trade-level metrics for evaluating strategy quality.

```python
def trade_metrics(pnl: pd.Series) -> dict:
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    # Profit factor = gross profits / gross losses
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 else float('inf')

    return {
        'total_trades': len(pnl),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': pnl.mean(),
    }
```

| Metric | Good Value | Meaning |
|--------|-----------|---------|
| Win Rate | > 50% | More winning trades than losing (depends on payoff ratio) |
| Profit Factor | > 1.5 | Gross profits are 1.5x gross losses |
| Expectancy | > 0 | Average trade is profitable |

{: .note }
A strategy can be profitable with a low win rate (30%) if the average win is much larger than the average loss. Win rate alone is not meaningful without the payoff ratio.

## Putting It All Together

```python
def strategy_report(equity_curve: pd.Series, benchmark: pd.Series) -> dict:
    returns = equity_curve.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()
    alpha, beta = alpha_beta(returns, bench_returns)

    return {
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
        'annualized_return': (1 + returns.mean()) ** 252 - 1,
        'annualized_volatility': annualized_volatility(returns),
        'sharpe_ratio': sharpe_ratio(returns),
        'max_drawdown': max_drawdown(equity_curve),
        'alpha': alpha,
        'beta': beta,
    }
```

## Summary

- **Returns**: Simple for reporting, log for statistical analysis
- **Volatility**: Annualized standard deviation of returns (√252 scaling)
- **Sharpe ratio**: Risk-adjusted return — the single most important metric
- **Drawdown**: Peak-to-trough loss — measures worst-case pain
- **Alpha/Beta**: Performance relative to a benchmark
- **Win rate / Profit factor**: Trade-level quality metrics

## Next Steps

The next chapter explores **historical market events** and their implications for algorithmic trading.
