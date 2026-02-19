---
layout: default
title: "Benchmark Comparison"
parent: "Part 25: Monitoring & Analytics"
nav_order: 2
---

# Benchmark Comparison

## Overview

A strategy that returns 15% annually sounds impressive -- until you learn the S&P 500 returned 20% over the same period. Benchmark comparison is the practice of measuring your strategy's performance relative to a passive alternative so you can determine whether the complexity of active management is justified.

The `BenchmarkComparison` class in Puffin computes the standard set of relative performance metrics and provides visualization tools for cumulative returns, excess returns, and rolling statistics.

{: .note }
The most common benchmark for US equity strategies is the SPDR S&P 500 ETF (SPY). For sector-specific or international strategies, choose a benchmark that reflects the investable universe of your strategy.

## Core Metrics

Before diving into code, it helps to understand what each metric tells you:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Alpha** | `mean(strategy) - beta * mean(benchmark)` | Excess return not explained by market exposure |
| **Beta** | `cov(strategy, benchmark) / var(benchmark)` | Sensitivity to benchmark moves |
| **Information Ratio** | `mean(excess) / std(excess)` | Risk-adjusted outperformance (higher is better) |
| **Tracking Error** | `std(excess)` | Volatility of the return difference |
| **Correlation** | `corr(strategy, benchmark)` | How closely strategy follows benchmark |
| **Outperformance** | `sum(strategy) - sum(benchmark)` | Total cumulative excess return |

{: .tip }
An information ratio above 0.5 is generally considered good; above 1.0 is exceptional. Most active managers struggle to sustain an IR above 0.5 after costs.

## Basic Comparison

Pass strategy and benchmark return series to `compare()` to get all metrics in a single dictionary.

```python
from puffin.monitor.benchmark import BenchmarkComparison
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

{: .warning }
Both series must be aligned on the same index (dates). The `compare()` method drops rows where either series has `NaN`, but misaligned indices will silently produce wrong results. Always verify alignment before calling `compare()`.

## Interpreting the Results

### Alpha and Beta

Alpha and beta come from the capital asset pricing model (CAPM). Beta tells you how much your strategy moves when the benchmark moves one percent. Alpha tells you how much excess return your strategy generates after adjusting for that market exposure.

```python
# A beta of 1.2 means:
# If SPY goes up 1%, your strategy goes up ~1.2%
# If SPY goes down 1%, your strategy goes down ~1.2%

# A positive alpha means:
# Your strategy earns more than what beta alone would predict
```

### Information Ratio and Tracking Error

The information ratio normalises your excess return by its volatility. A high IR means you are consistently beating the benchmark, not just occasionally getting lucky. The tracking error tells you how "different" your strategy is from the benchmark.

```python
# High tracking error + high IR = actively different strategy that works
# High tracking error + low IR = actively different but not adding value
# Low tracking error + any IR = closet indexing (not much differentiation)
```

## Visualization

The `plot_comparison()` method produces a four-panel figure: cumulative returns, excess returns, rolling correlation, and a metrics summary table.

```python
import matplotlib.pyplot as plt

# Plot comparison
fig = bc.plot_comparison(strategy_returns, benchmark_returns)
plt.show()
```

The comparison plot includes:
1. **Cumulative returns** -- strategy vs benchmark equity curves on the same axes
2. **Excess returns** -- bar chart of per-period outperformance or underperformance
3. **Rolling correlation** -- 20-period rolling window showing regime changes
4. **Performance metrics** -- text summary of all computed statistics

{: .tip }
Save the figure to disk for inclusion in client reports or strategy review decks: `fig.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')`.

## Rolling Metrics

Point-in-time metrics can be misleading because they depend on the start and end dates. Rolling metrics show how alpha, beta, and the information ratio evolve over time, helping you detect strategy decay or regime changes.

```python
# Calculate rolling performance metrics
rolling = bc.rolling_metrics(
    strategy_returns,
    benchmark_returns,
    window=60  # 60-day rolling window
)

print(rolling.tail())
```

### Plotting Rolling Alpha

```python
import matplotlib.pyplot as plt

rolling['alpha'].plot(figsize=(12, 6))
plt.title('Rolling 60-Day Alpha')
plt.ylabel('Alpha')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.show()
```

### Plotting Rolling Beta

```python
rolling['beta'].plot(figsize=(12, 6))
plt.title('Rolling 60-Day Beta')
plt.ylabel('Beta')
plt.axhline(y=1, color='black', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.show()
```

{: .note }
A rolling beta that drifts significantly from its historical average may indicate that your strategy is changing its market exposure, which could be intentional (dynamic hedging) or a bug.

## Integration with P&L Tracker

In practice, you derive your strategy returns from the `PnLTracker` and fetch benchmark returns from a data provider.

```python
from puffin.monitor.pnl import PnLTracker
from puffin.monitor.benchmark import BenchmarkComparison
import pandas as pd

# Assume tracker has accumulated history
tracker = PnLTracker(initial_cash=100_000.0)
# ... trades and updates happen here ...

# Derive daily strategy returns from P&L history
strategy_returns = tracker.daily_pnl() / tracker.initial_cash

# Fetch benchmark returns (e.g., from yfinance)
# benchmark_returns = get_spy_returns()
benchmark_returns = pd.Series([0.005, 0.01, -0.003, 0.008, 0.012])

bc = BenchmarkComparison()
metrics = bc.compare(strategy_returns, benchmark_returns)

print(f"Alpha: {metrics['alpha']:.4f}")
print(f"Sharpe vs Benchmark (IR): {metrics['ir']:.2f}")
```

## System Health Monitoring

Beyond benchmark comparison, Puffin includes a `SystemHealth` class for monitoring the operational health of your data feeds and broker connections.

### Data Feed Health

```python
from puffin.monitor.health import SystemHealth

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

### Custom Alerts and Overall Health

```python
# Send custom alerts
health.alert("Low liquidity detected", level='warning')
health.alert("Order execution failed", level='error')
health.alert("Daily loss limit reached", level='critical')

# Get overall system health
overall = health.get_overall_health()
print(f"Overall status: {overall['status']}")
print(f"Checks: {overall['checks']}")
```

{: .warning }
Always wire up an `alert_callback` in production. The default behaviour is to log only; without a callback you will miss critical alerts if you are not watching the log stream.

## Best Practices

1. **Benchmark Selection**
   - Choose a benchmark that reflects your investable universe
   - For long-short strategies, use a risk-free rate or zero as the benchmark
   - For sector strategies, use the corresponding sector ETF

2. **Rolling Windows**
   - Use a window large enough for statistical significance (60+ observations)
   - Track rolling metrics daily and flag significant regime changes
   - Monitor rolling correlation for unexpected decorrelation events

3. **System Health**
   - Check data feed and broker health on every bar
   - Set up alert notifications (email, Slack, SMS) for production
   - Monitor data feed latency -- stale data leads to stale signals
   - Track broker connection status and heartbeat intervals

## Source Code

The benchmark comparison and system health implementations live in the following files:

- [`puffin/monitor/benchmark.py`](https://github.com/MichaelTien8901/puffin/blob/main/puffin/monitor/benchmark.py) -- `BenchmarkComparison` with `compare()`, `plot_comparison()`, and `rolling_metrics()`
- [`puffin/monitor/health.py`](https://github.com/MichaelTien8901/puffin/blob/main/puffin/monitor/health.py) -- `SystemHealth` with data feed checks, broker checks, and alerting
