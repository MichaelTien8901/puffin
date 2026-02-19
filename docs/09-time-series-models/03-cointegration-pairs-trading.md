---
layout: default
title: "Cointegration & Pairs Trading"
parent: "Part 9: Time Series Models"
nav_order: 3
permalink: /09-time-series-models/cointegration-pairs-trading/
---

# Cointegration & Pairs Trading

Cointegration occurs when two or more non-stationary time series share a common stochastic trend. Their linear combination is stationary, making them suitable for mean-reversion strategies such as pairs trading.

---

## Cointegration Overview

{: .note }
> Two series are **cointegrated** if each is individually non-stationary (I(1)), but a linear combination of them is stationary (I(0)). This implies a long-run equilibrium relationship that the series reverts to over time.

Classic examples of cointegrated pairs in finance:
- Stocks of companies in the same sector (e.g., Coca-Cola and Pepsi)
- An ETF and its underlying basket of stocks
- Spot and futures prices of the same commodity

---

## Engle-Granger Test

The Engle-Granger two-step method tests two series for cointegration:
1. Regress one series on the other to obtain the residuals (spread)
2. Test the residuals for stationarity using the ADF test

```python
import numpy as np
import pandas as pd
from puffin.models import engle_granger_test

# Generate cointegrated pair
np.random.seed(42)
common_factor = np.random.randn(252).cumsum()
price1 = pd.Series(common_factor + np.random.randn(252) * 0.5)
price2 = pd.Series(2 * common_factor + np.random.randn(252) * 0.5)

# Test for cointegration
result = engle_granger_test(price1, price2)
print(f"Cointegrated: {result['is_cointegrated']}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Hedge ratio: {result['hedge_ratio']:.4f}")
```

The **hedge ratio** indicates how many units of the second asset to short for each unit of the first asset held long, so that the combined position is market-neutral with respect to the common factor.

---

## Finding Cointegrated Pairs

When working with a universe of assets, you can systematically search for all cointegrated pairs:

```python
from puffin.models import find_cointegrated_pairs

# Create price data for multiple assets
prices = pd.DataFrame({
    'AAPL': common_factor + np.random.randn(252) * 0.3,
    'MSFT': 1.5 * common_factor + np.random.randn(252) * 0.3,
    'GOOGL': np.random.randn(252).cumsum(),
    'META': np.random.randn(252).cumsum()
})

# Find all cointegrated pairs
pairs = find_cointegrated_pairs(prices, significance=0.05)

for ticker1, ticker2, p_value, hedge_ratio in pairs:
    print(f"{ticker1}-{ticker2}: p={p_value:.4f}, hedge={hedge_ratio:.4f}")
```

{: .warning }
> When testing many pairs, apply a multiple-testing correction (e.g., Bonferroni) to control the false discovery rate. With N assets, you are testing N*(N-1)/2 pairs, which inflates the chance of spurious results.

---

## Spread Analysis

Once you identify a cointegrated pair, analyze the spread to assess tradability:

```python
from puffin.models import calculate_spread, half_life, adf_test_spread

# Calculate spread
spread = calculate_spread(price1, price2, hedge_ratio=result['hedge_ratio'])

# Test spread for mean reversion
spread_test = adf_test_spread(spread)
print(f"Spread is mean-reverting: {spread_test['is_stationary']}")

# Calculate half-life of mean reversion
hl = half_life(spread)
print(f"Half-life: {hl:.2f} periods")
```

{: .important }
> The **half-life** measures how quickly the spread reverts to its mean. Shorter half-lives (5-60 days) are preferred for active trading. Very short half-lives may be dominated by transaction costs, while very long half-lives tie up capital with uncertain reversion.

---

## Johansen Test for Multiple Series

The Johansen test extends cointegration analysis to systems of more than two variables. It determines the number of cointegrating relationships (the cointegrating rank):

```python
from puffin.models import johansen_test

# Test for cointegration relationships
result = johansen_test(prices[['AAPL', 'MSFT', 'GOOGL']])
print(f"Number of cointegration relationships: {result['n_cointegrated']}")
print(f"Trace statistics:\n{result['trace_statistic']}")
```

The Johansen test uses two statistics:
- **Trace statistic**: Tests whether the number of cointegrating relations is at most r
- **Maximum eigenvalue statistic**: Tests r against the alternative of r+1

---

## Pairs Trading Strategy

Pairs trading exploits temporary deviations from the equilibrium relationship between cointegrated assets. When the spread widens beyond a threshold, you trade expecting it to revert.

### Strategy Setup

```python
from puffin.models import PairsTradingStrategy

# Create strategy
strategy = PairsTradingStrategy(
    entry_z=2.0,   # Enter when spread is 2 std devs from mean
    exit_z=0.5,    # Exit when spread returns to 0.5 std devs
    lookback=20    # Use 20-period rolling statistics
)

# Find tradable pairs
pairs = strategy.find_pairs(
    prices,
    significance=0.05,
    min_half_life=1,
    max_half_life=60
)

print(f"Found {len(pairs)} tradable pairs")
```

### Generate Trading Signals

```python
# Select a pair to trade
pair = ('AAPL', 'MSFT')

# Compute spread
spread = strategy.compute_spread(pair, prices)

# Generate signals
signals = strategy.generate_signals(spread)

# Signals: 1 (long spread), -1 (short spread), 0 (no position)
print(f"Current signal: {signals.iloc[-1]}")
```

The signal logic:
- **Long spread** (+1): Spread is below -entry_z standard deviations (spread is unusually low, expect reversion upward)
- **Short spread** (-1): Spread is above +entry_z standard deviations (spread is unusually high, expect reversion downward)
- **Exit** (0): Spread returns within exit_z standard deviations of the mean

---

## Backtesting Pairs

### Backtest a Single Pair

```python
# Backtest the pair
results = strategy.backtest_pair(
    pair,
    prices,
    transaction_cost=0.001  # 10 bps per trade
)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Number of Trades: {results['num_trades']}")
print(f"Win Rate: {results['win_rate']:.2%}")

# Plot cumulative returns
import matplotlib.pyplot as plt
results['cumulative_returns'].plot(title='Pair Trading Performance')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.show()
```

### Backtest a Portfolio of Pairs

Diversifying across multiple pairs reduces idiosyncratic risk:

```python
# Trade multiple pairs
pairs_to_trade = [('AAPL', 'MSFT'), ('GOOGL', 'META')]

portfolio_results = strategy.backtest_portfolio(
    pairs_to_trade,
    prices,
    transaction_cost=0.001
)

print(f"Portfolio Sharpe: {portfolio_results['sharpe_ratio']:.2f}")
print(f"Portfolio Return: {portfolio_results['total_return']:.2%}")

# Individual pair results
for pair_name, pair_stats in portfolio_results['pair_results'].items():
    print(f"\n{pair_name}:")
    print(f"  Sharpe: {pair_stats['sharpe_ratio']:.2f}")
    print(f"  Return: {pair_stats['total_return']:.2%}")
```

### Rank Pairs by Performance

```python
from puffin.models import rank_pairs_by_performance

# Evaluate and rank all pairs
rankings = rank_pairs_by_performance(
    pairs,
    prices,
    metric='sharpe_ratio'
)

print("\nTop 5 Pairs by Sharpe Ratio:")
print(rankings.head())
```

---

## Complete Example: Pairs Trading System

Here is a complete example combining cointegration analysis with the pairs trading strategy:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from puffin.models import (
    find_cointegrated_pairs,
    PairsTradingStrategy,
    check_stationarity
)

# 1. Generate synthetic price data
np.random.seed(42)
n = 500

# Common factors
factor1 = np.random.randn(n).cumsum()
factor2 = np.random.randn(n).cumsum()

# Create price series
prices = pd.DataFrame({
    'Stock_A': factor1 + np.random.randn(n) * 0.5,
    'Stock_B': 1.5 * factor1 + np.random.randn(n) * 0.5,
    'Stock_C': factor2 + np.random.randn(n) * 0.5,
    'Stock_D': 0.8 * factor2 + np.random.randn(n) * 0.5,
    'Stock_E': np.random.randn(n).cumsum()  # Independent
})
prices.index = pd.date_range('2020-01-01', periods=n, freq='D')

# 2. Find cointegrated pairs
print("Finding cointegrated pairs...")
pairs = find_cointegrated_pairs(prices, significance=0.05)

print(f"\nFound {len(pairs)} cointegrated pairs:")
for ticker1, ticker2, p_value, hedge_ratio in pairs:
    print(f"  {ticker1}-{ticker2}: p={p_value:.4f}, hedge={hedge_ratio:.4f}")

# 3. Set up pairs trading strategy
strategy = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, lookback=20)

# Filter pairs by half-life
tradable_pairs = strategy.find_pairs(
    prices,
    significance=0.05,
    min_half_life=5,
    max_half_life=100
)

print(f"\n{len(tradable_pairs)} pairs meet half-life criteria")

# 4. Backtest portfolio
if tradable_pairs:
    results = strategy.backtest_portfolio(
        tradable_pairs,
        prices,
        transaction_cost=0.001
    )

    # 5. Display results
    print("\n" + "="*60)
    print("PORTFOLIO PERFORMANCE")
    print("="*60)
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")

    # Plot performance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Cumulative returns
    results['cumulative_returns'].plot(ax=ax1, title='Portfolio Cumulative Returns')
    ax1.set_ylabel('Cumulative Returns')
    ax1.grid(True)

    # Daily returns distribution
    results['returns'].hist(bins=50, ax=ax2)
    ax2.set_title('Daily Returns Distribution')
    ax2.set_xlabel('Returns')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # 6. Individual pair analysis
    print("\n" + "="*60)
    print("INDIVIDUAL PAIR PERFORMANCE")
    print("="*60)
    for pair_name, pair_stats in results['pair_results'].items():
        print(f"\n{pair_name}:")
        print(f"  Sharpe Ratio: {pair_stats['sharpe_ratio']:.2f}")
        print(f"  Total Return: {pair_stats['total_return']:.2%}")
        print(f"  Max Drawdown: {pair_stats['max_drawdown']:.2%}")
        print(f"  Number of Trades: {pair_stats['num_trades']}")
        print(f"  Win Rate: {pair_stats['win_rate']:.2%}")
```

---

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
