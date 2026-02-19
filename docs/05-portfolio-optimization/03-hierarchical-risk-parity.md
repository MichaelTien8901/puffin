---
layout: default
title: "Hierarchical Risk Parity"
parent: "Part 5: Portfolio Optimization"
nav_order: 3
---

# Hierarchical Risk Parity

Hierarchical Risk Parity (HRP), developed by Marcos Lopez de Prado, combines hierarchical clustering with inverse-variance allocation to create more stable and robust portfolios. Unlike mean-variance optimization, HRP does not require matrix inversion of the covariance matrix, making it numerically stable even when the number of assets is large relative to the sample size.

## Overview

**Algorithm Steps:**

1. **Tree Clustering**: Group assets based on correlation distance
2. **Quasi-Diagonalization**: Reorder covariance matrix to group similar assets
3. **Recursive Bisection**: Allocate weights through the hierarchy using inverse-variance weighting

### Advantages

- More stable out-of-sample performance
- Less sensitive to estimation errors
- No need for matrix inversion (avoids numerical instability)
- Intuitive interpretation through hierarchical structure

{: .note }
> HRP was introduced in Lopez de Prado (2016), "Building Diversified Portfolios that Outperform Out of Sample". The paper demonstrates that HRP outperforms both mean-variance and risk parity in Monte Carlo experiments with estimation noise.

## Implementation

Puffin provides several HRP utilities in `puffin.portfolio`:

| Function | Description |
|:---------|:------------|
| `hrp_weights` | Returns a NumPy array of HRP weights |
| `hrp_weights_with_names` | Returns a pandas Series with asset names as the index |
| `plot_dendrogram` | Visualizes the hierarchical clustering tree |
| `hrp_allocation_stats` | Detailed per-asset allocation statistics |

### Computing HRP Weights

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from puffin.portfolio import (
    hrp_weights,
    hrp_weights_with_names,
    plot_dendrogram,
    hrp_allocation_stats
)

# Load historical returns
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Calculate HRP weights
hrp_w = hrp_weights_with_names(returns)
print("HRP Weights:")
print(hrp_w)

# Visualize hierarchical clustering
plt.figure(figsize=(12, 6))
linkage_matrix, dend = plot_dendrogram(returns, linkage_method='single')
plt.title('Asset Clustering Dendrogram')
plt.xlabel('Assets')
plt.ylabel('Distance')
plt.show()

# Get detailed allocation statistics
stats = hrp_allocation_stats(returns, hrp_w.values)
print("\nHRP Allocation Statistics:")
print(stats)
```

{: .tip }
> The dendrogram is one of the most useful diagnostic plots in HRP. Assets that merge at low distance are highly correlated and will share a subtree. If two assets you consider distinct merge early, investigate whether their correlation has increased recently.

### Comparing Linkage Methods

The choice of linkage method affects how the hierarchy is built and, consequently, the final weights. The four standard methods are:

- **Single**: Merges clusters by the minimum pairwise distance (can produce long chains)
- **Complete**: Merges by maximum pairwise distance (produces compact clusters)
- **Average**: Uses the mean pairwise distance (a compromise)
- **Ward**: Minimizes within-cluster variance (tends to produce balanced trees)

```python
# Test different clustering methods
methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, method in enumerate(methods):
    weights = hrp_weights(returns, linkage_method=method)

    axes[i].bar(range(len(returns.columns)), weights)
    axes[i].set_xticks(range(len(returns.columns)))
    axes[i].set_xticklabels(returns.columns, rotation=45)
    axes[i].set_title(f'HRP Weights ({method.capitalize()} Linkage)')
    axes[i].set_ylabel('Weight')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

{: .warning }
> Different linkage methods can produce materially different weight vectors. Always compare several methods and validate with out-of-sample backtests before committing to one.

## Performance Analysis with Tearsheets

Once portfolio weights are determined, it is essential to evaluate performance using standard risk-return metrics. Puffin provides a tearsheet module for this purpose.

### Computing Portfolio Statistics

```python
from puffin.portfolio import (
    compute_stats,
    generate_tearsheet,
    print_tearsheet_summary
)

# Create portfolio returns using HRP weights
portfolio_returns = (returns * hrp_w.values).sum(axis=1)

# Compute statistics
stats = compute_stats(portfolio_returns, risk_free_rate=0.02, periods_per_year=252)

print("Portfolio Performance:")
print(f"Annual Return: {stats['annual_return']:.2%}")
print(f"Annual Volatility: {stats['annual_vol']:.2%}")
print(f"Sharpe Ratio: {stats['sharpe']:.3f}")
print(f"Sortino Ratio: {stats['sortino']:.3f}")
print(f"Maximum Drawdown: {stats['max_dd']:.2%}")
print(f"Calmar Ratio: {stats['calmar']:.3f}")
print(f"VaR (95%): {stats['var_95']:.2%}")
print(f"CVaR (95%): {stats['cvar_95']:.2%}")
```

### Generating Tearsheets

A tearsheet aggregates cumulative returns, drawdowns, rolling statistics, and benchmark comparisons into a single report.

```python
# Generate comprehensive tearsheet
# If you have a benchmark (e.g., S&P 500 returns)
benchmark_returns = pd.read_csv('sp500_returns.csv', index_col=0, parse_dates=True)

tearsheet = generate_tearsheet(
    portfolio_returns,
    benchmark=benchmark_returns.squeeze(),
    risk_free_rate=0.02
)

# Print formatted summary
print_tearsheet_summary(tearsheet)

# Access specific metrics
print(f"\nBeta: {tearsheet.get('beta', 'N/A'):.3f}")
print(f"Alpha: {tearsheet.get('alpha', 'N/A'):.2%}")
print(f"Information Ratio: {tearsheet.get('information_ratio', 'N/A'):.3f}")
```

### Visualization

```python
from puffin.portfolio import (
    plot_returns,
    plot_drawdown,
    plot_monthly_returns,
    plot_rolling_metrics
)

# Cumulative returns
fig1 = plot_returns(portfolio_returns, benchmark=benchmark_returns.squeeze())
plt.show()

# Drawdown analysis
fig2 = plot_drawdown(portfolio_returns)
plt.show()

# Monthly returns heatmap
fig3 = plot_monthly_returns(portfolio_returns)
plt.show()

# Rolling performance metrics
fig4 = plot_rolling_metrics(portfolio_returns, window=252)
plt.show()
```

{: .tip }
> The rolling Sharpe chart (`plot_rolling_metrics`) is especially useful for detecting regime changes. A sustained drop in the 1-year rolling Sharpe may indicate that the correlation structure has shifted and the portfolio needs re-clustering.

## Portfolio Rebalancing

### Rebalancing Strategies

After constructing optimal weights, portfolios drift as asset prices move. Rebalancing restores the target allocation but incurs transaction costs. Puffin's `RebalanceEngine` supports several scheduling strategies.

```python
from puffin.portfolio import (
    RebalanceEngine,
    CostModel,
    rebalance_schedule,
    backtest_rebalancing
)

# Define transaction cost model
cost_model = CostModel(
    commission_pct=0.001,      # 0.1% commission
    commission_fixed=1.0,      # $1 fixed fee
    slippage_pct=0.0005,       # 0.05% slippage
    min_commission=1.0         # $1 minimum
)

# Create rebalancing engine
engine = RebalanceEngine(cost_model=cost_model)

# Define target weights (e.g., using HRP)
target_weights = dict(zip(returns.columns, hrp_w.values))

# Monthly rebalancing schedule
monthly_schedule = rebalance_schedule(strategy='monthly')

# Backtest the strategy
backtest_result = backtest_rebalancing(
    returns,
    target_weights,
    monthly_schedule,
    initial_value=100000.0,
    cost_model=cost_model
)

print("Rebalancing Backtest Results:")
print(f"Final Portfolio Value: ${backtest_result['portfolio_value'].iloc[-1]:,.2f}")
print(f"Total Transaction Costs: ${backtest_result['transaction_costs'].iloc[-1]:,.2f}")
print(f"Number of Rebalances: {backtest_result['rebalanced'].sum()}")
```

### Threshold-Based Rebalancing

Calendar-based rebalancing (monthly, quarterly) trades on a fixed schedule regardless of drift. Threshold-based rebalancing only triggers when weights deviate beyond a tolerance, often reducing turnover and costs.

```python
# Rebalance only when weights drift significantly
threshold_schedule = rebalance_schedule(strategy='threshold', threshold=0.05)

threshold_result = backtest_rebalancing(
    returns,
    target_weights,
    threshold_schedule,
    initial_value=100000.0,
    cost_model=cost_model
)

print(f"\nThreshold Rebalancing:")
print(f"Number of Rebalances: {threshold_result['rebalanced'].sum()}")
print(f"Total Transaction Costs: ${threshold_result['transaction_costs'].iloc[-1]:,.2f}")
```

### Cost-Aware Rebalancing

The most sophisticated approach performs a cost-benefit analysis before each potential rebalance, only executing trades when the expected improvement exceeds the transaction costs.

```python
# Current portfolio state (after drift)
current_weights = {
    'AAPL': 0.27,
    'GOOGL': 0.24,
    'MSFT': 0.26,
    'AMZN': 0.23
}

# Current prices
prices = {
    'AAPL': 150.0,
    'GOOGL': 2800.0,
    'MSFT': 300.0,
    'AMZN': 3200.0
}

# Decide whether to rebalance based on cost-benefit analysis
result = engine.optimize_with_costs(
    current_weights,
    target_weights,
    portfolio_value=100000.0,
    prices=prices,
    cost_threshold=0.001  # Only rebalance if benefit exceeds 0.1% of portfolio
)

print("\nCost-Benefit Analysis:")
print(f"Should Rebalance: {result['should_rebalance']}")
print(f"Expected Benefit: ${result['expected_benefit']:.2f}")
print(f"Transaction Costs: ${result['total_cost']:.2f}")
print(f"Benefit-Cost Ratio: {result['benefit_cost_ratio']:.2f}")

if result['should_rebalance']:
    print("\nProposed Trades:")
    for trade in result['trades']:
        action = "BUY" if trade.quantity > 0 else "SELL"
        print(f"  {action} {abs(trade.quantity):.2f} shares of {trade.symbol} @ ${trade.price:.2f}")
```

{: .warning }
> The cost-benefit analysis uses expected tracking error reduction as the "benefit". This estimate is only as good as the covariance matrix. In volatile markets, consider using a shorter lookback window for covariance estimation.

### Comparing Rebalancing Strategies

```python
from puffin.portfolio import compare_rebalancing_strategies

# Compare multiple strategies
strategies = ['monthly', 'quarterly', 'threshold']
comparison = compare_rebalancing_strategies(
    returns,
    target_weights,
    strategies=strategies,
    initial_value=100000.0,
    cost_model=cost_model
)

# Plot comparison
plt.figure(figsize=(12, 6))
for strategy, result in comparison.items():
    plt.plot(result.index, result['portfolio_value'], label=strategy.capitalize())

plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.title('Rebalancing Strategy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Summary statistics
print("\nStrategy Comparison:")
for strategy, result in comparison.items():
    final_value = result['portfolio_value'].iloc[-1]
    total_costs = result['transaction_costs'].iloc[-1]
    n_rebalances = result['rebalanced'].sum()

    print(f"{strategy.capitalize()}:")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total Costs: ${total_costs:,.2f}")
    print(f"  Rebalances: {n_rebalances}")
    print(f"  Return: {(final_value/100000 - 1)*100:.2f}%")
    print()
```

## Complete Example: Multi-Strategy Portfolio

This example brings together all three optimization methods -- mean-variance, risk parity, and HRP -- backtests each with monthly rebalancing, and compares their performance side by side.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from puffin.portfolio import (
    MeanVarianceOptimizer,
    risk_parity_weights,
    hrp_weights,
    compute_stats,
    plot_returns,
    RebalanceEngine,
    CostModel,
    rebalance_schedule,
    backtest_rebalancing
)

# Load data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Strategy 1: Maximum Sharpe Ratio
optimizer = MeanVarianceOptimizer()
max_sharpe = optimizer.max_sharpe(returns, risk_free_rate=0.02)
sharpe_weights = dict(zip(returns.columns, max_sharpe['weights']))

# Strategy 2: Risk Parity
rp_w = risk_parity_weights(returns)
rp_weights = dict(zip(returns.columns, rp_w))

# Strategy 3: Hierarchical Risk Parity
hrp_w = hrp_weights(returns)
hrp_weights_dict = dict(zip(returns.columns, hrp_w))

# Backtest each strategy
cost_model = CostModel(commission_pct=0.001, slippage_pct=0.0005)
schedule = rebalance_schedule(strategy='monthly')

strategies = {
    'Max Sharpe': sharpe_weights,
    'Risk Parity': rp_weights,
    'HRP': hrp_weights_dict
}

results = {}
for name, weights in strategies.items():
    result = backtest_rebalancing(
        returns,
        weights,
        schedule,
        initial_value=100000.0,
        cost_model=cost_model
    )
    results[name] = result

# Compare performance
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Portfolio values
for name, result in results.items():
    axes[0].plot(result.index, result['portfolio_value'], label=name, linewidth=2)

axes[0].set_xlabel('Date')
axes[0].set_ylabel('Portfolio Value ($)')
axes[0].set_title('Strategy Performance Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative returns
for name, result in results.items():
    cumulative = (result['portfolio_value'] / 100000 - 1) * 100
    axes[1].plot(result.index, cumulative, label=name, linewidth=2)

axes[1].set_xlabel('Date')
axes[1].set_ylabel('Cumulative Return (%)')
axes[1].set_title('Cumulative Returns')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("Strategy Performance Summary:")
print("=" * 70)
for name, result in results.items():
    # Calculate portfolio returns
    portfolio_returns = result['portfolio_value'].pct_change().dropna()
    stats = compute_stats(portfolio_returns, risk_free_rate=0.02/252, periods_per_year=252)

    print(f"\n{name}:")
    print(f"  Final Value: ${result['portfolio_value'].iloc[-1]:,.2f}")
    print(f"  Total Return: {(result['portfolio_value'].iloc[-1]/100000 - 1)*100:.2f}%")
    print(f"  Annual Return: {stats['annual_return']:.2%}")
    print(f"  Annual Volatility: {stats['annual_vol']:.2%}")
    print(f"  Sharpe Ratio: {stats['sharpe']:.3f}")
    print(f"  Max Drawdown: {stats['max_dd']:.2%}")
    print(f"  Transaction Costs: ${result['transaction_costs'].iloc[-1]:,.2f}")
```

## Source Code

Browse the implementation: [`puffin/portfolio/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/portfolio)
