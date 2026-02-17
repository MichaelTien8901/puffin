---
layout: default
title: "Part 5: Portfolio Optimization"
nav_order: 6
permalink: /05-portfolio-optimization/
---

# Portfolio Optimization

## Introduction

Portfolio optimization is the process of selecting the best portfolio allocation from a set of assets, balancing expected returns against risk. This chapter covers modern portfolio theory and advanced techniques for constructing optimal portfolios.

## Markowitz Mean-Variance Optimization

### Theory

Harry Markowitz's mean-variance optimization framework forms the foundation of modern portfolio theory. The key insight is that investors should consider both expected returns and the covariance structure of returns when constructing portfolios.

The optimization problem seeks to:
- **Minimize risk** (portfolio variance) for a given level of expected return
- **Maximize return** for a given level of risk
- **Maximize the Sharpe ratio** (risk-adjusted return)

**Mathematical Formulation:**

For a portfolio with weights $w = [w_1, w_2, ..., w_n]$:

- **Portfolio Return**: $R_p = w^T \mu$
- **Portfolio Variance**: $\sigma_p^2 = w^T \Sigma w$
- **Sharpe Ratio**: $SR = \frac{R_p - r_f}{\sigma_p}$

Where:
- $\mu$ = vector of expected returns
- $\Sigma$ = covariance matrix
- $r_f$ = risk-free rate

### Implementation

```python
import numpy as np
import pandas as pd
from puffin.portfolio import MeanVarianceOptimizer

# Load historical returns
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Initialize optimizer
optimizer = MeanVarianceOptimizer()

# Find minimum variance portfolio
min_var = optimizer.min_variance(returns)
print(f"Minimum Variance Portfolio:")
print(f"Expected Return: {min_var['return']:.2%}")
print(f"Risk (Volatility): {min_var['risk']:.2%}")
print(f"Weights: {dict(zip(returns.columns, min_var['weights']))}")

# Find maximum Sharpe ratio portfolio
max_sharpe = optimizer.max_sharpe(returns, risk_free_rate=0.02)
print(f"\nMaximum Sharpe Portfolio:")
print(f"Expected Return: {max_sharpe['return']:.2%}")
print(f"Risk: {max_sharpe['risk']:.2%}")
print(f"Sharpe Ratio: {max_sharpe['sharpe']:.3f}")

# Optimize for a target return
target_return = 0.10  # 10% annual return
result = optimizer.optimize(returns, target_return=target_return)
if result:
    print(f"\nPortfolio with {target_return:.1%} target return:")
    print(f"Risk: {result['risk']:.2%}")
```

### Efficient Frontier

The efficient frontier represents the set of portfolios that offer the highest expected return for each level of risk.

```python
# Compute efficient frontier
frontier = optimizer.efficient_frontier(returns, n_points=50)

# Plot efficient frontier
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(frontier['risk'], frontier['return'], c=frontier['return']/frontier['risk'],
            cmap='viridis', s=50)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risk (Volatility)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.grid(True, alpha=0.3)

# Mark special portfolios
plt.scatter(min_var['risk'], min_var['return'], color='red', s=200,
            marker='*', label='Min Variance', zorder=3)
plt.scatter(max_sharpe['risk'], max_sharpe['return'], color='gold', s=200,
            marker='*', label='Max Sharpe', zorder=3)
plt.legend()
plt.show()
```

## Risk Parity

### Concept

Risk parity portfolios allocate capital such that each asset contributes equally to the total portfolio risk, rather than having equal dollar weights. This approach often leads to better diversification.

**Risk Contribution:**

The risk contribution of asset $i$ is:
$$RC_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}$$

### Implementation

```python
from puffin.portfolio import (
    risk_parity_weights,
    inverse_volatility_weights,
    risk_contribution
)

# Risk parity portfolio (equal risk contribution)
rp_weights = risk_parity_weights(returns)
print("Risk Parity Weights:")
for asset, weight in zip(returns.columns, rp_weights):
    print(f"  {asset}: {weight:.2%}")

# Calculate risk contributions
cov_matrix = returns.cov().values
risk_contrib = risk_contribution(rp_weights, cov_matrix)
portfolio_variance = np.dot(rp_weights, np.dot(cov_matrix, rp_weights))

print("\nRisk Contributions (as % of portfolio variance):")
for asset, rc in zip(returns.columns, risk_contrib):
    print(f"  {asset}: {rc/portfolio_variance:.2%}")

# Inverse volatility portfolio (simpler alternative)
iv_weights = inverse_volatility_weights(returns)
print("\nInverse Volatility Weights:")
for asset, weight in zip(returns.columns, iv_weights):
    print(f"  {asset}: {weight:.2%}")
```

### Maximum Diversification

```python
from puffin.portfolio import maximum_diversification_weights, diversification_ratio

# Find maximum diversification portfolio
max_div_weights = maximum_diversification_weights(returns)

# Calculate diversification ratio
div_ratio = diversification_ratio(max_div_weights, returns)
print(f"Diversification Ratio: {div_ratio:.3f}")

# Compare with equal weights
equal_weights = np.ones(len(returns.columns)) / len(returns.columns)
equal_div_ratio = diversification_ratio(equal_weights, returns)
print(f"Equal Weight Diversification Ratio: {equal_div_ratio:.3f}")
```

## Hierarchical Risk Parity (HRP)

### Overview

Hierarchical Risk Parity (HRP), developed by Marcos Lopez de Prado, combines hierarchical clustering with inverse-variance allocation to create more stable and robust portfolios.

**Algorithm Steps:**

1. **Tree Clustering**: Group assets based on correlation distance
2. **Quasi-Diagonalization**: Reorder covariance matrix to group similar assets
3. **Recursive Bisection**: Allocate weights through the hierarchy using inverse-variance weighting

### Advantages

- More stable out-of-sample performance
- Less sensitive to estimation errors
- No need for matrix inversion (avoids numerical instability)
- Intuitive interpretation through hierarchical structure

### Implementation

```python
from puffin.portfolio import (
    hrp_weights,
    hrp_weights_with_names,
    plot_dendrogram,
    hrp_allocation_stats
)

# Calculate HRP weights
hrp_w = hrp_weights_with_names(returns)
print("HRP Weights:")
print(hrp_w)

# Visualize hierarchical clustering
import matplotlib.pyplot as plt

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

### Comparing Linkage Methods

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

## Performance Analysis with Tearsheets

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

## Portfolio Rebalancing

### Rebalancing Strategies

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

## Key Takeaways

1. **Mean-Variance Optimization** provides a mathematical framework for portfolio construction but can be sensitive to input estimation errors

2. **Risk Parity** focuses on equal risk contribution rather than equal capital allocation, often leading to better diversification

3. **Hierarchical Risk Parity** combines clustering with risk parity to create more stable portfolios that are robust to estimation errors

4. **Transaction Costs Matter**: Always consider rebalancing costs and implement cost-aware strategies

5. **Multiple Strategies**: Different optimization approaches work better in different market conditions; consider combining strategies or using ensemble approaches

## Further Reading

- Markowitz, H. (1952). ["Portfolio Selection"](https://doi.org/10.2307/2975974). Journal of Finance
- Lopez de Prado, M. (2016). ["Building Diversified Portfolios that Outperform Out of Sample"](https://doi.org/10.3905/jpm.2016.42.4.059)
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios"

## Next Steps

In the next chapter, we'll explore:
- Black-Litterman model for incorporating views
- Robust portfolio optimization techniques
- Multi-period portfolio optimization
- Factor-based portfolio construction
