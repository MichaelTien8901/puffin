---
layout: default
title: "Mean-Variance Optimization"
parent: "Part 5: Portfolio Optimization"
nav_order: 1
---

# Mean-Variance Optimization

Mean-variance optimization (MVO), introduced by Harry Markowitz in 1952, is the cornerstone of modern portfolio theory. It formalizes the trade-off between risk and return, allowing investors to construct portfolios that lie on the **efficient frontier** -- the set of allocations offering the highest expected return for each level of risk.

## Theory

The optimization problem seeks the weight vector $w$ that maximizes the Sharpe ratio or, equivalently, minimizes portfolio variance subject to a target return constraint.

**Mathematical Formulation:**

For a portfolio with weights $w = [w_1, w_2, ..., w_n]$:

- **Portfolio Return**: $R_p = w^T \mu$
- **Portfolio Variance**: $\sigma_p^2 = w^T \Sigma w$
- **Sharpe Ratio**: $SR = \frac{R_p - r_f}{\sigma_p}$

Where:
- $\mu$ = vector of expected returns
- $\Sigma$ = covariance matrix
- $r_f$ = risk-free rate

{: .warning }
> MVO is highly sensitive to estimation errors in the expected return vector $\mu$. Small perturbations in forecasted returns can lead to dramatically different optimal weights. In practice, consider shrinkage estimators (Ledoit-Wolf) or Bayesian approaches to stabilize inputs.

## Implementation

The `MeanVarianceOptimizer` class in Puffin wraps `scipy.optimize.minimize` and provides convenience methods for common portfolio targets.

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

{: .tip }
> The `max_sharpe` method uses the tangent portfolio derivation. If short-selling is not permitted, pass `allow_short=False` to add non-negativity constraints.

## Efficient Frontier

The efficient frontier represents the set of portfolios that offer the highest expected return for each level of risk. Portfolios below the frontier are sub-optimal because an investor could achieve higher return for the same risk (or the same return at lower risk) by moving to the frontier.

### Computing the Frontier

The optimizer traces the frontier by solving the minimum-variance problem for a grid of target returns between the minimum-variance portfolio return and the maximum achievable return.

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

### Interpreting the Frontier

Key observations when examining the efficient frontier:

- The **minimum-variance portfolio** sits at the leftmost point of the frontier (lowest risk).
- The **maximum Sharpe ratio portfolio** is the point where the capital market line (drawn from the risk-free rate) is tangent to the frontier.
- Portfolios on the lower half of the "bullet" (below the minimum-variance point) are inefficient -- they have higher risk for the same return as a portfolio on the upper half.

{: .note }
> In practice, the efficient frontier is estimated from historical data and is therefore subject to sampling error. Out-of-sample performance often disappoints relative to the in-sample frontier. This motivates the risk-parity and HRP approaches covered in subsequent pages.

## Practical Considerations

### Constraint Handling

Real-world portfolios often require constraints beyond the standard long-only or fully-invested conditions:

- **Sector exposure limits** -- no more than 30% in any single sector
- **Position size bounds** -- each weight between 1% and 15%
- **Turnover constraints** -- limit trading to reduce transaction costs

The `MeanVarianceOptimizer` accepts an optional `constraints` parameter compatible with `scipy.optimize` constraint dictionaries.

### Covariance Estimation

The sample covariance matrix is a noisy estimator when the number of assets $n$ approaches or exceeds the number of observations $T$. Common remedies include:

| Estimator | Description |
|:----------|:------------|
| **Ledoit-Wolf shrinkage** | Shrinks sample covariance toward a structured target |
| **Oracle Approximating Shrinkage** | Data-driven shrinkage intensity |
| **Factor models** | Express covariance through a small number of factors |
| **Exponentially weighted** | Down-weight older observations to capture regime changes |

{: .tip }
> Puffin's `MeanVarianceOptimizer` defaults to the sample covariance but accepts a pre-computed covariance matrix via the `cov_matrix` keyword argument. Use `sklearn.covariance.LedoitWolf` for a shrinkage estimator.

## Source Code

Browse the implementation: [`puffin/portfolio/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/portfolio)
