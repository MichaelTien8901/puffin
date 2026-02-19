---
layout: default
title: "Risk Parity"
parent: "Part 5: Portfolio Optimization"
nav_order: 2
---

# Risk Parity

Risk parity is an allocation philosophy that shifts the focus from capital weights to **risk weights**. Instead of asking "how much money should I put in each asset?", risk parity asks "how much risk should each asset contribute?". The answer -- equal risk contribution -- often leads to portfolios that are better diversified and more robust than mean-variance solutions.

## Concept

Risk parity portfolios allocate capital such that each asset contributes equally to the total portfolio risk, rather than having equal dollar weights. This approach often leads to better diversification.

**Risk Contribution:**

The risk contribution of asset $i$ is:
$$RC_i = w_i \cdot \frac{\partial \sigma_p}{\partial w_i} = w_i \cdot \frac{(\Sigma w)_i}{\sigma_p}$$

By construction, the sum of all risk contributions equals the portfolio volatility:
$$\sigma_p = \sum_{i=1}^{n} RC_i$$

A risk parity portfolio sets $RC_1 = RC_2 = \cdots = RC_n$, meaning each asset contributes $\sigma_p / n$ to total risk.

{: .note }
> Because low-volatility assets (e.g., bonds) need larger capital weights to match the risk contribution of high-volatility assets (e.g., equities), risk parity portfolios are typically bond-heavy in a stock/bond universe. Leverage is often applied to reach an equity-like return target.

## Implementation

Puffin provides three risk-based allocation functions in `puffin.portfolio`:

| Function | Description |
|:---------|:------------|
| `risk_parity_weights` | Solves for exact equal risk contribution via numerical optimization |
| `inverse_volatility_weights` | Fast closed-form approximation (weights proportional to $1/\sigma_i$) |
| `risk_contribution` | Decomposes portfolio risk into per-asset contributions |

### Equal Risk Contribution

```python
import numpy as np
import pandas as pd
from puffin.portfolio import (
    risk_parity_weights,
    inverse_volatility_weights,
    risk_contribution
)

# Load historical returns
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

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
```

{: .tip }
> After computing `risk_contribution`, verify that all values are approximately equal. If one asset dominates, the optimizer may not have converged -- try increasing the maximum number of iterations.

### Inverse Volatility Weights

The inverse-volatility approach is a simpler, closed-form approximation that works well when asset correlations are roughly uniform.

```python
# Inverse volatility portfolio (simpler alternative)
iv_weights = inverse_volatility_weights(returns)
print("\nInverse Volatility Weights:")
for asset, weight in zip(returns.columns, iv_weights):
    print(f"  {asset}: {weight:.2%}")
```

{: .warning }
> Inverse volatility ignores correlations entirely. It is a reasonable heuristic when assets have similar pairwise correlations but can be misleading when correlation structure varies significantly (e.g., a mix of equities and commodities).

## Maximum Diversification

The **maximum diversification portfolio** maximizes the diversification ratio:

$$DR = \frac{w^T \sigma}{\sqrt{w^T \Sigma w}}$$

where $\sigma$ is the vector of individual asset volatilities. The numerator is the weighted average volatility (as if assets were uncorrelated) and the denominator is the true portfolio volatility. A higher ratio indicates greater diversification benefit from correlations.

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

{: .note }
> A diversification ratio of 1.0 means no diversification benefit (perfectly correlated assets or a single-asset portfolio). Values above 1.0 indicate that correlations are reducing portfolio volatility below the weighted-average volatility.

## Comparing Risk-Based Approaches

The table below summarizes the key differences among the risk-based methods covered in this page:

| Method | Objective | Correlation-Aware? | Closed Form? |
|:-------|:----------|:--------------------|:-------------|
| Equal Weight | Minimize tracking error to cap-weight | No | Yes |
| Inverse Volatility | Approximate equal risk | No | Yes |
| Risk Parity | Exact equal risk contribution | Yes | No (numerical) |
| Maximum Diversification | Maximize diversification ratio | Yes | No (numerical) |

In practice, the choice depends on:

- **Universe size**: For a small universe (< 10 assets), the numerical approaches are fast and preferred. For hundreds of assets, inverse volatility scales better.
- **Rebalancing frequency**: Simpler methods are cheaper to rebalance.
- **Leverage policy**: Risk parity often requires leverage to achieve competitive absolute returns.

## Source Code

Browse the implementation: [`puffin/portfolio/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/portfolio)
