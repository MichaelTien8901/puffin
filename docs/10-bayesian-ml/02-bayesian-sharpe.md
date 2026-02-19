---
layout: default
title: "Bayesian Sharpe Ratio"
parent: "Part 10: Bayesian ML"
nav_order: 2
---

# Bayesian Sharpe Ratio

The Sharpe ratio is a key metric for strategy evaluation, but traditional estimates don't account for estimation uncertainty. A strategy with 252 days of data might have a Sharpe of 1.5, but how confident are we in that number? Bayesian methods provide the full distribution of possible Sharpe ratios, enabling principled strategy comparison and selection.

{: .note }
> The Bayesian Sharpe ratio uses a Student's t-distribution for returns rather than a normal distribution, making it more robust to the fat tails commonly observed in financial data.

## Basic Usage

```python
from puffin.models.bayesian import bayesian_sharpe
import numpy as np

# Simulate strategy returns
np.random.seed(42)
returns = np.random.randn(252) * 0.015 + 0.0008  # Daily returns

# Compute Bayesian Sharpe ratio
sharpe_stats = bayesian_sharpe(returns, samples=5000)

print(f"Posterior Mean Sharpe: {sharpe_stats['mean']:.2f}")
print(f"94% HDI: [{sharpe_stats['hdi_low']:.2f}, {sharpe_stats['hdi_high']:.2f}]")
print(f"P(Sharpe > 0): {sharpe_stats['prob_positive']:.1%}")
```

The `bayesian_sharpe` function returns a dictionary with:

| Key | Description |
|-----|-------------|
| `mean` | Posterior mean of the annualized Sharpe ratio |
| `std` | Posterior standard deviation |
| `hdi_low` | Lower bound of the 94% Highest Density Interval |
| `hdi_high` | Upper bound of the 94% Highest Density Interval |
| `prob_positive` | Posterior probability that the Sharpe ratio is positive |

{: .important }
> The width of the HDI depends on sample size. A strategy with only 60 days of data will have a much wider interval than one with 3 years of data -- reflecting the greater uncertainty in the estimate.

## Strategy Comparison

Comparing strategies using point estimates of the Sharpe ratio can be misleading. Bayesian comparison accounts for estimation uncertainty and tells you the probability that one strategy truly outperforms another.

```python
from puffin.models.bayesian import compare_strategies_bayesian
import numpy as np

# Simulate three strategies
np.random.seed(42)
momentum_returns = np.random.randn(252) * 0.012 + 0.001
mean_reversion_returns = np.random.randn(252) * 0.010 + 0.0008
ml_strategy_returns = np.random.randn(252) * 0.015 + 0.0012

# Compare strategies
comparison = compare_strategies_bayesian({
    'Momentum': momentum_returns,
    'Mean Reversion': mean_reversion_returns,
    'ML Strategy': ml_strategy_returns
}, samples=5000)

print(comparison)
```

Output:
```
   rank        strategy  sharpe_mean  sharpe_std  hdi_low  hdi_high  prob_positive
0     1     ML Strategy         1.27        0.18     0.92      1.61           1.00
1     2        Momentum         1.13        0.17     0.80      1.45           1.00
2     3  Mean Reversion         1.01        0.16     0.70      1.32           1.00
```

{: .highlight }
> Notice that while ML Strategy ranks first, its HDI overlaps with Momentum's. This means we cannot conclude with high confidence that ML Strategy is truly better -- a nuance that point estimates would miss entirely.

## Key Advantages

1. **Accounts for Estimation Uncertainty**: Small sample sizes have wider credible intervals
2. **Robust to Outliers**: Uses Student's t-distribution instead of normal
3. **Probability Statements**: Can ask "What's the probability this strategy has positive Sharpe?"
4. **Principled Comparison**: Compare strategies accounting for uncertainty

## Example: Strategy Selection with Uncertainty

This example compares five strategies with different risk-return characteristics and visualizes the results with credible intervals.

```python
from puffin.models.bayesian import bayesian_sharpe, compare_strategies_bayesian
import numpy as np
import pandas as pd

# Simulate 5 strategies with different characteristics
np.random.seed(42)
n_days = 252

strategies = {
    'High Sharpe Low Vol': np.random.randn(n_days) * 0.008 + 0.0010,
    'High Sharpe High Vol': np.random.randn(n_days) * 0.020 + 0.0025,
    'Low Sharpe Low Vol': np.random.randn(n_days) * 0.005 + 0.0003,
    'Negative Sharpe': np.random.randn(n_days) * 0.015 - 0.0005,
    'Zero Sharpe': np.random.randn(n_days) * 0.012,
}

# Compare all strategies
results = compare_strategies_bayesian(strategies, samples=5000)
print(results)

# Visualize results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(results))
ax.barh(x, results['sharpe_mean'])
ax.errorbar(
    results['sharpe_mean'],
    x,
    xerr=[
        results['sharpe_mean'] - results['hdi_low'],
        results['hdi_high'] - results['sharpe_mean']
    ],
    fmt='none',
    color='black',
    capsize=5
)

ax.set_yticks(x)
ax.set_yticklabels(results['strategy'])
ax.set_xlabel('Sharpe Ratio')
ax.set_title('Strategy Comparison (with 94% Credible Intervals)')
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

{: .warning }
> When comparing strategies, ensure the return series cover the same time period. Comparing a strategy tested during a bull market with one tested during a bear market will produce misleading results regardless of the statistical method used.

## When to Use Bayesian Sharpe

The Bayesian approach to Sharpe ratio estimation is particularly valuable when:

- **Limited data**: The strategy has less than 2 years of live or backtest data
- **Strategy selection**: Choosing among multiple candidate strategies for deployment
- **Risk budgeting**: Allocating capital proportional to confidence in each strategy
- **Regime changes**: Assessing whether a strategy's performance has genuinely shifted

For strategies with extensive track records (5+ years), the frequentist and Bayesian estimates will largely agree, but the Bayesian approach still provides the useful `prob_positive` metric.

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
