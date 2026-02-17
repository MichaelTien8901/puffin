---
layout: default
title: "Part 10: Bayesian ML"
nav_order: 11
---

# Bayesian Machine Learning for Trading

Bayesian methods provide a principled framework for incorporating uncertainty into trading decisions. Unlike traditional frequentist approaches that provide point estimates, Bayesian inference gives us full probability distributions over parameters and predictions, allowing for better risk management and decision-making under uncertainty.

## Table of Contents
- [Introduction to Bayesian Inference](#introduction-to-bayesian-inference)
- [PyMC Basics](#pymc-basics)
- [Bayesian Linear Regression](#bayesian-linear-regression)
- [Bayesian Sharpe Ratio](#bayesian-sharpe-ratio)
- [Dynamic Hedge Ratios for Pairs Trading](#dynamic-hedge-ratios)
- [Stochastic Volatility Models](#stochastic-volatility-models)
- [Practical Examples](#practical-examples)

## Introduction to Bayesian Inference

Bayesian inference is based on Bayes' theorem:

```
P(θ|data) = P(data|θ) × P(θ) / P(data)
```

Where:
- **P(θ|data)** is the **posterior**: what we learn about parameters θ after seeing the data
- **P(data|θ)** is the **likelihood**: how probable the data is given parameters θ
- **P(θ)** is the **prior**: our beliefs about θ before seeing the data
- **P(data)** is the **evidence**: normalizing constant

### Why Bayesian Methods for Trading?

1. **Uncertainty Quantification**: Get probability distributions, not just point estimates
2. **Incorporate Prior Knowledge**: Use domain expertise through informative priors
3. **Robust to Overfitting**: Bayesian regularization naturally prevents overfitting
4. **Adaptive Learning**: Update beliefs as new data arrives
5. **Better Risk Management**: Understand full distribution of possible outcomes

## PyMC Basics

PyMC is a Python library for probabilistic programming using Markov Chain Monte Carlo (MCMC) sampling.

### Installation

```bash
pip install pymc arviz
```

### Simple Example

```python
import pymc as pm
import numpy as np

# Generate some data
np.random.seed(42)
true_mean = 5.0
data = np.random.normal(true_mean, 2.0, 100)

# Build Bayesian model
with pm.Model() as model:
    # Prior on mean
    mu = pm.Normal('mu', mu=0, sigma=10)

    # Prior on standard deviation
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Likelihood
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)

    # Sample from posterior
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Analyze results
import arviz as az
print(az.summary(trace))
az.plot_posterior(trace)
```

## Bayesian Linear Regression

Traditional OLS gives point estimates. Bayesian linear regression provides full posterior distributions over coefficients, enabling uncertainty-aware predictions.

### Basic Usage

```python
from puffin.models.bayesian import BayesianLinearRegression
import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(42)
n = 200
X = np.random.randn(n, 3)
y = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.5

# Fit Bayesian model
model = BayesianLinearRegression()
model.fit(X, y, samples=2000, tune=1000)

# Get summary statistics
summary = model.summary()
print(summary)

# Make predictions with uncertainty
X_test = np.random.randn(10, 3)
mean_pred, (lower, upper) = model.predict(X_test, hdi_prob=0.94)

print(f"Predictions: {mean_pred}")
print(f"94% Credible Interval: [{lower}, {upper}]")
```

### Trading Application: Factor Model with Uncertainty

```python
import yfinance as yf
from puffin.models.bayesian import BayesianLinearRegression

# Download data
spy = yf.download('SPY', start='2020-01-01', end='2023-12-31')['Adj Close']
stock = yf.download('AAPL', start='2020-01-01', end='2023-12-31')['Adj Close']

# Calculate returns
spy_returns = spy.pct_change().dropna()
stock_returns = stock.pct_change().dropna()

# Align data
returns_df = pd.DataFrame({
    'stock': stock_returns,
    'market': spy_returns
}).dropna()

# Fit Bayesian factor model
X = returns_df[['market']].values
y = returns_df['stock'].values

factor_model = BayesianLinearRegression()
factor_model.fit(X, y, samples=2000, tune=1000)

# Get beta with uncertainty
summary = factor_model.summary()
print("Beta (market exposure):")
print(summary['beta'])

# Visualize posterior
factor_model.plot_posterior('beta')
```

## Bayesian Sharpe Ratio

The Sharpe ratio is a key metric for strategy evaluation, but traditional estimates don't account for estimation uncertainty. Bayesian methods provide the full distribution of possible Sharpe ratios.

### Basic Usage

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

### Strategy Comparison

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

### Key Advantages

1. **Accounts for Estimation Uncertainty**: Small sample sizes have wider credible intervals
2. **Robust to Outliers**: Uses Student's t-distribution instead of normal
3. **Probability Statements**: Can ask "What's the probability this strategy has positive Sharpe?"
4. **Principled Comparison**: Compare strategies accounting for uncertainty

## Dynamic Hedge Ratios

In pairs trading, the hedge ratio (beta between two assets) changes over time. Bayesian methods allow us to track this with uncertainty quantification.

### Basic Pairs Trading

```python
from puffin.models.bayesian import BayesianPairsTrading
import yfinance as yf

# Download pair
start_date = '2022-01-01'
end_date = '2023-12-31'

stock_y = yf.download('PEP', start=start_date, end=end_date)['Adj Close']
stock_x = yf.download('KO', start=start_date, end=end_date)['Adj Close']

# Initialize model
pairs_model = BayesianPairsTrading()

# Fit dynamic hedge ratios
hedge_ratios = pairs_model.fit_dynamic_hedge(
    stock_y,
    stock_x,
    window=60  # 60-day rolling window
)

print(hedge_ratios.tail())
```

Output:
```
            hedge_ratio_mean  hedge_ratio_std    spread
Date
2023-12-22             1.23             0.08      2.45
2023-12-23             1.24             0.09      2.31
2023-12-26             1.22             0.08      2.67
2023-12-27             1.23             0.08      2.52
2023-12-28             1.25             0.09      2.18
```

### Generate Trading Signals

```python
# Generate signals based on spread z-score
signals = pairs_model.generate_signals(
    entry_threshold=2.0,   # Enter when |z-score| > 2
    exit_threshold=0.5      # Exit when |z-score| < 0.5
)

# Backtest signals
returns_y = stock_y.pct_change()
returns_x = stock_x.pct_change()

# Strategy return: signal * (y - hedge_ratio * x)
strategy_returns = signals * (
    returns_y - hedge_ratios['hedge_ratio_mean'] * returns_x
)

print(f"Strategy Sharpe: {strategy_returns.mean() / strategy_returns.std() * np.sqrt(252):.2f}")
```

### Visualize Hedge Ratio Evolution

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot prices
axes[0].plot(stock_y.index, stock_y.values, label='PEP')
axes[0].plot(stock_x.index, stock_x.values, label='KO')
axes[0].set_title('Asset Prices')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot hedge ratio with uncertainty
axes[1].plot(hedge_ratios.index, hedge_ratios['hedge_ratio_mean'], label='Hedge Ratio')
axes[1].fill_between(
    hedge_ratios.index,
    hedge_ratios['hedge_ratio_mean'] - 2 * hedge_ratios['hedge_ratio_std'],
    hedge_ratios['hedge_ratio_mean'] + 2 * hedge_ratios['hedge_ratio_std'],
    alpha=0.3,
    label='95% Credible Interval'
)
axes[1].set_title('Dynamic Hedge Ratio')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot spread with entry/exit thresholds
spread_zscore = (hedge_ratios['spread'] - hedge_ratios['spread'].rolling(20).mean()) / \
                hedge_ratios['spread'].rolling(20).std()

axes[2].plot(spread_zscore.index, spread_zscore.values, label='Spread Z-Score')
axes[2].axhline(y=2.0, color='r', linestyle='--', label='Entry Threshold')
axes[2].axhline(y=-2.0, color='r', linestyle='--')
axes[2].axhline(y=0.5, color='g', linestyle='--', label='Exit Threshold')
axes[2].axhline(y=-0.5, color='g', linestyle='--')
axes[2].set_title('Spread Z-Score')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Stochastic Volatility Models

Stochastic volatility (SV) models treat volatility as a latent time-varying process, providing more flexible modeling than GARCH.

### Model Specification

The standard SV model:
```
r_t = σ_t × ε_t                    (returns equation)
log(σ_t²) = μ + φ(log(σ_{t-1}²) - μ) + η_t   (volatility equation)
```

Where:
- `σ_t`: volatility at time t (unobserved)
- `μ`: long-run mean of log-volatility
- `φ`: persistence (0 < φ < 1)
- `ε_t, η_t`: independent standard normal innovations

### Basic Usage

```python
from puffin.models.stochastic_vol import StochasticVolatilityModel
import yfinance as yf

# Download data
spy = yf.download('SPY', start='2022-01-01', end='2023-12-31')
returns = spy['Adj Close'].pct_change().dropna()

# Fit stochastic volatility model
sv_model = StochasticVolatilityModel()
sv_model.fit(returns, samples=2000, tune=1000)

# Extract volatility path
vol_path = sv_model.volatility_path
vol_forecast = sv_model.volatility_forecast

print(f"Current volatility: {vol_path[-1]:.4f}")
print(f"Next-period forecast: {vol_forecast:.4f}")

# Get parameter estimates
summary = sv_model.summary()
print(summary)
```

### Visualize Volatility

```python
# Plot volatility over time
fig = sv_model.plot_volatility()
plt.show()

# Plot parameter posteriors
fig = sv_model.plot_posterior()
plt.show()
```

### Quick Volatility Estimation

For faster estimation without storing the full model:

```python
from puffin.models.stochastic_vol import estimate_volatility_regime

# Quick estimation
vol_df = estimate_volatility_regime(returns, samples=1000)

print(vol_df.tail())
```

Output:
```
            volatility  vol_lower  vol_upper
Date
2023-12-22      0.0092     0.0078     0.0108
2023-12-23      0.0095     0.0080     0.0112
2023-12-26      0.0089     0.0075     0.0105
2023-12-27      0.0091     0.0077     0.0107
2023-12-28      0.0093     0.0079     0.0109
```

### Trading Applications

#### 1. Volatility-Adjusted Position Sizing

```python
# Scale positions by inverse volatility
target_vol = 0.01  # 1% daily vol target

position_sizes = target_vol / vol_df['volatility']

# Cap maximum leverage
position_sizes = position_sizes.clip(upper=2.0)
```

#### 2. Dynamic Stop Losses

```python
# Set stops at 2 standard deviations
stop_distance = 2 * vol_df['volatility']

# Wider stops in high vol, tighter in low vol
```

#### 3. Volatility Regime Detection

```python
# Identify high/low volatility regimes
vol_median = vol_df['volatility'].median()

regime = pd.Series('normal', index=vol_df.index)
regime[vol_df['volatility'] > 1.5 * vol_median] = 'high_vol'
regime[vol_df['volatility'] < 0.5 * vol_median] = 'low_vol'

print(regime.value_counts())
```

## Practical Examples

### Example 1: Strategy Selection with Uncertainty

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

### Example 2: Walk-Forward Pairs Trading

```python
from puffin.models.bayesian import BayesianPairsTrading
import yfinance as yf
import pandas as pd

# Download multiple pairs
pairs = [
    ('XOM', 'CVX'),   # Energy
    ('JPM', 'BAC'),   # Banks
    ('PG', 'KO')      # Consumer
]

results = []

for stock1, stock2 in pairs:
    # Download data
    data1 = yf.download(stock1, start='2022-01-01', end='2023-12-31')['Adj Close']
    data2 = yf.download(stock2, start='2022-01-01', end='2023-12-31')['Adj Close']

    # Fit pairs model
    pairs_model = BayesianPairsTrading()
    hedge_df = pairs_model.fit_dynamic_hedge(data1, data2, window=60)

    # Generate signals
    signals = pairs_model.generate_signals(entry_threshold=2.0)

    # Calculate returns
    ret1 = data1.pct_change()
    ret2 = data2.pct_change()
    strategy_ret = signals * (ret1 - hedge_df['hedge_ratio_mean'] * ret2)

    # Compute performance
    sharpe = strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)

    results.append({
        'pair': f'{stock1}-{stock2}',
        'sharpe': sharpe,
        'mean_hedge': hedge_df['hedge_ratio_mean'].mean(),
        'hedge_std': hedge_df['hedge_ratio_std'].mean()
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('sharpe', ascending=False))
```

### Example 3: Volatility-Scaled Portfolio

```python
from puffin.models.stochastic_vol import estimate_volatility_regime
import yfinance as yf
import numpy as np

# Download multiple assets
tickers = ['SPY', 'QQQ', 'IWM', 'EFA']
data = yf.download(tickers, start='2022-01-01', end='2023-12-31')['Adj Close']
returns = data.pct_change().dropna()

# Estimate volatility for each asset
volatilities = {}
for ticker in tickers:
    vol_df = estimate_volatility_regime(returns[ticker], samples=1000)
    volatilities[ticker] = vol_df['volatility']

vol_df = pd.DataFrame(volatilities)

# Equal risk contribution portfolio
# Weight inversely proportional to volatility
weights = (1 / vol_df).div((1 / vol_df).sum(axis=1), axis=0)

# Calculate portfolio returns
portfolio_returns = (returns * weights).sum(axis=1)

# Compare to equal-weight
equal_weight_returns = returns.mean(axis=1)

print(f"Equal Risk Sharpe: {portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252):.2f}")
print(f"Equal Weight Sharpe: {equal_weight_returns.mean() / equal_weight_returns.std() * np.sqrt(252):.2f}")
```

## Best Practices

### 1. Prior Selection

- **Weakly Informative Priors**: Balance between regularization and data influence
- **Domain Knowledge**: Use reasonable ranges based on market behavior
- **Sensitivity Analysis**: Check robustness to prior choice

### 2. MCMC Diagnostics

```python
import arviz as az

# Check convergence
print(az.summary(trace, var_names=['beta', 'sigma']))

# Look for:
# - r_hat close to 1.0 (< 1.01)
# - High effective sample size (ess_bulk, ess_tail)

# Visual diagnostics
az.plot_trace(trace)
az.plot_rank(trace)
```

### 3. Computational Efficiency

- **Start Small**: Use fewer samples for development, scale up for production
- **Parallel Chains**: Use `cores=4` parameter in `pm.sample()`
- **Vectorization**: Process multiple assets in parallel
- **Caching**: Store fitted models for reuse

### 4. Production Considerations

```python
# Save fitted model
import pickle

with open('bayesian_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load later
with open('bayesian_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

## Summary

Bayesian methods provide powerful tools for trading:

1. **Uncertainty Quantification**: Full probability distributions enable better risk management
2. **Adaptive Learning**: Update beliefs as new data arrives
3. **Robust Inference**: Handle outliers and fat tails naturally
4. **Principled Comparison**: Compare strategies accounting for estimation uncertainty

Key implementations in Puffin:
- `BayesianLinearRegression`: Factor models with uncertainty
- `bayesian_sharpe()`: Robust strategy evaluation
- `BayesianPairsTrading`: Dynamic hedge ratios
- `StochasticVolatilityModel`: Time-varying volatility

In the next chapter, we'll explore advanced Bayesian techniques including hierarchical models and Bayesian optimization for strategy selection.

## Further Reading

- [*Bayesian Methods for Hackers*](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) by Cameron Davidson-Pilon
- [*Statistical Rethinking*](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath
- [PyMC Documentation](https://www.pymc.io/)
- [ArviZ Documentation](https://arviz-devs.github.io/arviz/)
- "Bayesian Methods in Finance" by Rachev et al.
