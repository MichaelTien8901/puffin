---
layout: default
title: "Stochastic Volatility"
parent: "Part 10: Bayesian ML"
nav_order: 3
---

# Stochastic Volatility & Bayesian Pairs Trading

This page covers two advanced Bayesian applications for trading: dynamic hedge ratios for pairs trading and stochastic volatility models. Both leverage PyMC's MCMC sampling to provide uncertainty-aware estimates that update as new data arrives.

{: .note }
> Unlike GARCH models that specify volatility as a deterministic function of past returns, stochastic volatility models treat volatility as a latent random process -- providing more flexible modeling of volatility dynamics.

## Dynamic Hedge Ratios for Pairs Trading

In pairs trading, the hedge ratio (beta between two assets) changes over time. Bayesian methods allow us to track this with uncertainty quantification, so we know not only the current hedge ratio but also how confident we are in it.

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

{: .highlight }
> The `hedge_ratio_std` column tells you how uncertain the hedge ratio estimate is. When uncertainty is high, consider reducing position sizes or widening entry thresholds.

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

## Trading Applications

### 1. Volatility-Adjusted Position Sizing

```python
# Scale positions by inverse volatility
target_vol = 0.01  # 1% daily vol target

position_sizes = target_vol / vol_df['volatility']

# Cap maximum leverage
position_sizes = position_sizes.clip(upper=2.0)
```

### 2. Dynamic Stop Losses

```python
# Set stops at 2 standard deviations
stop_distance = 2 * vol_df['volatility']

# Wider stops in high vol, tighter in low vol
```

### 3. Volatility Regime Detection

```python
# Identify high/low volatility regimes
vol_median = vol_df['volatility'].median()

regime = pd.Series('normal', index=vol_df.index)
regime[vol_df['volatility'] > 1.5 * vol_median] = 'high_vol'
regime[vol_df['volatility'] < 0.5 * vol_median] = 'low_vol'

print(regime.value_counts())
```

{: .important }
> Volatility regime detection can be used as a meta-strategy: reduce position sizes or switch to mean-reversion strategies during high-volatility regimes, and increase trend-following exposure during low-volatility regimes.

## Practical Examples

### Walk-Forward Pairs Trading

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

### Volatility-Scaled Portfolio

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

### Prior Selection

- **Weakly Informative Priors**: Balance between regularization and data influence
- **Domain Knowledge**: Use reasonable ranges based on market behavior
- **Sensitivity Analysis**: Check robustness to prior choice

### Computational Efficiency

- **Start Small**: Use fewer samples for development, scale up for production
- **Parallel Chains**: Use `cores=4` parameter in `pm.sample()`
- **Vectorization**: Process multiple assets in parallel
- **Caching**: Store fitted models for reuse

### Production Considerations

```python
# Save fitted model
import pickle

with open('bayesian_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load later
with open('bayesian_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

{: .warning }
> Bayesian models can be computationally expensive. For production use, consider fitting models during off-market hours and caching results. The `estimate_volatility_regime` function provides a faster alternative when full posterior analysis is not needed.

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
