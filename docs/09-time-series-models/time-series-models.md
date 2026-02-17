---
layout: default
title: "Part 9: Time Series Models"
nav_order: 10
permalink: /09-time-series-models/
---

# Time Series Models for Trading

Time series models are essential tools for analyzing and forecasting financial data. This chapter covers key time series techniques for algorithmic trading, including stationarity testing, ARIMA, VAR, GARCH, cointegration, and pairs trading strategies.

## Table of Contents
- [Stationarity Testing](#stationarity-testing)
- [ARIMA Models](#arima-models)
- [VAR Models](#var-models)
- [GARCH Models](#garch-models)
- [Cointegration](#cointegration)
- [Pairs Trading Strategy](#pairs-trading-strategy)


## Stationarity Testing

Stationarity is a fundamental concept in time series analysis. A stationary time series has statistical properties (mean, variance, autocorrelation) that don't change over time.

### Why Stationarity Matters

Most time series models assume stationarity. Non-stationary series can lead to:
- Spurious regressions
- Invalid statistical inference
- Poor forecasting performance

### Testing for Stationarity

The Puffin library provides two main stationarity tests:

1. **Augmented Dickey-Fuller (ADF) Test**: Null hypothesis is non-stationarity
2. **KPSS Test**: Null hypothesis is stationarity (opposite of ADF)

```python
import numpy as np
import pandas as pd
from puffin.models import test_stationarity, test_kpss, check_stationarity

# Generate example data
np.random.seed(42)
returns = pd.Series(np.random.randn(252))  # Stationary
prices = pd.Series(np.random.randn(252).cumsum())  # Non-stationary

# Test returns (should be stationary)
adf_result = test_stationarity(returns)
print(f"Returns - Stationary: {adf_result['is_stationary']}")
print(f"ADF p-value: {adf_result['p_value']:.4f}")

# Test prices (should be non-stationary)
price_result = test_stationarity(prices)
print(f"Prices - Stationary: {price_result['is_stationary']}")

# Comprehensive check using both tests
check_stationarity(returns, verbose=True)
```

### Time Series Decomposition

Decompose a series into trend, seasonal, and residual components:

```python
from puffin.models import decompose_series

# Generate seasonal data
t = np.arange(500)
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 52)
noise = np.random.randn(500)
series = pd.Series(trend + seasonal + noise)
series.index = pd.date_range('2020-01-01', periods=500, freq='D')

# Decompose
components = decompose_series(series, period=52)

print("Components:", components.keys())
# Components: trend, seasonal, residual, observed
```

### Autocorrelation Analysis

Analyze autocorrelation to understand time dependencies:

```python
from puffin.models import plot_acf_pacf, autocorrelation
import matplotlib.pyplot as plt

# Plot ACF and PACF
fig = plot_acf_pacf(returns, nlags=20)
plt.show()

# Get autocorrelation values
acf_values = autocorrelation(returns, nlags=40)
print(f"First-order autocorrelation: {acf_values[1]:.3f}")
```


## ARIMA Models

ARIMA (AutoRegressive Integrated Moving Average) models combine three components:
- **AR(p)**: Autoregression - uses past values
- **I(d)**: Integration - differencing to achieve stationarity
- **MA(q)**: Moving Average - uses past forecast errors

### Basic ARIMA Usage

```python
from puffin.models import ARIMAModel

# Create and fit ARIMA model
model = ARIMAModel(order=(1, 0, 1))  # ARIMA(1,0,1)
model.fit(returns)

# Generate forecasts
forecast = model.predict(steps=10)
print(f"10-step forecast:\n{forecast}")

# Forecast with confidence intervals
forecast_df = model.forecast(returns, horizon=10, confidence=0.95)
print(forecast_df)
```

### Automatic Order Selection

Let the model automatically select the best order using AIC:

```python
from puffin.models import auto_arima

# Automatically select best ARIMA model
model = auto_arima(returns, max_p=5, max_d=2, max_q=5)
print(f"Selected order: {model.order_}")
print(f"AIC: {model.aic_:.2f}")

# Make predictions
predictions = model.predict(steps=20)
```

### Model Diagnostics

```python
# Get model residuals
residuals = model.residuals()

# Check if residuals are white noise
residual_test = test_stationarity(residuals)
print(f"Residuals stationary: {residual_test['is_stationary']}")

# View model summary
print(model.summary())
```


## VAR Models

Vector Autoregression (VAR) models analyze relationships between multiple time series. Each variable is modeled as a function of its own past values and past values of all other variables.

### Basic VAR Usage

```python
from puffin.models import VARModel

# Create multivariate data
data = pd.DataFrame({
    'returns_spy': np.random.randn(252),
    'returns_qqq': np.random.randn(252),
    'returns_iwm': np.random.randn(252)
})

# Fit VAR model
var_model = VARModel()
var_model.fit(data, max_lags=5)

print(f"Selected lags: {var_model.lags_}")

# Forecast all variables
forecast = var_model.predict(steps=10)
print(forecast)
```

### Impulse Response Analysis

Analyze how shocks in one variable affect others:

```python
# Calculate impulse response function
irf = var_model.impulse_response(periods=20)

# Response of all variables to shock in first variable
print(f"IRF shape: {irf.shape}")
```

### Granger Causality Testing

Test whether one variable helps predict another:

```python
# Test if returns_qqq Granger-causes returns_spy
causality = var_model.granger_causality(
    caused='returns_spy',
    causing='returns_qqq',
    max_lag=5
)

for lag, tests in causality.items():
    print(f"Lag {lag}: p-value = {tests['ssr_ftest_pvalue']:.4f}")
```

### Test All Pairs for Causality

```python
from puffin.models import test_granger_causality_matrix

# Create causality matrix for all variable pairs
causality_matrix = test_granger_causality_matrix(data, max_lag=5)
print(causality_matrix)
```


## GARCH Models

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models capture time-varying volatility and volatility clustering in financial returns.

### Basic GARCH Usage

```python
from puffin.models import GARCHModel

# Generate returns with volatility clustering
np.random.seed(42)
volatility = np.ones(252)
for i in range(1, 252):
    volatility[i] = 0.1 + 0.8 * volatility[i-1]**2
returns_with_vol = np.random.randn(252) * volatility
returns_series = pd.Series(returns_with_vol)

# Fit GARCH(1,1) model
garch_model = GARCHModel(p=1, q=1, model='garch')
garch_model.fit(returns_series)

# Get conditional volatility
conditional_vol = garch_model.conditional_volatility
print(f"Current volatility: {conditional_vol.iloc[-1]:.4f}")
```

### Volatility Forecasting

```python
# Forecast volatility
vol_forecast = garch_model.forecast_volatility(horizon=10)
print("Volatility forecast:")
print(vol_forecast)

# Full forecast with mean and variance
full_forecast = garch_model.forecast(horizon=10)
print(full_forecast)
```

### Model Variants

```python
# EGARCH model (captures asymmetry)
egarch = GARCHModel(p=1, q=1, model='egarch')
egarch.fit(returns_series)

# GJR-GARCH model (captures leverage effect)
gjr = GARCHModel(p=1, q=1, model='gjr-garch')
gjr.fit(returns_series)
```

### Comparing Multiple GARCH Models

```python
from puffin.models import fit_garch_models

# Fit multiple specifications
models = fit_garch_models(returns_series, max_p=2, max_q=2)

# Find best model by AIC
best_model_name = min(models.keys(), key=lambda k: models[k].results_.aic)
best_model = models[best_model_name]
print(f"Best model: {best_model_name}")
print(f"AIC: {best_model.results_.aic:.2f}")
```

### Rolling Volatility Forecasts

```python
from puffin.models import rolling_volatility_forecast

# Generate rolling 1-step ahead volatility forecasts
rolling_vol = rolling_volatility_forecast(
    returns_series,
    window=252,
    horizon=1,
    p=1,
    q=1
)

print(f"Rolling volatility forecasts:\n{rolling_vol.tail()}")
```


## Cointegration

Cointegration occurs when two or more non-stationary time series share a common stochastic trend. Their linear combination is stationary, making them suitable for pairs trading.

### Engle-Granger Test

Test two series for cointegration:

```python
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

### Finding Cointegrated Pairs

Search for cointegrated pairs in a universe of assets:

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

### Spread Analysis

Calculate and analyze the spread between cointegrated pairs:

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

### Johansen Test for Multiple Series

Test for cointegration among multiple series:

```python
from puffin.models import johansen_test

# Test for cointegration relationships
result = johansen_test(prices[['AAPL', 'MSFT', 'GOOGL']])
print(f"Number of cointegration relationships: {result['n_cointegrated']}")
print(f"Trace statistics:\n{result['trace_statistic']}")
```


## Pairs Trading Strategy

Pairs trading exploits temporary deviations from the equilibrium relationship between cointegrated assets.

### Basic Pairs Trading

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


## Complete Example: Pairs Trading System

Here's a complete example combining all components:

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


## Key Takeaways

1. **Stationarity is fundamental**: Always test for stationarity before applying time series models.

2. **ARIMA for forecasting**: Use ARIMA models to forecast univariate time series. Auto-selection helps find optimal parameters.

3. **VAR for multivariate analysis**: VAR models capture relationships between multiple time series and test for Granger causality.

4. **GARCH for volatility**: GARCH models are essential for volatility forecasting and risk management.

5. **Cointegration for pairs trading**: Cointegration identifies pairs with long-term equilibrium relationships suitable for statistical arbitrage.

6. **Mean reversion**: Pairs trading exploits mean reversion in spreads between cointegrated assets.

7. **Half-life matters**: Shorter half-lives indicate faster mean reversion and more trading opportunities.

8. **Risk management**: Always include transaction costs and use proper position sizing in pairs trading.


## Next Steps

- Explore advanced time series models (state-space models, regime-switching models)
- Implement dynamic hedge ratio estimation
- Add risk management layers (stop-loss, position limits)
- Develop multi-pair portfolio optimization
- Integrate with live trading systems

For more details, see the [Puffin API documentation](https://github.com/yourusername/puffin) and explore the examples in the repository.
