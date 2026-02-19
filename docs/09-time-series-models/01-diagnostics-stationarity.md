---
layout: default
title: "Diagnostics & Stationarity"
parent: "Part 9: Time Series Models"
nav_order: 1
permalink: /09-time-series-models/diagnostics-stationarity/
---

# Diagnostics & Stationarity

Stationarity is a fundamental concept in time series analysis. A stationary time series has statistical properties (mean, variance, autocorrelation) that do not change over time. Before fitting any time series model, you must verify that your data meets the stationarity assumption.

---

## Why Stationarity Matters

{: .warning }
> Most time series models (ARIMA, VAR) assume stationarity. Fitting these models to non-stationary data leads to **spurious regressions**, invalid statistical inference, and poor forecasting performance.

Non-stationary series exhibit trends, changing variance, or structural breaks. Common examples include:
- **Stock prices**: Exhibit random-walk behavior (non-stationary)
- **Stock returns**: Typically stationary (differences of log-prices)
- **Volatility**: May be stationary but with clustering effects

---

## Testing for Stationarity

The Puffin library provides two main stationarity tests that are complementary:

1. **Augmented Dickey-Fuller (ADF) Test**: Null hypothesis is non-stationarity (unit root present)
2. **KPSS Test**: Null hypothesis is stationarity (opposite of ADF)

{: .note }
> Using both tests together provides a more robust assessment. If ADF rejects and KPSS does not reject, the series is likely stationary. If both reject or neither rejects, the result is ambiguous.

### ADF Test

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
```

### KPSS Test

```python
# KPSS test (null hypothesis is stationarity)
kpss_result = test_kpss(returns)
print(f"Returns - KPSS stationary: {kpss_result['is_stationary']}")
print(f"KPSS p-value: {kpss_result['p_value']:.4f}")
```

### Comprehensive Stationarity Check

Use `check_stationarity` to run both tests and get a combined assessment:

```python
# Comprehensive check using both tests
check_stationarity(returns, verbose=True)
```

This function runs both ADF and KPSS tests and reports whether they agree on the stationarity diagnosis.

---

## Time Series Decomposition

Decomposing a time series into its structural components helps you understand the underlying patterns before modeling.

A time series can be decomposed into:
- **Trend**: Long-term movement in the level of the series
- **Seasonal**: Repeating patterns at fixed intervals
- **Residual**: Random variation after removing trend and seasonality

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

{: .important }
> The choice of decomposition method (additive vs. multiplicative) depends on whether the seasonal effect is constant (additive) or proportional to the level (multiplicative). Financial returns typically use additive decomposition.

---

## Autocorrelation Analysis

Autocorrelation measures the correlation of a time series with its own lagged values. It is essential for:
- Identifying the order of AR and MA components in ARIMA models
- Detecting seasonality and cyclical patterns
- Checking whether model residuals are white noise

### ACF and PACF

- **ACF (Autocorrelation Function)**: Shows correlation at each lag, including indirect effects through intermediate lags
- **PACF (Partial Autocorrelation Function)**: Shows the direct correlation at each lag, controlling for shorter lags

| Pattern | ACF | PACF | Suggested Model |
|:--------|:----|:-----|:----------------|
| Cuts off at lag q | Tails off | -- | MA(q) |
| Tails off | -- | Cuts off at lag p | AR(p) |
| Tails off | Tails off | -- | ARMA(p,q) |

### Plotting ACF and PACF

```python
from puffin.models import plot_acf_pacf, autocorrelation
import matplotlib.pyplot as plt

# Plot ACF and PACF
fig = plot_acf_pacf(returns, nlags=20)
plt.show()
```

### Computing Autocorrelation Values

```python
# Get autocorrelation values
acf_values = autocorrelation(returns, nlags=40)
print(f"First-order autocorrelation: {acf_values[1]:.3f}")
```

A first-order autocorrelation near zero suggests that returns are not predictable from the immediately preceding value -- consistent with weak-form market efficiency. Significant autocorrelation at specific lags can indicate exploitable patterns.

---

## Practical Guidelines

{: .highlight }
> **Rule of thumb**: If the ADF p-value is below 0.05 and the KPSS p-value is above 0.05, you can treat the series as stationary and proceed with modeling.

1. **Always test raw data first** before differencing or transforming.
2. **Apply log-returns** for price data -- this typically produces stationary series.
3. **Check residuals** after fitting any model to confirm they resemble white noise.
4. **Beware of structural breaks** -- a series may appear non-stationary due to regime changes rather than unit roots.

---

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
