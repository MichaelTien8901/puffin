---
layout: default
title: "ARIMA, VAR & GARCH"
parent: "Part 9: Time Series Models"
nav_order: 2
permalink: /09-time-series-models/arima-var-garch/
---

# ARIMA, VAR & GARCH Models

This page covers three families of time series models commonly used in quantitative finance: ARIMA for univariate forecasting, VAR for multivariate analysis, and GARCH for volatility modeling.

---

## ARIMA Models

ARIMA (AutoRegressive Integrated Moving Average) models combine three components:
- **AR(p)**: Autoregression -- uses past values of the series
- **I(d)**: Integration -- differencing to achieve stationarity
- **MA(q)**: Moving Average -- uses past forecast errors

{: .note }
> The "integrated" part (d) handles non-stationarity by differencing. An ARIMA(1,1,1) model first differences the series once, then fits an ARMA(1,1) to the differenced data.

### Basic ARIMA Usage

```python
import numpy as np
import pandas as pd
from puffin.models import ARIMAModel

# Create and fit ARIMA model
np.random.seed(42)
returns = pd.Series(np.random.randn(252))

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

Manually choosing the (p, d, q) order can be tedious. The `auto_arima` function searches over a grid of parameters and selects the best model by AIC (Akaike Information Criterion):

```python
from puffin.models import auto_arima

# Automatically select best ARIMA model
model = auto_arima(returns, max_p=5, max_d=2, max_q=5)
print(f"Selected order: {model.order_}")
print(f"AIC: {model.aic_:.2f}")

# Make predictions
predictions = model.predict(steps=20)
```

{: .important }
> Lower AIC values indicate a better trade-off between model fit and complexity. Always compare AIC across candidate models rather than relying on a single specification.

### Model Diagnostics

After fitting an ARIMA model, check that the residuals behave like white noise:

```python
from puffin.models import test_stationarity

# Get model residuals
residuals = model.residuals()

# Check if residuals are white noise
residual_test = test_stationarity(residuals)
print(f"Residuals stationary: {residual_test['is_stationary']}")

# View model summary
print(model.summary())
```

If residuals show significant autocorrelation, the model order may be insufficient.

---

## VAR Models

Vector Autoregression (VAR) models analyze relationships between multiple time series simultaneously. Each variable is modeled as a function of its own past values and past values of all other variables.

{: .note }
> VAR models are particularly useful in finance for studying how shocks in one market (e.g., bonds) transmit to another (e.g., equities).

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

Impulse response functions (IRF) show how a one-standard-deviation shock to one variable affects all variables over time:

```python
# Calculate impulse response function
irf = var_model.impulse_response(periods=20)

# Response of all variables to shock in first variable
print(f"IRF shape: {irf.shape}")
```

This analysis reveals lead-lag relationships and the speed at which shocks are absorbed across markets.

### Granger Causality Testing

Granger causality tests whether the past values of one variable contain information useful for predicting another, beyond what is contained in the target's own past:

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

{: .warning }
> Granger causality is a statistical concept, not true causality. A significant result means one series has predictive power for another, not that it causes changes in the other.

### Test All Pairs for Causality

```python
from puffin.models import test_granger_causality_matrix

# Create causality matrix for all variable pairs
causality_matrix = test_granger_causality_matrix(data, max_lag=5)
print(causality_matrix)
```

---

## GARCH Models

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models capture **volatility clustering** -- the empirical observation that large price changes tend to be followed by large changes, and small changes by small changes.

### Why Volatility Modeling Matters

- **Risk management**: Accurate volatility estimates are essential for Value-at-Risk (VaR) and position sizing.
- **Options pricing**: Implied volatility drives options values; GARCH provides realized volatility forecasts.
- **Portfolio construction**: Dynamic volatility estimates improve portfolio optimization.

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

Different GARCH variants capture different properties of financial returns:

| Model | Key Feature | Use Case |
|:------|:------------|:---------|
| GARCH | Symmetric volatility response | General volatility modeling |
| EGARCH | Asymmetric response, no positivity constraint | Leverage effects in equities |
| GJR-GARCH | Separate coefficients for positive/negative shocks | Asymmetric volatility |

```python
# EGARCH model (captures asymmetry)
egarch = GARCHModel(p=1, q=1, model='egarch')
egarch.fit(returns_series)

# GJR-GARCH model (captures leverage effect)
gjr = GARCHModel(p=1, q=1, model='gjr-garch')
gjr.fit(returns_series)
```

{: .note }
> The **leverage effect** refers to the tendency for volatility to increase more after negative returns than after positive returns of the same magnitude. EGARCH and GJR-GARCH models capture this asymmetry.

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

For out-of-sample evaluation, use rolling window volatility forecasts:

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

Rolling forecasts provide a realistic assessment of how the model would perform in practice, since each forecast uses only data available at the time.

---

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
