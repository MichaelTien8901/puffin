---
layout: default
title: "PyMC Fundamentals"
parent: "Part 10: Bayesian ML"
nav_order: 1
---

# PyMC Fundamentals

PyMC is a Python library for probabilistic programming using Markov Chain Monte Carlo (MCMC) sampling. This page covers the basics of Bayesian inference with PyMC and demonstrates how to build Bayesian linear regression models for factor-based trading strategies.

{: .note }
> PyMC uses the No-U-Turn Sampler (NUTS), a variant of Hamiltonian Monte Carlo, which provides efficient exploration of complex posterior distributions without manual tuning.

## Installation

```bash
pip install pymc arviz
```

## Simple Example

Before diving into trading applications, let's build a basic Bayesian model to estimate the mean and standard deviation of a dataset.

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

{: .important }
> Always check MCMC convergence diagnostics: `r_hat` should be close to 1.0 (below 1.01), and effective sample size (`ess_bulk`, `ess_tail`) should be sufficiently large.

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

The `hdi_prob` parameter controls the width of the Highest Density Interval (HDI). A 94% HDI means there is a 94% probability that the true value lies within the interval, given the model and data.

### Trading Application: Factor Model with Uncertainty

A Bayesian factor model estimates a stock's beta (market exposure) with full uncertainty quantification. This is valuable because beta is not a fixed number -- it varies over time and has estimation error.

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

{: .highlight }
> The posterior distribution of beta tells you not just the most likely market exposure, but the full range of plausible values. A wide posterior means more uncertainty in the estimate.

## MCMC Diagnostics

After fitting any Bayesian model, always check convergence diagnostics:

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

Key diagnostics to check:

| Diagnostic | Good Value | Meaning |
|-----------|-----------|---------|
| `r_hat` | < 1.01 | Chains have converged to the same distribution |
| `ess_bulk` | > 400 | Enough effective samples for reliable bulk statistics |
| `ess_tail` | > 400 | Enough effective samples for reliable tail estimates |
| Divergences | 0 | No numerical issues during sampling |

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
