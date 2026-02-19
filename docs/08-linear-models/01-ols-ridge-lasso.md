---
layout: default
title: "OLS, Ridge & Lasso"
parent: "Part 8: Linear Models"
nav_order: 1
---

# OLS, Ridge & Lasso Regression

This section covers the three core linear regression techniques for return prediction: Ordinary Least Squares (OLS), Ridge regression (L2 regularization), and Lasso regression (L1 regularization). Each method trades off bias and variance differently, making them suitable for different trading scenarios.

## OLS Regression for Return Prediction

Ordinary Least Squares (OLS) minimizes the sum of squared residuals to estimate coefficients. It is the simplest linear model and the starting point for most regression-based trading strategies.

### Basic Example

```python
import pandas as pd
import numpy as np
from puffin.models.linear import OLSModel
from puffin.data import YFinanceProvider

# Fetch historical data
symbol = 'AAPL'
df = YFinanceProvider().fetch(symbol, start='2020-01-01', end='2023-12-31')

# Calculate returns and features
df['returns'] = df['close'].pct_change()
df['momentum_5'] = df['close'].pct_change(5)
df['momentum_20'] = df['close'].pct_change(20)
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
df['volatility'] = df['returns'].rolling(20).std()

# Prepare data (shift features to avoid look-ahead bias)
df['target'] = df['returns'].shift(-1)  # Next day's return
df = df.dropna()

features = ['momentum_5', 'momentum_20', 'volume_ratio', 'volatility']
X = df[features]
y = df['target']

# Split into train/test
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Fit OLS model
model = OLSModel(add_constant=True)
model.fit(X_train, y_train)

# Examine results
print("Model Summary:")
summary = model.summary()
print(f"R-squared: {summary['r_squared']:.4f}")
print(f"Adjusted R-squared: {summary['adj_r_squared']:.4f}")
print(f"RMSE: {summary['rmse']:.6f}")

print("\nCoefficients:")
print(model.coefficients)

print("\nP-values:")
print(model.p_values)

# Make predictions
y_pred = model.predict(X_test)

# Calculate prediction accuracy
correlation = np.corrcoef(y_test, y_pred)[0, 1]
print(f"\nPrediction Correlation: {correlation:.4f}")
```

### Interpreting OLS Results

**Coefficients:**
- Positive coefficient: Feature increase leads to return increase
- Negative coefficient: Feature increase leads to return decrease
- Magnitude: Effect size (e.g., beta=0.01 means 1 unit increase leads to 0.01 return increase)

**Statistical Tests:**
- **P-value < 0.05**: Feature is statistically significant
- **R-squared**: Proportion of variance explained (higher is better)
- **Residuals**: Should be randomly distributed (check for autocorrelation)

{: .warning }
> OLS is sensitive to outliers and multicollinearity. Always check the Variance Inflation Factor (VIF) for correlated features, and examine residual plots before relying on coefficient estimates.

**Common Issues:**
- **Multicollinearity**: Correlated features cause unstable coefficients
- **Heteroscedasticity**: Non-constant variance violates OLS assumptions
- **Autocorrelation**: Time-series dependencies violate independence assumption

## Regularization: Ridge and Lasso

Regularization adds penalties to prevent overfitting, especially with many features. Both Ridge and Lasso shrink coefficient magnitudes, but they do so differently, leading to distinct practical trade-offs.

### Ridge Regression (L2 Regularization)

Ridge adds a penalty proportional to the squared coefficients:

```
minimize ||y - Xb||^2 + alpha * ||b||^2
```

This shrinks all coefficients toward zero but never sets them exactly to zero.

```python
from puffin.models.linear import RidgeModel

# Ridge with cross-validated alpha selection
ridge = RidgeModel(alphas=np.logspace(-3, 3, 50), cv=5, normalize=True)
ridge.fit(X_train, y_train)

print(f"Selected alpha: {ridge.alpha:.4f}")
print("\nCoefficients:")
print(ridge.coefficients)

# Feature importance
print("\nFeature Importance:")
print(ridge.feature_importance())

# Predictions
y_pred_ridge = ridge.predict(X_test)
```

{: .tip }
> Ridge regression is often the best default choice when you have many correlated features (e.g., overlapping momentum windows). It stabilizes coefficient estimates without discarding any feature entirely.

**When to use Ridge:**
- Many correlated features
- Want to keep all features
- Multicollinearity present
- Need stable coefficient estimates

### Lasso Regression (L1 Regularization)

Lasso adds a penalty proportional to the absolute value of coefficients:

```
minimize ||y - Xb||^2 + alpha * ||b||_1
```

The key property of Lasso is that it can set coefficients exactly to zero, performing automatic feature selection.

```python
from puffin.models.linear import LassoModel

# Lasso with cross-validated alpha selection
lasso = LassoModel(alphas=np.logspace(-4, 0, 50), cv=5, normalize=True)
lasso.fit(X_train, y_train)

print(f"Selected alpha: {lasso.alpha:.6f}")
print("\nCoefficients:")
print(lasso.coefficients)

# Selected features (non-zero coefficients)
print("\nSelected Features:")
print(lasso.selected_features)

# Feature importance
print("\nFeature Importance:")
print(lasso.feature_importance())
```

**When to use Lasso:**
- Feature selection needed
- Many irrelevant features
- Want sparse model
- Interpretability important

{: .note }
> In practice, Lasso is particularly useful when constructing alpha factors from a large pool of candidate signals. It automatically identifies which signals contribute to return prediction and discards the rest.

## Comparing OLS, Ridge, and Lasso

The following code compares all three models on the same test set, evaluating mean squared error, R-squared, and prediction correlation:

```python
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'OLS': model,
    'Ridge': ridge,
    'Lasso': lasso,
}

print("Model Comparison:")
print("-" * 60)
for name, m in models.items():
    y_pred = m.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    print(f"{name:10s} - MSE: {mse:.6f}, R2: {r2:.4f}, Corr: {corr:.4f}")
```

### Quick Comparison Guide

| Property | OLS | Ridge | Lasso |
|----------|-----|-------|-------|
| Regularization | None | L2 (squared) | L1 (absolute) |
| Feature selection | No | No | Yes |
| Coefficient behavior | Unbiased | Shrunk toward zero | Shrunk, some exactly zero |
| Best for | Few features, no collinearity | Many correlated features | Many irrelevant features |
| Overfitting risk | High with many features | Low | Low |

{: .tip }
> A common workflow: start with OLS to establish a baseline, switch to Ridge if you see multicollinearity or overfitting, and use Lasso when you suspect many features are irrelevant and want an automatically sparse model.

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
