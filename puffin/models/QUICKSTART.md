# Linear Models Quick Start Guide

This guide provides quick examples to get you started with linear models in Puffin.

## Installation

The linear models require the `ml` optional dependencies:

```bash
pip install -e ".[ml]"
```

## Basic Usage

### 1. OLS Regression

Predict returns using technical indicators:

```python
from puffin.models.linear import OLSModel
import pandas as pd

# Prepare your features and target
X = df[['momentum', 'volume_ratio', 'volatility']]
y = df['returns']

# Fit model
model = OLSModel(add_constant=True)
model.fit(X, y)

# Check results
print(f"R²: {model.r_squared:.4f}")
print("\nCoefficients:")
print(model.coefficients)

# Make predictions
predictions = model.predict(X)
```

### 2. Ridge Regression (with many features)

```python
from puffin.models.linear import RidgeModel
import numpy as np

# Ridge automatically selects best alpha via cross-validation
ridge = RidgeModel(
    alphas=np.logspace(-3, 3, 50),  # Try 50 alpha values
    cv=5,                            # 5-fold cross-validation
    normalize=True                   # Normalize features
)

ridge.fit(X, y)
print(f"Selected alpha: {ridge.alpha:.4f}")

# See which features are most important
print("\nFeature Importance:")
print(ridge.feature_importance())
```

### 3. Lasso Regression (for feature selection)

```python
from puffin.models.linear import LassoModel

# Lasso performs automatic feature selection
lasso = LassoModel(cv=5, normalize=True)
lasso.fit(X, y)

# See which features were selected
print(f"Selected {len(lasso.selected_features)} out of {X.shape[1]} features")
print("Selected features:", lasso.selected_features)

# Non-selected features have zero coefficients
print("\nCoefficients:")
print(lasso.coefficients)
```

### 4. Direction Classification

```python
from puffin.models.linear import DirectionClassifier

# Create direction labels (1 = up, 0 = down)
y_direction = (df['returns'] > 0).astype(int)

# Fit classifier
classifier = DirectionClassifier(
    class_weight='balanced',  # Handle imbalanced classes
    normalize=True
)
classifier.fit(X, y_direction)

# Predict direction
directions = classifier.predict(X_test)

# Get probabilities for confidence-based trading
probabilities = classifier.predict_proba(X_test)
prob_up = probabilities[:, 1]

# Trade only when confident
strong_buy = prob_up > 0.6
strong_sell = prob_up < 0.4
```

### 5. CAPM Analysis

```python
from puffin.models.factor_models import FamaFrenchModel

ff = FamaFrenchModel()

# Analyze your strategy returns
capm = ff.fit_capm(
    returns=strategy_returns,
    start='2020-01-01',
    end='2023-12-31'
)

print(f"Alpha: {capm['alpha']:.6f} (p={capm['alpha_pvalue']:.4f})")
print(f"Beta:  {capm['beta']:.4f}")

if capm['alpha'] > 0 and capm['alpha_pvalue'] < 0.05:
    print("Significant outperformance! 🎉")
```

### 6. Fama-French Factor Analysis

```python
# 3-factor model
ff3 = ff.fit_three_factor(
    returns=asset_returns,
    start='2020-01-01',
    end='2023-12-31'
)

print(f"Market Beta: {ff3['beta_mkt']:.4f}")
print(f"Size Beta (SMB): {ff3['beta_smb']:.4f}")
print(f"Value Beta (HML): {ff3['beta_hml']:.4f}")

# 5-factor model (includes profitability and investment)
ff5 = ff.fit_five_factor(returns=asset_returns, start='2020-01-01', end='2023-12-31')
print("\nFactor Loadings:")
for factor, beta in ff5['betas'].items():
    print(f"  {factor}: {beta:.4f}")
```

## Common Patterns

### Pattern 1: Return Prediction Pipeline

```python
from puffin.models.linear import RidgeModel
from sklearn.model_selection import train_test_split

# Calculate features
df['momentum'] = df['close'].pct_change(10)
df['volatility'] = df['close'].pct_change().rolling(20).std()
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

# Create target (next day's return)
df['target'] = df['close'].pct_change().shift(-1)
df = df.dropna()

# Split data
X = df[['momentum', 'volatility', 'volume_ratio']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit and predict
model = RidgeModel(normalize=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y_test, predictions):.4f}")
```

### Pattern 2: Direction-Based Trading Strategy

```python
from puffin.models.linear import DirectionClassifier

# Prepare data
y_direction = (df['target'] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_direction, test_size=0.2, shuffle=False
)

# Fit classifier
classifier = DirectionClassifier(class_weight='balanced')
classifier.fit(X_train, y_train)

# Generate trading signals
proba = classifier.predict_proba(X_test)[:, 1]
signals = pd.Series(0, index=X_test.index)
signals[proba > 0.55] = 1   # Buy when confident
signals[proba < 0.45] = -1  # Sell when confident

# Calculate returns
test_returns = df.loc[X_test.index, 'target']
strategy_returns = signals * test_returns

print(f"Strategy Return: {strategy_returns.sum():.2%}")
print(f"Buy & Hold Return: {test_returns.sum():.2%}")
```

### Pattern 3: Rolling Factor Analysis

```python
from puffin.models.factor_models import FamaFrenchModel
import pandas as pd

ff = FamaFrenchModel()
window = 252  # 1 year

results = []
for i in range(window, len(returns)):
    window_returns = returns.iloc[i-window:i]
    start = window_returns.index[0]
    end = window_returns.index[-1]

    capm = ff.fit_capm(window_returns, start=start, end=end)
    results.append({
        'date': end,
        'alpha': capm['alpha'],
        'beta': capm['beta'],
        'r_squared': capm['r_squared']
    })

results_df = pd.DataFrame(results).set_index('date')

# Plot time-varying beta
import matplotlib.pyplot as plt
results_df['beta'].plot(title='Rolling Beta')
plt.axhline(y=1, color='r', linestyle='--', label='Market')
plt.legend()
plt.show()
```

## Tips and Best Practices

### 1. Feature Scaling
Always normalize features for Ridge, Lasso, and Logistic Regression:
```python
model = RidgeModel(normalize=True)  # ✓ Good
```

### 2. Avoid Look-Ahead Bias
Shift target variable properly:
```python
df['target'] = df['returns'].shift(-1)  # ✓ Predict next period
# NOT: df['target'] = df['returns']     # ✗ Uses future information
```

### 3. Check Statistical Significance
Don't rely on insignificant coefficients:
```python
summary = model.summary()
print(summary['p_values'])
# Use only features with p_value < 0.05
```

### 4. Handle Imbalanced Classes
For direction classification:
```python
classifier = DirectionClassifier(class_weight='balanced')  # ✓ Good
```

### 5. Use Cross-Validation
Let Ridge/Lasso select optimal alpha:
```python
ridge = RidgeModel(alphas=np.logspace(-3, 3, 50), cv=5)  # ✓ Good
# NOT: Ridge(alpha=1.0)  # ✗ Fixed alpha might not be optimal
```

## Running Tests

Verify your installation:
```bash
# Test linear models
pytest tests/models/test_linear.py -v

# Test factor models
pytest tests/models/test_factor_models.py -v

# Run demo
python examples/linear_models_demo.py
```

## Getting Help

- **Documentation**: See `docs/08-linear-models/index.md` for complete tutorial
- **Examples**: Check `examples/linear_models_demo.py` for working code
- **Tests**: Review `tests/models/test_*.py` for usage patterns

## Common Issues

### Issue: "Model must be fitted before making predictions"
**Solution**: Call `fit()` before `predict()`:
```python
model = OLSModel()
model.fit(X_train, y_train)  # Fit first!
predictions = model.predict(X_test)
```

### Issue: Poor predictive performance
**Solutions**:
1. Add more features
2. Use regularization (Ridge/Lasso)
3. Check for look-ahead bias
4. Increase training data size
5. Try different time periods

### Issue: Multicollinearity warnings
**Solution**: Use Ridge regression instead of OLS:
```python
ridge = RidgeModel(normalize=True)  # Ridge handles correlated features
```

### Issue: Lasso selects too few features
**Solution**: Decrease alpha range:
```python
lasso = LassoModel(alphas=np.logspace(-5, -1, 50))  # Lower alpha values
```

## Next Steps

1. **Part 9: Time Series Models** - ARIMA, GARCH, Kalman filters
2. **Part 10: Machine Learning** - Random forests, gradient boosting
3. **Part 11: Deep Learning** - Neural networks, LSTMs, transformers

---

Happy Trading! 📈
