---
layout: default
title: "Random Forests"
parent: "Part 11: Tree Ensembles"
nav_order: 1
permalink: /11-tree-ensembles/01-random-forests
---

# Random Forests

Random Forests build multiple decision trees using bootstrap sampling and random feature selection, then aggregate their predictions. This section reviews decision tree fundamentals, walks through the `RandomForestTrader` implementation, and covers hyperparameter tuning for financial data.

## Decision Trees Review

Decision trees split the feature space into regions and make predictions based on the majority class (classification) or mean value (regression) in each region.

### Key Concepts

- **Splitting criteria**: Information gain (classification) or variance reduction (regression)
- **Tree depth**: Controls model complexity
- **Leaf size**: Minimum samples required in a leaf node
- **Pruning**: Removing branches to prevent overfitting

### Limitations of Single Trees

- **High variance**: Small changes in data can produce very different trees
- **Overfitting**: Deep trees memorize training data
- **Instability**: Sensitive to data changes

Ensemble methods address these limitations by combining multiple trees.

{: .warning }
> A single decision tree almost always overfits financial data. Never use a standalone tree for production trading signals -- always prefer an ensemble approach.

## How Random Forests Work

Random Forests combine two sources of randomness to produce a diverse set of trees:

1. **Bootstrap sampling**: Each tree is trained on a random sample (with replacement) of the data
2. **Random feature selection**: At each split, only a random subset of features is considered
3. **Aggregation**: Predictions are averaged (regression) or voted (classification)

This process reduces variance while maintaining low bias. Because each tree sees a different view of the data, the ensemble's errors tend to cancel out rather than compound.

{: .note }
> The name "Random Forest" comes from two sources of randomness: random row sampling (bootstrap) and random column sampling (feature subsets). Together these ensure that individual trees are decorrelated.

## Implementation with Puffin

The `RandomForestTrader` class wraps scikit-learn's `RandomForestClassifier` and `RandomForestRegressor` with trading-specific defaults and utilities.

```python
import pandas as pd
import numpy as np
from puffin.ensembles import RandomForestTrader
from puffin.data import YFinanceProvider

# Load data
client = YFinanceProvider()
df = client.get_stock_prices("AAPL", start="2020-01-01", end="2023-12-31")

# Create features
df["return_5d"] = df["close"].pct_change(5)
df["return_20d"] = df["close"].pct_change(20)
df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
df["rsi"] = compute_rsi(df["close"], window=14)

# Create target: next 5-day return
df["forward_return"] = df["close"].pct_change(5).shift(-5)
df["signal"] = (df["forward_return"] > 0).astype(int)

# Prepare data
features = df[["return_5d", "return_20d", "volatility_20d", "rsi"]].dropna()
target = df.loc[features.index, "signal"]

# Train model
model = RandomForestTrader(task="classification", random_state=42)
model.fit(features, target, n_estimators=100, max_depth=10)

# Cross-validate
cv_results = model.cross_validate(features, target)
print(f"Cross-validation accuracy: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")

# Feature importance
importance = model.feature_importance()
print("\nFeature Importance:")
print(importance)
```

### Understanding the Output

The `feature_importance()` method returns a pandas Series sorted by importance. For Random Forests, this is based on the mean decrease in impurity (Gini importance) across all trees. For a more robust measure, use SHAP values (covered in the [SHAP Interpretation](03-shap-interpretation) page).

## Hyperparameter Tuning

Key hyperparameters for Random Forests:

| Parameter | Description | Typical Range | Trading Guidance |
|:----------|:------------|:--------------|:-----------------|
| `n_estimators` | Number of trees | 100 -- 1000 | Higher is better but slower; 200--500 is a good range |
| `max_depth` | Maximum tree depth | 5 -- 15 | Keep shallow (5--10) to avoid overfitting noisy data |
| `min_samples_split` | Minimum samples to split a node | 10 -- 50 | Higher values add regularization |
| `max_features` | Features considered at each split | `"sqrt"`, `"log2"`, or float | `"sqrt"` is the default; try `0.3`--`0.5` for wider search |

{: .important }
> For financial data, err on the side of more regularization. Shallow trees (`max_depth=5--8`) with higher `min_samples_split` (20+) tend to generalize better to unseen market regimes.

### Grid Search Example

```python
from sklearn.model_selection import TimeSeriesSplit

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 8, 10],
    "min_samples_split": [10, 20, 50],
    "max_features": ["sqrt", 0.5],
}

# Use time-series cross-validation (not random CV)
tscv = TimeSeriesSplit(n_splits=5)

best_params = model.tune_hyperparameters(
    features, target,
    param_grid=param_grid,
    cv=tscv,
)
print("Best parameters:", best_params)
```

{: .warning }
> Always use `TimeSeriesSplit` (or a similar walk-forward approach) for financial cross-validation. Standard k-fold CV leaks future information into training data, inflating performance estimates.

## When to Use Random Forests

Random Forests are an excellent starting point for any trading ML project:

- **Baseline model**: Quick to train, minimal tuning, robust results
- **Feature selection**: Use importance scores to identify useful predictors before training more complex models
- **Stable ensembles**: Less sensitive to hyperparameter choices than gradient boosting
- **Parallel training**: Trees are independent and can be trained in parallel

For higher accuracy with more careful tuning, consider the [Gradient Boosting](02-gradient-boosting) methods covered in the next section.

## Source Code

Browse the implementation: [`puffin/ensembles/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ensembles)
