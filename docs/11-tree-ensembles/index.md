---
layout: default
title: "Part 11: Tree Ensembles"
nav_order: 12
---

# Tree Ensembles for Trading

Tree ensemble methods combine multiple decision trees to create powerful, robust predictive models. These methods are particularly effective for trading applications due to their ability to capture non-linear patterns, handle mixed data types, and provide interpretable feature importances.

## Introduction to Tree Ensembles

### Why Tree Ensembles for Trading?

Tree ensemble methods offer several advantages for financial prediction:

1. **Non-linear pattern recognition**: Capture complex relationships between features and returns
2. **Feature interactions**: Automatically discover interactions between technical and fundamental factors
3. **Robustness**: Less sensitive to outliers than linear models
4. **Mixed data types**: Handle both numerical and categorical features
5. **Feature importance**: Provide interpretable insights into what drives predictions
6. **No feature scaling required**: Trees are invariant to monotonic transformations

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

## Random Forests

Random Forests build multiple decision trees using bootstrap sampling and random feature selection, then average their predictions.

### How Random Forests Work

1. **Bootstrap sampling**: Each tree is trained on a random sample (with replacement) of the data
2. **Random feature selection**: At each split, only a random subset of features is considered
3. **Aggregation**: Predictions are averaged (regression) or voted (classification)

This process reduces variance while maintaining low bias.

### Implementation with Puffin

```python
import pandas as pd
import numpy as np
from puffin.ensembles import RandomForestTrader
from puffin.data import YFinanceClient

# Load data
client = YFinanceClient()
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

### Hyperparameter Tuning

Key hyperparameters for Random Forests:

- `n_estimators`: Number of trees (higher is better, but slower)
- `max_depth`: Maximum tree depth (controls overfitting)
- `min_samples_split`: Minimum samples to split a node
- `max_features`: Number of features to consider at each split

## Gradient Boosting

Gradient boosting builds trees sequentially, where each tree corrects the errors of the previous trees. This approach often achieves better accuracy than Random Forests.

### Key Differences from Random Forests

- **Sequential training**: Trees are built one at a time
- **Error correction**: Each tree fits the residual errors
- **Lower learning rate**: Small steps prevent overfitting
- **Typically uses shallow trees**: Complexity comes from the ensemble

### Three Major Implementations

1. **XGBoost**: Optimized implementation with regularization
2. **LightGBM**: Fast, efficient, memory-optimized
3. **CatBoost**: Native categorical feature handling

## XGBoost

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting library with additional regularization.

### Implementation

```python
from puffin.ensembles import XGBoostTrader

# Initialize with financial-optimized defaults
model = XGBoostTrader(task="classification", random_state=42)

# Fit with default parameters
model.fit(features, target)

# Or specify custom parameters
custom_params = {
    "learning_rate": 0.01,  # Lower = more conservative
    "max_depth": 5,
    "min_child_weight": 5,  # Higher = more regularization
    "subsample": 0.8,  # Row sampling
    "colsample_bytree": 0.8,  # Column sampling
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
    "n_estimators": 200,
}

model.fit(features, target, params=custom_params)

# Make predictions
predictions = model.predict(features)

# Plot feature importance
fig = model.plot_importance(max_features=10)
fig.savefig("xgboost_importance.png")
```

### Hyperparameter Tuning

```python
# Define parameter grid
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "n_estimators": [100, 200, 300],
}

# Tune hyperparameters
best_params = model.tune_hyperparameters(features, target, param_grid=param_grid, cv=5)
print("Best parameters:", best_params)
```

## LightGBM

LightGBM uses histogram-based algorithms for faster training and lower memory usage. It's particularly effective for large datasets.

### Key Features

- **Leaf-wise tree growth**: Grows trees leaf-by-leaf rather than level-by-level
- **Histogram-based**: Bins continuous features for faster splitting
- **Native categorical support**: No need for one-hot encoding
- **Efficient memory usage**: Can handle large datasets

### Implementation

```python
from puffin.ensembles import LightGBMTrader

# Initialize model
model = LightGBMTrader(task="classification", random_state=42)

# Fit with categorical features
categorical_features = ["sector", "market_cap_category"]
model.fit(features, target, categorical_features=categorical_features)

# Tune hyperparameters
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 30],
    "n_estimators": [100, 200],
}

best_params = model.tune_hyperparameters(
    features, target, param_grid=param_grid, cv=5, categorical_features=categorical_features
)
```

## CatBoost

CatBoost (Categorical Boosting) provides native categorical feature handling without preprocessing, making it ideal for trading data with sector, industry, or rating categories.

### Key Features

- **Native categorical encoding**: Handles categories without one-hot encoding
- **Ordered boosting**: Reduces overfitting through clever ordering
- **Symmetric trees**: Faster prediction
- **Built-in regularization**: Less prone to overfitting

### Implementation

```python
from puffin.ensembles import CatBoostTrader

# Prepare data with categorical features
features = df[["return_5d", "volatility_20d", "sector", "market_cap"]].copy()
features["sector"] = features["sector"].astype(str)  # Ensure it's string type

# Initialize model
model = CatBoostTrader(task="classification", random_state=42)

# Fit with categorical features
cat_features = ["sector"]
model.fit(features, target, cat_features=cat_features)

# Get feature importance
importance = model.feature_importance()
print(importance)

# Cross-validate
cv_results = model.cross_validate(features, target, cv=5, cat_features=cat_features)
print(f"CV Score: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
```

## SHAP Interpretation

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance based on game theory, showing how each feature contributes to individual predictions.

### Why SHAP for Trading?

- **Model-agnostic**: Works with any tree ensemble
- **Local explanations**: Understand individual predictions
- **Global insights**: Aggregate to see overall patterns
- **Interaction effects**: Discover feature interactions

### Basic SHAP Analysis

```python
from puffin.ensembles import ModelInterpreter

# Initialize interpreter
interpreter = ModelInterpreter()

# Calculate SHAP values
shap_values = interpreter.shap_values(model.model, features)

# Summary plot (beeswarm)
fig = interpreter.plot_summary(model.model, features, plot_type="dot", max_display=20)
fig.savefig("shap_summary.png")

# Bar plot of mean absolute SHAP values
fig = interpreter.plot_summary(model.model, features, plot_type="bar", max_display=10)
fig.savefig("shap_bar.png")
```

### Dependence Plots

Dependence plots show how a feature's value affects the prediction, including interaction effects.

```python
# Plot dependence for a specific feature
fig = interpreter.plot_dependence(
    model.model,
    features,
    feature="volatility_20d",
    interaction_feature="auto"  # Automatically select interaction feature
)
fig.savefig("shap_dependence_volatility.png")
```

### Waterfall Plots

Waterfall plots explain individual predictions by showing each feature's contribution.

```python
# Explain a specific prediction
fig = interpreter.plot_waterfall(model.model, features, index=100)
fig.savefig("shap_waterfall.png")
```

### Comparing Feature Importance Across Models

```python
# Train multiple models
xgb_model = XGBoostTrader(task="classification", random_state=42)
xgb_model.fit(features, target)

lgb_model = LightGBMTrader(task="classification", random_state=42)
lgb_model.fit(features, target)

cat_model = CatBoostTrader(task="classification", random_state=42)
cat_model.fit(features, target)

# Compare feature importance using SHAP
models_dict = {
    "XGBoost": xgb_model.model,
    "LightGBM": lgb_model.model,
    "CatBoost": cat_model.model,
}

importance_df = interpreter.feature_importance_comparison(
    models_dict, features, method="shap"
)
print(importance_df)

# Plot comparison
fig = interpreter.plot_importance_comparison(
    models_dict, features, max_features=15, method="shap"
)
fig.savefig("importance_comparison.png")
```

## Long-Short Strategy

Combine multiple ensemble models to create a long-short equity strategy.

### Strategy Overview

1. Train multiple ensemble models
2. Generate predictions for all assets
3. Go long on top predicted performers
4. Go short on bottom predicted performers
5. Rebalance periodically

### Implementation

```python
from puffin.ensembles import EnsembleLongShort, RandomForestTrader, XGBoostTrader

# Prepare cross-sectional data (multiple assets)
# Each row represents an asset at a point in time
features = pd.DataFrame({
    "momentum_1m": [...],
    "momentum_3m": [...],
    "value_score": [...],
    "quality_score": [...],
})
forward_returns = pd.Series([...])  # Forward returns

# Train multiple models
rf_model = RandomForestTrader(task="classification", random_state=42)
xgb_model = XGBoostTrader(task="classification", random_state=43)

# Create signals (binary: positive/negative return)
signals = (forward_returns > 0).astype(int)

# Create ensemble
ensemble = EnsembleLongShort(models={
    "random_forest": rf_model,
    "xgboost": xgb_model,
})

# Fit all models
ensemble.fit(features, signals)

# Generate trading signals
trading_signals = ensemble.generate_signals(
    features,
    top_pct=0.2,  # Long top 20%
    bottom_pct=0.2,  # Short bottom 20%
)

print(trading_signals.head())
```

### Backtesting the Strategy

```python
# Backtest performance
results = ensemble.backtest_signals(
    features,
    forward_returns,
    top_pct=0.2,
    bottom_pct=0.2
)

print("Backtest Results:")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Hit Rate: {results['hit_rate']:.2%}")
print(f"Number of Trades: {results['n_trades']}")

# View model weights
weights = ensemble.get_model_weights()
print("\nModel Weights:")
print(weights)
```

### Advanced: Per-Model Predictions

```python
# Get predictions from all models
all_predictions = ensemble.predict_ensemble(features)
print(all_predictions.head())

# Generate signals using a specific model
rf_signals = ensemble.generate_signals(features, method="random_forest")
xgb_signals = ensemble.generate_signals(features, method="xgboost")
ensemble_signals = ensemble.generate_signals(features, method="ensemble")

# Compare strategies
rf_results = ensemble.backtest_signals(features, forward_returns, top_pct=0.2, bottom_pct=0.2)
# Note: backtest_signals always uses ensemble, so we need to create separate ensembles
```

## Complete Example: Multi-Asset Strategy

```python
import pandas as pd
import numpy as np
from puffin.data import YFinanceClient
from puffin.ensembles import (
    XGBoostTrader, LightGBMTrader, CatBoostTrader,
    ModelInterpreter, EnsembleLongShort
)

# 1. Load data for multiple assets
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
client = YFinanceClient()

data = []
for ticker in tickers:
    df = client.get_stock_prices(ticker, start="2020-01-01", end="2023-12-31")

    # Create features
    df["ticker"] = ticker
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)
    df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Target: next 5-day return
    df["forward_return"] = df["close"].pct_change(5).shift(-5)

    data.append(df)

# Combine data
all_data = pd.concat(data, ignore_index=True).dropna()

# Prepare features and target
feature_cols = ["return_5d", "return_20d", "volatility_20d", "volume_ratio"]
features = all_data[feature_cols]
forward_returns = all_data["forward_return"]
signals = (forward_returns > 0).astype(int)

# 2. Train models
xgb_model = XGBoostTrader(task="classification", random_state=42)
lgb_model = LightGBMTrader(task="classification", random_state=43)
cat_model = CatBoostTrader(task="classification", random_state=44)

# 3. Create ensemble
ensemble = EnsembleLongShort(models={
    "XGBoost": xgb_model,
    "LightGBM": lgb_model,
    "CatBoost": cat_model,
})

# 4. Fit ensemble
ensemble.fit(features, signals)

# 5. Generate and backtest signals
results = ensemble.backtest_signals(
    features, forward_returns, top_pct=0.2, bottom_pct=0.2
)

print("=" * 60)
print("ENSEMBLE LONG-SHORT STRATEGY RESULTS")
print("=" * 60)
print(f"Total Return:    {results['total_return']:>10.2%}")
print(f"Mean Return:     {results['mean_return']:>10.4%}")
print(f"Std Return:      {results['std_return']:>10.4%}")
print(f"Sharpe Ratio:    {results['sharpe_ratio']:>10.2f}")
print(f"Hit Rate:        {results['hit_rate']:>10.2%}")
print(f"Number of Trades: {results['n_trades']:>9,}")
print("=" * 60)

# 6. Interpret models with SHAP
interpreter = ModelInterpreter()

models_dict = {
    "XGBoost": xgb_model.model,
    "LightGBM": lgb_model.model,
    "CatBoost": cat_model.model,
}

# Compare feature importance
importance_df = interpreter.feature_importance_comparison(
    models_dict, features, method="shap"
)
print("\nFeature Importance (SHAP):")
print(importance_df)

# Create visualizations
fig = interpreter.plot_importance_comparison(models_dict, features, method="shap")
fig.savefig("ensemble_importance.png")

fig = interpreter.plot_summary(xgb_model.model, features, plot_type="dot")
fig.savefig("xgboost_shap_summary.png")
```

## Best Practices

### Model Selection

- **Random Forest**: Good baseline, robust, easy to tune
- **XGBoost**: Best for structured data with careful tuning
- **LightGBM**: Best for large datasets, fast training
- **CatBoost**: Best for data with many categorical features

### Hyperparameter Tuning

1. Start with default parameters
2. Use time-series cross-validation
3. Tune learning rate first (lower is more stable)
4. Then tune tree structure (depth, leaves)
5. Finally tune regularization parameters

### Feature Engineering

- Use domain knowledge to create meaningful features
- Include technical indicators, fundamental ratios
- Consider time-based features (day of week, month)
- Create interaction features for key relationships

### Avoiding Overfitting

- Use aggressive regularization for financial data
- Lower learning rates (0.01-0.05)
- Shallow trees (depth 3-7)
- Time-series cross-validation (not random CV)
- Monitor out-of-sample performance

### Production Considerations

- Retrain models regularly (monthly or quarterly)
- Monitor feature distributions for drift
- Track prediction accuracy over time
- Use ensemble averaging for stability
- Implement proper position sizing

## Summary

Tree ensemble methods provide powerful tools for trading signal generation:

- **Random Forests** offer robustness through bagging
- **Gradient boosting** (XGBoost, LightGBM, CatBoost) provides state-of-the-art accuracy
- **SHAP values** enable model interpretation and feature analysis
- **Ensemble strategies** combine multiple models for improved performance

The combination of high predictive power, interpretability, and robustness makes tree ensembles essential tools for quantitative trading.

## Further Reading

- Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine."
- [Breiman, L. (2001). "Random forests."](https://doi.org/10.1023/A:1010933404324)
- [Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system."](https://doi.org/10.1145/2939672.2939785)
- [Ke, G., et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree."](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features."](https://arxiv.org/abs/1706.09516)
- [Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions."](https://arxiv.org/abs/1705.07874)
