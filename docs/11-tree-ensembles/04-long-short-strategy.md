---
layout: default
title: "Long-Short Strategy"
parent: "Part 11: Tree Ensembles"
nav_order: 4
permalink: /11-tree-ensembles/04-long-short-strategy
---

# Long-Short Strategy

The `EnsembleLongShort` class combines multiple ensemble models to create a long-short equity strategy. By aggregating predictions from diverse models, the strategy achieves more robust signal generation than any single model.

## Strategy Overview

The ensemble long-short approach follows a systematic pipeline:

1. **Train multiple ensemble models** (Random Forest, XGBoost, LightGBM, CatBoost)
2. **Generate predictions** for all assets in the universe
3. **Go long** on top predicted performers (highest predicted probability of positive returns)
4. **Go short** on bottom predicted performers (lowest predicted probability)
5. **Rebalance periodically** based on updated predictions

{: .note }
> Long-short strategies are market-neutral by design: the long and short legs offset market exposure, leaving returns driven primarily by the model's stock-selection ability.

## Basic Implementation

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

### Signal Output

The `generate_signals()` method returns a Series with values:
- `1` for long positions (top quantile)
- `-1` for short positions (bottom quantile)
- `0` for neutral (middle 60% in the example above)

## Backtesting the Strategy

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

{: .warning }
> Backtesting results are always optimistic. Be skeptical of high Sharpe ratios and always validate with out-of-sample data. Common pitfalls include look-ahead bias, survivorship bias, and transaction cost underestimation.

## Per-Model Predictions

You can examine predictions from individual models to understand how each contributes to the ensemble.

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

{: .note }
> Comparing per-model signals helps identify which models are contributing positively. If one model consistently disagrees with the ensemble and has lower accuracy, consider downweighting or removing it.

## Complete Example: Multi-Asset Strategy

This end-to-end example demonstrates loading data for multiple tickers, engineering features, training an ensemble, backtesting, and interpreting results with SHAP.

```python
import pandas as pd
import numpy as np
from puffin.data import YFinanceProvider
from puffin.ensembles import (
    XGBoostTrader, LightGBMTrader, CatBoostTrader,
    ModelInterpreter, EnsembleLongShort
)

# 1. Load data for multiple assets
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
client = YFinanceProvider()

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

| Model | Strengths | Best For |
|:------|:----------|:---------|
| Random Forest | Robust, easy to tune, good baseline | First pass, feature selection |
| XGBoost | Best structured-data accuracy with tuning | Careful research strategies |
| LightGBM | Fastest training, lowest memory | Large cross-sectional datasets |
| CatBoost | Native categorical handling | Data with sectors, ratings, categories |

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

{: .warning }
> The single most common mistake in financial ML is overfitting. If your backtest Sharpe ratio exceeds 3.0, be extremely skeptical. Validate with multiple out-of-sample periods and realistic transaction costs.

### Production Considerations

- Retrain models regularly (monthly or quarterly)
- Monitor feature distributions for drift
- Track prediction accuracy over time
- Use ensemble averaging for stability
- Implement proper position sizing

{: .important }
> In production, always compare live performance against backtest expectations. Significant degradation may indicate regime change, feature drift, or data pipeline issues. Automate this monitoring with the tools in [Part 25: Monitoring & Analytics]({{ site.baseurl }}/25-monitoring-analytics/).

## Source Code

Browse the implementation: [`puffin/ensembles/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ensembles)
