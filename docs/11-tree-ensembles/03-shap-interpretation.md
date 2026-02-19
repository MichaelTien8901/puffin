---
layout: default
title: "SHAP Interpretation"
parent: "Part 11: Tree Ensembles"
nav_order: 3
permalink: /11-tree-ensembles/03-shap-interpretation
---

# SHAP Interpretation

SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance based on cooperative game theory, showing how each feature contributes to individual predictions. For trading models, SHAP is invaluable: it tells you not just *what* a model predicts, but *why*.

## Why SHAP for Trading?

Traditional feature importance (e.g., Gini importance in Random Forests) only tells you which features matter *on average*. SHAP goes further:

- **Model-agnostic**: Works with any tree ensemble (Random Forest, XGBoost, LightGBM, CatBoost)
- **Local explanations**: Understand why a model predicts "buy" for a specific stock on a specific day
- **Global insights**: Aggregate local explanations to see overall patterns
- **Interaction effects**: Discover how features combine (e.g., high momentum + low volatility)
- **Consistency**: Features that contribute more to predictions always receive higher SHAP values

{: .note }
> SHAP values are grounded in Shapley values from cooperative game theory. Each feature is treated as a "player" in a game, and its SHAP value represents its fair contribution to the prediction. This mathematical foundation ensures consistency and additivity.

## The ModelInterpreter Class

Puffin provides the `ModelInterpreter` class as a unified interface for SHAP-based model analysis.

```python
from puffin.ensembles import ModelInterpreter

# Initialize interpreter
interpreter = ModelInterpreter()
```

## Basic SHAP Analysis

### Computing SHAP Values

```python
# Calculate SHAP values for a trained model
shap_values = interpreter.shap_values(model.model, features)
```

The returned `shap_values` array has the same shape as `features` -- one SHAP value per feature per observation. Positive values push the prediction higher; negative values push it lower.

### Summary Plot (Beeswarm)

The beeswarm summary plot is the most informative single visualization for understanding a model. Each dot represents one observation; its position on the x-axis shows the SHAP value, and its color shows the feature value.

```python
# Summary plot (beeswarm)
fig = interpreter.plot_summary(model.model, features, plot_type="dot", max_display=20)
fig.savefig("shap_summary.png")
```

### Bar Plot

The bar plot shows the mean absolute SHAP value for each feature -- a global measure of importance.

```python
# Bar plot of mean absolute SHAP values
fig = interpreter.plot_summary(model.model, features, plot_type="bar", max_display=10)
fig.savefig("shap_bar.png")
```

{: .important }
> Always examine both the beeswarm and bar plots. The bar plot tells you *which* features matter most; the beeswarm tells you *how* they matter (direction and non-linearity).

## Dependence Plots

Dependence plots show how a single feature's value affects the model's prediction, revealing non-linear relationships and interaction effects.

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

### Reading Dependence Plots

- **X-axis**: Feature value (e.g., 20-day volatility)
- **Y-axis**: SHAP value (contribution to prediction)
- **Color**: Value of the interaction feature
- **Trend**: Shows the non-linear relationship between the feature and the prediction

For trading, dependence plots are particularly useful for understanding regime behavior. For example, you might discover that momentum is predictive only when volatility is low, or that value factors work differently across market-cap segments.

{: .note }
> Set `interaction_feature="auto"` to let SHAP automatically identify the strongest interaction partner. This often reveals non-obvious relationships in your data.

## Waterfall Plots

Waterfall plots explain individual predictions by decomposing them into feature contributions. This is essential for understanding specific trading signals.

```python
# Explain a specific prediction
fig = interpreter.plot_waterfall(model.model, features, index=100)
fig.savefig("shap_waterfall.png")
```

### When to Use Waterfall Plots

- **Signal investigation**: Why did the model predict "buy" for AAPL on 2023-06-15?
- **Risk analysis**: Which features drove an unusually large position?
- **Model debugging**: Check if the model is relying on sensible features for specific predictions
- **Compliance**: Provide explainable rationale for trading decisions

{: .warning }
> Waterfall plots for individual predictions can be misleading if viewed in isolation. Always combine them with global analysis (summary and dependence plots) to confirm that patterns are systematic, not anecdotal.

## Comparing Feature Importance Across Models

One of the most powerful SHAP applications is comparing how different models use features. If all models agree on feature importance, you can be more confident in those features. If they disagree, it may indicate instability or overfitting.

```python
from puffin.ensembles import XGBoostTrader, LightGBMTrader, CatBoostTrader

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

### Interpreting the Comparison

The comparison DataFrame shows SHAP-based importance for each model side by side. Look for:

- **Consensus features**: High importance across all models -- these are reliable predictors
- **Model-specific features**: High importance in one model but not others -- may indicate overfitting
- **Rank correlation**: High Spearman rank correlation between models suggests they learned similar structures

{: .important }
> Features that rank highly across all three frameworks (XGBoost, LightGBM, CatBoost) are the most trustworthy signals. Consider building your trading strategy primarily around these consensus features.

## Practical Workflow

A recommended SHAP analysis workflow for trading models:

1. **Train models** on your feature set
2. **Bar plot** to identify the top 10-15 most important features
3. **Beeswarm plot** to understand direction and non-linearity
4. **Dependence plots** for each top feature to find interactions
5. **Cross-model comparison** to validate feature robustness
6. **Waterfall plots** for specific signals of interest
7. **Iterate**: Remove unimportant features, add new ones, retrain

## Source Code

Browse the implementation: [`puffin/ensembles/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ensembles)
