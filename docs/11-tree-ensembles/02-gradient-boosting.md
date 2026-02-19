---
layout: default
title: "Gradient Boosting"
parent: "Part 11: Tree Ensembles"
nav_order: 2
permalink: /11-tree-ensembles/02-gradient-boosting
---

# Gradient Boosting

Gradient boosting builds trees sequentially, where each tree corrects the errors of the previous trees. This approach often achieves better accuracy than Random Forests, especially with careful hyperparameter tuning.

## Key Differences from Random Forests

| Aspect | Random Forest | Gradient Boosting |
|:-------|:-------------|:-----------------|
| Training | Parallel (independent trees) | Sequential (error correction) |
| Bias-Variance | Reduces variance | Reduces bias |
| Tree depth | Can use deep trees | Typically shallow (3--7) |
| Learning rate | N/A | Critical hyperparameter |
| Overfitting risk | Lower | Higher (requires regularization) |

{: .note }
> Gradient boosting's sequential nature means each tree is "specialized" in correcting specific errors. This produces a more accurate but potentially more fragile model. Always use regularization and early stopping with financial data.

## Three Major Implementations

Puffin supports three battle-tested gradient boosting frameworks:

1. **XGBoost**: Optimized implementation with L1/L2 regularization
2. **LightGBM**: Fast, memory-efficient histogram-based algorithm
3. **CatBoost**: Native categorical feature handling with ordered boosting

## XGBoost

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting library with built-in regularization that has dominated structured-data ML competitions since 2015.

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

{: .important }
> Start with a low learning rate (0.01--0.05) and increase `n_estimators` to compensate. A lower learning rate with more trees generally produces better generalization.

## LightGBM

LightGBM uses histogram-based algorithms for faster training and lower memory usage. It is particularly effective for large datasets common in cross-sectional equity strategies.

### Key Features

- **Leaf-wise tree growth**: Grows trees leaf-by-leaf rather than level-by-level, finding better splits
- **Histogram-based**: Bins continuous features for faster splitting decisions
- **Native categorical support**: No need for one-hot encoding
- **Efficient memory usage**: Can handle datasets with millions of rows

{: .note }
> LightGBM's leaf-wise growth can overfit small datasets more easily than XGBoost's level-wise approach. Use `num_leaves` carefully -- it should be less than `2^max_depth`.

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

### LightGBM vs XGBoost

For most trading applications:
- **Speed**: LightGBM trains 2-10x faster on large datasets
- **Memory**: LightGBM uses significantly less RAM
- **Accuracy**: Comparable; LightGBM may slightly edge out on large cross-sectional data
- **Tuning**: XGBoost is somewhat easier to tune for beginners

## CatBoost

CatBoost (Categorical Boosting) provides native categorical feature handling without preprocessing, making it ideal for trading data with sector, industry, or rating categories.

### Key Features

- **Native categorical encoding**: Handles categories without one-hot encoding
- **Ordered boosting**: Reduces overfitting through clever ordering of training examples
- **Symmetric trees**: Faster prediction at inference time
- **Built-in regularization**: Less prone to overfitting out of the box

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

{: .warning }
> CatBoost's ordered boosting uses permutation-based target encoding internally. While this reduces overfitting, it means training is slower than LightGBM. The tradeoff is worthwhile when you have many categorical features (sector, exchange, rating, etc.).

## Choosing Between Frameworks

| Use Case | Recommended Framework |
|:---------|:---------------------|
| Quick baseline | XGBoost (well-documented, easy to tune) |
| Large datasets (1M+ rows) | LightGBM (fastest training, lowest memory) |
| Many categorical features | CatBoost (native support, no encoding needed) |
| Production inference speed | CatBoost (symmetric trees) or LightGBM |
| Kaggle-style competitions | XGBoost or LightGBM |

{: .important }
> In practice, the best approach is to train all three and combine them in an [ensemble long-short strategy](04-long-short-strategy). Model diversity improves robustness -- each framework learns slightly different patterns from the same data.

## Source Code

Browse the implementation: [`puffin/ensembles/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/ensembles)
