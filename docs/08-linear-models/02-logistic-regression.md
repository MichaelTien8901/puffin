---
layout: default
title: "Logistic Regression"
parent: "Part 8: Linear Models"
nav_order: 2
---

# Logistic Regression for Direction Prediction

While OLS and Ridge predict continuous return values, many trading strategies only need to know whether the price will go up or down. Logistic regression is the natural choice for this binary classification task. It outputs probabilities, which can be directly used to size positions and set confidence thresholds.

## Why Direction Prediction?

In practice, predicting the exact magnitude of returns is extremely difficult. However, predicting the **direction** of the next move -- even with a modest edge above 50% -- is sufficient to build a profitable strategy when combined with proper position sizing and risk management.

{: .note }
> A direction classifier with 53% accuracy can be highly profitable if the average winning trade is larger than the average losing trade. The key is combining directional accuracy with favorable risk-reward ratios.

## Building a Direction Classifier

The `DirectionClassifier` wraps scikit-learn's logistic regression with features tailored for trading: balanced class weights (to handle the near-50/50 split in market direction), feature normalization, and probability calibration.

```python
from puffin.models.linear import DirectionClassifier
import pandas as pd
import numpy as np
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

# Prepare data (shift to avoid look-ahead bias)
df['target'] = df['returns'].shift(-1)
df = df.dropna()

features = ['momentum_5', 'momentum_20', 'volume_ratio', 'volatility']
X = df[features]

# Create direction labels
df['direction'] = (df['target'] > 0).astype(int)  # 1 = up, 0 = down

# Split into train/test
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_direction_train = df['direction'][:split_idx]
y_direction_test = df['direction'][split_idx:]

# Fit direction classifier
classifier = DirectionClassifier(class_weight='balanced', normalize=True)
classifier.fit(X_train, y_direction_train)

# Predictions
y_pred_direction = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

accuracy = accuracy_score(y_direction_test, y_pred_direction)
auc = roc_auc_score(y_direction_test, y_pred_proba[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_direction_test, y_pred_direction))

# Feature importance
print("\nFeature Importance:")
print(classifier.feature_importance())
```

### Understanding the Output

**Key Metrics:**
- **Accuracy**: Proportion of correct predictions. For direction prediction, anything consistently above 52% is noteworthy.
- **AUC (Area Under ROC Curve)**: Measures the model's ability to discriminate between up and down days. An AUC of 0.5 is random; above 0.55 is useful.
- **Precision / Recall**: Precision tells you what fraction of predicted "up" days were actually up; recall tells you what fraction of actual "up" days you caught.

{: .warning }
> Be careful with accuracy as a metric. If the market goes up 55% of days, a naive "always predict up" classifier achieves 55% accuracy. Always compare against the base rate and use AUC for a more robust evaluation.

**Feature Importance:**
Logistic regression coefficients indicate the log-odds impact of each feature. Larger absolute values mean stronger influence on the predicted direction.

## Trading with Direction Predictions

The real value of logistic regression is that it outputs **probabilities**, not just hard predictions. We can use these probabilities to:
- Trade only when confidence is high (probability threshold)
- Scale position size proportionally to confidence
- Avoid trading during uncertain periods

### Probability-Threshold Strategy

```python
# Use probability threshold for trading signals
threshold = 0.55  # Trade only when confident

df_test = df[split_idx:].copy()
df_test['pred_proba'] = y_pred_proba[:, 1]
df_test['signal'] = 0
df_test.loc[df_test['pred_proba'] > threshold, 'signal'] = 1  # Long
df_test.loc[df_test['pred_proba'] < (1 - threshold), 'signal'] = -1  # Short

# Calculate strategy returns
df_test['strategy_returns'] = df_test['signal'] * df_test['target']

# Performance metrics
total_return = (1 + df_test['strategy_returns']).prod() - 1
sharpe_ratio = df_test['strategy_returns'].mean() / df_test['strategy_returns'].std() * np.sqrt(252)

print(f"\nStrategy Performance:")
print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Win Rate: {(df_test['strategy_returns'] > 0).mean():.2%}")
```

{: .tip }
> Tuning the probability threshold is a trade-off between frequency and quality. A higher threshold (e.g., 0.60) generates fewer trades but with higher conviction. A lower threshold (e.g., 0.52) trades more often but with lower per-trade edge. Backtest across multiple thresholds to find the sweet spot for your strategy.

### Combining Return Magnitude and Direction

A powerful approach is to combine logistic regression (direction) with Ridge regression (magnitude) to generate composite trading signals:

```python
from puffin.models.linear import RidgeModel, DirectionClassifier

# Fit both models
return_model = RidgeModel(normalize=True)
return_model.fit(X_train, df['target'][:split_idx])

direction_model = DirectionClassifier(class_weight='balanced', normalize=True)
direction_model.fit(X_train, y_direction_train)

# Composite signal: trade only when both agree
pred_return = return_model.predict(X_test)
pred_proba = direction_model.predict_proba(X_test)[:, 1]

df_test = df[split_idx:].copy()
df_test['signal'] = 0

# Both models agree on direction AND high confidence
long_signal = (pred_return > 0) & (pred_proba > 0.55)
short_signal = (pred_return < 0) & (pred_proba < 0.45)

df_test.loc[long_signal, 'signal'] = 1
df_test.loc[short_signal, 'signal'] = -1

# Scale position by predicted magnitude
df_test['position_size'] = abs(pred_return) / abs(pred_return).max()
df_test['strategy_returns'] = df_test['signal'] * df_test['position_size'] * df_test['target']

composite_return = (1 + df_test['strategy_returns']).prod() - 1
composite_sharpe = (
    df_test['strategy_returns'].mean()
    / df_test['strategy_returns'].std()
    * np.sqrt(252)
)

print(f"Composite Strategy Return: {composite_return:.2%}")
print(f"Composite Sharpe Ratio: {composite_sharpe:.2f}")
```

{: .note }
> The composite approach tends to outperform either model alone because it requires agreement from two independent signals. This reduces false positives at the cost of fewer total trades.

### Practical Considerations

**Refit Frequency**: Market regimes change. Refit the classifier periodically (e.g., every 20 trading days) using a rolling window of recent data to adapt to evolving conditions.

**Class Balance**: Financial markets are roughly balanced between up and down days, but not exactly. Using `class_weight='balanced'` in `DirectionClassifier` ensures the model does not simply learn the majority class.

**Feature Engineering**: The quality of direction predictions depends heavily on the input features. Experiment with:
- Multiple momentum windows (5, 10, 20, 60 days)
- Volatility measures (realized, implied if available)
- Volume-based features (volume ratio, on-balance volume)
- Cross-asset signals (sector momentum, VIX level)

## Source Code

Browse the implementation: [`puffin/models/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/models)
