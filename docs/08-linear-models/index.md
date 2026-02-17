---
layout: default
title: "Part 8: Linear Models"
nav_order: 9
---

# Linear Models for Trading

Linear models are fundamental tools in quantitative finance, providing interpretable and computationally efficient methods for return prediction, risk analysis, and factor modeling. This chapter covers the application of linear regression techniques to trading strategies.

## Table of Contents
- [Introduction to Linear Models](#introduction-to-linear-models)
- [OLS Regression for Return Prediction](#ols-regression-for-return-prediction)
- [Regularization: Ridge and Lasso](#regularization-ridge-and-lasso)
- [Logistic Regression for Direction Prediction](#logistic-regression-for-direction-prediction)
- [Capital Asset Pricing Model (CAPM)](#capital-asset-pricing-model-capm)
- [Fama-French Factor Models](#fama-french-factor-models)
- [Fama-MacBeth Regression](#fama-macbeth-regression)
- [Complete Trading Example](#complete-trading-example)

## Introduction to Linear Models

Linear models assume that the target variable is a linear combination of input features plus noise:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

**Advantages:**
- Interpretability: Coefficients directly show feature importance
- Computational efficiency: Fast to train and predict
- Statistical inference: P-values, confidence intervals available
- Well-understood theory: Decades of financial research

**Key Applications in Trading:**
1. **Return Prediction**: Forecast future returns from technical indicators and fundamentals
2. **Factor Analysis**: Decompose returns into systematic risk factors
3. **Direction Classification**: Predict up/down movements for directional strategies
4. **Risk Attribution**: Understand sources of portfolio risk

## OLS Regression for Return Prediction

Ordinary Least Squares (OLS) minimizes the sum of squared residuals to estimate coefficients.

### Basic Example

```python
import pandas as pd
import numpy as np
from puffin.models.linear import OLSModel
from puffin.data.sources import get_bars

# Fetch historical data
symbol = 'AAPL'
df = get_bars(symbol, start='2020-01-01', end='2023-12-31')

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
- Positive coefficient: Feature increase → return increase
- Negative coefficient: Feature increase → return decrease
- Magnitude: Effect size (e.g., β=0.01 means 1 unit increase → 0.01 return increase)

**Statistical Tests:**
- **P-value < 0.05**: Feature is statistically significant
- **R-squared**: Proportion of variance explained (higher is better)
- **Residuals**: Should be randomly distributed (check for autocorrelation)

**Common Issues:**
- **Multicollinearity**: Correlated features cause unstable coefficients
- **Heteroscedasticity**: Non-constant variance violates OLS assumptions
- **Autocorrelation**: Time-series dependencies violate independence assumption

## Regularization: Ridge and Lasso

Regularization adds penalties to prevent overfitting, especially with many features.

### Ridge Regression (L2 Regularization)

Ridge adds penalty proportional to squared coefficients: minimize ||y - Xβ||² + α||β||²

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

**When to use Ridge:**
- Many correlated features
- Want to keep all features
- Multicollinearity present
- Need stable coefficient estimates

### Lasso Regression (L1 Regularization)

Lasso adds penalty proportional to absolute coefficients: minimize ||y - Xβ||² + α||β||₁

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

### Comparing OLS, Ridge, and Lasso

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

    print(f"{name:10s} - MSE: {mse:.6f}, R²: {r2:.4f}, Corr: {corr:.4f}")
```

## Logistic Regression for Direction Prediction

For directional trading strategies, we predict whether the price will move up or down.

```python
from puffin.models.linear import DirectionClassifier

# Create direction labels
df['direction'] = (df['target'] > 0).astype(int)  # 1 = up, 0 = down

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

### Trading with Direction Predictions

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

## Capital Asset Pricing Model (CAPM)

CAPM relates an asset's expected return to its systematic risk (beta):

```
E[Rᵢ - Rf] = α + β(E[Rₘ - Rf])
```

Where:
- Rᵢ: Asset return
- Rf: Risk-free rate
- Rₘ: Market return
- β: Systematic risk (market sensitivity)
- α: Jensen's alpha (excess return)

### Fitting CAPM

```python
from puffin.models.factor_models import FamaFrenchModel

# Initialize model
ff_model = FamaFrenchModel()

# Get asset returns
asset_returns = df['returns']

# Fit CAPM
capm_results = ff_model.fit_capm(
    returns=asset_returns,
    start='2020-01-01',
    end='2023-12-31'
)

print("CAPM Results:")
print(f"Alpha: {capm_results['alpha']:.6f} (p={capm_results['alpha_pvalue']:.4f})")
print(f"Beta: {capm_results['beta']:.4f} (p={capm_results['beta_pvalue']:.4f})")
print(f"R-squared: {capm_results['r_squared']:.4f}")

# Interpret results
if capm_results['alpha_pvalue'] < 0.05:
    if capm_results['alpha'] > 0:
        print("\nSignificant positive alpha: Outperforming market!")
    else:
        print("\nSignificant negative alpha: Underperforming market.")
else:
    print("\nAlpha not significant: Performance explained by market exposure.")

if capm_results['beta'] > 1:
    print(f"High beta ({capm_results['beta']:.2f}): More volatile than market")
elif capm_results['beta'] < 1:
    print(f"Low beta ({capm_results['beta']:.2f}): Less volatile than market")
```

### Beta Interpretation

- **β = 1**: Moves with market
- **β > 1**: Amplifies market moves (aggressive)
- **β < 1**: Dampens market moves (defensive)
- **β < 0**: Moves opposite to market (rare)

## Fama-French Factor Models

Fama-French models extend CAPM with additional factors capturing size and value effects.

### Three-Factor Model

```
E[Rᵢ - Rf] = α + β₁(Rₘ - Rf) + β₂·SMB + β₃·HML
```

Where:
- **Mkt-RF**: Market excess return
- **SMB** (Small Minus Big): Size factor (small cap - large cap)
- **HML** (High Minus Low): Value factor (value - growth)

```python
# Fit 3-factor model
ff3_results = ff_model.fit_three_factor(
    returns=asset_returns,
    start='2020-01-01',
    end='2023-12-31'
)

print("Fama-French 3-Factor Results:")
print(f"Alpha: {ff3_results['alpha']:.6f} (p={ff3_results['pvalues']['alpha']:.4f})")
print(f"Market Beta: {ff3_results['beta_mkt']:.4f} (p={ff3_results['pvalues']['Mkt-RF']:.4f})")
print(f"SMB Beta: {ff3_results['beta_smb']:.4f} (p={ff3_results['pvalues']['SMB']:.4f})")
print(f"HML Beta: {ff3_results['beta_hml']:.4f} (p={ff3_results['pvalues']['HML']:.4f})")
print(f"R-squared: {ff3_results['r_squared']:.4f}")
```

### Five-Factor Model

The 5-factor model adds profitability (RMW) and investment (CMA) factors:

```
E[Rᵢ - Rf] = α + β₁(Rₘ - Rf) + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA
```

Where:
- **RMW** (Robust Minus Weak): Profitability factor
- **CMA** (Conservative Minus Aggressive): Investment factor

```python
# Fit 5-factor model
ff5_results = ff_model.fit_five_factor(
    returns=asset_returns,
    start='2020-01-01',
    end='2023-12-31'
)

print("Fama-French 5-Factor Results:")
print(f"Alpha: {ff5_results['alpha']:.6f}")
print(f"\nFactor Exposures:")
for factor, beta in ff5_results['betas'].items():
    pval = ff5_results['pvalues'][factor]
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"  {factor:10s}: {beta:7.4f} {sig}")
print(f"\nR-squared: {ff5_results['r_squared']:.4f}")
print(f"Adjusted R-squared: {ff5_results['adj_r_squared']:.4f}")
```

### Portfolio Attribution

Use factor models to understand portfolio risk sources:

```python
# Multiple assets
symbols = ['AAPL', 'MSFT', 'JPM', 'XOM']
factor_exposures = {}

for symbol in symbols:
    df_asset = get_bars(symbol, start='2020-01-01', end='2023-12-31')
    returns = df_asset['close'].pct_change().dropna()

    results = ff_model.fit_five_factor(returns, start='2020-01-01', end='2023-12-31')
    factor_exposures[symbol] = results['betas']

# Create factor exposure DataFrame
exposure_df = pd.DataFrame(factor_exposures).T
print("\nFactor Exposures by Asset:")
print(exposure_df.round(3))

# Visualize
import matplotlib.pyplot as plt
exposure_df.plot(kind='bar', figsize=(12, 6))
plt.title('Factor Exposures by Asset')
plt.ylabel('Beta')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('factor_exposures.png')
```

## Fama-MacBeth Regression

Fama-MacBeth is a two-step procedure for estimating factor risk premiums in cross-section:

1. **Time-series regression**: Estimate factor loadings (betas) for each asset
2. **Cross-sectional regression**: Regress returns on betas at each time period
3. **Average**: Average cross-sectional coefficients to get risk premiums

### Example with Panel Data

```python
from puffin.models.factor_models import fama_macbeth

# Prepare panel data with multiple assets and time periods
panel_data = []

for symbol in ['AAPL', 'MSFT', 'JPM', 'XOM', 'PG', 'JNJ', 'V', 'MA']:
    df_asset = get_bars(symbol, start='2020-01-01', end='2023-12-31')
    df_asset['returns'] = df_asset['close'].pct_change()

    # Calculate factor exposures (from prior period)
    df_asset['beta_mkt'] = df_asset['returns'].rolling(60).cov(market_returns) / market_returns.rolling(60).var()
    df_asset['beta_smb'] = calculate_smb_beta(df_asset)  # Custom function
    df_asset['beta_hml'] = calculate_hml_beta(df_asset)  # Custom function

    df_asset['asset'] = symbol
    df_asset = df_asset.dropna()

    panel_data.append(df_asset[['date', 'asset', 'returns', 'beta_mkt', 'beta_smb', 'beta_hml']])

panel_df = pd.concat(panel_data, ignore_index=True)

# Run Fama-MacBeth
fm_results = fama_macbeth(
    panel_data=panel_df,
    factors=['beta_mkt', 'beta_smb', 'beta_hml'],
    return_col='returns',
    time_col='date',
    asset_col='asset'
)

print("Fama-MacBeth Results:")
print("\nRisk Premiums:")
for factor, premium in fm_results['risk_premiums'].items():
    t_stat = fm_results['t_stats'][factor]
    sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.65 else ""
    print(f"  {factor:15s}: {premium:8.6f} (t={t_stat:6.2f}) {sig}")

print(f"\nAverage R-squared: {fm_results['r_squared']:.4f}")
print(f"Number of periods: {fm_results['n_periods']}")
```

### Interpreting Fama-MacBeth Results

- **Risk Premium**: Expected excess return per unit of factor exposure
- **T-statistic**: Statistical significance (|t| > 1.96 for 5% level)
- **Cross-sectional R²**: How well factors explain return differences across assets

## Complete Trading Example

Let's build a complete trading system using linear models:

```python
import pandas as pd
import numpy as np
from puffin.data.sources import get_bars
from puffin.models.linear import RidgeModel, DirectionClassifier
from puffin.models.factor_models import FamaFrenchModel
from puffin.backtest import Backtest, Strategy
from puffin.risk import calculate_metrics

class LinearModelStrategy(Strategy):
    """Trading strategy using linear models."""

    def __init__(self, lookback=60, refit_freq=20):
        self.lookback = lookback
        self.refit_freq = refit_freq
        self.return_model = None
        self.direction_model = None
        self.days_since_fit = 0

    def calculate_features(self, df):
        """Calculate technical features."""
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_20'] = df['close'].pct_change(20)
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['trend'] = (df['close'] / df['close'].rolling(50).mean()) - 1
        return df

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def on_data(self, data):
        """Called on each bar."""
        if len(data) < self.lookback + 30:
            return

        # Calculate features
        df = self.calculate_features(data.copy())
        df = df.dropna()

        if len(df) < self.lookback:
            return

        # Refit models periodically
        self.days_since_fit += 1
        if self.return_model is None or self.days_since_fit >= self.refit_freq:
            self.fit_models(df)
            self.days_since_fit = 0

        # Make predictions
        features = ['momentum_5', 'momentum_20', 'rsi', 'volume_ratio',
                   'volatility', 'trend']
        X_latest = df[features].iloc[-1:].values

        # Predict return magnitude
        pred_return = self.return_model.predict(X_latest)[0]

        # Predict direction probability
        pred_proba = self.direction_model.predict_proba(X_latest)[0, 1]

        # Generate signal
        if pred_proba > 0.55 and pred_return > 0:
            # Strong buy signal
            self.set_position(1.0)
        elif pred_proba < 0.45 and pred_return < 0:
            # Strong sell signal
            self.set_position(-1.0)
        elif pred_proba > 0.52:
            # Weak buy signal
            self.set_position(0.5)
        elif pred_proba < 0.48:
            # Weak sell signal
            self.set_position(-0.5)
        else:
            # No clear signal
            self.set_position(0.0)

    def fit_models(self, df):
        """Fit return and direction models."""
        # Prepare training data
        features = ['momentum_5', 'momentum_20', 'rsi', 'volume_ratio',
                   'volatility', 'trend']

        df['returns'] = df['close'].pct_change()
        df['target_return'] = df['returns'].shift(-1)
        df['target_direction'] = (df['target_return'] > 0).astype(int)

        train_df = df.dropna().iloc[-self.lookback:]
        X_train = train_df[features]
        y_return = train_df['target_return']
        y_direction = train_df['target_direction']

        # Fit return prediction model (Ridge)
        self.return_model = RidgeModel(normalize=True)
        self.return_model.fit(X_train, y_return)

        # Fit direction prediction model (Logistic)
        self.direction_model = DirectionClassifier(
            class_weight='balanced',
            normalize=True
        )
        self.direction_model.fit(X_train, y_direction)

# Run backtest
df = get_bars('AAPL', start='2020-01-01', end='2023-12-31')

strategy = LinearModelStrategy(lookback=120, refit_freq=20)
backtest = Backtest(data=df, strategy=strategy, cash=100000)

results = backtest.run()

# Display results
print("\nBacktest Results:")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Total Trades: {results['total_trades']}")

# Compare to buy-and-hold
bh_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
print(f"\nBuy-and-Hold Return: {bh_return:.2%}")
print(f"Excess Return: {results['total_return'] - bh_return:.2%}")

# Plot results
backtest.plot()
```

## Key Takeaways

1. **Linear models provide interpretability** - Understand which features drive returns

2. **Regularization prevents overfitting** - Use Ridge/Lasso with many features

3. **Direction prediction differs from return prediction** - Logistic regression for classification

4. **Factor models explain systematic risk** - CAPM and Fama-French decompose returns

5. **Statistical significance matters** - Check p-values before trusting results

6. **Refit models periodically** - Market dynamics change over time

7. **Combine predictions** - Use both magnitude and direction for better signals

8. **Monitor residuals** - Ensure model assumptions hold

## Next Steps

- **Part 9: Time Series Models** - ARIMA, GARCH, state-space models
- **Part 10: Machine Learning** - Non-linear models, ensemble methods
- **Part 11: Deep Learning** - Neural networks for trading

## References

- Fama, E. F., & French, K. R. (1993). [Common risk factors in the returns on stocks and bonds](https://doi.org/10.1016/0304-405X(93)90023-5). *Journal of Financial Economics*.
- Fama, E. F., & MacBeth, J. D. (1973). [Risk, return, and equilibrium: Empirical tests](https://doi.org/10.1086/260061). *Journal of Political Economy*.
- Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press.
