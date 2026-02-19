---
layout: default
title: "Factor Evaluation"
parent: "Part 4: Alpha Factors"
nav_order: 3
---

# Factor Evaluation

Discovering a factor is only half the battle. Before deploying any alpha signal in a live strategy, you need to rigorously evaluate its predictive power, stability, and implementability. This page covers the Puffin tools for factor evaluation, WorldQuant-style formulaic alphas, and techniques for combining multiple factors into a composite signal.

---

## Factor Evaluation with Alphalens

Evaluating factor quality is crucial before deploying it in a strategy. Puffin's `FactorEvaluator` follows the Alphalens methodology: sort assets into quantile buckets by factor value, then measure whether the top-ranked bucket outperforms the bottom-ranked bucket over various forward-return horizons.

```python
from puffin.factors import FactorEvaluator
import pandas as pd
import numpy as np

# Create factor with MultiIndex (date, symbol)
factor_data = []
for date in pd.date_range('2024-01-01', periods=50, freq='D'):
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        factor_data.append({
            'date': date,
            'symbol': symbol,
            'factor': np.random.randn()  # Replace with real factor
        })

factor = pd.DataFrame(factor_data).set_index(['date', 'symbol'])['factor']

# Price data
prices = pd.DataFrame(
    np.random.randn(60, 3).cumsum(axis=0) + 100,
    index=pd.date_range('2024-01-01', periods=60, freq='D'),
    columns=['AAPL', 'MSFT', 'GOOGL']
)

# Evaluate factor
evaluator = FactorEvaluator(quantiles=5, periods=[1, 5, 21])

# Compute full tearsheet
tearsheet = evaluator.full_tearsheet(factor, prices)

print("Mean IC:", tearsheet['ic_mean'])
print("IC Std:", tearsheet['ic_std'])
print("Information Ratio:", tearsheet['ic_ir'])
print("Mean Turnover:", tearsheet['mean_turnover'])

if 'mean_returns' in tearsheet:
    print("\nFactor Returns by Period:")
    print(tearsheet['mean_returns'])
```

### Key Evaluation Metrics

**Information Coefficient (IC):**
- Correlation between factor and forward returns
- Good factors: |IC| > 0.05
- Excellent factors: |IC| > 0.10

**Information Ratio (IR):**
- IC mean divided by IC std
- Measures consistency of predictive power
- Good factors: IR > 0.5

**Factor Returns:**
- Returns of long-short portfolio (top quintile - bottom quintile)
- Should be positive and significant

**Turnover:**
- How much factor rankings change over time
- High turnover = high transaction costs
- Good factors: Low turnover with high returns

{: .warning }
> A factor with a high IC but also high turnover may not be profitable after transaction costs. Always evaluate net-of-cost returns. As a rule of thumb, if turnover exceeds 50% per period, the factor needs very strong returns to remain viable.

---

## WorldQuant-Style Formulaic Alphas

WorldQuant popularized expressing alpha factors as mathematical formulas. Puffin supports this approach through `AlphaExpression`, which parses and evaluates string-based factor definitions.

```python
from puffin.factors import AlphaExpression, evaluate_alpha, ALPHA_LIBRARY
import pandas as pd

# Prepare data
data = {
    'open': pd.DataFrame({'AAPL': [100, 101, 102, 103, 104]}),
    'high': pd.DataFrame({'AAPL': [102, 103, 104, 105, 106]}),
    'low': pd.DataFrame({'AAPL': [99, 100, 101, 102, 103]}),
    'close': pd.DataFrame({'AAPL': [101, 102, 103, 104, 105]}),
    'volume': pd.DataFrame({'AAPL': [1000, 1100, 1050, 1200, 1150]})
}

# Define and evaluate alpha expression
alpha = AlphaExpression("rank(delta(close, 1))")
factor = alpha.evaluate(data)

# More complex example
alpha2 = AlphaExpression("rank(ts_mean(close, 5) / close - 1)")
factor2 = alpha2.evaluate(data)

# Use predefined alphas from library
print("Available alphas:", list(ALPHA_LIBRARY.keys()))

# Evaluate alpha from library
alpha3 = evaluate_alpha(ALPHA_LIBRARY['alpha001'], data)
```

### Common Alpha Operators

**Cross-Sectional:**
- `rank(x)`: Percentile rank across assets
- `scale(x)`: Normalize to sum to 1

**Time-Series:**
- `delay(x, d)`: Value d periods ago
- `delta(x, d)`: Change over d periods
- `ts_mean(x, d)`: Rolling mean
- `ts_std(x, d)`: Rolling standard deviation
- `ts_rank(x, d)`: Rank within rolling window

**Dual:**
- `correlation(x, y, d)`: Rolling correlation
- `covariance(x, y, d)`: Rolling covariance

### Example Alphas

```python
from puffin.factors import evaluate_alpha

# Alpha 1: Reversal
# "Buy yesterday's losers"
alpha1 = "rank(-delta(close, 1))"

# Alpha 2: Momentum
# "Buy assets above moving average"
alpha2 = "rank(close / ts_mean(close, 20) - 1)"

# Alpha 3: Volume-Price
# "Buy when volume increases with price"
alpha3 = "rank(correlation(close, volume, 10))"

# Alpha 4: Volatility-adjusted momentum
# "Momentum divided by volatility"
alpha4 = "rank(delta(close, 5) / ts_std(close, 5))"

# Evaluate all
for expr in [alpha1, alpha2, alpha3, alpha4]:
    factor = evaluate_alpha(expr, data)
    print(f"Alpha: {expr}")
    print(f"Latest values:\n{factor.iloc[-1]}\n")
```

{: .tip }
> The formulaic alpha approach makes it easy to iterate rapidly. You can define dozens of candidate alphas as one-line strings, evaluate them all against the same data, and keep only the ones that pass your IC/IR thresholds.

---

## Combining Multiple Factors

Combining multiple factors often improves performance through diversification. The intuition is the same as portfolio diversification: if individual factors are imperfectly correlated, the combination is more stable than any single factor.

```python
from puffin.factors import combine_alphas, neutralize_factor
from puffin.factors import evaluate_alpha
import pandas as pd

# Evaluate multiple alphas
factors = {
    'momentum': evaluate_alpha("rank(delta(close, 5))", data),
    'reversal': evaluate_alpha("rank(-delta(close, 1))", data),
    'volume': evaluate_alpha("rank(correlation(close, volume, 10))", data)
}

# Equal-weighted combination
combined_equal = combine_alphas(factors)

# Custom weights
combined_weighted = combine_alphas(
    factors,
    weights={'momentum': 0.5, 'reversal': 0.2, 'volume': 0.3}
)

# Neutralize factor against market beta
market_beta = pd.DataFrame({'AAPL': [1.2, 1.1, 1.2, 1.1, 1.2]})
neutral_factor = neutralize_factor(combined_weighted, market_beta)
```

{: .tip }
> **Plain English:** Beta measures how much a stock moves *with* the market. If the market sneezes, does your stock catch a cold (high Beta) or stay healthy (low Beta)? A Beta of 1.5 means your stock moves 50% more than the market -- exciting on the way up, painful on the way down.

{: .note }
> Factor neutralization removes unwanted exposures (e.g., market beta, sector, size) so that the composite signal captures pure alpha rather than systematic risk premiums. This is especially important for market-neutral strategies.

---

## Complete Factor Research Workflow

Here's a complete example of researching a new alpha factor from raw data through evaluation:

```python
import pandas as pd
import numpy as np
from puffin.factors import (
    compute_momentum_factors,
    TechnicalIndicators,
    wavelet_denoise,
    FactorEvaluator,
    AlphaExpression
)

# 1. Load data
prices = pd.DataFrame({
    'AAPL': np.random.randn(252).cumsum() + 100,
    'MSFT': np.random.randn(252).cumsum() + 200,
    'GOOGL': np.random.randn(252).cumsum() + 150
}, index=pd.date_range('2024-01-01', periods=252, freq='D'))

ohlcv = {
    'close': prices,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'open': prices.shift(1).fillna(prices.iloc[0]),
    'volume': pd.DataFrame(
        np.random.uniform(1000, 2000, (252, 3)),
        index=prices.index,
        columns=prices.columns
    )
}

# 2. Compute raw factors
momentum = compute_momentum_factors(prices, windows=[5, 21, 63])

# 3. Denoise signals
for symbol in prices.columns:
    prices[symbol] = wavelet_denoise(prices[symbol], level=2)

# 4. Create custom alpha
alpha = AlphaExpression("rank(ts_mean(close, 21) / close - 1)")
factor_df = alpha.evaluate(ohlcv)

# 5. Convert to MultiIndex for evaluation
from puffin.factors import to_multiindex_series
factor = to_multiindex_series(factor_df)

# 6. Evaluate factor
evaluator = FactorEvaluator(quantiles=5, periods=[1, 5, 21])
tearsheet = evaluator.full_tearsheet(factor, prices)

# 7. Analyze results
print("=" * 50)
print("FACTOR EVALUATION RESULTS")
print("=" * 50)
print(f"Mean IC: {tearsheet['ic_mean']:.4f}")
print(f"IC Std: {tearsheet['ic_std']:.4f}")
print(f"Information Ratio: {tearsheet['ic_ir']:.4f}")
print(f"Mean Turnover: {tearsheet['mean_turnover']:.2f}%")

if 'mean_returns' in tearsheet:
    print("\nFactor Returns:")
    print(tearsheet['mean_returns'])

# 8. Deploy if factor is good
if tearsheet['ic_ir'] > 0.5 and tearsheet['mean_turnover'] < 50:
    print("\nFactor passed evaluation criteria!")
    print("Ready for backtesting and deployment.")
else:
    print("\nFactor needs improvement.")
```

---

## Best Practices

1. **Start Simple**: Begin with well-known factors (momentum, value) before creating complex custom factors.

2. **Avoid Overfitting**:
   - Use out-of-sample testing
   - Limit the number of parameters
   - Test across different time periods and market regimes

3. **Consider Transaction Costs**:
   - High turnover factors may not be profitable after costs
   - Factor returns should exceed implementation costs

4. **Diversify**:
   - Combine multiple low-correlation factors
   - Don't rely on a single factor type

5. **Monitor Decay**:
   - Factors can stop working due to crowding
   - Regularly re-evaluate factor performance
   - Be prepared to retire or modify factors

6. **Use Proper Data**:
   - Ensure data is free from look-ahead bias
   - Handle survivorship bias
   - Account for corporate actions (splits, dividends)

{: .warning }
> Look-ahead bias is the most dangerous pitfall in factor research. Always ensure that the data available to your factor at time *t* was genuinely known at time *t*. Fundamental data, in particular, is released with a lag -- using Q1 earnings to generate a January signal is a classic mistake.

---

**Practice Exercise**: Create a custom alpha factor combining momentum and volatility, evaluate it using the FactorEvaluator, and interpret the results. Try to achieve an Information Ratio above 0.5 with turnover below 50%.

---

## Source Code

Browse the implementation: [`puffin/factors/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/factors)
