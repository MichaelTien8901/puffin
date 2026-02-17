---
layout: default
title: "Part 4: Alpha Factors"
nav_order: 5
permalink: /04-alpha-factors/
---

# Alpha Factor Research

Alpha factors are the building blocks of quantitative trading strategies. They are signals that predict future asset returns and form the basis of systematic investment decisions. In this chapter, we'll explore how to compute, evaluate, and combine various types of alpha factors using the Puffin framework.

## What Are Alpha Factors?

An alpha factor is a quantitative signal that captures some aspect of market behavior or asset characteristics that may predict future returns. The term "alpha" refers to excess returns above a benchmark.

{: .tip }
> **Plain English:** Alpha is your "skill" versus "luck." If the tide lifts all boats (the market goes up), Alpha is having a motor that makes you go *even faster*. Specifically: if the market returns 10% and you return 15%, your Alpha is that extra 5%.

Key characteristics of good alpha factors:
- **Predictive Power**: Strong correlation with future returns
- **Stability**: Consistent performance across different time periods
- **Low Correlation**: Independent from other factors in your portfolio
- **Implementability**: Can be traded with reasonable transaction costs

## Types of Alpha Factors

### 1. Momentum Factors

Momentum factors capture the tendency of assets to continue moving in the same direction. They are based on the observation that "winners keep winning and losers keep losing" (at least in the short to medium term).

{: .tip }
> **Plain English:** Momentum is like a freight train — it takes a long time to speed up and a long time to slow down. Stocks that have been going up tend to *keep* going up for a while, and vice versa. Momentum strategies ride that train.

```python
import pandas as pd
import numpy as np
from puffin.factors import compute_momentum_factors

# Load price data
prices = pd.DataFrame({
    'AAPL': [100, 102, 105, 103, 108, 110, 112],
    'MSFT': [200, 198, 202, 205, 207, 210, 208],
    'GOOGL': [150, 152, 151, 155, 158, 160, 162]
}, index=pd.date_range('2024-01-01', periods=7, freq='D'))

# Compute momentum factors over multiple time horizons
momentum = compute_momentum_factors(
    prices,
    windows=[5, 10, 21, 63, 252]  # Week, 2-weeks, month, quarter, year
)

print(momentum.head())
```

**Momentum Factor Variants:**
- **Simple Momentum**: Return over lookback period
- **Momentum Ratio**: Short-term vs long-term momentum
- **Acceleration**: Second derivative of momentum
- **Trend Strength**: Percentage of positive days

### 2. Value Factors

Value factors identify undervalued or overvalued securities based on fundamental metrics. The core idea is "buy cheap, sell expensive."

```python
from puffin.factors import compute_value_factors

# Fundamental data with MultiIndex (date, symbol)
fundamentals = pd.DataFrame({
    'price': [150, 155, 152],
    'earnings': [10, 12, 11],
    'book_value': [80, 85, 82],
    'enterprise_value': [2000, 2100, 2050],
    'ebitda': [150, 160, 155],
    'revenue': [500, 520, 510],
    'market_cap': [1500, 1550, 1525]
}, index=pd.MultiIndex.from_product(
    [pd.date_range('2024-01-01', periods=3, freq='Q'), ['AAPL']],
    names=['date', 'symbol']
))

# Compute value factors
value = compute_value_factors(fundamentals)

print(value)
# Output includes: P/E, P/B, EV/EBITDA, earnings yield, book yield
```

**Value Factor Variants:**
- **P/E Ratio**: Price to Earnings
- **P/B Ratio**: Price to Book Value
- **EV/EBITDA**: Enterprise Value to EBITDA
- **Earnings Yield**: E/P (inverse of P/E)

### 3. Volatility Factors

Volatility factors measure the variability of asset returns. They can be used for risk management or to exploit the "low volatility anomaly" (low-vol stocks often outperform high-vol stocks).

```python
from puffin.factors import compute_volatility_factors

# With OHLC data for advanced volatility estimators
ohlcv = {
    'close': prices,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'open': prices.shift(1).fillna(prices.iloc[0])
}

# Compute volatility factors
volatility = compute_volatility_factors(
    ohlcv,
    windows=[21, 63]  # Month and quarter
)

print(volatility.head())
```

**Volatility Estimators:**
- **Realized Volatility**: Close-to-close standard deviation
- **Parkinson Estimator**: Uses high-low range (more efficient)
- **Garman-Klass**: Uses OHLC data (most efficient)
- **Volatility Ratio**: Short-term vs long-term vol (regime detection)

### 4. Quality Factors

Quality factors measure the financial health and profitability of companies. They identify companies with strong fundamentals.

```python
from puffin.factors import compute_quality_factors

# Financial statement data
financials = pd.DataFrame({
    'net_income': [1000, 1100, 1050],
    'revenue': [5000, 5200, 5100],
    'assets': [10000, 10500, 10300],
    'equity': [6000, 6200, 6100],
    'operating_cash_flow': [1200, 1300, 1250],
    'total_accruals': [-50, -30, -40],
    'liabilities': [4000, 4300, 4200]
}, index=pd.MultiIndex.from_product(
    [pd.date_range('2024-01-01', periods=3, freq='Q'), ['AAPL']],
    names=['date', 'symbol']
))

# Compute quality factors
quality = compute_quality_factors(financials)

print(quality)
# Output: ROE, ROA, profit margin, asset turnover, accruals ratio, etc.
```

**Quality Metrics:**
- **ROE**: Return on Equity
- **ROA**: Return on Assets
- **Profit Margin**: Net income / Revenue
- **Accruals Ratio**: Lower = better earnings quality
- **Cash Flow to Income**: Higher = better quality

## Technical Indicators with TA-Lib

The Puffin framework provides a unified interface for technical indicators with automatic fallback to pure Python if TA-Lib is not installed.

```python
from puffin.factors import TechnicalIndicators

# Create OHLCV data
ohlcv = {
    'open': pd.Series([100, 102, 101, 103, 105]),
    'high': pd.Series([103, 104, 103, 106, 108]),
    'low': pd.Series([99, 101, 100, 102, 104]),
    'close': pd.Series([102, 101, 102, 105, 107]),
    'volume': pd.Series([1000, 1100, 1050, 1200, 1150])
}

# Initialize technical indicators calculator
ti = TechnicalIndicators()

# Compute all indicators
all_indicators = ti.compute_all(ohlcv)

# Or compute specific categories
overlap = ti.compute_overlap(ohlcv)  # SMA, EMA, BBANDS, SAR
momentum = ti.compute_momentum(ohlcv)  # RSI, MACD, STOCH, ADX, CCI, MFI
volume = ti.compute_volume(ohlcv)  # OBV, AD, ADOSC
volatility = ti.compute_volatility(ohlcv)  # ATR, NATR, TRANGE

print("RSI:", momentum['rsi'].iloc[-1])
print("MACD:", momentum['macd'].iloc[-1])
print("Bollinger Upper:", overlap['bb_upper'].iloc[-1])
```

### Popular Technical Indicators

**Overlap Studies (Trend Following):**
- **SMA**: Simple Moving Average
- **EMA**: Exponential Moving Average
- **Bollinger Bands**: Volatility bands around moving average
- **Parabolic SAR**: Stop and Reverse indicator

**Momentum Indicators:**
- **RSI**: Relative Strength Index (overbought/oversold)
- **MACD**: Moving Average Convergence Divergence
- **Stochastic**: Momentum oscillator comparing close to high-low range
- **ADX**: Average Directional Index (trend strength)

## Signal Denoising with Kalman Filters

Financial data is noisy. Kalman filters provide optimal estimates of the true signal by filtering out noise.

```python
from puffin.factors import KalmanFilter, extract_trend, dynamic_hedge_ratio

# Basic Kalman filtering
prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])

kf = KalmanFilter(
    process_covariance=1e-5,  # How much we expect signal to change
    observation_covariance=1e-2  # Noise level in observations
)

# Filter signal (forward pass only)
filtered = kf.filter(prices)

# Smooth signal (forward-backward pass - better estimates)
smoothed = kf.smooth(prices)

print("Original:", prices.iloc[-1])
print("Filtered:", filtered.iloc[-1])
print("Smoothed:", smoothed.iloc[-1])
```

### Trend Extraction

```python
# Extract trend from noisy price series
trend = extract_trend(
    prices,
    process_variance=1e-5,  # Lower = smoother trend
    observation_variance=1e-2  # Higher = more smoothing
)

# Use for trend-following strategy
signal = (prices > trend).astype(int)  # 1 when above trend, 0 when below
```

### Dynamic Hedge Ratio for Pairs Trading

```python
# Calculate time-varying hedge ratio for pairs trading
stock1 = pd.Series([100, 101, 102, 103, 104])
stock2 = pd.Series([50, 51, 50, 52, 53])

hedge_ratio = dynamic_hedge_ratio(stock1, stock2, delta=1e-5)

# Construct spread
spread = stock1 - hedge_ratio * stock2

# Trade when spread deviates from mean
```

## Signal Denoising with Wavelets

Wavelets provide multi-resolution decomposition of signals, allowing you to separate different time scales.

```python
from puffin.factors import (
    wavelet_denoise,
    wavelet_decompose,
    multiscale_decomposition
)

# Denoise signal using wavelet thresholding
noisy_signal = pd.Series([1, 2, 1.5, 3, 2.5, 4, 3.5, 5])

denoised = wavelet_denoise(
    noisy_signal,
    wavelet='db4',  # Daubechies 4 wavelet
    level=3,  # Decomposition depth
    threshold_method='soft'  # Soft thresholding
)

# Decompose into multiple time scales
prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

components = wavelet_decompose(prices, wavelet='db4', level=3)

print("Trend:", components['approximation'])
print("High-frequency noise:", components['detail_1'])
print("Medium-frequency:", components['detail_2'])

# Multi-scale trading
scales = multiscale_decomposition(prices, level=4)

# Trade on different time scales
# scales['trend'] - Long-term position trading
# scales['D4'] - Swing trading (days to weeks)
# scales['D1'] - Day trading (intraday)
```

## Factor Evaluation with Alphalens

Evaluating factor quality is crucial before deploying it in a strategy.

```python
from puffin.factors import FactorEvaluator

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

## WorldQuant-Style Formulaic Alphas

WorldQuant popularized expressing alpha factors as mathematical formulas. Puffin supports this approach.

```python
from puffin.factors import AlphaExpression, evaluate_alpha, ALPHA_LIBRARY

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

## Combining Multiple Factors

Combining multiple factors often improves performance through diversification.

```python
from puffin.factors import combine_alphas, neutralize_factor

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
> **Plain English:** Beta measures how much a stock moves *with* the market. If the market sneezes, does your stock catch a cold (high Beta) or stay healthy (low Beta)? A Beta of 1.5 means your stock moves 50% more than the market — exciting on the way up, painful on the way down.

## Complete Factor Research Workflow

Here's a complete example of researching a new alpha factor:

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
    print("\n✓ Factor passed evaluation criteria!")
    print("Ready for backtesting and deployment.")
else:
    print("\n✗ Factor needs improvement.")
```

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

## Next Steps

Now that you understand alpha factor research, you can:
- Learn about machine learning-based factors in Part 5
- Backtest your factors in Part 7
- Implement risk management in Part 8
- Deploy factors in live trading in Part 9

## Further Reading

- Kakushadze, Z. (2016). ["101 Formulaic Alphas"](https://arxiv.org/abs/1601.00991). Wilmott Magazine.
- Stambaugh, R. F., & Yuan, Y. (2017). "Mispricing Factors". Review of Financial Studies.
- Hou, K., Xue, C., & Zhang, L. (2015). ["Digesting Anomalies"](https://doi.org/10.1093/rfs/hhu068). Review of Financial Studies.
- Prado, M. L. (2018). ["Advances in Financial Machine Learning"](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086). Wiley.


**Practice Exercise**: Create a custom alpha factor combining momentum and volatility, evaluate it using the FactorEvaluator, and interpret the results. Try to achieve an Information Ratio above 0.5 with turnover below 50%.
