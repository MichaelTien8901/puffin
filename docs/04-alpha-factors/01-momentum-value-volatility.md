---
layout: default
title: "Momentum, Value & Volatility Factors"
parent: "Part 4: Alpha Factors"
nav_order: 1
---

# Momentum, Value & Volatility Factors

This page covers the four classic factor families -- momentum, value, volatility, and quality -- along with TA-Lib technical indicators. These are the workhorses of quantitative investing: well-studied, broadly used, and still effective when combined thoughtfully.

---

## Momentum Factors

Momentum factors capture the tendency of assets to continue moving in the same direction. They are based on the observation that "winners keep winning and losers keep losing" (at least in the short to medium term).

{: .tip }
> **Plain English:** Momentum is like a freight train -- it takes a long time to speed up and a long time to slow down. Stocks that have been going up tend to *keep* going up for a while, and vice versa. Momentum strategies ride that train.

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

---

## Value Factors

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

---

## Volatility Factors

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

---

## Quality Factors

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

---

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

{: .note }
> TA-Lib is an optional dependency. If it is not installed, `TechnicalIndicators` falls back to equivalent pure-Python implementations so that code runs everywhere without modification.

---

## Source Code

Browse the implementation: [`puffin/factors/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/factors)
