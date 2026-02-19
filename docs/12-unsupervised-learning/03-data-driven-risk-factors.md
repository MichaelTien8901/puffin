---
layout: default
title: "Data-Driven Risk Factors"
parent: "Part 12: Unsupervised Learning"
nav_order: 3
---

# Data-Driven Risk Factors

Traditional factor models (Fama-French, Barra) use pre-defined factors such as market, size, and value. PCA extracts factors directly from return data, producing a purely statistical decomposition. This page covers factor extraction, exposure analysis, attribution, and trading signals built on data-driven factors.

{: .note }
> Data-driven factors complement traditional models. They can capture structural patterns
> (e.g., sector rotations, crowded trades) that predefined factors miss.

## Extract Risk Factors

Use PCA to extract latent factors from a cross-section of returns. Each factor is a weighted combination of all assets, chosen to maximize explanatory power.

```python
from puffin.unsupervised import extract_risk_factors, factor_exposures

# Extract 5 data-driven factors
factors = extract_risk_factors(returns, n_factors=5)
print(factors.head())
#             Factor_1  Factor_2  Factor_3  Factor_4  Factor_5
# 2020-01-01     0.012     0.005    -0.003     0.001     0.002
# 2020-01-02    -0.008     0.010     0.001    -0.004     0.000
```

Each row is one trading day. Each column is the return of one statistical factor on that day. Factor 1 explains the most variance (typically the market factor), Factor 2 the next most, and so on.

## Factor Exposures

Factor exposures (also called loadings or betas) measure how sensitive each asset is to each factor.

```python
# Compute factor exposures (betas)
loadings = factor_exposures(returns, factors)
print(loadings)
#        Factor_1  Factor_2  Factor_3  Factor_4  Factor_5
# AAPL       0.85      0.23      0.15      0.05      0.01
# GOOGL      0.80      0.30      0.10      0.08      0.02
# MSFT       0.75      0.35      0.05      0.10      0.03
```

{: .important }
> A loading of 0.85 on Factor 1 means that a 1% move in Factor 1 explains a 0.85% move in that
> stock's return, all else equal. High Factor 1 loadings across all stocks confirm it is the market factor.

**Interpreting the loading matrix**:
- **Factor 1**: Near-uniform loadings suggest the market factor
- **Factor 2**: Positive/negative splits may indicate sector or style tilts
- **Higher factors**: Increasingly specific patterns (may be noise in small samples)

## Factor Attribution

Decompose returns into the portion explained by common factors and the idiosyncratic (stock-specific) residual.

```python
from puffin.unsupervised import factor_attribution, specific_risk

# Attribute returns to factors
attribution = factor_attribution(returns, factors, loadings)

# Specific (idiosyncratic) risk
spec_risk = specific_risk(returns, attribution)
print(spec_risk)
# AAPL     0.15
# GOOGL    0.18
# MSFT     0.12
```

The attribution DataFrame contains the factor-explained component of each asset's returns. The difference between actual and attributed returns is the specific return -- the portion unique to each asset.

{: .note }
> Low specific risk means the asset moves primarily with common factors (high systematic risk).
> High specific risk means idiosyncratic events (earnings surprises, management changes) dominate.

## Variance Decomposition

Quantify how much of each asset's total variance comes from common factors versus idiosyncratic sources.

```python
from puffin.unsupervised import factor_variance_decomposition

decomp = factor_variance_decomposition(returns, n_factors=5)
print(decomp)
#        total_variance  factor_variance  specific_variance  pct_factor  pct_specific
# AAPL            0.040            0.032              0.008        80.0          20.0
# GOOGL           0.050            0.038              0.012        76.0          24.0
# MSFT            0.045            0.036              0.009        80.0          20.0
```

**Interpretation**: 80% of AAPL's variance is explained by common factors, 20% is stock-specific. This ratio varies by:
- **Market cap**: Large caps tend to have higher factor variance (more correlated with the market)
- **Sector**: Utility stocks may have lower factor variance than tech
- **Time period**: Factor explanatory power increases during crises (correlations rise)

## Factor Mimicking Portfolio

Create a tradeable portfolio that replicates a specific statistical factor. This converts an abstract PCA factor into an implementable strategy.

```python
from puffin.unsupervised import factor_mimicking_portfolio

# Replicate Factor 1 (usually market factor)
portfolio = factor_mimicking_portfolio(returns, target_factor_idx=0, n_factors=5)
print(portfolio)
# AAPL     0.22
# GOOGL    0.20
# MSFT     0.19
# AMZN     0.21
# TSLA     0.18
```

The weights represent the portfolio that best tracks the target factor's returns. For Factor 1, the near-equal weights confirm it is the market factor. For Factor 2, you would see a long-short split reflecting the style or sector tilt it captures.

{: .warning }
> Mimicking portfolios for higher-order factors can be noisy and may have high turnover.
> Consider using only Factors 1-3 for practical portfolio construction.

## Dynamic Factor Exposure

Factor exposures are not static. Track how an asset's sensitivity to each factor evolves over time using rolling windows.

```python
from puffin.unsupervised import dynamic_factor_exposure
import matplotlib.pyplot as plt

# Rolling 252-day window
exposures = dynamic_factor_exposure(returns, window=252, n_factors=3)

# exposures is a dict: Factor_1 -> DataFrame of rolling betas
factor_1_betas = exposures["Factor_1"]
print(factor_1_betas.head())
#             AAPL  GOOGL  MSFT  AMZN  TSLA
# 2021-01-01  0.82   0.78  0.75  0.80  0.85
# 2021-01-02  0.83   0.79  0.76  0.81  0.86

# Plot AAPL's exposure to Factor 1
plt.plot(factor_1_betas.index, factor_1_betas["AAPL"])
plt.title("AAPL Exposure to Factor 1")
plt.show()
```

**Use cases for dynamic exposures**:
- **Risk monitoring**: Detect when a stock's market beta is drifting
- **Hedging**: Adjust hedge ratios as exposures change
- **Regime detection**: Sharp changes in exposure may signal regime shifts

## Factor Timing Signal

Generate trading signals based on factor momentum. If a factor has performed well recently, tilt toward assets with high exposure to it; if it has performed poorly, tilt away.

```python
from puffin.unsupervised import factor_timing_signal

# Signal based on Factor 1's 21-day momentum
signals = factor_timing_signal(returns, factor_idx=0, n_factors=3, lookback=21)
print(signals.tail())
# 2020-12-27    1   # Long
# 2020-12-28    1   # Long
# 2020-12-29    0   # Neutral
# 2020-12-30   -1   # Short
# 2020-12-31   -1   # Short
```

The signal is +1 (long) when the factor's recent return is above its historical average, -1 (short) when below, and 0 (neutral) otherwise. Combine this with factor exposures to construct a factor-timed portfolio.

{: .important }
> Factor timing is notoriously difficult. Backtest carefully and account for transaction costs.
> Many practitioners use factor timing only to adjust tilts, not as a standalone strategy.

## Practical Example: Portfolio Construction

Combine clustering and PCA for a robust, diversified portfolio.

```python
import numpy as np
import pandas as pd
from puffin.unsupervised import (
    cluster_assets,
    cluster_summary,
    extract_risk_factors,
    factor_exposures,
)

# Load returns
returns = pd.read_csv("sp500_returns.csv", index_col=0, parse_dates=True)

# Step 1: Cluster assets into 5 groups
labels = cluster_assets(returns, n_clusters=5, method='kmeans')
summary = cluster_summary(returns, labels)
print(summary)

# Step 2: Select best asset from each cluster (highest Sharpe)
selected_assets = []
for cluster_id in summary['cluster']:
    cluster_assets_col = returns.columns[labels == cluster_id]
    cluster_returns = returns[cluster_assets_col]

    # Compute Sharpe ratio
    mean_ret = cluster_returns.mean() * 252
    std_ret = cluster_returns.std() * np.sqrt(252)
    sharpe = mean_ret / std_ret

    # Pick best
    best_asset = sharpe.idxmax()
    selected_assets.append(best_asset)

print(f"Selected assets: {selected_assets}")

# Step 3: Extract factors for risk management
factors = extract_risk_factors(returns[selected_assets], n_factors=3)
loadings = factor_exposures(returns[selected_assets], factors)
print("Factor exposures:")
print(loadings)

# Step 4: Construct equal-weighted portfolio
portfolio_weights = pd.Series(1/len(selected_assets), index=selected_assets)
print("\nPortfolio weights:")
print(portfolio_weights)
```

This pipeline selects one representative asset per cluster (maximizing diversification) and then analyzes the resulting portfolio's factor exposures to ensure no unintended concentration.

## Source Code

Browse the implementation: [`puffin/unsupervised/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/unsupervised)
