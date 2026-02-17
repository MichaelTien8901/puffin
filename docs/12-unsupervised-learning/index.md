---
layout: default
title: "Part 12: Unsupervised Learning"
nav_order: 13
---

# Unsupervised Learning for Trading

## Overview

Unsupervised learning discovers hidden patterns in data without labeled targets. In trading, it helps with:

- **Dimensionality reduction**: PCA identifies dominant market factors
- **Asset grouping**: Clustering finds assets with similar behavior
- **Risk factor extraction**: Data-driven alternatives to traditional factor models
- **Regime detection**: Identifying market states automatically

Unlike supervised learning, unsupervised methods don't predict future values directly. Instead, they reveal structure that informs portfolio construction and risk management.

## Principal Component Analysis (PCA)

PCA finds directions of maximum variance in return data. The first principal component captures the most variation, the second captures the next most (orthogonal to the first), and so on.

### Basic Usage

```python
from puffin.unsupervised import MarketPCA
import pandas as pd

# Load returns data
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)

# Fit PCA
pca = MarketPCA(n_components=5)
pca.fit(returns)

# Check explained variance
print(pca.explained_variance_ratio)
# [0.35, 0.18, 0.12, 0.09, 0.06]

# How many components for 95% variance?
print(pca.n_components_95)
# 8
```

### Transform Returns

```python
# Project returns onto principal components
transformed = pca.transform(returns)
# Shape: (n_days, n_components)

# Or fit and transform in one step
transformed = pca.fit_transform(returns)
```

### Variance Explained Plot

```python
import matplotlib.pyplot as plt

plot_data = pca.explained_variance_plot()
print(plot_data)
#    component  variance_explained  cumulative_variance
# 0          1                0.35                 0.35
# 1          2                0.18                 0.53
# 2          3                0.12                 0.65

# Plot cumulative variance
plt.plot(plot_data["component"], plot_data["cumulative_variance"], marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")
plt.legend()
plt.show()
```

## Eigenportfolios

Eigenportfolios are portfolios formed from principal components. The first eigenportfolio represents the dominant market mode (often market-wide movement).

```python
# Extract top 3 eigenportfolios
portfolios = pca.eigenportfolios(returns, n=3)
print(portfolios)
#          AAPL  GOOGL  MSFT  AMZN  TSLA
# PC1      0.22   0.21  0.20  0.19  0.18
# PC2      0.35   0.15  0.25  0.15  0.10
# PC3      0.10   0.30  0.15  0.25  0.20

# Weights sum to 1 (long-only by default)
print(portfolios.sum(axis=1))
# PC1    1.0
# PC2    1.0
# PC3    1.0
```

### Reconstructing Returns

You can approximate returns using only the top N components:

```python
# Reconstruct using top 3 components
reconstructed = pca.reconstruct(returns, n_components=3)

# Compare original vs reconstructed
original_vol = returns.std().mean()
reconstructed_vol = reconstructed.std().mean()
print(f"Original: {original_vol:.4f}, Reconstructed: {reconstructed_vol:.4f}")
```

## K-Means Clustering

Group assets with similar return patterns into clusters.

```python
from puffin.unsupervised import cluster_assets, cluster_summary

# Cluster into 3 groups
labels = cluster_assets(returns, n_clusters=3, method='kmeans')
print(labels)
# [0, 0, 1, 1, 2, 2]  # Cluster ID for each asset

# Get cluster statistics
summary = cluster_summary(returns, labels)
print(summary)
#    cluster  n_assets  mean_return  volatility  sharpe_ratio  assets
# 0        0         2         0.15        0.25          0.60  AAPL, GOOGL
# 1        1         2         0.12        0.20          0.60  MSFT, AMZN
# 2        2         2         0.08        0.30          0.27  TSLA, ...
```

### Finding Optimal Number of Clusters

```python
from puffin.unsupervised import optimal_clusters

# Use silhouette method
optimal_k = optimal_clusters(returns, max_k=10, method='silhouette')
print(f"Optimal clusters: {optimal_k}")

# Or use elbow method
optimal_k_elbow = optimal_clusters(returns, max_k=10, method='elbow')
```

**Silhouette method**: Measures how similar an asset is to its own cluster versus other clusters. Higher is better.

**Elbow method**: Finds the point where adding more clusters doesn't reduce within-cluster variance much.

## Hierarchical Clustering

Builds a tree of clusters, useful for understanding relationships between assets.

```python
from puffin.unsupervised import hierarchical_cluster, plot_dendrogram

# Cluster with Ward linkage
labels = hierarchical_cluster(returns, method='ward', n_clusters=4)

# Visualize dendrogram
fig = plot_dendrogram(returns, method='ward')
plt.show()
```

**Linkage methods**:
- `ward`: Minimizes within-cluster variance (default)
- `complete`: Maximum distance between clusters
- `average`: Average distance between clusters
- `single`: Minimum distance between clusters

## DBSCAN

DBSCAN finds clusters of arbitrary shape and identifies outliers automatically.

```python
from puffin.unsupervised import dbscan_cluster

# eps: Maximum distance between neighbors
# min_samples: Minimum cluster size
labels = dbscan_cluster(returns, eps=0.5, min_samples=3)

# Label -1 = noise/outliers
outliers = returns.columns[labels == -1]
print(f"Outliers: {outliers.tolist()}")
```

**When to use DBSCAN**:
- Irregular cluster shapes
- Want to identify outliers
- Don't know number of clusters in advance

## Gaussian Mixture Models (GMM)

GMM assigns soft cluster membership (probabilities) rather than hard labels.

```python
from puffin.unsupervised import gmm_cluster

labels, probabilities = gmm_cluster(returns, n_components=3, covariance_type='full')

print("Hard labels:", labels)
# [0, 0, 1, 2, 2]

print("Soft probabilities:")
print(probabilities)
#        Cluster 0  Cluster 1  Cluster 2
# AAPL        0.85       0.10       0.05
# GOOGL       0.80       0.15       0.05
# MSFT        0.10       0.85       0.05
```

**Covariance types**:
- `full`: Each cluster has own covariance matrix (most flexible)
- `tied`: All clusters share covariance matrix
- `diag`: Diagonal covariances (features independent)
- `spherical`: Single variance per cluster

## Cluster Correlation

Analyze correlation structure between clusters:

```python
from puffin.unsupervised import cluster_correlation

labels = cluster_assets(returns, n_clusters=3, method='kmeans')
corr_matrix = cluster_correlation(returns, labels)

print(corr_matrix)
#             Cluster 0  Cluster 1  Cluster 2
# Cluster 0       0.75       0.35       0.20
# Cluster 1       0.35       0.80       0.25
# Cluster 2       0.20       0.25       0.70

# High diagonal = within-cluster correlation
# Low off-diagonal = clusters are distinct
```

## Data-Driven Risk Factors

Traditional factor models (Fama-French) use pre-defined factors. PCA extracts factors directly from data.

### Extract Factors

```python
from puffin.unsupervised import extract_risk_factors, factor_exposures

# Extract 5 data-driven factors
factors = extract_risk_factors(returns, n_factors=5)
print(factors.head())
#             Factor_1  Factor_2  Factor_3  Factor_4  Factor_5
# 2020-01-01     0.012     0.005    -0.003     0.001     0.002
# 2020-01-02    -0.008     0.010     0.001    -0.004     0.000

# Compute factor exposures (betas)
loadings = factor_exposures(returns, factors)
print(loadings)
#        Factor_1  Factor_2  Factor_3  Factor_4  Factor_5
# AAPL       0.85      0.23      0.15      0.05      0.01
# GOOGL      0.80      0.30      0.10      0.08      0.02
# MSFT       0.75      0.35      0.05      0.10      0.03
```

### Factor Attribution

Decompose returns into factor and specific components:

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

### Variance Decomposition

```python
from puffin.unsupervised import factor_variance_decomposition

decomp = factor_variance_decomposition(returns, n_factors=5)
print(decomp)
#        total_variance  factor_variance  specific_variance  pct_factor  pct_specific
# AAPL            0.040            0.032              0.008        80.0          20.0
# GOOGL           0.050            0.038              0.012        76.0          24.0
# MSFT            0.045            0.036              0.009        80.0          20.0
```

**Interpretation**: 80% of AAPL's variance is explained by common factors, 20% is stock-specific.

### Factor Mimicking Portfolio

Create a portfolio that replicates a specific factor:

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

### Dynamic Factor Exposure

Track how factor exposures change over time:

```python
from puffin.unsupervised import dynamic_factor_exposure

# Rolling 252-day window
exposures = dynamic_factor_exposure(returns, window=252, n_factors=3)

# exposures is a dict: Factor_1 -> DataFrame of rolling betas
factor_1_betas = exposures["Factor_1"]
print(factor_1_betas.head())
#             AAPL  GOOGL  MSFT  AMZN  TSLA
# 2021-01-01  0.82   0.78  0.75  0.80  0.85
# 2021-01-02  0.83   0.79  0.76  0.81  0.86

# Plot AAPL's exposure to Factor 1
import matplotlib.pyplot as plt
plt.plot(factor_1_betas.index, factor_1_betas["AAPL"])
plt.title("AAPL Exposure to Factor 1")
plt.show()
```

### Factor Timing

Generate trading signals based on factor momentum:

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

## Practical Example: Portfolio Construction

Combine clustering and PCA for a robust portfolio:

```python
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
    cluster_assets = returns.columns[labels == cluster_id]
    cluster_returns = returns[cluster_assets]

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

## Common Pitfalls

1. **Overfitting with too many components**: Use explained variance threshold (e.g., 95%) rather than arbitrary numbers.

2. **Ignoring time structure**: PCA treats all observations equally. For time series, consider rolling windows or exponential weighting.

3. **Interpreting PCA factors**: Principal components are linear combinations of assets. They're mathematically optimal but not always economically meaningful.

4. **Cluster instability**: Small changes in data can flip cluster labels. Use hierarchical clustering or GMM for more stable results.

5. **Correlation vs causation**: Clustering finds correlation, not causation. Assets may cluster due to omitted variables.

## Exercises

1. Load S&P 500 returns and apply PCA. How many components explain 90% of variance?

2. Cluster tech stocks (AAPL, GOOGL, MSFT, etc.) using k-means. Do the clusters make sense?

3. Extract 3 data-driven risk factors from a portfolio. What percentage of variance is factor vs specific?

4. Use DBSCAN to find outlier assets in a sector. Which stocks are flagged?

5. Construct a factor-mimicking portfolio for the first principal component. How does it compare to equal-weighted?

## Summary

- **PCA** reduces dimensionality and identifies dominant market modes
- **Eigenportfolios** provide interpretable portfolios from principal components
- **K-means** groups assets with similar behavior
- **Hierarchical clustering** reveals asset relationships via dendrograms
- **DBSCAN** finds irregular clusters and outliers
- **GMM** provides soft cluster membership probabilities
- **Data-driven factors** extract risk factors directly from returns
- **Factor attribution** decomposes returns into common and specific components

Unsupervised learning complements supervised models by revealing structure in unlabeled data. Use it for portfolio construction, risk management, and regime detection.

## Next Steps

Part 13 explores **NLP for trading**: sentiment analysis, news processing, and text-based signals.
