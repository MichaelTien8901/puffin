---
layout: default
title: "PCA & Eigenportfolios"
parent: "Part 12: Unsupervised Learning"
nav_order: 1
---

# PCA & Eigenportfolios

Principal Component Analysis (PCA) finds directions of maximum variance in return data. The first principal component captures the most variation, the second captures the next most (orthogonal to the first), and so on. Eigenportfolios are the portfolio-weight interpretations of these components.

{: .note }
> PCA is the most widely used dimensionality reduction technique in quantitative finance.
> It underpins statistical risk models at firms like MSCI Barra and Axioma.

## MarketPCA: Basic Usage

The `MarketPCA` class wraps scikit-learn's PCA with finance-specific convenience methods.

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

{: .important }
> The number of components needed for 95% variance depends on the asset universe.
> A diversified set of 500 stocks may need 15-20 components; a single-sector set may need only 3-5.

## Transform Returns

Project returns onto principal components to work in a lower-dimensional space.

```python
# Project returns onto principal components
transformed = pca.transform(returns)
# Shape: (n_days, n_components)

# Or fit and transform in one step
transformed = pca.fit_transform(returns)
```

Each column of `transformed` is a time series of factor returns for one principal component. These are uncorrelated by construction, which simplifies downstream analysis.

## Variance Explained Plot

Visualize how much variance each component captures and how quickly cumulative variance grows.

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

A steep initial curve that flattens quickly indicates that a few dominant factors drive most of the variation -- typical for equity markets where the "market factor" dominates.

## Eigenportfolios

Eigenportfolios are portfolios formed from principal components. The first eigenportfolio represents the dominant market mode (often market-wide movement). Higher-order eigenportfolios capture sector rotations, value/growth tilts, or other structural patterns.

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

{: .note }
> **Interpreting eigenportfolios**: PC1 typically has roughly equal weights across all assets (the "market portfolio").
> PC2 often separates assets into two groups (e.g., growth vs value), making it a long-short portfolio.

## Reconstructing Returns

You can approximate returns using only the top N components. This separates the systematic signal from idiosyncratic noise.

```python
# Reconstruct using top 3 components
reconstructed = pca.reconstruct(returns, n_components=3)

# Compare original vs reconstructed
original_vol = returns.std().mean()
reconstructed_vol = reconstructed.std().mean()
print(f"Original: {original_vol:.4f}, Reconstructed: {reconstructed_vol:.4f}")
```

The reconstruction error (difference between original and reconstructed returns) represents the idiosyncratic component -- the portion of returns not explained by common factors. This is useful for:

- **Risk decomposition**: Separating systematic vs specific risk
- **Denoising**: Removing noise from covariance matrices
- **Anomaly detection**: Large reconstruction errors signal unusual stock-specific events

## Practical Tips

| Consideration | Recommendation |
|---|---|
| Number of components | Use the 95% cumulative variance threshold as a starting point |
| Standardization | Always standardize returns before PCA (mean-zero, unit-variance) |
| Rolling PCA | Re-estimate periodically (e.g., quarterly) to capture structural changes |
| Outliers | Robust PCA variants exist for data with outliers; consider winsorizing first |

## Source Code

Browse the implementation: [`puffin/unsupervised/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/unsupervised)
