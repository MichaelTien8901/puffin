# Unsupervised Learning Module

This module provides unsupervised learning algorithms for algorithmic trading, including PCA, clustering, and data-driven risk factor extraction.

## Features

### 1. Principal Component Analysis (PCA)

**Class: `MarketPCA`**

Dimensionality reduction for market returns data.

```python
from puffin.unsupervised import MarketPCA

pca = MarketPCA(n_components=5)
pca.fit(returns)

# Check variance explained
print(pca.explained_variance_ratio)
print(f"Components for 95% variance: {pca.n_components_95}")

# Transform returns
transformed = pca.transform(returns)

# Extract eigenportfolios
portfolios = pca.eigenportfolios(returns, n=3)
```

**Key Methods:**
- `fit(returns)` - Fit PCA on returns data
- `transform(returns)` - Transform returns to PC space
- `eigenportfolios(returns, n)` - Extract top N eigenportfolios
- `reconstruct(returns, n_components)` - Reconstruct returns with limited components

**Key Properties:**
- `explained_variance_ratio` - Variance explained by each component
- `components` - Principal components (eigenvectors)
- `n_components_95` - Components needed for 95% variance

### 2. Clustering Algorithms

**Functions:**

```python
from puffin.unsupervised import (
    cluster_assets,
    optimal_clusters,
    cluster_summary,
    hierarchical_cluster,
    plot_dendrogram,
    dbscan_cluster,
    gmm_cluster,
    cluster_correlation,
)

# K-means clustering
labels = cluster_assets(returns, n_clusters=3, method='kmeans')

# Find optimal number of clusters
optimal_k = optimal_clusters(returns, max_k=10, method='silhouette')

# Cluster statistics
summary = cluster_summary(returns, labels)

# Hierarchical clustering
labels = hierarchical_cluster(returns, method='ward', n_clusters=4)
fig = plot_dendrogram(returns, method='ward')

# DBSCAN (density-based)
labels = dbscan_cluster(returns, eps=0.5, min_samples=5)

# Gaussian Mixture Model
labels, probabilities = gmm_cluster(returns, n_components=3)

# Inter-cluster correlation
corr = cluster_correlation(returns, labels)
```

**Supported Methods:**
- **K-means**: Partitions assets into K clusters
- **Hierarchical**: Builds cluster hierarchy (ward, complete, average, single linkage)
- **DBSCAN**: Density-based clustering with outlier detection
- **GMM**: Probabilistic clustering with soft assignments

### 3. Data-Driven Risk Factors

**Functions:**

```python
from puffin.unsupervised import (
    extract_risk_factors,
    factor_exposures,
    factor_attribution,
    specific_risk,
    factor_variance_decomposition,
    factor_mimicking_portfolio,
    dynamic_factor_exposure,
    factor_timing_signal,
)

# Extract factors using PCA
factors = extract_risk_factors(returns, n_factors=5)

# Compute factor loadings (betas)
loadings = factor_exposures(returns, factors)

# Attribute returns to factors
attribution = factor_attribution(returns, factors, loadings)

# Idiosyncratic risk
spec_risk = specific_risk(returns, attribution)

# Variance decomposition
decomp = factor_variance_decomposition(returns, n_factors=5)

# Factor-replicating portfolio
portfolio = factor_mimicking_portfolio(returns, target_factor_idx=0)

# Rolling factor exposures
exposures = dynamic_factor_exposure(returns, window=252, n_factors=3)

# Factor timing signals
signals = factor_timing_signal(returns, factor_idx=0, lookback=21)
```

## Installation

The module requires:
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib

## Quick Start

```python
import pandas as pd
from puffin.unsupervised import (
    MarketPCA,
    cluster_assets,
    cluster_summary,
    extract_risk_factors,
    factor_exposures,
)

# Load returns
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)

# 1. PCA Analysis
pca = MarketPCA()
pca.fit(returns)
print(f"Top 3 PCs explain {pca.explained_variance_ratio[:3].sum():.2%} of variance")

eigenportfolios = pca.eigenportfolios(returns, n=3)
print(eigenportfolios)

# 2. Cluster Assets
labels = cluster_assets(returns, n_clusters=5, method='kmeans')
summary = cluster_summary(returns, labels)
print(summary)

# 3. Extract Risk Factors
factors = extract_risk_factors(returns, n_factors=3)
loadings = factor_exposures(returns, factors)
print("Factor exposures:")
print(loadings)
```

## Examples

See `examples/unsupervised_example.py` for comprehensive usage examples.

## Testing

Run tests with:
```bash
pytest tests/unsupervised/
```

## Use Cases

### Portfolio Construction
- Use PCA to identify dominant market modes
- Cluster assets for diversification
- Extract data-driven risk factors for factor-based investing

### Risk Management
- Decompose variance into factor and specific components
- Monitor dynamic factor exposures
- Identify outlier assets with DBSCAN

### Regime Detection
- Cluster market states using GMM
- Track factor timing signals
- Analyze inter-cluster correlations

## References

- **PCA**: Jolliffe, I. T. (2002). Principal Component Analysis
- **K-means**: MacQueen, J. (1967). Some methods for classification and analysis
- **Hierarchical Clustering**: Ward, J. H. (1963). Hierarchical grouping to optimize an objective function
- **DBSCAN**: Ester, M. et al. (1996). A density-based algorithm for discovering clusters
- **GMM**: Reynolds, D. A. (2009). Gaussian Mixture Models

## License

Part of the Puffin algorithmic trading framework.
