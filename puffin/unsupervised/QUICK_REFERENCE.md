# Unsupervised Learning - Quick Reference

## Import Statement

```python
from puffin.unsupervised import (
    # PCA
    MarketPCA,
    # Clustering
    cluster_assets,
    optimal_clusters,
    cluster_summary,
    hierarchical_cluster,
    plot_dendrogram,
    dbscan_cluster,
    gmm_cluster,
    cluster_correlation,
    # Risk Factors
    extract_risk_factors,
    factor_exposures,
    factor_attribution,
    specific_risk,
    factor_variance_decomposition,
    factor_mimicking_portfolio,
    dynamic_factor_exposure,
    factor_timing_signal,
)
```

## Common Workflows

### Workflow 1: PCA Analysis

```python
# Initialize and fit
pca = MarketPCA(n_components=5)
pca.fit(returns)

# Check variance explained
print(pca.explained_variance_ratio)
print(f"Components for 95%: {pca.n_components_95}")

# Get eigenportfolios
portfolios = pca.eigenportfolios(returns, n=3)
```

### Workflow 2: Asset Clustering

```python
# Find optimal clusters
optimal_k = optimal_clusters(returns, max_k=10)

# Cluster assets
labels = cluster_assets(returns, n_clusters=optimal_k, method='kmeans')

# Get statistics
summary = cluster_summary(returns, labels)
print(summary)
```

### Workflow 3: Risk Factor Analysis

```python
# Extract factors
factors = extract_risk_factors(returns, n_factors=5)

# Compute exposures
loadings = factor_exposures(returns, factors)

# Variance decomposition
decomp = factor_variance_decomposition(returns, n_factors=5)
print(decomp)
```

### Workflow 4: Portfolio Construction

```python
# Step 1: Cluster for diversification
labels = cluster_assets(returns, n_clusters=5, method='kmeans')

# Step 2: Select best from each cluster
summary = cluster_summary(returns, labels)
# (Select assets with highest Sharpe)

# Step 3: Weight by factor exposure
factors = extract_risk_factors(selected_returns, n_factors=3)
loadings = factor_exposures(selected_returns, factors)
# (Adjust weights based on factor exposures)
```

## Method Comparison

### Clustering Methods

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| K-means | General clustering | Fast, simple | Need K, spherical clusters |
| Hierarchical | Understanding relationships | Dendrogram, no K needed | Slower, sensitive to noise |
| DBSCAN | Irregular clusters | Finds outliers, no K | Need eps/min_samples tuning |
| GMM | Soft assignments | Probabilistic | Need K, slower |

### When to Use Each

- **K-means**: Default choice for asset grouping
- **Hierarchical**: Exploring asset relationships
- **DBSCAN**: Finding outlier assets
- **GMM**: Need probability of cluster membership

## Parameter Guidelines

### PCA
- `n_components=None` → Keep all components
- `n_components=5` → Keep 5 components
- Use `n_components_95` for 95% variance threshold

### K-means
- `n_clusters=3-10` → Typical range for assets
- Use `optimal_clusters()` to find best K

### Hierarchical
- `method='ward'` → Minimize variance (default)
- `method='complete'` → Maximum distance
- `method='average'` → Average distance

### DBSCAN
- `eps=0.5-2.0` → Distance threshold
- `min_samples=3-5` → Minimum cluster size
- Tune on your data

### GMM
- `n_components=3-10` → Number of clusters
- `covariance_type='full'` → Most flexible (default)
- `covariance_type='diag'` → Faster, less flexible

### Risk Factors
- `n_factors=3-10` → Number of factors
- `n_factors=5` → Common choice
- Use variance decomposition to assess

## Common Patterns

### Pattern 1: Variance-Based Component Selection

```python
pca = MarketPCA()
pca.fit(returns)

# Use components that explain 95% variance
n_comp = pca.n_components_95
factors = extract_risk_factors(returns, n_factors=n_comp)
```

### Pattern 2: Multi-Method Consensus

```python
# Compare clustering methods
methods = ['kmeans', 'hierarchical', 'gmm']
labels_dict = {}

for method in methods:
    labels_dict[method] = cluster_assets(returns, n_clusters=3, method=method)

# Find assets consistently grouped together
```

### Pattern 3: Dynamic Factor Monitoring

```python
# Track factor exposures over time
exposures = dynamic_factor_exposure(returns, window=252, n_factors=3)

# Monitor specific asset
asset_factor_1 = exposures["Factor_1"]["AAPL"]

# Generate signals
signals = factor_timing_signal(returns, factor_idx=0, lookback=21)
```

### Pattern 4: Factor-Based Portfolio

```python
# Extract factors
factors = extract_risk_factors(returns, n_factors=5)

# Create mimicking portfolio for Factor 1
portfolio = factor_mimicking_portfolio(returns, target_factor_idx=0)

# Check factor exposure
loadings = factor_exposures(pd.DataFrame(portfolio).T, factors)
```

## Interpretation Tips

### PCA
- **First PC**: Usually market-wide movement (beta)
- **Second PC**: Often size or sector effect
- **High explained variance**: Data is low-dimensional
- **Low explained variance**: Complex, high-dimensional structure

### Clustering
- **High within-cluster correlation**: Good clustering
- **Similar cluster sizes**: Balanced grouping
- **Many outliers (DBSCAN)**: Consider different eps
- **High silhouette score**: Well-separated clusters

### Risk Factors
- **High factor variance %**: Driven by common factors
- **High specific variance %**: Idiosyncratic risk
- **High factor loading**: Sensitive to that factor
- **Low factor loading**: Independent of that factor

## Performance Tips

1. **PCA**: Pre-standardize if assets have very different scales
2. **Clustering**: Use sample of data for large universes
3. **Risk Factors**: Cache factor extractions for repeated use
4. **Dynamic Analysis**: Use multiprocessing for rolling windows

## Troubleshooting

### Issue: PCA explains low variance
- **Solution**: Returns may be too noisy or high-dimensional
- **Action**: Increase observation period or reduce assets

### Issue: Clustering gives one large cluster
- **Solution**: Increase K or use hierarchical to see structure
- **Action**: Check correlation matrix for natural groupings

### Issue: DBSCAN all noise (-1)
- **Solution**: eps too small or min_samples too large
- **Action**: Plot distance distribution, tune parameters

### Issue: Factor exposures unstable
- **Solution**: Short observation window or noisy data
- **Action**: Increase window size or use regularization

## Example: Complete Analysis

```python
import pandas as pd
from puffin.unsupervised import *

# Load data
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)

# 1. Dimensionality
pca = MarketPCA()
pca.fit(returns)
print(f"Dimensions: {pca.n_components_95} components for 95% variance")

# 2. Grouping
optimal_k = optimal_clusters(returns, max_k=10)
labels = cluster_assets(returns, n_clusters=optimal_k)
summary = cluster_summary(returns, labels)
print(f"\nClusters: {optimal_k} groups")
print(summary)

# 3. Risk factors
factors = extract_risk_factors(returns, n_factors=5)
decomp = factor_variance_decomposition(returns, n_factors=5)
print(f"\nAvg factor variance: {decomp['pct_factor'].mean():.1f}%")

# 4. Outliers
dbscan_labels = dbscan_cluster(returns, eps=1.0, min_samples=3)
outliers = returns.columns[dbscan_labels == -1]
print(f"\nOutliers: {list(outliers)}")
```

## Resources

- **Full Documentation**: `puffin/unsupervised/README.md`
- **Tutorial**: `docs/12-unsupervised-learning/01-unsupervised-learning.md`
- **Examples**: `examples/unsupervised_example.py`
- **Tests**: `tests/unsupervised/`

## Next Steps

After unsupervised analysis, consider:
1. **Portfolio Optimization**: Use clusters for constraint groups
2. **Risk Management**: Monitor factor exposures in real-time
3. **Strategy Development**: Factor timing signals
4. **Feature Engineering**: PCA components as ML features
