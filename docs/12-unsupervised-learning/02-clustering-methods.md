---
layout: default
title: "Clustering Methods"
parent: "Part 12: Unsupervised Learning"
nav_order: 2
---

# Clustering Methods

Clustering groups assets with similar return patterns, enabling diversified portfolio construction, sector analysis, and outlier detection. This page covers four complementary approaches available in the `puffin.unsupervised` module.

## K-Means Clustering

K-means partitions assets into K groups by minimizing within-cluster variance. It is fast, simple, and a good starting point for asset grouping.

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

{: .note }
> K-means uses Euclidean distance on return vectors. For correlation-based grouping, consider
> transforming the correlation matrix into a distance matrix first: `d = sqrt(2 * (1 - corr))`.

### Finding Optimal Number of Clusters

Choosing K is critical. Too few clusters miss structure; too many overfit noise.

```python
from puffin.unsupervised import optimal_clusters

# Use silhouette method
optimal_k = optimal_clusters(returns, max_k=10, method='silhouette')
print(f"Optimal clusters: {optimal_k}")

# Or use elbow method
optimal_k_elbow = optimal_clusters(returns, max_k=10, method='elbow')
```

**Silhouette method**: Measures how similar an asset is to its own cluster versus other clusters. Higher is better. A silhouette score above 0.5 indicates well-separated clusters.

**Elbow method**: Finds the point where adding more clusters doesn't reduce within-cluster variance much. Look for the "elbow" in the inertia curve.

{: .important }
> Always validate clusters qualitatively. If tech stocks end up in the same cluster as utilities,
> the features or distance metric may need adjustment.

## Hierarchical Clustering

Hierarchical clustering builds a tree (dendrogram) of nested clusters. Unlike K-means, it does not require specifying K upfront -- you can cut the tree at any level.

```python
from puffin.unsupervised import hierarchical_cluster, plot_dendrogram
import matplotlib.pyplot as plt

# Cluster with Ward linkage
labels = hierarchical_cluster(returns, method='ward', n_clusters=4)

# Visualize dendrogram
fig = plot_dendrogram(returns, method='ward')
plt.show()
```

### Linkage Methods

The linkage method determines how distances between clusters are computed when merging.

| Method | Strategy | Best for |
|---|---|---|
| `ward` | Minimizes within-cluster variance | General purpose (default) |
| `complete` | Maximum distance between clusters | Compact, equal-sized clusters |
| `average` | Average distance between clusters | Balanced approach |
| `single` | Minimum distance between clusters | Detecting elongated clusters |

{: .note }
> Hierarchical clustering is used internally by the Hierarchical Risk Parity (HRP)
> portfolio optimizer covered in [Part 5](../05-portfolio-optimization/). The dendrogram
> determines the quasi-diagonal ordering of the covariance matrix.

## DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters of arbitrary shape and automatically identifies outliers. It does not require specifying the number of clusters.

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
- Irregular cluster shapes (not all clusters are spherical)
- Want to identify outliers automatically
- Don't know number of clusters in advance

{: .warning }
> DBSCAN is sensitive to `eps` and `min_samples`. Use the k-distance graph to choose `eps`:
> compute the distance to each point's k-th nearest neighbor, sort, and look for an elbow.

### Tuning DBSCAN Parameters

| Parameter | Effect of increasing | Guidance |
|---|---|---|
| `eps` | Larger clusters, fewer outliers | Start with the elbow of the k-distance plot |
| `min_samples` | Denser clusters required, more outliers | Typically set to `2 * n_features` |

## Gaussian Mixture Models (GMM)

GMM assigns soft cluster membership (probabilities) rather than hard labels. Each cluster is modeled as a multivariate Gaussian distribution. This is useful when assets may belong to multiple groups.

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

### Covariance Types

The covariance type controls the flexibility of each cluster's shape.

| Type | Description | Parameters | Best for |
|---|---|---|---|
| `full` | Each cluster has own covariance matrix | Most flexible | Small to medium datasets |
| `tied` | All clusters share one covariance matrix | Moderate | When clusters have similar shape |
| `diag` | Diagonal covariances (features independent) | Fewer parameters | High-dimensional data |
| `spherical` | Single variance per cluster | Fewest parameters | Isotropic clusters |

{: .note }
> Use BIC (Bayesian Information Criterion) to select both the number of components and the
> covariance type. Lower BIC indicates a better model.

## Cluster Correlation

After assigning cluster labels, analyze the correlation structure between clusters to verify they capture distinct behaviors.

```python
from puffin.unsupervised import cluster_assets, cluster_correlation

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

**Interpreting the matrix**:
- **High diagonal values** (e.g., 0.75-0.80): Assets within each cluster move together -- clusters are internally cohesive.
- **Low off-diagonal values** (e.g., 0.20-0.35): Clusters move independently -- good for diversification.
- If off-diagonal values are high, consider increasing K or switching methods.

## Method Comparison

| Method | Requires K? | Soft labels? | Finds outliers? | Scalability |
|---|---|---|---|---|
| K-Means | Yes | No | No | Excellent |
| Hierarchical | Optional | No | No | Moderate |
| DBSCAN | No | No | Yes | Good |
| GMM | Yes | Yes | No | Moderate |

## Source Code

Browse the implementation: [`puffin/unsupervised/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/unsupervised)
