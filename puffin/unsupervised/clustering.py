"""Clustering algorithms for asset grouping and regime detection."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from typing import Literal, Optional, Union


def cluster_assets(
    returns: pd.DataFrame,
    n_clusters: int,
    method: Literal["kmeans", "hierarchical", "dbscan", "gmm"] = "kmeans",
    **kwargs
) -> np.ndarray:
    """Cluster assets based on return patterns.

    Args:
        returns: DataFrame of asset returns (rows=dates, cols=assets).
        n_clusters: Number of clusters (ignored for DBSCAN).
        method: Clustering algorithm to use.
        **kwargs: Additional arguments passed to clustering algorithm.

    Returns:
        Array of cluster labels for each asset.
    """
    returns_clean = returns.dropna().T  # Transpose so rows are assets

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = model.fit_predict(returns_clean)

    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        labels = model.fit_predict(returns_clean)

    elif method == "dbscan":
        eps = kwargs.pop("eps", 0.5)
        min_samples = kwargs.pop("min_samples", 5)
        labels = dbscan_cluster(returns, eps=eps, min_samples=min_samples)

    elif method == "gmm":
        covariance_type = kwargs.pop("covariance_type", "full")
        labels, _ = gmm_cluster(returns, n_clusters, covariance_type=covariance_type)

    else:
        raise ValueError(f"Unknown method: {method}")

    return labels


def optimal_clusters(
    returns: pd.DataFrame,
    max_k: int = 15,
    method: Literal["silhouette", "elbow"] = "silhouette"
) -> int:
    """Find optimal number of clusters using silhouette or elbow method.

    Args:
        returns: DataFrame of asset returns.
        max_k: Maximum number of clusters to test.
        method: Method to use ('silhouette' or 'elbow').

    Returns:
        Optimal number of clusters.
    """
    returns_clean = returns.dropna().T

    if method == "silhouette":
        scores = []
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(returns_clean)
            score = silhouette_score(returns_clean, labels)
            scores.append(score)
        return int(np.argmax(scores) + 2)  # +2 because we started at k=2

    elif method == "elbow":
        inertias = []
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(returns_clean)
            inertias.append(model.inertia_)

        # Find elbow using second derivative
        inertias = np.array(inertias)
        diff1 = np.diff(inertias)
        diff2 = np.diff(diff1)
        elbow_idx = np.argmin(diff2) + 2  # +2 for offset
        return int(elbow_idx)

    else:
        raise ValueError(f"Unknown method: {method}")


def cluster_summary(returns: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Compute summary statistics for each cluster.

    Args:
        returns: DataFrame of asset returns.
        labels: Array of cluster labels.

    Returns:
        DataFrame with mean return, volatility, and Sharpe ratio per cluster.
    """
    returns_clean = returns.dropna()

    summaries = []
    for label in np.unique(labels):
        cluster_assets = returns.columns[labels == label]
        cluster_returns = returns_clean[cluster_assets]

        mean_return = cluster_returns.mean().mean() * 252  # Annualized
        volatility = cluster_returns.std().mean() * np.sqrt(252)  # Annualized
        sharpe = mean_return / volatility if volatility > 0 else 0

        summaries.append({
            "cluster": label,
            "n_assets": len(cluster_assets),
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "assets": ", ".join(cluster_assets[:5])  # First 5 assets
        })

    return pd.DataFrame(summaries)


def hierarchical_cluster(
    returns: pd.DataFrame,
    method: Literal["ward", "complete", "average", "single"] = "ward",
    n_clusters: Optional[int] = None
) -> np.ndarray:
    """Perform hierarchical clustering on assets.

    Args:
        returns: DataFrame of asset returns.
        method: Linkage method ('ward', 'complete', 'average', 'single').
        n_clusters: Number of clusters. None = use dendrogram to decide.

    Returns:
        Array of cluster labels.
    """
    returns_clean = returns.dropna().T

    if n_clusters is None:
        # Default to sqrt(n_assets) if not specified
        n_clusters = int(np.sqrt(len(returns.columns)))

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(returns_clean)

    return labels


def plot_dendrogram(
    returns: pd.DataFrame,
    method: Literal["ward", "complete", "average", "single"] = "ward",
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """Create a dendrogram for hierarchical clustering.

    Args:
        returns: DataFrame of asset returns.
        method: Linkage method.
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure object.
    """
    returns_clean = returns.dropna().T

    # Compute correlation distance
    corr = returns_clean.T.corr()
    distance = 1 - corr
    distance = np.clip(distance, 0, 2)  # Ensure valid distance metric

    # Convert to condensed distance matrix
    condensed = squareform(distance, checks=False)

    # Compute linkage
    linkage_matrix = hierarchy.linkage(condensed, method=method)

    # Create dendrogram
    fig, ax = plt.subplots(figsize=figsize)
    hierarchy.dendrogram(
        linkage_matrix,
        labels=returns.columns.tolist(),
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8
    )
    ax.set_title(f"Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Distance")

    plt.tight_layout()
    return fig


def dbscan_cluster(
    returns: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5
) -> np.ndarray:
    """Perform DBSCAN clustering on assets.

    Args:
        returns: DataFrame of asset returns.
        eps: Maximum distance between samples in a neighborhood.
        min_samples: Minimum samples in a neighborhood to form a cluster.

    Returns:
        Array of cluster labels (-1 = noise/outliers).
    """
    returns_clean = returns.dropna().T

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(returns_clean)

    return labels


def gmm_cluster(
    returns: pd.DataFrame,
    n_components: int,
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
) -> tuple[np.ndarray, np.ndarray]:
    """Perform Gaussian Mixture Model clustering.

    Args:
        returns: DataFrame of asset returns.
        n_components: Number of mixture components (clusters).
        covariance_type: Type of covariance parameters.

    Returns:
        Tuple of (labels, probabilities). Probabilities is (n_assets, n_components).
    """
    returns_clean = returns.dropna().T

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42
    )
    model.fit(returns_clean)

    labels = model.predict(returns_clean)
    probabilities = model.predict_proba(returns_clean)

    return labels, probabilities


def cluster_correlation(
    returns: pd.DataFrame,
    labels: np.ndarray
) -> pd.DataFrame:
    """Compute average correlation within and between clusters.

    Args:
        returns: DataFrame of asset returns.
        labels: Array of cluster labels.

    Returns:
        DataFrame of correlation matrix (cluster x cluster).
    """
    returns_clean = returns.dropna()
    unique_labels = np.unique(labels)

    corr_matrix = np.zeros((len(unique_labels), len(unique_labels)))

    for i, label_i in enumerate(unique_labels):
        assets_i = returns.columns[labels == label_i]
        for j, label_j in enumerate(unique_labels):
            assets_j = returns.columns[labels == label_j]

            # Compute average correlation between clusters
            cross_corr = returns_clean[assets_i].corrwith(
                returns_clean[assets_j], axis=0
            )
            corr_matrix[i, j] = cross_corr.mean()

    return pd.DataFrame(
        corr_matrix,
        index=[f"Cluster {l}" for l in unique_labels],
        columns=[f"Cluster {l}" for l in unique_labels]
    )
