"""Tests for clustering algorithms."""

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from puffin.unsupervised.clustering import (
    cluster_assets,
    optimal_clusters,
    cluster_summary,
    hierarchical_cluster,
    plot_dendrogram,
    dbscan_cluster,
    gmm_cluster,
    cluster_correlation,
)


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")

    # Create 3 groups of correlated assets
    group1 = np.random.multivariate_normal([0.001, 0.0008], [[0.04, 0.03], [0.03, 0.04]], 252)
    group2 = np.random.multivariate_normal([0.0005, 0.0007], [[0.02, 0.015], [0.015, 0.02]], 252)
    group3 = np.random.multivariate_normal([0.0012, 0.001], [[0.05, 0.04], [0.04, 0.05]], 252)

    returns = np.hstack([group1, group2, group3])
    assets = ["A1", "A2", "B1", "B2", "C1", "C2"]

    return pd.DataFrame(returns, index=dates, columns=assets)


def test_cluster_assets_kmeans(sample_returns):
    """Test k-means clustering."""
    labels = cluster_assets(sample_returns, n_clusters=3, method="kmeans")

    assert len(labels) == 6
    assert len(np.unique(labels)) <= 3
    assert all(0 <= label < 3 for label in labels)


def test_cluster_assets_hierarchical(sample_returns):
    """Test hierarchical clustering."""
    labels = cluster_assets(sample_returns, n_clusters=3, method="hierarchical")

    assert len(labels) == 6
    assert len(np.unique(labels)) <= 3


def test_cluster_assets_dbscan(sample_returns):
    """Test DBSCAN clustering."""
    labels = cluster_assets(sample_returns, n_clusters=3, method="dbscan", eps=1.0, min_samples=2)

    assert len(labels) == 6
    # DBSCAN can have noise points (-1)
    assert all(label >= -1 for label in labels)


def test_cluster_assets_gmm(sample_returns):
    """Test GMM clustering."""
    labels = cluster_assets(sample_returns, n_clusters=3, method="gmm")

    assert len(labels) == 6
    assert len(np.unique(labels)) <= 3


def test_cluster_assets_invalid_method(sample_returns):
    """Test error for invalid clustering method."""
    with pytest.raises(ValueError, match="Unknown method"):
        cluster_assets(sample_returns, n_clusters=3, method="invalid")


def test_optimal_clusters_silhouette(sample_returns):
    """Test optimal cluster detection with silhouette."""
    optimal_k = optimal_clusters(sample_returns, max_k=6, method="silhouette")

    assert 2 <= optimal_k <= 6
    assert isinstance(optimal_k, int)


def test_optimal_clusters_elbow(sample_returns):
    """Test optimal cluster detection with elbow method."""
    optimal_k = optimal_clusters(sample_returns, max_k=6, method="elbow")

    assert 2 <= optimal_k <= 6
    assert isinstance(optimal_k, int)


def test_optimal_clusters_invalid_method(sample_returns):
    """Test error for invalid optimal clusters method."""
    with pytest.raises(ValueError, match="Unknown method"):
        optimal_clusters(sample_returns, method="invalid")


def test_cluster_summary(sample_returns):
    """Test cluster summary statistics."""
    labels = cluster_assets(sample_returns, n_clusters=3, method="kmeans")
    summary = cluster_summary(sample_returns, labels)

    assert isinstance(summary, pd.DataFrame)
    assert "cluster" in summary.columns
    assert "n_assets" in summary.columns
    assert "mean_return" in summary.columns
    assert "volatility" in summary.columns
    assert "sharpe_ratio" in summary.columns
    assert "assets" in summary.columns

    assert len(summary) <= 3
    assert summary["n_assets"].sum() == 6


def test_hierarchical_cluster(sample_returns):
    """Test hierarchical clustering function."""
    labels = hierarchical_cluster(sample_returns, method="ward", n_clusters=3)

    assert len(labels) == 6
    assert len(np.unique(labels)) == 3


def test_hierarchical_cluster_methods(sample_returns):
    """Test different linkage methods."""
    for method in ["ward", "complete", "average", "single"]:
        labels = hierarchical_cluster(sample_returns, method=method, n_clusters=2)
        assert len(labels) == 6
        assert len(np.unique(labels)) == 2


def test_hierarchical_cluster_auto_n(sample_returns):
    """Test hierarchical clustering with automatic n_clusters."""
    labels = hierarchical_cluster(sample_returns, method="ward", n_clusters=None)

    assert len(labels) == 6
    # Should default to sqrt(n_assets) = sqrt(6) ≈ 2
    assert 1 <= len(np.unique(labels)) <= 6


def test_plot_dendrogram(sample_returns):
    """Test dendrogram plotting."""
    fig = plot_dendrogram(sample_returns, method="ward")

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1

    plt.close(fig)


def test_plot_dendrogram_methods(sample_returns):
    """Test dendrogram with different linkage methods."""
    for method in ["ward", "complete", "average"]:
        fig = plot_dendrogram(sample_returns, method=method)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_dbscan_cluster(sample_returns):
    """Test DBSCAN clustering function."""
    labels = dbscan_cluster(sample_returns, eps=1.0, min_samples=2)

    assert len(labels) == 6
    # -1 indicates noise
    assert all(label >= -1 for label in labels)


def test_dbscan_cluster_parameters(sample_returns):
    """Test DBSCAN with different parameters."""
    # Tight clustering (more noise)
    labels_tight = dbscan_cluster(sample_returns, eps=0.3, min_samples=3)

    # Loose clustering (fewer clusters)
    labels_loose = dbscan_cluster(sample_returns, eps=2.0, min_samples=2)

    assert len(labels_tight) == 6
    assert len(labels_loose) == 6


def test_gmm_cluster(sample_returns):
    """Test GMM clustering function."""
    labels, probabilities = gmm_cluster(sample_returns, n_components=3)

    assert len(labels) == 6
    assert probabilities.shape == (6, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1


def test_gmm_cluster_covariance_types(sample_returns):
    """Test GMM with different covariance types."""
    for cov_type in ["full", "tied", "diag", "spherical"]:
        labels, probs = gmm_cluster(sample_returns, n_components=2, covariance_type=cov_type)
        assert len(labels) == 6
        assert probs.shape == (6, 2)


def test_cluster_correlation(sample_returns):
    """Test cluster correlation matrix."""
    labels = cluster_assets(sample_returns, n_clusters=3, method="kmeans")
    corr_matrix = cluster_correlation(sample_returns, labels)

    assert isinstance(corr_matrix, pd.DataFrame)
    n_clusters = len(np.unique(labels))
    assert corr_matrix.shape == (n_clusters, n_clusters)

    # Diagonal should have high correlation (within cluster)
    assert all(corr_matrix.iloc[i, i] > 0 for i in range(n_clusters))


def test_cluster_correlation_values(sample_returns):
    """Test cluster correlation values are reasonable."""
    labels = cluster_assets(sample_returns, n_clusters=2, method="kmeans")
    corr_matrix = cluster_correlation(sample_returns, labels)

    # All correlations should be between -1 and 1
    assert (corr_matrix >= -1).all().all()
    assert (corr_matrix <= 1).all().all()


def test_clustering_with_nans():
    """Test clustering handles NaN values."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.DataFrame(
        np.random.randn(100, 4),
        index=dates,
        columns=["A", "B", "C", "D"]
    )
    # Add some NaNs
    returns.iloc[0, 0] = np.nan
    returns.iloc[5, 1] = np.nan

    labels = cluster_assets(returns, n_clusters=2, method="kmeans")
    assert len(labels) == 4


def test_cluster_reproducibility(sample_returns):
    """Test clustering reproducibility with random seed."""
    labels1 = cluster_assets(sample_returns, n_clusters=3, method="kmeans")
    labels2 = cluster_assets(sample_returns, n_clusters=3, method="kmeans")

    # Should get same results with same random seed in KMeans
    assert np.array_equal(labels1, labels2)
