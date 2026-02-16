"""Example usage of unsupervised learning for trading.

This script demonstrates:
1. PCA and eigenportfolio analysis
2. Asset clustering
3. Data-driven risk factor extraction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)


def generate_sample_data(n_days=252, n_assets=10):
    """Generate sample return data with realistic correlation structure."""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    # Create 3 groups of correlated assets
    # Group 1: Tech stocks (high correlation)
    group1 = np.random.multivariate_normal(
        [0.001, 0.0008, 0.0009],
        [[0.04, 0.03, 0.032], [0.03, 0.04, 0.035], [0.032, 0.035, 0.041]],
        n_days
    )

    # Group 2: Financial stocks (medium correlation)
    group2 = np.random.multivariate_normal(
        [0.0005, 0.0006, 0.0007],
        [[0.03, 0.02, 0.021], [0.02, 0.035, 0.023], [0.021, 0.023, 0.036]],
        n_days
    )

    # Group 3: Energy stocks (low correlation with others)
    group3 = np.random.multivariate_normal(
        [0.0003, 0.0004, 0.0005, 0.0006],
        [
            [0.05, 0.03, 0.032, 0.028],
            [0.03, 0.052, 0.035, 0.030],
            [0.032, 0.035, 0.048, 0.033],
            [0.028, 0.030, 0.033, 0.050],
        ],
        n_days
    )

    # Combine all groups
    returns = np.hstack([group1, group2, group3])

    assets = [
        "TECH1", "TECH2", "TECH3",
        "FIN1", "FIN2", "FIN3",
        "ENRG1", "ENRG2", "ENRG3", "ENRG4"
    ]

    return pd.DataFrame(returns, index=dates, columns=assets)


def example_pca():
    """Example: PCA and eigenportfolios."""
    print("=" * 60)
    print("EXAMPLE 1: PCA and Eigenportfolios")
    print("=" * 60)

    from puffin.unsupervised import MarketPCA

    # Generate data
    returns = generate_sample_data()

    # Fit PCA
    pca = MarketPCA()
    pca.fit(returns)

    print("\nExplained Variance Ratio:")
    for i, var in enumerate(pca.explained_variance_ratio, 1):
        print(f"  PC{i}: {var:.4f} ({var*100:.2f}%)")

    print(f"\nComponents needed for 95% variance: {pca.n_components_95}")

    # Extract eigenportfolios
    portfolios = pca.eigenportfolios(returns, n=3)
    print("\nTop 3 Eigenportfolios:")
    print(portfolios.round(3))

    # Variance plot
    plot_data = pca.explained_variance_plot()
    plt.figure(figsize=(10, 6))
    plt.bar(plot_data["component"], plot_data["variance_explained"], alpha=0.6, label="Individual")
    plt.plot(plot_data["component"], plot_data["cumulative_variance"], 'ro-', label="Cumulative")
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.title("PCA Variance Explanation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pca_variance.png")
    print("\nVariance plot saved as 'pca_variance.png'")


def example_clustering():
    """Example: Asset clustering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Asset Clustering")
    print("=" * 60)

    from puffin.unsupervised import (
        cluster_assets,
        optimal_clusters,
        cluster_summary,
        plot_dendrogram,
    )

    # Generate data
    returns = generate_sample_data()

    # Find optimal clusters
    optimal_k = optimal_clusters(returns, max_k=8, method="silhouette")
    print(f"\nOptimal number of clusters: {optimal_k}")

    # Cluster assets
    labels = cluster_assets(returns, n_clusters=3, method="kmeans")
    print("\nCluster assignments:")
    for asset, label in zip(returns.columns, labels):
        print(f"  {asset}: Cluster {label}")

    # Cluster summary
    summary = cluster_summary(returns, labels)
    print("\nCluster Summary:")
    print(summary.to_string(index=False))

    # Dendrogram
    fig = plot_dendrogram(returns, method="ward")
    plt.savefig("dendrogram.png")
    print("\nDendrogram saved as 'dendrogram.png'")
    plt.close(fig)


def example_risk_factors():
    """Example: Data-driven risk factor extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Data-Driven Risk Factors")
    print("=" * 60)

    from puffin.unsupervised import (
        extract_risk_factors,
        factor_exposures,
        factor_variance_decomposition,
        factor_mimicking_portfolio,
    )

    # Generate data
    returns = generate_sample_data()

    # Extract factors
    factors = extract_risk_factors(returns, n_factors=3)
    print("\nExtracted 3 risk factors")
    print(f"Factor returns shape: {factors.shape}")
    print("\nFirst 5 days of factor returns:")
    print(factors.head())

    # Compute factor exposures
    loadings = factor_exposures(returns, factors)
    print("\nFactor Exposures (Betas):")
    print(loadings.round(3))

    # Variance decomposition
    decomp = factor_variance_decomposition(returns, n_factors=3)
    print("\nVariance Decomposition:")
    print(decomp.round(4))

    # Factor mimicking portfolio
    portfolio = factor_mimicking_portfolio(returns, target_factor_idx=0, n_factors=3)
    print("\nFactor 1 Mimicking Portfolio:")
    print(portfolio.round(3))

    # Plot factor exposures
    fig, ax = plt.subplots(figsize=(10, 6))
    loadings.plot(kind='bar', ax=ax)
    ax.set_title("Factor Exposures by Asset")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Loading")
    ax.legend(title="Factor")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("factor_exposures.png")
    print("\nFactor exposures plot saved as 'factor_exposures.png'")
    plt.close(fig)


def example_clustering_comparison():
    """Example: Compare different clustering methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Clustering Method Comparison")
    print("=" * 60)

    from puffin.unsupervised import cluster_assets, gmm_cluster

    # Generate data
    returns = generate_sample_data()

    methods = ["kmeans", "hierarchical", "dbscan"]

    print("\nComparing clustering methods:")
    for method in methods:
        if method == "dbscan":
            labels = cluster_assets(returns, n_clusters=3, method=method, eps=1.0, min_samples=2)
        else:
            labels = cluster_assets(returns, n_clusters=3, method=method)

        n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise (-1)
        print(f"\n{method.upper()}:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Cluster sizes: {np.bincount(labels[labels >= 0])}")

    # GMM with probabilities
    print("\nGMM (soft clustering):")
    labels, probs = gmm_cluster(returns, n_components=3)
    print("Cluster probabilities for first 3 assets:")
    print(pd.DataFrame(
        probs[:3],
        index=returns.columns[:3],
        columns=["Cluster 0", "Cluster 1", "Cluster 2"]
    ).round(3))


if __name__ == "__main__":
    print("Unsupervised Learning for Trading - Examples\n")

    # Run examples
    example_pca()
    example_clustering()
    example_risk_factors()
    example_clustering_comparison()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
