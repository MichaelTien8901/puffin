"""Unsupervised learning for trading: PCA, clustering, and risk factors."""

from .pca import MarketPCA
from .clustering import (
    cluster_assets,
    optimal_clusters,
    cluster_summary,
    hierarchical_cluster,
    plot_dendrogram,
    dbscan_cluster,
    gmm_cluster,
    cluster_correlation,
)
from .risk_factors import (
    extract_risk_factors,
    factor_exposures,
    factor_attribution,
    specific_risk,
    factor_variance_decomposition,
    factor_mimicking_portfolio,
    dynamic_factor_exposure,
    factor_timing_signal,
)

__all__ = [
    # PCA
    "MarketPCA",
    # Clustering
    "cluster_assets",
    "optimal_clusters",
    "cluster_summary",
    "hierarchical_cluster",
    "plot_dendrogram",
    "dbscan_cluster",
    "gmm_cluster",
    "cluster_correlation",
    # Risk factors
    "extract_risk_factors",
    "factor_exposures",
    "factor_attribution",
    "specific_risk",
    "factor_variance_decomposition",
    "factor_mimicking_portfolio",
    "dynamic_factor_exposure",
    "factor_timing_signal",
]
