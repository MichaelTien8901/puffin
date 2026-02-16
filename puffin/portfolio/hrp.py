"""
Hierarchical Risk Parity (HRP) Portfolio Optimization

Implements HRP algorithm by Marcos Lopez de Prado for portfolio construction
using hierarchical clustering to create more stable portfolios.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform
from typing import List, Optional, Tuple


def hrp_weights(
    returns: pd.DataFrame,
    linkage_method: str = 'single'
) -> np.ndarray:
    """
    Calculate portfolio weights using Hierarchical Risk Parity.

    The HRP algorithm follows three steps:
    1. Tree clustering: cluster assets based on correlation distance
    2. Quasi-diagonalization: reorder the covariance matrix
    3. Recursive bisection: allocate weights through the hierarchy

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset (rows: time, columns: assets)
    linkage_method : str, optional
        Linkage method for hierarchical clustering. Options: 'single', 'complete',
        'average', 'ward'. Default is 'single'.

    Returns
    -------
    np.ndarray
        HRP portfolio weights
    """
    # Step 1: Tree clustering
    corr_matrix = returns.corr()
    dist_matrix = _correlation_to_distance(corr_matrix)
    link = _tree_clustering(dist_matrix, method=linkage_method)

    # Step 2: Quasi-diagonalization
    sorted_indices = _quasi_diag(link, len(returns.columns))

    # Step 3: Recursive bisection
    cov_matrix = returns.cov()
    weights = _recursive_bisection(cov_matrix, sorted_indices)

    # Create weights array in original order
    weight_array = np.zeros(len(returns.columns))
    for i, idx in enumerate(sorted_indices):
        weight_array[idx] = weights[i]

    return weight_array


def _correlation_to_distance(corr_matrix: pd.DataFrame) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.

    Uses the formula: distance = sqrt(0.5 * (1 - correlation))

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix

    Returns
    -------
    np.ndarray
        Distance matrix
    """
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    return dist_matrix.values


def _tree_clustering(
    dist_matrix: np.ndarray,
    method: str = 'single'
) -> np.ndarray:
    """
    Perform hierarchical clustering on the distance matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Distance matrix (must be square and symmetric)
    method : str, optional
        Linkage method, by default 'single'

    Returns
    -------
    np.ndarray
        Linkage matrix from scipy.cluster.hierarchy.linkage
    """
    # Convert to condensed distance matrix (1D array of upper triangle)
    dist_condensed = squareform(dist_matrix, checks=False)

    # Perform hierarchical clustering
    link = linkage(dist_condensed, method=method)

    return link


def _quasi_diag(link: np.ndarray, num_items: int) -> List[int]:
    """
    Reorder assets to quasi-diagonalize the covariance matrix.

    This recursively traverses the clustering tree to determine the optimal
    ordering of assets.

    Parameters
    ----------
    link : np.ndarray
        Linkage matrix from hierarchical clustering
    num_items : int
        Number of original items (assets)

    Returns
    -------
    List[int]
        Sorted indices representing the reordered assets
    """
    # Convert linkage matrix to tree
    tree = to_tree(link, rd=False)

    # Get sorted order through tree traversal
    sorted_indices = _get_tree_order(tree, num_items)

    return sorted_indices


def _get_tree_order(node, num_items: int) -> List[int]:
    """
    Recursively traverse tree to get leaf order.

    Parameters
    ----------
    node : ClusterNode
        Current node in the tree
    num_items : int
        Number of original items

    Returns
    -------
    List[int]
        Ordered list of leaf indices
    """
    if node.is_leaf():
        return [node.id]

    # Recursively get order from left and right subtrees
    left_order = _get_tree_order(node.left, num_items)
    right_order = _get_tree_order(node.right, num_items)

    return left_order + right_order


def _recursive_bisection(
    cov_matrix: pd.DataFrame,
    sorted_indices: List[int]
) -> np.ndarray:
    """
    Recursively allocate weights through hierarchical bisection.

    At each level of the hierarchy, variance is computed for the left and right
    clusters, and weights are allocated inversely proportional to variance.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix of returns
    sorted_indices : List[int]
        Reordered asset indices from quasi-diagonalization

    Returns
    -------
    np.ndarray
        Portfolio weights in the sorted order
    """
    weights = np.ones(len(sorted_indices))

    # Recursive bisection
    _recursive_bisection_helper(cov_matrix, sorted_indices, weights, 0, len(sorted_indices))

    return weights


def _recursive_bisection_helper(
    cov_matrix: pd.DataFrame,
    sorted_indices: List[int],
    weights: np.ndarray,
    start: int,
    end: int
):
    """
    Helper function for recursive bisection.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix
    sorted_indices : List[int]
        Sorted asset indices
    weights : np.ndarray
        Weight array to modify in-place
    start : int
        Start index of current cluster
    end : int
        End index of current cluster
    """
    if end - start <= 1:
        return

    # Split point
    mid = (start + end) // 2

    # Get indices for left and right clusters
    left_indices = sorted_indices[start:mid]
    right_indices = sorted_indices[mid:end]

    # Calculate cluster variances
    left_var = _cluster_variance(cov_matrix, left_indices)
    right_var = _cluster_variance(cov_matrix, right_indices)

    # Allocate weight inversely proportional to variance
    total_var = left_var + right_var
    if total_var > 0:
        left_weight = 1.0 - left_var / total_var
        right_weight = 1.0 - right_var / total_var

        # Normalize
        total_weight = left_weight + right_weight
        left_weight /= total_weight
        right_weight /= total_weight
    else:
        left_weight = 0.5
        right_weight = 0.5

    # Update weights
    weights[start:mid] *= left_weight
    weights[mid:end] *= right_weight

    # Recurse on left and right clusters
    _recursive_bisection_helper(cov_matrix, sorted_indices, weights, start, mid)
    _recursive_bisection_helper(cov_matrix, sorted_indices, weights, mid, end)


def _cluster_variance(cov_matrix: pd.DataFrame, indices: List[int]) -> float:
    """
    Calculate the variance of a cluster.

    For a cluster, we use the inverse-variance portfolio (IVP) to compute
    the cluster's variance.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix
    indices : List[int]
        Asset indices in the cluster

    Returns
    -------
    float
        Cluster variance
    """
    if len(indices) == 0:
        return 0.0

    if len(indices) == 1:
        # Single asset: return its variance
        return cov_matrix.iloc[indices[0], indices[0]]

    # Get sub-covariance matrix for the cluster
    cov_slice = cov_matrix.iloc[indices, indices].values

    # Inverse variance weights for the cluster
    inv_diag = 1.0 / np.diag(cov_slice)
    weights = inv_diag / np.sum(inv_diag)

    # Cluster variance
    cluster_var = np.dot(weights, np.dot(cov_slice, weights))

    return cluster_var


def hrp_weights_with_names(
    returns: pd.DataFrame,
    linkage_method: str = 'single'
) -> pd.Series:
    """
    Calculate HRP weights and return as a pandas Series with asset names.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset
    linkage_method : str, optional
        Linkage method for hierarchical clustering, by default 'single'

    Returns
    -------
    pd.Series
        HRP weights indexed by asset names
    """
    weights = hrp_weights(returns, linkage_method=linkage_method)
    return pd.Series(weights, index=returns.columns)


def plot_dendrogram(
    returns: pd.DataFrame,
    linkage_method: str = 'single',
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Plot dendrogram for hierarchical clustering of assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset
    linkage_method : str, optional
        Linkage method, by default 'single'
    **kwargs
        Additional arguments passed to scipy.cluster.hierarchy.dendrogram

    Returns
    -------
    tuple
        (linkage_matrix, dendrogram_dict)
    """
    corr_matrix = returns.corr()
    dist_matrix = _correlation_to_distance(corr_matrix)
    link = _tree_clustering(dist_matrix, method=linkage_method)

    # Plot dendrogram
    dend = dendrogram(link, labels=returns.columns.tolist(), **kwargs)

    return link, dend


def hrp_allocation_stats(
    returns: pd.DataFrame,
    weights: np.ndarray
) -> pd.DataFrame:
    """
    Calculate allocation statistics for an HRP portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns for each asset
    weights : np.ndarray
        Portfolio weights

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: weight, risk_contribution, return_contribution
    """
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    # Portfolio statistics
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

    # Risk contributions
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib

    # Return contributions
    return_contrib = weights * mean_returns

    # Create DataFrame
    stats = pd.DataFrame({
        'weight': weights,
        'risk_contribution': risk_contrib / portfolio_variance if portfolio_variance > 0 else 0,
        'return_contribution': return_contrib / portfolio_return if portfolio_return > 0 else 0
    }, index=returns.columns)

    return stats
