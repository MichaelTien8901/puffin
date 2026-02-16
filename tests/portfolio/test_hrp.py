"""
Tests for Hierarchical Risk Parity (HRP) portfolio optimization.
"""

import pytest
import numpy as np
import pandas as pd
from puffin.portfolio.hrp import (
    hrp_weights,
    hrp_weights_with_names,
    hrp_allocation_stats,
    _correlation_to_distance,
    _tree_clustering,
    _cluster_variance
)


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Create returns with different correlation structures
    n_periods = 252

    # Two clusters of correlated assets
    cluster1 = np.random.randn(n_periods, 2) * 0.01 + 0.0005
    cluster2 = np.random.randn(n_periods, 2) * 0.015 + 0.0007

    # Add some inter-cluster correlation
    common_factor = np.random.randn(n_periods, 1) * 0.005
    cluster1 += common_factor * 0.3
    cluster2 += common_factor * 0.2

    returns = np.hstack([cluster1, cluster2])

    return pd.DataFrame(
        returns,
        index=dates,
        columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    )


def test_hrp_weights_basic(sample_returns):
    """Test basic HRP weights computation."""
    weights = hrp_weights(sample_returns)

    # Should return numpy array
    assert isinstance(weights, np.ndarray)

    # Should have correct length
    assert len(weights) == len(sample_returns.columns)

    # Weights should sum to 1
    assert np.isclose(weights.sum(), 1.0)

    # All weights should be non-negative
    assert np.all(weights >= 0)


def test_hrp_weights_with_names(sample_returns):
    """Test HRP weights with asset names."""
    weights = hrp_weights_with_names(sample_returns)

    # Should return pandas Series
    assert isinstance(weights, pd.Series)

    # Should have correct index
    assert list(weights.index) == list(sample_returns.columns)

    # Weights should sum to 1
    assert np.isclose(weights.sum(), 1.0)


def test_hrp_different_linkage_methods(sample_returns):
    """Test HRP with different linkage methods."""
    methods = ['single', 'complete', 'average', 'ward']

    for method in methods:
        weights = hrp_weights(sample_returns, linkage_method=method)

        # Should produce valid weights
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0)


def test_hrp_allocation_stats(sample_returns):
    """Test HRP allocation statistics."""
    weights = hrp_weights(sample_returns)
    stats = hrp_allocation_stats(sample_returns, weights)

    # Should return DataFrame
    assert isinstance(stats, pd.DataFrame)

    # Should have correct columns
    assert 'weight' in stats.columns
    assert 'risk_contribution' in stats.columns
    assert 'return_contribution' in stats.columns

    # Risk contributions should sum to approximately 1
    assert np.isclose(stats['risk_contribution'].sum(), 1.0, atol=0.01)


def test_correlation_to_distance(sample_returns):
    """Test correlation to distance conversion."""
    corr_matrix = sample_returns.corr()
    dist_matrix = _correlation_to_distance(corr_matrix)

    # Should be numpy array
    assert isinstance(dist_matrix, np.ndarray)

    # Should be same shape as correlation matrix
    assert dist_matrix.shape == corr_matrix.shape

    # Diagonal should be zero (distance from asset to itself)
    assert np.allclose(np.diag(dist_matrix), 0.0, atol=0.01)

    # All distances should be non-negative
    assert np.all(dist_matrix >= 0)


def test_tree_clustering(sample_returns):
    """Test hierarchical tree clustering."""
    corr_matrix = sample_returns.corr()
    dist_matrix = _correlation_to_distance(corr_matrix)
    linkage_matrix = _tree_clustering(dist_matrix)

    # Should return linkage matrix
    assert isinstance(linkage_matrix, np.ndarray)

    # Linkage matrix should have shape (n-1, 4)
    n_assets = len(sample_returns.columns)
    assert linkage_matrix.shape == (n_assets - 1, 4)


def test_cluster_variance(sample_returns):
    """Test cluster variance computation."""
    cov_matrix = sample_returns.cov()

    # Single asset variance
    var_single = _cluster_variance(cov_matrix, [0])
    assert var_single > 0
    assert np.isclose(var_single, cov_matrix.iloc[0, 0])

    # Two assets variance
    var_two = _cluster_variance(cov_matrix, [0, 1])
    assert var_two > 0

    # All assets variance
    var_all = _cluster_variance(cov_matrix, list(range(len(sample_returns.columns))))
    assert var_all > 0


def test_hrp_vs_equal_weights(sample_returns):
    """Test that HRP produces different weights than equal weighting."""
    hrp_w = hrp_weights(sample_returns)
    equal_w = np.ones(len(sample_returns.columns)) / len(sample_returns.columns)

    # HRP should produce different weights (unless assets are identical)
    assert not np.allclose(hrp_w, equal_w, atol=0.01)


def test_hrp_stability():
    """Test HRP stability with similar data."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Generate two similar datasets
    returns1 = pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        index=dates,
        columns=['A', 'B', 'C']
    )

    np.random.seed(43)
    returns2 = pd.DataFrame(
        np.random.randn(252, 3) * 0.01,
        index=dates,
        columns=['A', 'B', 'C']
    )

    weights1 = hrp_weights(returns1)
    weights2 = hrp_weights(returns2)

    # Weights should be somewhat similar (not too different)
    assert np.corrcoef(weights1, weights2)[0, 1] > 0.5


def test_hrp_single_asset():
    """Test HRP with single asset."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.DataFrame(
        np.random.randn(100, 1) * 0.01,
        index=dates,
        columns=['AAPL']
    )

    weights = hrp_weights(returns)

    # Should have 100% weight on the single asset
    assert np.isclose(weights[0], 1.0)


def test_hrp_two_assets():
    """Test HRP with two assets."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')

    returns = pd.DataFrame(
        np.random.randn(100, 2) * 0.01,
        index=dates,
        columns=['A', 'B']
    )

    weights = hrp_weights(returns)

    # Should produce valid weights
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)
    assert len(weights) == 2


def test_hrp_uncorrelated_assets():
    """Test HRP with uncorrelated assets of equal volatility."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Create uncorrelated assets with similar volatility
    returns = pd.DataFrame(
        np.random.randn(252, 4) * 0.01,
        index=dates,
        columns=['A', 'B', 'C', 'D']
    )

    weights = hrp_weights(returns)

    # With uncorrelated assets of equal volatility, weights should be similar
    assert np.std(weights) < 0.1


def test_hrp_high_low_vol_assets():
    """Test HRP with high and low volatility assets."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Create assets with different volatilities
    returns = pd.DataFrame({
        'low_vol': np.random.randn(252) * 0.005,
        'high_vol': np.random.randn(252) * 0.03
    }, index=dates)

    weights = hrp_weights(returns)

    # Low volatility asset should get higher weight
    assert weights[0] > weights[1]
