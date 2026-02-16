"""
Tests for mean-variance portfolio optimization.
"""

import pytest
import numpy as np
import pandas as pd
from puffin.portfolio.mean_variance import MeanVarianceOptimizer


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Create correlated returns for 4 assets
    n_assets = 4
    n_periods = 252

    # Generate correlation matrix
    corr = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.3],
        [0.3, 0.4, 1.0, 0.5],
        [0.2, 0.3, 0.5, 1.0]
    ])

    # Generate returns
    std_devs = np.array([0.01, 0.012, 0.015, 0.018])
    means = np.array([0.0005, 0.0006, 0.0007, 0.0008])

    L = np.linalg.cholesky(corr)
    uncorrelated = np.random.randn(n_periods, n_assets)
    correlated = uncorrelated @ L.T

    returns = correlated * std_devs + means

    return pd.DataFrame(
        returns,
        index=dates,
        columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    )


def test_mean_variance_optimizer_init():
    """Test optimizer initialization."""
    optimizer = MeanVarianceOptimizer()
    assert optimizer is not None


def test_min_variance(sample_returns):
    """Test minimum variance portfolio."""
    optimizer = MeanVarianceOptimizer()
    result = optimizer.min_variance(sample_returns)

    assert 'weights' in result
    assert 'return' in result
    assert 'risk' in result
    assert 'sharpe' in result

    # Weights should sum to 1
    assert np.isclose(result['weights'].sum(), 1.0)

    # All weights should be non-negative (no short selling)
    assert np.all(result['weights'] >= 0)

    # Risk should be positive
    assert result['risk'] > 0


def test_max_sharpe(sample_returns):
    """Test maximum Sharpe ratio portfolio."""
    optimizer = MeanVarianceOptimizer()
    result = optimizer.max_sharpe(sample_returns, risk_free_rate=0.0)

    assert 'weights' in result
    assert 'return' in result
    assert 'risk' in result
    assert 'sharpe' in result

    # Weights should sum to 1
    assert np.isclose(result['weights'].sum(), 1.0)

    # All weights should be non-negative
    assert np.all(result['weights'] >= 0)

    # Should have positive return and risk
    assert result['return'] > 0
    assert result['risk'] > 0


def test_optimize_target_return(sample_returns):
    """Test optimization with target return constraint."""
    optimizer = MeanVarianceOptimizer()

    target_return = 0.0006
    result = optimizer.optimize(sample_returns, target_return=target_return)

    if result is not None:
        # Weights should sum to 1
        assert np.isclose(result['weights'].sum(), 1.0)

        # Should approximately achieve target return
        assert np.isclose(result['return'], target_return, rtol=0.01)

        # All weights should be non-negative
        assert np.all(result['weights'] >= 0)


def test_efficient_frontier(sample_returns):
    """Test efficient frontier computation."""
    optimizer = MeanVarianceOptimizer()
    frontier = optimizer.efficient_frontier(sample_returns, n_points=10)

    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) > 0

    # Should have return and risk columns
    assert 'return' in frontier.columns
    assert 'risk' in frontier.columns

    # Should have columns for each asset
    for asset in sample_returns.columns:
        assert asset in frontier.columns

    # Returns should be increasing or stable along frontier
    assert frontier['return'].is_monotonic_increasing or len(frontier['return'].unique()) < len(frontier)

    # Weights should sum to 1 for each point
    for idx, row in frontier.iterrows():
        weights = row[sample_returns.columns].values
        assert np.isclose(weights.sum(), 1.0, atol=0.01)


def test_portfolio_stats_computation(sample_returns):
    """Test internal portfolio statistics computation."""
    optimizer = MeanVarianceOptimizer()

    mean_returns = sample_returns.mean().values
    cov_matrix = sample_returns.cov().values

    # Equal weights
    weights = np.ones(len(sample_returns.columns)) / len(sample_returns.columns)

    portfolio_return, portfolio_risk = optimizer._compute_portfolio_stats(
        weights, mean_returns, cov_matrix
    )

    assert portfolio_return > 0
    assert portfolio_risk > 0


def test_min_variance_vs_max_sharpe(sample_returns):
    """Test that max Sharpe has higher return than min variance."""
    optimizer = MeanVarianceOptimizer()

    min_var = optimizer.min_variance(sample_returns)
    max_sharpe = optimizer.max_sharpe(sample_returns)

    # Max Sharpe should have higher or equal return
    assert max_sharpe['return'] >= min_var['return']

    # Max Sharpe should have higher Sharpe ratio
    assert max_sharpe['sharpe'] >= min_var['sharpe']


def test_zero_risk_free_rate(sample_returns):
    """Test with zero risk-free rate."""
    optimizer = MeanVarianceOptimizer()
    result = optimizer.max_sharpe(sample_returns, risk_free_rate=0.0)

    assert result['sharpe'] > 0


def test_positive_risk_free_rate(sample_returns):
    """Test with positive risk-free rate."""
    optimizer = MeanVarianceOptimizer()

    result_zero = optimizer.max_sharpe(sample_returns, risk_free_rate=0.0)
    result_positive = optimizer.max_sharpe(sample_returns, risk_free_rate=0.02)

    # Sharpe ratio should be lower with higher risk-free rate
    assert result_positive['sharpe'] <= result_zero['sharpe']


def test_single_asset():
    """Test with single asset (edge case)."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.DataFrame(
        np.random.randn(100, 1) * 0.01,
        index=dates,
        columns=['AAPL']
    )

    optimizer = MeanVarianceOptimizer()
    result = optimizer.min_variance(returns)

    # Should have 100% weight on the single asset
    assert np.isclose(result['weights'][0], 1.0)


def test_highly_correlated_assets():
    """Test with highly correlated assets."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')

    # Create highly correlated returns
    base = np.random.randn(100) * 0.01
    returns = pd.DataFrame({
        'A': base + np.random.randn(100) * 0.001,
        'B': base + np.random.randn(100) * 0.001,
        'C': base + np.random.randn(100) * 0.001
    }, index=dates)

    optimizer = MeanVarianceOptimizer()
    result = optimizer.min_variance(returns)

    # Should still work and produce valid weights
    assert np.isclose(result['weights'].sum(), 1.0)
    assert np.all(result['weights'] >= 0)
