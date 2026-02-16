"""Tests for PCA and eigenportfolio analysis."""

import numpy as np
import pandas as pd
import pytest
from puffin.unsupervised.pca import MarketPCA


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Generate correlated returns
    cov = np.array([
        [0.04, 0.02, 0.03, 0.01, 0.015],
        [0.02, 0.05, 0.025, 0.015, 0.02],
        [0.03, 0.025, 0.045, 0.02, 0.025],
        [0.01, 0.015, 0.02, 0.06, 0.01],
        [0.015, 0.02, 0.025, 0.01, 0.08],
    ])

    returns = np.random.multivariate_normal(
        mean=[0.0005] * 5,
        cov=cov,
        size=252
    )

    return pd.DataFrame(returns, index=dates, columns=assets)


def test_market_pca_fit(sample_returns):
    """Test PCA fit method."""
    pca = MarketPCA(n_components=3)
    pca.fit(sample_returns)

    assert pca._pca is not None
    assert len(pca.explained_variance_ratio) == 3
    assert pca.components.shape == (3, 5)


def test_market_pca_transform(sample_returns):
    """Test PCA transform method."""
    pca = MarketPCA(n_components=3)
    transformed = pca.fit_transform(sample_returns)

    assert transformed.shape == (252, 3)
    assert isinstance(transformed, np.ndarray)


def test_market_pca_explained_variance(sample_returns):
    """Test explained variance ratio."""
    pca = MarketPCA()
    pca.fit(sample_returns)

    variance_ratio = pca.explained_variance_ratio
    assert len(variance_ratio) == 5
    assert np.allclose(variance_ratio.sum(), 1.0)
    assert all(variance_ratio[i] >= variance_ratio[i+1] for i in range(len(variance_ratio)-1))


def test_market_pca_n_components_95(sample_returns):
    """Test n_components_95 property."""
    pca = MarketPCA()
    pca.fit(sample_returns)

    n_comp = pca.n_components_95
    assert 1 <= n_comp <= 5
    assert np.cumsum(pca.explained_variance_ratio)[n_comp-1] >= 0.95


def test_eigenportfolios(sample_returns):
    """Test eigenportfolio extraction."""
    pca = MarketPCA()
    portfolios = pca.eigenportfolios(sample_returns, n=3)

    assert portfolios.shape == (3, 5)
    assert all(portfolios.columns == sample_returns.columns)
    assert all(portfolios.index == ["PC1", "PC2", "PC3"])

    # Check weights sum to 1
    for i in range(3):
        assert np.isclose(portfolios.iloc[i].sum(), 1.0)


def test_reconstruct(sample_returns):
    """Test reconstruction with limited components."""
    pca = MarketPCA()
    reconstructed = pca.reconstruct(sample_returns, n_components=3)

    assert reconstructed.shape == sample_returns.dropna().shape
    assert all(reconstructed.columns == sample_returns.columns)


def test_explained_variance_plot(sample_returns):
    """Test explained variance plot data."""
    pca = MarketPCA()
    pca.fit(sample_returns)

    plot_data = pca.explained_variance_plot()

    assert "component" in plot_data.columns
    assert "variance_explained" in plot_data.columns
    assert "cumulative_variance" in plot_data.columns
    assert len(plot_data) == 5
    assert plot_data["cumulative_variance"].iloc[-1] <= 1.0


def test_pca_error_before_fit():
    """Test that methods raise error before fit is called."""
    pca = MarketPCA()

    with pytest.raises(ValueError, match="Must call fit"):
        _ = pca.explained_variance_ratio

    with pytest.raises(ValueError, match="Must call fit"):
        _ = pca.components

    with pytest.raises(ValueError, match="Must call fit"):
        _ = pca.n_components_95


def test_pca_with_nans():
    """Test PCA handles NaN values."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.DataFrame(
        np.random.randn(100, 3),
        index=dates,
        columns=["A", "B", "C"]
    )
    # Add some NaNs
    returns.iloc[0, 0] = np.nan
    returns.iloc[5, 1] = np.nan

    pca = MarketPCA(n_components=2)
    pca.fit(returns)

    assert pca._pca is not None
    assert len(pca.explained_variance_ratio) == 2


def test_pca_method_chaining(sample_returns):
    """Test that fit returns self for method chaining."""
    pca = MarketPCA(n_components=2)
    result = pca.fit(sample_returns)

    assert result is pca
    assert pca._pca is not None
