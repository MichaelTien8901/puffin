"""Tests for risk factor extraction."""

import numpy as np
import pandas as pd
import pytest
from puffin.unsupervised.risk_factors import (
    extract_risk_factors,
    factor_exposures,
    factor_attribution,
    specific_risk,
    factor_variance_decomposition,
    factor_mimicking_portfolio,
    dynamic_factor_exposure,
    factor_timing_signal,
)


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


def test_extract_risk_factors(sample_returns):
    """Test risk factor extraction."""
    factors = extract_risk_factors(sample_returns, n_factors=3)

    assert isinstance(factors, pd.DataFrame)
    assert factors.shape == (252, 3)
    assert all(factors.columns == ["Factor_1", "Factor_2", "Factor_3"])


def test_extract_risk_factors_n_factors(sample_returns):
    """Test extraction with different number of factors."""
    for n in [2, 3, 5]:
        factors = extract_risk_factors(sample_returns, n_factors=n)
        assert factors.shape[1] == n


def test_factor_exposures(sample_returns):
    """Test factor exposure calculation."""
    factors = extract_risk_factors(sample_returns, n_factors=3)
    loadings = factor_exposures(sample_returns, factors)

    assert isinstance(loadings, pd.DataFrame)
    assert loadings.shape == (5, 3)  # 5 assets x 3 factors
    assert all(loadings.index == sample_returns.columns)
    assert all(loadings.columns == factors.columns)


def test_factor_attribution(sample_returns):
    """Test factor attribution."""
    factors = extract_risk_factors(sample_returns, n_factors=3)
    loadings = factor_exposures(sample_returns, factors)
    attribution = factor_attribution(sample_returns, factors, loadings)

    assert isinstance(attribution, pd.DataFrame)
    assert attribution.shape == (252, 5)
    assert all(attribution.columns == sample_returns.columns)


def test_specific_risk(sample_returns):
    """Test specific risk calculation."""
    factors = extract_risk_factors(sample_returns, n_factors=3)
    loadings = factor_exposures(sample_returns, factors)
    attribution = factor_attribution(sample_returns, factors, loadings)

    spec_risk = specific_risk(sample_returns, attribution)

    assert isinstance(spec_risk, pd.Series)
    assert len(spec_risk) == 5
    assert all(spec_risk > 0)  # Risk is always positive


def test_factor_variance_decomposition(sample_returns):
    """Test variance decomposition."""
    decomp = factor_variance_decomposition(sample_returns, n_factors=3)

    assert isinstance(decomp, pd.DataFrame)
    assert decomp.shape == (5, 5)  # 5 assets x 5 metrics
    assert "total_variance" in decomp.columns
    assert "factor_variance" in decomp.columns
    assert "specific_variance" in decomp.columns
    assert "pct_factor" in decomp.columns
    assert "pct_specific" in decomp.columns

    # Percentages should sum to approximately 100
    for asset in decomp.index:
        total_pct = decomp.loc[asset, "pct_factor"] + decomp.loc[asset, "pct_specific"]
        assert np.isclose(total_pct, 100, atol=0.1)


def test_factor_mimicking_portfolio(sample_returns):
    """Test factor mimicking portfolio."""
    portfolio = factor_mimicking_portfolio(sample_returns, target_factor_idx=0, n_factors=3)

    assert isinstance(portfolio, pd.Series)
    assert len(portfolio) == 5
    assert all(portfolio >= 0)  # Long-only
    assert np.isclose(portfolio.sum(), 1.0)  # Sums to 1


def test_factor_mimicking_portfolio_indices(sample_returns):
    """Test mimicking portfolios for different factors."""
    for idx in [0, 1, 2]:
        portfolio = factor_mimicking_portfolio(sample_returns, target_factor_idx=idx, n_factors=3)
        assert len(portfolio) == 5
        assert np.isclose(portfolio.sum(), 1.0)


def test_dynamic_factor_exposure(sample_returns):
    """Test dynamic factor exposure calculation."""
    exposures = dynamic_factor_exposure(sample_returns, window=100, n_factors=2)

    assert isinstance(exposures, dict)
    assert len(exposures) == 2  # 2 factors
    assert "Factor_1" in exposures
    assert "Factor_2" in exposures

    for factor_name, df in exposures.items():
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 5  # 5 assets
        assert df.shape[0] == 252 - 100  # Rolling window


def test_dynamic_factor_exposure_windows(sample_returns):
    """Test dynamic exposure with different windows."""
    for window in [50, 100, 150]:
        exposures = dynamic_factor_exposure(sample_returns, window=window, n_factors=2)
        for df in exposures.values():
            assert df.shape[0] == 252 - window


def test_factor_timing_signal(sample_returns):
    """Test factor timing signal generation."""
    signals = factor_timing_signal(sample_returns, factor_idx=0, n_factors=3, lookback=21)

    assert isinstance(signals, pd.Series)
    assert len(signals) == 252
    assert all(signals.isin([-1, 0, 1]))


def test_factor_timing_signal_parameters(sample_returns):
    """Test timing signals with different parameters."""
    for lookback in [10, 21, 63]:
        signals = factor_timing_signal(sample_returns, factor_idx=0, n_factors=3, lookback=lookback)
        assert len(signals) == 252
        assert all(signals.isin([-1, 0, 1]))


def test_risk_factors_with_nans():
    """Test risk factor extraction handles NaN values."""
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

    factors = extract_risk_factors(returns, n_factors=2)
    # Should drop NaN rows
    assert factors.shape[0] < 100
    assert factors.shape[1] == 2


def test_factor_exposures_consistency(sample_returns):
    """Test that factor exposures are consistent."""
    factors = extract_risk_factors(sample_returns, n_factors=3)
    loadings1 = factor_exposures(sample_returns, factors)
    loadings2 = factor_exposures(sample_returns, factors)

    # Should get same results
    assert np.allclose(loadings1.values, loadings2.values)


def test_variance_decomposition_completeness(sample_returns):
    """Test that variance decomposition accounts for total variance."""
    decomp = factor_variance_decomposition(sample_returns, n_factors=3)

    for asset in decomp.index:
        # Factor + specific should approximately equal total
        reconstructed = (
            decomp.loc[asset, "factor_variance"] +
            decomp.loc[asset, "specific_variance"]
        )
        assert np.isclose(
            reconstructed,
            decomp.loc[asset, "total_variance"],
            rtol=0.05  # Allow 5% relative tolerance
        )


def test_factor_attribution_shape(sample_returns):
    """Test that factor attribution maintains proper shape."""
    factors = extract_risk_factors(sample_returns, n_factors=2)
    loadings = factor_exposures(sample_returns, factors)
    attribution = factor_attribution(sample_returns, factors, loadings)

    # Attribution should have same shape as original returns
    assert attribution.shape == sample_returns.dropna().shape
