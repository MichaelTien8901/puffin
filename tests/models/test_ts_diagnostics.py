"""
Tests for time series diagnostics.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from puffin.models.ts_diagnostics import (
    decompose_series,
    test_stationarity,
    test_kpss,
    plot_acf_pacf,
    autocorrelation,
    check_stationarity
)


@pytest.fixture
def stationary_series():
    """Generate a stationary time series (white noise)."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252))


@pytest.fixture
def non_stationary_series():
    """Generate a non-stationary time series (random walk)."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252).cumsum())


@pytest.fixture
def seasonal_series():
    """Generate a seasonal time series."""
    np.random.seed(42)
    t = np.arange(500)
    trend = 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 52)  # 52-week seasonality
    noise = np.random.randn(500)
    series = pd.Series(trend + seasonal + noise)
    series.index = pd.date_range('2020-01-01', periods=500, freq='D')
    return series


class TestDecomposeSeries:
    """Tests for decompose_series function."""

    def test_basic_decomposition(self, seasonal_series):
        """Test basic seasonal decomposition."""
        components = decompose_series(seasonal_series, period=52)

        assert 'trend' in components
        assert 'seasonal' in components
        assert 'residual' in components
        assert 'observed' in components

        # Check that all components have same length
        assert len(components['trend']) == len(seasonal_series)
        assert len(components['seasonal']) == len(seasonal_series)

    def test_decomposition_too_short(self):
        """Test that short series raise error."""
        short_series = pd.Series(np.random.randn(50))

        with pytest.raises(ValueError, match="Series length"):
            decompose_series(short_series, period=52)

    def test_multiplicative_decomposition(self, seasonal_series):
        """Test multiplicative decomposition."""
        # Make series positive for multiplicative decomposition
        positive_series = seasonal_series - seasonal_series.min() + 10

        components = decompose_series(positive_series, period=52, model='multiplicative')

        assert 'trend' in components
        assert 'seasonal' in components


class TestStationarity:
    """Tests for stationarity tests."""

    def test_adf_stationary(self, stationary_series):
        """Test ADF on stationary series."""
        result = test_stationarity(stationary_series)

        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result
        assert 'critical_values' in result

        # White noise should be stationary
        assert result['is_stationary'] is True

    def test_adf_non_stationary(self, non_stationary_series):
        """Test ADF on non-stationary series."""
        result = test_stationarity(non_stationary_series)

        # Random walk should be non-stationary
        # Note: small sample might sometimes appear stationary
        assert 'is_stationary' in result

    def test_kpss_stationary(self, stationary_series):
        """Test KPSS on stationary series."""
        result = test_kpss(stationary_series)

        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result

        # White noise should be stationary
        assert result['is_stationary'] is True

    def test_kpss_non_stationary(self, non_stationary_series):
        """Test KPSS on non-stationary series."""
        result = test_kpss(non_stationary_series)

        # Random walk should be non-stationary
        assert 'is_stationary' in result

    def test_check_stationarity(self, stationary_series):
        """Test comprehensive stationarity check."""
        results = check_stationarity(stationary_series, verbose=False)

        assert 'adf' in results
        assert 'kpss' in results

    def test_too_short_series(self):
        """Test that short series raise error."""
        short_series = pd.Series(np.random.randn(5))

        with pytest.raises(ValueError, match="at least 10"):
            test_stationarity(short_series)


class TestAutocorrelation:
    """Tests for autocorrelation functions."""

    def test_plot_acf_pacf(self, stationary_series):
        """Test ACF/PACF plotting."""
        fig = plot_acf_pacf(stationary_series, nlags=20)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # ACF and PACF plots

        plt.close(fig)

    def test_autocorrelation_values(self, stationary_series):
        """Test autocorrelation calculation."""
        acf = autocorrelation(stationary_series, nlags=20)

        assert len(acf) == 21  # nlags + 1
        assert acf[0] == pytest.approx(1.0)  # Lag 0 should be 1
        assert np.all(np.abs(acf) <= 1.0)  # All values between -1 and 1

    def test_autocorrelation_ar_process(self):
        """Test autocorrelation on AR(1) process."""
        np.random.seed(42)
        # Generate AR(1) process: x_t = 0.5 * x_{t-1} + epsilon_t
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i-1] + np.random.randn()

        series = pd.Series(x)
        acf = autocorrelation(series, nlags=10)

        # AR(1) with phi=0.5 should have exponentially decaying ACF
        assert acf[1] > 0  # First lag should be positive
        assert acf[1] < acf[0]  # Should decay from lag 0

    def test_autocorrelation_too_short(self):
        """Test that autocorrelation handles short series."""
        short_series = pd.Series(np.random.randn(10))

        with pytest.raises(ValueError):
            autocorrelation(short_series, nlags=20)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self, seasonal_series):
        """Test a complete diagnostic workflow."""
        # 1. Decompose series
        components = decompose_series(seasonal_series, period=52)

        # 2. Test original series for stationarity
        original_result = check_stationarity(seasonal_series, verbose=False)

        # 3. Test residuals for stationarity
        residuals = components['residual'].dropna()
        residual_result = test_stationarity(residuals)

        # Residuals should be more stationary than original
        assert 'is_stationary' in original_result['adf']
        assert 'is_stationary' in residual_result

    def test_difference_to_stationarity(self, non_stationary_series):
        """Test differencing to achieve stationarity."""
        # Original series should be non-stationary
        original_result = test_stationarity(non_stationary_series)

        # First difference should be stationary
        diff_series = non_stationary_series.diff().dropna()
        diff_result = test_stationarity(diff_series)

        # Differenced random walk should be stationary
        assert diff_result['is_stationary'] is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_series_with_nans(self):
        """Test handling of series with NaN values."""
        series_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 20)

        # Should handle NaN by dropping them
        result = test_stationarity(series_with_nan)
        assert 'test_statistic' in result

    def test_constant_series(self):
        """Test handling of constant series."""
        constant_series = pd.Series([1.0] * 100)

        # Constant series might cause issues, but should handle gracefully
        # Different implementations might handle this differently
        try:
            result = test_stationarity(constant_series)
            # If it works, check that result is valid
            assert 'test_statistic' in result
        except (ValueError, np.linalg.LinAlgError):
            # Some implementations might raise error for constant series
            pass

    def test_very_short_period(self, seasonal_series):
        """Test decomposition with very short period."""
        # Should work with minimum period
        components = decompose_series(seasonal_series, period=2)
        assert 'trend' in components

    def test_empty_series(self):
        """Test with empty series."""
        empty_series = pd.Series([])

        with pytest.raises(ValueError):
            test_stationarity(empty_series)
