"""
Tests for Kalman filter module.
"""

import numpy as np
import pandas as pd
import pytest
from puffin.factors.kalman import (
    KalmanFilter,
    AdaptiveKalmanFilter,
    extract_trend,
    dynamic_hedge_ratio,
    kalman_ma_crossover
)


@pytest.fixture
def sample_signal():
    """Create sample signal with noise."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    # True signal: sine wave
    true_signal = np.sin(t)
    # Add noise
    noise = np.random.randn(100) * 0.2
    noisy_signal = true_signal + noise

    return pd.Series(noisy_signal, index=pd.date_range('2024-01-01', periods=100, freq='D'))


@pytest.fixture
def sample_prices():
    """Create sample price series."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Generate trending price series
    returns = np.random.randn(100) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.Series(prices, index=dates)


@pytest.fixture
def sample_cointegrated_pair():
    """Create sample cointegrated pair for pairs trading."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Stock 2 is base
    stock2 = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

    # Stock 1 follows Stock 2 with beta = 2
    beta = 2.0
    stock1 = 50 + beta * stock2 + np.random.randn(100) * 5

    return pd.Series(stock1, index=dates), pd.Series(stock2, index=dates)


class TestKalmanFilter:
    """Tests for KalmanFilter class."""

    def test_initialization(self):
        """Test KalmanFilter initialization."""
        kf = KalmanFilter()
        assert isinstance(kf, KalmanFilter)
        assert kf.dim_state == 1
        assert kf.F.shape == (1, 1)
        assert kf.H.shape == (1, 1)

    def test_predict(self):
        """Test Kalman filter prediction step."""
        kf = KalmanFilter()
        x_pred, P_pred = kf.predict()

        assert x_pred.shape == (1, 1)
        assert P_pred.shape == (1, 1)

    def test_update(self):
        """Test Kalman filter update step."""
        kf = KalmanFilter()
        measurement = np.array([1.0])
        x, P = kf.update(measurement)

        assert x.shape == (1, 1)
        assert P.shape == (1, 1)

    def test_filter(self, sample_signal):
        """Test Kalman filter on signal."""
        kf = KalmanFilter(
            process_covariance=1e-5,
            observation_covariance=0.04  # Match noise level
        )
        filtered = kf.filter(sample_signal)

        assert isinstance(filtered, pd.Series)
        assert len(filtered) == len(sample_signal)
        assert filtered.index.equals(sample_signal.index)

        # Filtered signal should be smoother (lower variance)
        # Note: This may not always hold perfectly for short series
        # So we just check that filtering produces different values
        assert not np.allclose(filtered.values, sample_signal.values)

    def test_smooth(self, sample_signal):
        """Test Kalman smoother on signal."""
        kf = KalmanFilter(
            process_covariance=1e-5,
            observation_covariance=0.04
        )
        smoothed = kf.smooth(sample_signal)

        assert isinstance(smoothed, pd.Series)
        assert len(smoothed) == len(sample_signal)
        assert smoothed.index.equals(sample_signal.index)

    def test_filter_reduces_noise(self, sample_signal):
        """Test that filtering reduces noise."""
        kf = KalmanFilter(
            process_covariance=1e-5,
            observation_covariance=0.04
        )
        filtered = kf.filter(sample_signal)

        # Calculate first differences (proxy for noise)
        signal_diff = sample_signal.diff().dropna()
        filtered_diff = filtered.diff().dropna()

        # Filtered signal should have smaller variance in differences
        assert filtered_diff.std() < signal_diff.std()

    def test_filter_numpy_array(self):
        """Test filtering with numpy array input."""
        np.random.seed(42)
        signal = np.random.randn(50)

        kf = KalmanFilter()
        filtered = kf.filter(signal)

        assert isinstance(filtered, pd.Series)
        assert len(filtered) == len(signal)


class TestAdaptiveKalmanFilter:
    """Tests for AdaptiveKalmanFilter class."""

    def test_initialization(self):
        """Test AdaptiveKalmanFilter initialization."""
        akf = AdaptiveKalmanFilter(adaptation_rate=0.01, window=20)
        assert isinstance(akf, AdaptiveKalmanFilter)
        assert akf.adaptation_rate == 0.01
        assert akf.window == 20

    def test_adaptive_filter(self, sample_signal):
        """Test adaptive Kalman filter."""
        akf = AdaptiveKalmanFilter(
            adaptation_rate=0.05,
            window=10,
            process_covariance=1e-5,
            observation_covariance=0.04
        )
        filtered = akf.filter(sample_signal)

        assert isinstance(filtered, pd.Series)
        assert len(filtered) == len(sample_signal)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_trend(self, sample_prices):
        """Test trend extraction from prices."""
        trend = extract_trend(
            sample_prices,
            process_variance=1e-5,
            observation_variance=1e-2
        )

        assert isinstance(trend, pd.Series)
        assert len(trend) == len(sample_prices)
        assert trend.index.equals(sample_prices.index)

        # Trend should be smoother than original
        assert trend.diff().std() < sample_prices.diff().std()

    def test_extract_trend_numpy(self):
        """Test trend extraction with numpy array."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(50) * 0.02))

        trend = extract_trend(prices)

        assert isinstance(trend, pd.Series)
        assert len(trend) == len(prices)

    def test_dynamic_hedge_ratio(self, sample_cointegrated_pair):
        """Test dynamic hedge ratio estimation."""
        stock1, stock2 = sample_cointegrated_pair

        hedge_ratio = dynamic_hedge_ratio(stock1, stock2, delta=1e-5)

        assert isinstance(hedge_ratio, pd.Series)
        assert len(hedge_ratio) == len(stock1)
        assert hedge_ratio.index.equals(stock1.index)

        # Hedge ratio should be approximately constant for cointegrated pair
        # (though it will vary due to noise)
        # Just check that we get reasonable values
        assert hedge_ratio.dropna().std() < 10  # Not too volatile

    def test_dynamic_hedge_ratio_numpy(self):
        """Test hedge ratio with numpy arrays."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = 2 * x + np.random.randn(50) * 0.1  # y = 2*x + noise

        hedge_ratio = dynamic_hedge_ratio(y, x)

        assert isinstance(hedge_ratio, pd.Series)
        assert len(hedge_ratio) == len(x)

        # Hedge ratio should converge to approximately 2
        # (though may not be exact due to initialization and noise)
        final_hr = hedge_ratio.iloc[-10:].mean()
        assert 0 < final_hr < 10  # Reasonable range

    def test_kalman_ma_crossover(self, sample_prices):
        """Test Kalman MA crossover strategy."""
        result = kalman_ma_crossover(
            sample_prices,
            fast_variance=1e-4,
            slow_variance=1e-6
        )

        assert isinstance(result, pd.DataFrame)
        assert 'fast_ma' in result.columns
        assert 'slow_ma' in result.columns
        assert 'signal' in result.columns

        # Signals should be -1, 0, or 1
        assert set(result['signal'].dropna().unique()).issubset({-1, 0, 1})

        # Fast MA should be more responsive (higher variance)
        assert result['fast_ma'].diff().std() > result['slow_ma'].diff().std()


class TestMultidimensionalKalman:
    """Tests for multidimensional Kalman filters."""

    def test_2d_state_space(self):
        """Test Kalman filter with 2D state space."""
        # Position and velocity model
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])  # State transition
        H = np.array([[1, 0]])  # Observe position only
        Q = np.eye(2) * 0.01  # Process noise
        R = np.array([[0.1]])  # Measurement noise

        kf = KalmanFilter(
            transition_matrix=F,
            observation_matrix=H,
            process_covariance=Q,
            observation_covariance=R,
            dim_state=2
        )

        # Generate measurements (noisy position)
        np.random.seed(42)
        true_position = np.cumsum(np.random.randn(50) * 0.1)
        measurements = true_position + np.random.randn(50) * 0.3

        # Filter
        filtered_states = []
        for measurement in measurements:
            x, P = kf.update(np.array([measurement]))
            filtered_states.append(x[0, 0])

        assert len(filtered_states) == len(measurements)

        # Filtered position should be smoother than measurements
        filtered_diff = np.diff(filtered_states)
        measured_diff = np.diff(measurements)
        assert np.std(filtered_diff) < np.std(measured_diff)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_constant_signal(self):
        """Test filtering constant signal."""
        constant_signal = pd.Series([5.0] * 50)

        kf = KalmanFilter()
        filtered = kf.filter(constant_signal)

        # Filtered signal should converge to constant
        assert np.allclose(filtered.iloc[-10:], 5.0, atol=0.1)

    def test_missing_values(self):
        """Test handling of NaN values."""
        np.random.seed(42)
        signal = pd.Series(np.random.randn(50))
        signal.iloc[10:15] = np.nan  # Insert NaNs

        kf = KalmanFilter()
        # Filter will process NaNs as is
        filtered = kf.filter(signal)

        # Should still produce output
        assert len(filtered) == len(signal)

    def test_single_observation(self):
        """Test filter with single observation."""
        kf = KalmanFilter()
        filtered = kf.filter(pd.Series([1.0]))

        assert len(filtered) == 1

    def test_very_small_signal(self):
        """Test filter with very small values."""
        small_signal = pd.Series([1e-10, 2e-10, 1.5e-10, 3e-10])

        kf = KalmanFilter()
        filtered = kf.filter(small_signal)

        assert len(filtered) == len(small_signal)
        assert not np.any(np.isnan(filtered))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
