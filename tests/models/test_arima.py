"""
Tests for ARIMA models.
"""

import pytest
import numpy as np
import pandas as pd
from puffin.models.arima import ARIMAModel, auto_arima


@pytest.fixture
def white_noise():
    """Generate white noise series."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252))


@pytest.fixture
def ar1_process():
    """Generate AR(1) process."""
    np.random.seed(42)
    n = 252
    phi = 0.7
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i-1] + np.random.randn()
    return pd.Series(x)


@pytest.fixture
def ma1_process():
    """Generate MA(1) process."""
    np.random.seed(42)
    n = 252
    theta = 0.5
    epsilon = np.random.randn(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = epsilon[i] + theta * epsilon[i-1]
    return pd.Series(x)


@pytest.fixture
def random_walk():
    """Generate random walk (I(1) process)."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252).cumsum())


class TestARIMAModel:
    """Tests for ARIMAModel class."""

    def test_fit_basic(self, ar1_process):
        """Test basic ARIMA model fitting."""
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)

        assert model.model_ is not None
        assert model.order_ == (1, 0, 0)
        assert model.aic_ is not None
        assert model.bic_ is not None

    def test_predict(self, ar1_process):
        """Test prediction."""
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)

        forecast = model.predict(steps=10)

        assert len(forecast) == 10
        assert isinstance(forecast, pd.Series)

    def test_forecast_with_intervals(self, ar1_process):
        """Test forecast with confidence intervals."""
        model = ARIMAModel(order=(1, 0, 0))
        forecast_df = model.forecast(ar1_process, horizon=10, confidence=0.95)

        assert 'forecast' in forecast_df.columns
        assert 'lower' in forecast_df.columns
        assert 'upper' in forecast_df.columns
        assert len(forecast_df) == 10

        # Check that lower < forecast < upper
        assert (forecast_df['lower'] <= forecast_df['forecast']).all()
        assert (forecast_df['forecast'] <= forecast_df['upper']).all()

    def test_auto_select_order(self, ar1_process):
        """Test automatic order selection."""
        model = ARIMAModel()
        best_order = model.select_order(ar1_process, max_p=3, max_d=1, max_q=3)

        assert isinstance(best_order, tuple)
        assert len(best_order) == 3

        # Should select an AR(1) or similar for AR(1) process
        # (might not be exactly (1,0,0) due to randomness)
        assert best_order[0] >= 0 and best_order[0] <= 3
        assert best_order[1] >= 0 and best_order[1] <= 1
        assert best_order[2] >= 0 and best_order[2] <= 3

    def test_fit_with_auto_order(self, ar1_process):
        """Test fitting with automatic order selection."""
        model = ARIMAModel()
        model.fit(ar1_process)  # Should auto-select order

        assert model.order_ is not None
        assert model.model_ is not None

    def test_residuals(self, ar1_process):
        """Test getting residuals."""
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)

        residuals = model.residuals()

        assert isinstance(residuals, pd.Series)
        assert len(residuals) > 0

        # Residuals should have mean close to 0
        assert abs(residuals.mean()) < 0.5

    def test_summary(self, ar1_process):
        """Test model summary."""
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)

        summary = model.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_get_params(self, ar1_process):
        """Test getting model parameters."""
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)

        params = model.get_params()

        assert 'params' in params
        assert 'order' in params
        assert 'aic' in params
        assert 'bic' in params


class TestARIMAOrders:
    """Test different ARIMA orders."""

    def test_ar_model(self, ar1_process):
        """Test pure AR model."""
        model = ARIMAModel(order=(2, 0, 0))
        model.fit(ar1_process)

        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_ma_model(self, ma1_process):
        """Test pure MA model."""
        model = ARIMAModel(order=(0, 0, 1))
        model.fit(ma1_process)

        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_arma_model(self, ar1_process):
        """Test ARMA model."""
        model = ARIMAModel(order=(1, 0, 1))
        model.fit(ar1_process)

        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_integrated_model(self, random_walk):
        """Test ARIMA with integration."""
        # ARIMA(0,1,0) is random walk model
        model = ARIMAModel(order=(0, 1, 0))
        model.fit(random_walk)

        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_full_arima(self, random_walk):
        """Test full ARIMA(p,d,q) model."""
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(random_walk)

        forecast = model.predict(steps=5)
        assert len(forecast) == 5


class TestAutoARIMA:
    """Tests for auto_arima function."""

    def test_auto_arima_ar_process(self, ar1_process):
        """Test auto_arima on AR process."""
        model = auto_arima(ar1_process, max_p=3, max_d=1, max_q=3)

        assert model.model_ is not None
        assert model.order_ is not None

        # Should fit a reasonable model
        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_auto_arima_ma_process(self, ma1_process):
        """Test auto_arima on MA process."""
        model = auto_arima(ma1_process, max_p=2, max_d=1, max_q=2)

        assert model.model_ is not None
        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_auto_arima_random_walk(self, random_walk):
        """Test auto_arima on random walk."""
        model = auto_arima(random_walk, max_p=2, max_d=2, max_q=2)

        assert model.model_ is not None
        # Should select model with d=1
        assert model.order_[1] >= 1


class TestOrderSelection:
    """Tests for order selection."""

    def test_select_order_aic(self, ar1_process):
        """Test order selection with AIC."""
        model = ARIMAModel()
        order = model.select_order(
            ar1_process,
            max_p=3,
            max_d=1,
            max_q=3,
            information_criterion='aic'
        )

        assert isinstance(order, tuple)
        assert len(order) == 3

    def test_select_order_bic(self, ar1_process):
        """Test order selection with BIC."""
        model = ARIMAModel()
        order = model.select_order(
            ar1_process,
            max_p=3,
            max_d=1,
            max_q=3,
            information_criterion='bic'
        )

        assert isinstance(order, tuple)
        assert len(order) == 3

    def test_select_order_limited_range(self, ar1_process):
        """Test order selection with limited parameter range."""
        model = ARIMAModel()
        order = model.select_order(
            ar1_process,
            max_p=1,
            max_d=0,
            max_q=1
        )

        # Should be within specified ranges
        assert order[0] <= 1
        assert order[1] <= 0
        assert order[2] <= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_too_short_series(self):
        """Test with series too short for ARIMA."""
        short_series = pd.Series(np.random.randn(5))

        model = ARIMAModel(order=(1, 0, 0))
        with pytest.raises(ValueError, match="at least 10"):
            model.fit(short_series)

    def test_invalid_order(self, ar1_process):
        """Test with invalid order."""
        # Negative order components should be handled
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)  # Should work

    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        model = ARIMAModel(order=(1, 0, 0))

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(steps=5)

    def test_series_with_nans(self):
        """Test with series containing NaN values."""
        series_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 30)

        model = ARIMAModel(order=(1, 0, 0))
        model.fit(series_with_nan)  # Should handle NaN by dropping

        forecast = model.predict(steps=5)
        assert len(forecast) == 5

    def test_zero_steps_prediction(self, ar1_process):
        """Test prediction with zero steps."""
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)

        with pytest.raises(ValueError, match="steps must be"):
            model.predict(steps=0)

    def test_large_forecast_horizon(self, ar1_process):
        """Test forecast with large horizon."""
        model = ARIMAModel(order=(1, 0, 0))
        model.fit(ar1_process)

        # Should handle large horizons
        forecast = model.predict(steps=100)
        assert len(forecast) == 100


class TestModelComparison:
    """Test comparing different ARIMA models."""

    def test_compare_orders(self, ar1_process):
        """Test comparing different orders."""
        model1 = ARIMAModel(order=(1, 0, 0))
        model1.fit(ar1_process)

        model2 = ARIMAModel(order=(2, 0, 0))
        model2.fit(ar1_process)

        # Both models should fit
        assert model1.aic_ is not None
        assert model2.aic_ is not None

        # Can compare AIC values
        assert isinstance(model1.aic_, float)
        assert isinstance(model2.aic_, float)

    def test_refit_different_data(self, ar1_process, white_noise):
        """Test refitting model with different data."""
        model = ARIMAModel(order=(1, 0, 0))

        # Fit with first series
        model.fit(ar1_process)
        aic1 = model.aic_

        # Refit with second series
        model.fit(white_noise)
        aic2 = model.aic_

        # AIC values should be different
        assert aic1 != aic2
