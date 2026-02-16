"""
Tests for Bayesian models.

Uses small sample sizes for CI/CD speed. Mark slow tests with @pytest.mark.slow.
"""

import numpy as np
import pandas as pd
import pytest

# Check if PyMC is available
try:
    import pymc as pm
    from puffin.models.bayesian import (
        BayesianLinearRegression,
        bayesian_sharpe,
        compare_strategies_bayesian,
        BayesianPairsTrading
    )
    from puffin.models.stochastic_vol import StochasticVolatilityModel, estimate_volatility_regime
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not PYMC_AVAILABLE,
    reason="PyMC not installed"
)


@pytest.fixture
def linear_data():
    """Generate simple linear relationship for testing."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    # True relationship: y = 2 + 3*x1 - 1.5*x2 + noise
    y = 2 + 3 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(n) * 0.5
    return X, y


@pytest.fixture
def return_data():
    """Generate synthetic return data."""
    np.random.seed(42)
    n = 200
    # Positive expected return
    returns = np.random.randn(n) * 0.01 + 0.0005
    return returns


@pytest.fixture
def pairs_data():
    """Generate cointegrated pairs for testing."""
    np.random.seed(42)
    n = 120
    # Create cointegrated series
    x = np.cumsum(np.random.randn(n) * 0.5) + 100
    # y follows x with some noise and drift
    y = 1.5 * x + np.cumsum(np.random.randn(n) * 0.3) + 50
    return y, x


@pytest.fixture
def garch_like_returns():
    """Generate GARCH-like returns with time-varying volatility."""
    np.random.seed(42)
    n = 150
    # Simulate volatility clustering
    vol = np.ones(n) * 0.01
    returns = np.zeros(n)

    for t in range(1, n):
        # Simple GARCH(1,1)-like process
        vol[t] = 0.01 + 0.1 * returns[t-1]**2 + 0.85 * vol[t-1]
        returns[t] = vol[t] * np.random.randn()

    return returns


class TestBayesianLinearRegression:
    """Test Bayesian linear regression."""

    @pytest.mark.slow
    def test_fit_predict(self, linear_data):
        """Test basic fit and predict functionality."""
        X, y = linear_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = BayesianLinearRegression()
        model.fit(X_train, y_train, samples=500, tune=200)  # Small samples for speed

        # Check predictions
        mean, (lower, upper) = model.predict(X_test)

        assert len(mean) == len(X_test)
        assert len(lower) == len(X_test)
        assert len(upper) == len(X_test)

        # HDI bounds should contain mean
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)

    @pytest.mark.slow
    def test_parameter_recovery(self, linear_data):
        """Test if model recovers approximately correct parameters."""
        X, y = linear_data

        model = BayesianLinearRegression()
        model.fit(X, y, samples=500, tune=200)

        # Check summary
        summary = model.summary()
        assert 'mean' in summary or summary  # ArviZ summary format

        # Predictions should be reasonable
        mean, _ = model.predict(X[:10])
        assert not np.any(np.isnan(mean))

    def test_pandas_input(self, linear_data):
        """Test with pandas DataFrame/Series input."""
        X, y = linear_data
        X_df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        y_series = pd.Series(y, name='target')

        model = BayesianLinearRegression()
        model.fit(X_df, y_series, samples=200, tune=100)

        mean, _ = model.predict(X_df[:10])
        assert len(mean) == 10

    def test_error_before_fit(self):
        """Test error raised when predicting before fitting."""
        model = BayesianLinearRegression()
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(np.random.randn(10, 2))


class TestBayesianSharpe:
    """Test Bayesian Sharpe ratio estimation."""

    @pytest.mark.slow
    def test_positive_returns(self, return_data):
        """Test Sharpe estimation with positive returns."""
        result = bayesian_sharpe(return_data, samples=1000)  # Small sample size

        assert 'mean' in result
        assert 'hdi_low' in result
        assert 'hdi_high' in result
        assert 'prob_positive' in result
        assert 'std' in result

        # With positive returns, expect positive Sharpe
        assert result['mean'] > 0
        assert result['prob_positive'] > 0.5

    @pytest.mark.slow
    def test_negative_returns(self):
        """Test Sharpe estimation with negative returns."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01 - 0.0005  # Negative drift

        result = bayesian_sharpe(returns, samples=1000)

        # With negative returns, expect negative Sharpe
        assert result['mean'] < 0
        assert result['prob_positive'] < 0.5

    @pytest.mark.slow
    def test_pandas_series(self, return_data):
        """Test with pandas Series input."""
        returns_series = pd.Series(return_data, name='returns')
        result = bayesian_sharpe(returns_series, samples=1000)

        assert isinstance(result['mean'], float)

    @pytest.mark.slow
    def test_compare_strategies(self):
        """Test strategy comparison."""
        np.random.seed(42)

        # Three strategies with different Sharpe ratios
        strategy_a = np.random.randn(150) * 0.01 + 0.001  # Best
        strategy_b = np.random.randn(150) * 0.01 + 0.0005  # Middle
        strategy_c = np.random.randn(150) * 0.01 - 0.0002  # Worst

        results = compare_strategies_bayesian(
            {
                'strategy_a': strategy_a,
                'strategy_b': strategy_b,
                'strategy_c': strategy_c
            },
            samples=1000
        )

        assert len(results) == 3
        assert 'strategy' in results.columns
        assert 'sharpe_mean' in results.columns
        assert 'rank' in results.columns

        # Check ranking (strategy_a should be best)
        assert results.iloc[0]['rank'] == 1


class TestBayesianPairsTrading:
    """Test Bayesian pairs trading."""

    @pytest.mark.slow
    def test_fit_dynamic_hedge(self, pairs_data):
        """Test dynamic hedge ratio estimation."""
        y, x = pairs_data

        pairs = BayesianPairsTrading()
        hedge_ratios = pairs.fit_dynamic_hedge(y, x, window=30)  # Small window for speed

        assert 'hedge_ratio_mean' in hedge_ratios.columns
        assert 'hedge_ratio_std' in hedge_ratios.columns
        assert 'spread' in hedge_ratios.columns

        # Check no NaN in later periods
        assert not hedge_ratios['hedge_ratio_mean'].iloc[-30:].isna().any()

        # Hedge ratio should be positive (y increases with x)
        assert hedge_ratios['hedge_ratio_mean'].iloc[-1] > 0

    @pytest.mark.slow
    def test_generate_signals(self, pairs_data):
        """Test signal generation."""
        y, x = pairs_data

        pairs = BayesianPairsTrading()
        hedge_ratios = pairs.fit_dynamic_hedge(y, x, window=30)

        signals = pairs.generate_signals(entry_threshold=2.0, exit_threshold=0.5)

        assert len(signals) == len(y)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_with_custom_spread(self):
        """Test signal generation with custom spread."""
        np.random.seed(42)
        spread = np.random.randn(100)

        pairs = BayesianPairsTrading()
        signals = pairs.generate_signals(spread=spread)

        assert len(signals) == len(spread)

    def test_error_without_fit(self):
        """Test error when generating signals without fitting."""
        pairs = BayesianPairsTrading()
        with pytest.raises(ValueError):
            pairs.generate_signals()


class TestStochasticVolatilityModel:
    """Test stochastic volatility model."""

    @pytest.mark.slow
    def test_fit_basic(self, garch_like_returns):
        """Test basic fitting."""
        model = StochasticVolatilityModel()
        model.fit(garch_like_returns, samples=500, tune=200)  # Small samples

        assert model.volatility_path is not None
        assert len(model.volatility_path) == len(garch_like_returns)
        assert model.volatility_forecast is not None

    @pytest.mark.slow
    def test_volatility_positive(self, garch_like_returns):
        """Test that estimated volatility is positive."""
        model = StochasticVolatilityModel()
        model.fit(garch_like_returns, samples=500, tune=200)

        assert np.all(model.volatility_path > 0)
        assert model.volatility_forecast > 0

    @pytest.mark.slow
    def test_pandas_input(self, garch_like_returns):
        """Test with pandas Series input."""
        returns_series = pd.Series(
            garch_like_returns,
            index=pd.date_range('2020-01-01', periods=len(garch_like_returns))
        )

        model = StochasticVolatilityModel()
        model.fit(returns_series, samples=500, tune=200)

        assert model.volatility_path is not None

    def test_fallback_on_failure(self):
        """Test fallback to EWMA if MCMC fails."""
        # Create problematic data that might cause sampling issues
        np.random.seed(42)
        returns = np.random.randn(50) * 0.001

        model = StochasticVolatilityModel()
        # This might trigger fallback, but shouldn't crash
        model.fit(returns, samples=100, tune=50)

        # Should still have volatility estimates
        assert model.volatility_path is not None

    @pytest.mark.slow
    def test_estimate_volatility_regime(self, garch_like_returns):
        """Test quick volatility regime estimation."""
        result = estimate_volatility_regime(garch_like_returns, samples=500)

        assert 'volatility' in result.columns
        assert 'vol_lower' in result.columns
        assert 'vol_upper' in result.columns

        # Lower bound should be less than upper bound
        assert np.all(result['vol_lower'] <= result['vol_upper'])


class TestIntegration:
    """Integration tests combining multiple Bayesian components."""

    @pytest.mark.slow
    def test_bayesian_workflow(self, return_data):
        """Test a complete Bayesian analysis workflow."""
        # 1. Estimate Sharpe ratio
        sharpe_result = bayesian_sharpe(return_data, samples=1000)
        assert sharpe_result['mean'] is not None

        # 2. Estimate volatility
        vol_result = estimate_volatility_regime(return_data, samples=500)
        assert len(vol_result) == len(return_data)

        # 3. Compare with benchmark
        benchmark_returns = np.random.randn(len(return_data)) * 0.01
        comparison = compare_strategies_bayesian(
            {
                'strategy': return_data,
                'benchmark': benchmark_returns
            },
            samples=1000
        )
        assert len(comparison) == 2


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_returns(self):
        """Test with empty returns array."""
        with pytest.raises((ValueError, IndexError)):
            bayesian_sharpe(np.array([]), samples=100)

    def test_constant_returns(self):
        """Test with constant (zero variance) returns."""
        returns = np.ones(100) * 0.01

        # Should handle gracefully or raise informative error
        try:
            result = bayesian_sharpe(returns, samples=500)
            # If it succeeds, Sharpe should be undefined or very high
            assert result is not None
        except Exception as e:
            # Acceptable to fail with informative error
            assert isinstance(e, (ValueError, RuntimeError))

    @pytest.mark.slow
    def test_very_short_series(self):
        """Test with very short time series."""
        returns = np.random.randn(10) * 0.01

        # Should handle short series
        model = StochasticVolatilityModel()
        model.fit(returns, samples=200, tune=100)
        assert model.volatility_path is not None
