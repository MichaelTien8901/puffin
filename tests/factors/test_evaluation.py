"""
Tests for factor evaluation module.
"""

import numpy as np
import pandas as pd
import pytest
from puffin.factors.evaluation import (
    FactorEvaluator,
    factor_autocorrelation,
    factor_rank_autocorrelation,
    quantile_returns_analysis
)


@pytest.fixture
def sample_factor():
    """Create sample factor data with MultiIndex."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    np.random.seed(42)
    data = []

    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'symbol': symbol,
                'factor': np.random.randn()
            })

    df = pd.DataFrame(data)
    return df.set_index(['date', 'symbol'])['factor']


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    np.random.seed(42)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.randn(60, 5) * 0.02, axis=0)),
        index=dates,
        columns=symbols
    )

    return prices


@pytest.fixture
def sample_predictive_factor(sample_prices):
    """Create a factor that predicts returns (for testing)."""
    # Factor = future 1-day return (with noise)
    forward_returns = sample_prices.pct_change(1).shift(-1)

    data = []
    for date in forward_returns.index[:-1]:  # Skip last date
        for symbol in forward_returns.columns:
            data.append({
                'date': date,
                'symbol': symbol,
                'factor': forward_returns.loc[date, symbol] + np.random.randn() * 0.01
            })

    df = pd.DataFrame(data)
    return df.set_index(['date', 'symbol'])['factor']


class TestFactorEvaluator:
    """Tests for FactorEvaluator class."""

    def test_initialization(self):
        """Test FactorEvaluator initialization."""
        evaluator = FactorEvaluator(quantiles=5, periods=[1, 5, 21])
        assert isinstance(evaluator, FactorEvaluator)
        assert evaluator.quantiles == 5
        assert evaluator.periods == [1, 5, 21]

    def test_compute_ic(self, sample_factor, sample_prices):
        """Test Information Coefficient computation."""
        evaluator = FactorEvaluator()

        # Compute forward returns
        forward_returns = sample_prices.pct_change(1).shift(-1)

        ic = evaluator.compute_ic(sample_factor, forward_returns, method='pearson')

        assert isinstance(ic, pd.Series)
        # Should have IC for each date where both factor and returns are available
        assert len(ic) > 0

        # IC should be correlation coefficient, so in [-1, 1]
        ic_values = ic.dropna()
        if len(ic_values) > 0:
            assert (ic_values >= -1).all()
            assert (ic_values <= 1).all()

    def test_compute_ic_spearman(self, sample_factor, sample_prices):
        """Test Spearman IC computation."""
        evaluator = FactorEvaluator()
        forward_returns = sample_prices.pct_change(1).shift(-1)

        ic = evaluator.compute_ic(sample_factor, forward_returns, method='spearman')

        assert isinstance(ic, pd.Series)
        ic_values = ic.dropna()
        if len(ic_values) > 0:
            assert (ic_values >= -1).all()
            assert (ic_values <= 1).all()

    def test_compute_factor_returns(self, sample_factor, sample_prices):
        """Test factor returns computation."""
        evaluator = FactorEvaluator(quantiles=5, periods=[1, 5])
        factor_returns = evaluator.compute_factor_returns(
            sample_factor,
            sample_prices,
            periods=[1, 5]
        )

        assert isinstance(factor_returns, pd.DataFrame)

        if not factor_returns.empty:
            assert 'return_1d' in factor_returns.columns
            assert 'return_5d' in factor_returns.columns

    def test_compute_turnover(self, sample_factor):
        """Test turnover computation."""
        evaluator = FactorEvaluator(quantiles=5)
        turnover = evaluator.compute_turnover(sample_factor)

        assert isinstance(turnover, pd.Series)

        if len(turnover) > 0:
            # Turnover should be percentage, so in [0, 100]
            assert (turnover >= 0).all()
            assert (turnover <= 100).all()

    def test_full_tearsheet(self, sample_factor, sample_prices):
        """Test full tearsheet generation."""
        evaluator = FactorEvaluator(quantiles=5, periods=[1, 5])
        tearsheet = evaluator.full_tearsheet(sample_factor, sample_prices)

        assert isinstance(tearsheet, dict)

        # Should have IC metrics
        assert 'ic_pearson' in tearsheet
        assert 'ic_spearman' in tearsheet
        assert 'ic_mean' in tearsheet
        assert 'ic_std' in tearsheet
        assert 'ic_ir' in tearsheet

        # Should have turnover
        assert 'turnover' in tearsheet
        assert 'mean_turnover' in tearsheet

        # Should have summary
        assert 'summary' in tearsheet

    def test_full_tearsheet_with_returns(self, sample_predictive_factor, sample_prices):
        """Test tearsheet with predictive factor (should have factor returns)."""
        evaluator = FactorEvaluator(quantiles=3, periods=[1])
        tearsheet = evaluator.full_tearsheet(sample_predictive_factor, sample_prices)

        # Should have factor returns
        if 'factor_returns' in tearsheet and not tearsheet['factor_returns'].empty:
            assert 'mean_returns' in tearsheet
            assert 'std_returns' in tearsheet
            assert 'sharpe_ratios' in tearsheet


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_factor_autocorrelation(self, sample_factor):
        """Test factor autocorrelation computation."""
        autocorr = factor_autocorrelation(sample_factor, lags=5)

        assert isinstance(autocorr, pd.Series)
        assert len(autocorr) == 5

        # Autocorrelation should be in [-1, 1]
        autocorr_values = autocorr.dropna()
        if len(autocorr_values) > 0:
            assert (autocorr_values >= -1).all()
            assert (autocorr_values <= 1).all()

    def test_factor_rank_autocorrelation(self, sample_factor):
        """Test factor rank autocorrelation."""
        rank_autocorr = factor_rank_autocorrelation(sample_factor, lag=1)

        assert isinstance(rank_autocorr, (float, np.floating, type(np.nan)))

        if not np.isnan(rank_autocorr):
            assert -1 <= rank_autocorr <= 1

    def test_quantile_returns_analysis(self, sample_factor, sample_prices):
        """Test quantile returns analysis."""
        quantile_returns = quantile_returns_analysis(
            sample_factor,
            sample_prices,
            quantiles=5,
            period=1
        )

        if not quantile_returns.empty:
            assert isinstance(quantile_returns, pd.DataFrame)
            # Should have data for each quantile (0 to 4 for 5 quantiles)
            assert len(quantile_returns) <= 5


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_date_factor(self, sample_prices):
        """Test with factor from single date."""
        factor = pd.Series(
            [0.5, -0.2, 0.8, 0.1, -0.5],
            index=pd.MultiIndex.from_product(
                [[pd.Timestamp('2024-01-01')], ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']],
                names=['date', 'symbol']
            )
        )

        evaluator = FactorEvaluator()
        forward_returns = sample_prices.pct_change(1).shift(-1)

        ic = evaluator.compute_ic(factor, forward_returns)

        # Should have at most one IC value
        assert len(ic) <= 1

    def test_mismatched_dates(self, sample_factor):
        """Test with mismatched dates between factor and prices."""
        # Create prices with different dates
        different_dates = pd.date_range('2023-01-01', periods=50, freq='D')
        prices = pd.DataFrame(
            np.random.randn(50, 5),
            index=different_dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        )

        evaluator = FactorEvaluator()
        forward_returns = prices.pct_change(1).shift(-1)

        ic = evaluator.compute_ic(sample_factor, forward_returns)

        # Should have empty or very small IC series (no date overlap)
        assert len(ic) == 0 or ic.isna().all()

    def test_missing_symbols(self, sample_prices):
        """Test with factor having different symbols than prices."""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        different_symbols = ['TSLA', 'NVDA', 'AMD']

        data = []
        for date in dates:
            for symbol in different_symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'factor': np.random.randn()
                })

        factor = pd.DataFrame(data).set_index(['date', 'symbol'])['factor']

        evaluator = FactorEvaluator()
        forward_returns = sample_prices.pct_change(1).shift(-1)

        ic = evaluator.compute_ic(factor, forward_returns)

        # Should have empty IC (no symbol overlap)
        assert len(ic) == 0 or ic.isna().all()

    def test_constant_factor(self, sample_prices):
        """Test with constant factor values."""
        dates = sample_prices.index[:20]
        symbols = sample_prices.columns

        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'factor': 1.0  # Constant
                })

        constant_factor = pd.DataFrame(data).set_index(['date', 'symbol'])['factor']

        evaluator = FactorEvaluator()
        forward_returns = sample_prices.pct_change(1).shift(-1)

        ic = evaluator.compute_ic(constant_factor, forward_returns)

        # Constant factor should have zero correlation with returns
        # (or NaN if std is 0)
        assert len(ic) == 0 or ic.isna().all() or np.allclose(ic.dropna(), 0, atol=1e-10)

    def test_few_assets(self):
        """Test with very few assets."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        symbols = ['AAPL', 'MSFT']  # Only 2 assets

        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'factor': np.random.randn()
                })

        factor = pd.DataFrame(data).set_index(['date', 'symbol'])['factor']

        prices = pd.DataFrame(
            np.random.randn(15, 2),
            index=pd.date_range('2024-01-01', periods=15, freq='D'),
            columns=symbols
        )

        evaluator = FactorEvaluator(quantiles=2)  # Only 2 quantiles for 2 assets
        forward_returns = prices.pct_change(1).shift(-1)

        ic = evaluator.compute_ic(factor, forward_returns)

        # Should still compute (though may not be very meaningful)
        assert isinstance(ic, pd.Series)

    def test_non_multiindex_factor(self):
        """Test that non-MultiIndex factor raises error."""
        factor = pd.Series([1, 2, 3])
        prices = pd.DataFrame([[100, 200], [101, 202]])

        evaluator = FactorEvaluator()

        with pytest.raises(ValueError, match="MultiIndex"):
            evaluator.compute_ic(factor, prices)


class TestStatisticalProperties:
    """Tests for statistical properties of factor metrics."""

    def test_ic_ir_calculation(self, sample_factor, sample_prices):
        """Test that Information Ratio is calculated correctly."""
        evaluator = FactorEvaluator()
        tearsheet = evaluator.full_tearsheet(sample_factor, sample_prices)

        ic_mean = tearsheet['ic_mean']
        ic_std = tearsheet['ic_std']
        ic_ir = tearsheet['ic_ir']

        # IR should be mean / std
        if ic_std > 1e-8:
            expected_ir = ic_mean / ic_std
            assert np.isclose(ic_ir, expected_ir, rtol=1e-6)

    def test_turnover_range(self, sample_factor):
        """Test that turnover is in valid range."""
        evaluator = FactorEvaluator(quantiles=5)
        turnover = evaluator.compute_turnover(sample_factor)

        # Turnover should be percentage in [0, 100]
        assert (turnover >= 0).all()
        assert (turnover <= 100).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
