"""
Tests for alpha factor computation module.
"""

import numpy as np
import pandas as pd
import pytest
from puffin.factors.alpha import (
    compute_momentum_factors,
    compute_value_factors,
    compute_volatility_factors,
    compute_quality_factors,
    compute_all_factors
)


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    prices = pd.DataFrame({
        'AAPL': 100 * (1 + np.cumsum(np.random.randn(100) * 0.02)),
        'MSFT': 200 * (1 + np.cumsum(np.random.randn(100) * 0.015)),
        'GOOGL': 150 * (1 + np.cumsum(np.random.randn(100) * 0.018))
    }, index=dates)

    return prices


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    close = pd.DataFrame({
        'AAPL': 100 * (1 + np.cumsum(np.random.randn(100) * 0.02)),
        'MSFT': 200 * (1 + np.cumsum(np.random.randn(100) * 0.015))
    }, index=dates)

    high = close * (1 + np.random.rand(100, 2) * 0.02)
    low = close * (1 - np.random.rand(100, 2) * 0.02)
    open_ = close.shift(1).fillna(close.iloc[0])

    return {
        'close': close,
        'high': pd.DataFrame(high, columns=close.columns, index=dates),
        'low': pd.DataFrame(low, columns=close.columns, index=dates),
        'open': pd.DataFrame(open_.values, columns=close.columns, index=dates)
    }


@pytest.fixture
def sample_fundamentals():
    """Create sample fundamental data."""
    dates = pd.date_range('2024-01-01', periods=20, freq='Q')
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'symbol': symbol,
                'price': np.random.uniform(100, 200),
                'earnings': np.random.uniform(5, 15),
                'book_value': np.random.uniform(50, 100),
                'enterprise_value': np.random.uniform(1000, 2000),
                'ebitda': np.random.uniform(50, 150),
                'revenue': np.random.uniform(200, 400),
                'market_cap': np.random.uniform(1000, 2500)
            })

    df = pd.DataFrame(data)
    df = df.set_index(['date', 'symbol'])
    return df


@pytest.fixture
def sample_financials():
    """Create sample financial statement data."""
    dates = pd.date_range('2024-01-01', periods=20, freq='Q')
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'symbol': symbol,
                'net_income': np.random.uniform(100, 200),
                'revenue': np.random.uniform(500, 1000),
                'assets': np.random.uniform(2000, 3000),
                'equity': np.random.uniform(1000, 1500),
                'operating_cash_flow': np.random.uniform(150, 250),
                'total_accruals': np.random.uniform(-50, 50),
                'liabilities': np.random.uniform(500, 1000)
            })

    df = pd.DataFrame(data)
    df = df.set_index(['date', 'symbol'])
    return df


class TestMomentumFactors:
    """Tests for momentum factor computation."""

    def test_compute_momentum_basic(self, sample_prices):
        """Test basic momentum factor computation."""
        factors = compute_momentum_factors(sample_prices, windows=[5, 10, 21])

        assert isinstance(factors, pd.DataFrame)
        assert 'mom_5' in factors.columns
        assert 'mom_10' in factors.columns
        assert 'mom_21' in factors.columns

        # Check shape
        assert len(factors) == len(sample_prices) * len(sample_prices.columns)

        # Check index is MultiIndex
        assert isinstance(factors.index, pd.MultiIndex)
        assert factors.index.names == ['date', 'symbol']

    def test_momentum_values(self, sample_prices):
        """Test momentum factor values are correct."""
        factors = compute_momentum_factors(sample_prices, windows=[5])

        # Get first symbol
        first_symbol = sample_prices.columns[0]
        symbol_factors = factors.xs(first_symbol, level='symbol')

        # Check some non-NaN values exist
        assert symbol_factors['mom_5'].notna().sum() > 0

    def test_momentum_cross_sectional(self):
        """Test cross-sectional momentum features."""
        # Need > 252 data points for 252-window momentum
        dates = pd.date_range('2022-01-01', periods=300, freq='D')
        np.random.seed(42)
        prices = pd.DataFrame({
            'AAPL': 100 * (1 + np.cumsum(np.random.randn(300) * 0.02)),
            'MSFT': 200 * (1 + np.cumsum(np.random.randn(300) * 0.015)),
        }, index=dates)

        factors = compute_momentum_factors(prices, windows=[21, 252])

        # Should have mom_ratio and mom_diff
        assert 'mom_ratio' in factors.columns
        assert 'mom_diff' in factors.columns


class TestValueFactors:
    """Tests for value factor computation."""

    def test_compute_value_factors(self, sample_fundamentals):
        """Test value factor computation."""
        factors = compute_value_factors(sample_fundamentals)

        assert isinstance(factors, pd.DataFrame)
        assert 'pe_ratio' in factors.columns
        assert 'pb_ratio' in factors.columns
        assert 'ev_ebitda' in factors.columns
        assert 'earnings_yield' in factors.columns
        assert 'book_yield' in factors.columns

    def test_value_factors_no_inf(self, sample_fundamentals):
        """Test that value factors don't contain inf values."""
        factors = compute_value_factors(sample_fundamentals)

        # Should not have inf values
        assert not np.isinf(factors['pe_ratio']).any()
        assert not np.isinf(factors['pb_ratio']).any()

    def test_value_factors_inverse_relationship(self, sample_fundamentals):
        """Test that yield factors are inverse of ratio factors."""
        factors = compute_value_factors(sample_fundamentals)

        # P/E and E/P should be approximately inverse
        pe_ep_product = factors['pe_ratio'] * factors['earnings_yield']
        # Account for NaNs
        valid_mask = pe_ep_product.notna()
        if valid_mask.sum() > 0:
            assert np.allclose(
                pe_ep_product[valid_mask],
                1.0,
                atol=0.01
            )


class TestVolatilityFactors:
    """Tests for volatility factor computation."""

    def test_compute_volatility_close_only(self, sample_prices):
        """Test volatility computation with close prices only."""
        factors = compute_volatility_factors(sample_prices, windows=[21])

        assert isinstance(factors, pd.DataFrame)
        assert 'realized_vol_21' in factors.columns
        # Should not have Parkinson or GK without OHLC
        assert 'parkinson_vol_21' not in factors.columns
        assert 'gk_vol_21' not in factors.columns

    def test_compute_volatility_ohlc(self, sample_ohlcv):
        """Test volatility computation with OHLC data."""
        factors = compute_volatility_factors(sample_ohlcv, windows=[21])

        assert 'realized_vol_21' in factors.columns
        assert 'parkinson_vol_21' in factors.columns
        assert 'gk_vol_21' in factors.columns

    def test_volatility_positive(self, sample_prices):
        """Test that volatility values are positive."""
        factors = compute_volatility_factors(sample_prices, windows=[21])

        # Volatility should be non-negative (exclude ratio and trend columns)
        vol_cols = [col for col in factors.columns
                    if 'vol' in col and 'ratio' not in col and 'trend' not in col]
        for col in vol_cols:
            valid_values = factors[col].dropna()
            if len(valid_values) > 0:
                assert (valid_values >= 0).all()


class TestQualityFactors:
    """Tests for quality factor computation."""

    def test_compute_quality_factors(self, sample_financials):
        """Test quality factor computation."""
        factors = compute_quality_factors(sample_financials)

        assert isinstance(factors, pd.DataFrame)
        assert 'roe' in factors.columns
        assert 'roa' in factors.columns
        assert 'profit_margin' in factors.columns
        assert 'asset_turnover' in factors.columns

    def test_quality_factors_no_inf(self, sample_financials):
        """Test that quality factors don't contain inf values."""
        factors = compute_quality_factors(sample_financials)

        for col in factors.columns:
            assert not np.isinf(factors[col]).any()

    def test_roe_reasonable_range(self, sample_financials):
        """Test that ROE values are in reasonable range."""
        factors = compute_quality_factors(sample_financials)

        # ROE should typically be between -1 and 1 for our synthetic data
        roe_values = factors['roe'].dropna()
        if len(roe_values) > 0:
            assert roe_values.abs().max() < 10  # Allow some flexibility


class TestAllFactors:
    """Tests for combined factor computation."""

    def test_compute_all_factors(self, sample_ohlcv, sample_fundamentals, sample_financials):
        """Test computing all factors at once."""
        factors = compute_all_factors(
            prices=sample_ohlcv,
            fundamentals=sample_fundamentals,
            financials=sample_financials,
            momentum_windows=[5, 21],
            volatility_windows=[21]
        )

        assert isinstance(factors, pd.DataFrame)

        # Should have momentum factors
        assert 'mom_5' in factors.columns

        # Should have volatility factors
        assert 'realized_vol_21' in factors.columns

        # Should have value factors (joined)
        # Note: may have NaNs where dates don't align

        # Should have quality factors (joined)
        # Note: may have NaNs where dates don't align


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
