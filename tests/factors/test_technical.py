"""
Tests for technical indicators module.
"""

import numpy as np
import pandas as pd
import pytest
from puffin.factors.technical import TechnicalIndicators


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    close = pd.Series(100 * (1 + np.cumsum(np.random.randn(100) * 0.02)), index=dates)
    high = close * (1 + np.random.rand(100) * 0.02)
    low = close * (1 - np.random.rand(100) * 0.02)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.uniform(1000, 2000, 100), index=dates)

    return {
        'close': close,
        'high': high,
        'low': low,
        'open': open_,
        'volume': volume
    }


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""

    def test_initialization(self):
        """Test TechnicalIndicators initialization."""
        ti = TechnicalIndicators()
        assert isinstance(ti, TechnicalIndicators)

    def test_compute_overlap(self, sample_ohlcv):
        """Test overlap indicators computation."""
        ti = TechnicalIndicators()
        overlap = ti.compute_overlap(sample_ohlcv)

        assert isinstance(overlap, pd.DataFrame)
        assert 'sma_20' in overlap.columns
        assert 'sma_50' in overlap.columns
        assert 'ema_12' in overlap.columns
        assert 'ema_26' in overlap.columns
        assert 'bb_upper' in overlap.columns
        assert 'bb_middle' in overlap.columns
        assert 'bb_lower' in overlap.columns

    def test_bollinger_bands_order(self, sample_ohlcv):
        """Test that Bollinger Bands maintain proper order."""
        ti = TechnicalIndicators()
        overlap = ti.compute_overlap(sample_ohlcv)

        # Upper should be >= Middle >= Lower (where not NaN)
        valid_mask = (
            overlap['bb_upper'].notna() &
            overlap['bb_middle'].notna() &
            overlap['bb_lower'].notna()
        )

        if valid_mask.sum() > 0:
            assert (overlap.loc[valid_mask, 'bb_upper'] >= overlap.loc[valid_mask, 'bb_middle']).all()
            assert (overlap.loc[valid_mask, 'bb_middle'] >= overlap.loc[valid_mask, 'bb_lower']).all()

    def test_compute_momentum(self, sample_ohlcv):
        """Test momentum indicators computation."""
        ti = TechnicalIndicators()
        momentum = ti.compute_momentum(sample_ohlcv)

        assert isinstance(momentum, pd.DataFrame)
        assert 'rsi' in momentum.columns
        assert 'macd' in momentum.columns
        assert 'macd_signal' in momentum.columns
        assert 'macd_hist' in momentum.columns
        assert 'stoch_k' in momentum.columns
        assert 'stoch_d' in momentum.columns
        assert 'adx' in momentum.columns
        assert 'cci' in momentum.columns
        assert 'mfi' in momentum.columns

    def test_rsi_range(self, sample_ohlcv):
        """Test that RSI is in valid range [0, 100]."""
        ti = TechnicalIndicators()
        momentum = ti.compute_momentum(sample_ohlcv)

        rsi_values = momentum['rsi'].dropna()
        if len(rsi_values) > 0:
            assert (rsi_values >= 0).all()
            assert (rsi_values <= 100).all()

    def test_mfi_range(self, sample_ohlcv):
        """Test that MFI is in valid range [0, 100]."""
        ti = TechnicalIndicators()
        momentum = ti.compute_momentum(sample_ohlcv)

        mfi_values = momentum['mfi'].dropna()
        if len(mfi_values) > 0:
            assert (mfi_values >= 0).all()
            assert (mfi_values <= 100).all()

    def test_compute_volume(self, sample_ohlcv):
        """Test volume indicators computation."""
        ti = TechnicalIndicators()
        volume = ti.compute_volume(sample_ohlcv)

        assert isinstance(volume, pd.DataFrame)
        assert 'obv' in volume.columns
        assert 'ad' in volume.columns
        assert 'adosc' in volume.columns

    def test_obv_cumulative(self, sample_ohlcv):
        """Test that OBV is cumulative."""
        ti = TechnicalIndicators()
        volume = ti.compute_volume(sample_ohlcv)

        obv_values = volume['obv'].dropna()
        # OBV should change over time (not all same value)
        if len(obv_values) > 1:
            assert obv_values.std() > 0

    def test_compute_volatility(self, sample_ohlcv):
        """Test volatility indicators computation."""
        ti = TechnicalIndicators()
        volatility = ti.compute_volatility(sample_ohlcv)

        assert isinstance(volatility, pd.DataFrame)
        assert 'atr' in volatility.columns
        assert 'natr' in volatility.columns
        assert 'trange' in volatility.columns

    def test_atr_positive(self, sample_ohlcv):
        """Test that ATR is positive."""
        ti = TechnicalIndicators()
        volatility = ti.compute_volatility(sample_ohlcv)

        atr_values = volatility['atr'].dropna()
        if len(atr_values) > 0:
            assert (atr_values >= 0).all()

    def test_compute_all(self, sample_ohlcv):
        """Test computing all indicators at once."""
        ti = TechnicalIndicators()
        all_indicators = ti.compute_all(sample_ohlcv)

        assert isinstance(all_indicators, pd.DataFrame)

        # Should have indicators from all categories
        assert 'sma_20' in all_indicators.columns  # Overlap
        assert 'rsi' in all_indicators.columns  # Momentum
        assert 'obv' in all_indicators.columns  # Volume
        assert 'atr' in all_indicators.columns  # Volatility

    def test_compute_all_selected_categories(self, sample_ohlcv):
        """Test computing only selected indicator categories."""
        ti = TechnicalIndicators()
        indicators = ti.compute_all(sample_ohlcv, categories=['overlap', 'momentum'])

        # Should have overlap and momentum
        assert 'sma_20' in indicators.columns
        assert 'rsi' in indicators.columns

        # Should not have volume or volatility
        assert 'obv' not in indicators.columns
        assert 'atr' not in indicators.columns

    def test_use_talib_flag(self, sample_ohlcv):
        """Test that use_talib flag works."""
        # Force pure Python mode
        ti = TechnicalIndicators(use_talib=False)
        indicators = ti.compute_overlap(sample_ohlcv)

        # Should still compute indicators
        assert 'sma_20' in indicators.columns
        assert 'ema_12' in indicators.columns

    def test_consistent_index(self, sample_ohlcv):
        """Test that all indicators have consistent index."""
        ti = TechnicalIndicators()
        all_indicators = ti.compute_all(sample_ohlcv)

        # Index should match input index
        assert len(all_indicators) == len(sample_ohlcv['close'])
        assert all_indicators.index.equals(sample_ohlcv['close'].index)


class TestPurePythonImplementations:
    """Tests for pure Python fallback implementations."""

    def test_rsi_python(self, sample_ohlcv):
        """Test pure Python RSI implementation."""
        ti = TechnicalIndicators(use_talib=False)
        rsi = ti._calculate_rsi_python(sample_ohlcv['close'], period=14)

        assert isinstance(rsi, pd.Series)
        # RSI should have some valid values
        assert rsi.notna().sum() > 0
        # RSI should be in [0, 100]
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()

    def test_macd_python(self, sample_ohlcv):
        """Test pure Python MACD implementation."""
        ti = TechnicalIndicators(use_talib=False)
        macd_data = ti._calculate_macd_python(sample_ohlcv['close'])

        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'hist' in macd_data

        # Histogram should be MACD - Signal
        valid_mask = (
            macd_data['macd'].notna() &
            macd_data['signal'].notna() &
            macd_data['hist'].notna()
        )

        if valid_mask.sum() > 0:
            expected_hist = macd_data['macd'][valid_mask] - macd_data['signal'][valid_mask]
            assert np.allclose(macd_data['hist'][valid_mask], expected_hist, atol=1e-6)

    def test_stoch_python(self, sample_ohlcv):
        """Test pure Python Stochastic implementation."""
        ti = TechnicalIndicators(use_talib=False)
        stoch_data = ti._calculate_stoch_python(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )

        assert 'k' in stoch_data
        assert 'd' in stoch_data

        # %K and %D should be in [0, 100]
        k_valid = stoch_data['k'].dropna()
        d_valid = stoch_data['d'].dropna()

        if len(k_valid) > 0:
            assert (k_valid >= 0).all()
            assert (k_valid <= 100).all()

        if len(d_valid) > 0:
            assert (d_valid >= 0).all()
            assert (d_valid <= 100).all()

    def test_obv_python(self, sample_ohlcv):
        """Test pure Python OBV implementation."""
        ti = TechnicalIndicators(use_talib=False)
        obv = ti._calculate_obv_python(sample_ohlcv['close'], sample_ohlcv['volume'])

        assert isinstance(obv, pd.Series)
        assert len(obv) == len(sample_ohlcv['close'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
