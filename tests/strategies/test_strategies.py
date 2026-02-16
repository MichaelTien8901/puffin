"""Tests for trading strategies."""

import numpy as np
import pandas as pd
import pytest

from puffin.strategies.momentum import MomentumStrategy
from puffin.strategies.mean_reversion import MeanReversionStrategy
from puffin.strategies.market_making import MarketMakingStrategy
from puffin.strategies.registry import get_strategy, list_strategies


@pytest.fixture
def trending_data():
    """Uptrending price data."""
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    prices = 100 + np.arange(100) * 0.5 + np.random.randn(100) * 0.5
    return pd.DataFrame(
        {
            "Open": prices - 0.2,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": np.random.randint(1000, 5000, 100),
        },
        index=dates,
    )


@pytest.fixture
def mean_reverting_data():
    """Mean-reverting price data oscillating around 100."""
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    prices = 100 + 5 * np.sin(np.linspace(0, 6 * np.pi, 100)) + np.random.randn(100) * 0.3
    return pd.DataFrame(
        {
            "Open": prices - 0.2,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": np.random.randint(1000, 5000, 100),
        },
        index=dates,
    )


class TestMomentumStrategy:
    def test_generates_signals(self, trending_data):
        strategy = MomentumStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(trending_data)
        assert "signal" in signals.columns
        assert "confidence" in signals.columns
        assert len(signals) == len(trending_data)

    def test_buy_signal_in_uptrend(self, trending_data):
        strategy = MomentumStrategy(short_window=10, long_window=30)
        signals = strategy.generate_signals(trending_data)
        # In a strong uptrend, later signals should be positive
        late_signals = signals["signal"].iloc[50:]
        assert (late_signals >= 0).mean() > 0.7

    def test_parameters(self):
        strategy = MomentumStrategy(short_window=5, long_window=20, ma_type="ema")
        params = strategy.get_parameters()
        assert params["short_window"] == 5
        assert params["long_window"] == 20
        assert params["ma_type"] == "ema"

    def test_ema_variant(self, trending_data):
        strategy = MomentumStrategy(short_window=10, long_window=30, ma_type="ema")
        signals = strategy.generate_signals(trending_data)
        assert len(signals) == len(trending_data)


class TestMeanReversionStrategy:
    def test_generates_signals(self, mean_reverting_data):
        strategy = MeanReversionStrategy(window=20)
        signals = strategy.generate_signals(mean_reverting_data)
        assert "signal" in signals.columns
        assert len(signals) == len(mean_reverting_data)

    def test_parameters(self):
        strategy = MeanReversionStrategy(window=30, num_std=2.5)
        params = strategy.get_parameters()
        assert params["window"] == 30
        assert params["num_std"] == 2.5


class TestMarketMakingStrategy:
    def test_generates_signals(self, mean_reverting_data):
        strategy = MarketMakingStrategy(spread_bps=10)
        signals = strategy.generate_signals(mean_reverting_data)
        assert "signal" in signals.columns
        assert len(signals) == len(mean_reverting_data)

    def test_quote_prices(self):
        strategy = MarketMakingStrategy(spread_bps=10)
        quotes = strategy.get_quote_prices(100.0)
        assert quotes["bid"] < 100.0
        assert quotes["ask"] > 100.0
        assert quotes["bid"] < quotes["ask"]

    def test_parameters(self):
        strategy = MarketMakingStrategy(spread_bps=20, max_inventory=50)
        params = strategy.get_parameters()
        assert params["spread_bps"] == 20
        assert params["max_inventory"] == 50


class TestRegistry:
    def test_list_strategies(self):
        strategies = list_strategies()
        assert "momentum" in strategies
        assert "mean_reversion" in strategies
        assert "stat_arb" in strategies
        assert "market_making" in strategies

    def test_get_strategy(self):
        strategy = get_strategy("momentum", short_window=5)
        assert isinstance(strategy, MomentumStrategy)
        assert strategy.get_parameters()["short_window"] == 5

    def test_unknown_strategy(self):
        with pytest.raises(KeyError):
            get_strategy("nonexistent")
